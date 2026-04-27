"""
claim_extractor.py
------------------
Decomposes an LLM answer into atomic, verifiable factual claims.
Each claim asserts exactly one fact with an explicit subject (no pronouns).
"""
from __future__ import annotations

import os
import sys

# Allow running as a script from the src/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass

from llm_client import call_llm_json, EXTRACT_MODEL

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_CLAIMS = 15       # cap — enough to cover all key facts without bloat
MIN_CLAIM_WORDS = 5   # discard trivially short fragments

# ── Prompts ────────────────────────────────────────────────────────────────────

CLAIM_EXTRACTION_SYSTEM_PROMPT = """You are a precise fact-extraction assistant. \
Your job is to decompose a text answer into the most specific, verifiable factual claims.

Each claim must:
- Assert exactly ONE fact
- Be self-contained — include the full subject (do NOT use pronouns like \
"it", "they", "he", "she", "this", "that")
- Be stated as a positive assertion (not a question or hypothesis)
- Be directly extractable from the given answer without inference

Priority: extract claims containing specific, checkable values — \
exact names, numbers, prices, dates, percentages, counts. \
Skip vague or descriptive sentences that cannot be verified against a database.
Limit to the 15 most important claims maximum.

Return ONLY valid JSON with this exact shape:
{"claims": [{"text": "<atomic claim sentence>", "source_span": "<exact substring from answer>"}]}

If the answer contains no verifiable facts, return: {"claims": []}"""

CLAIM_EXTRACTION_USER_PROMPT = """Question: {question}

Answer: {answer}

Extract up to 15 atomic, verifiable factual claims from this answer (prioritise specific numbers, names, dates). Return valid JSON only."""


# ── Dataclass ──────────────────────────────────────────────────────────────────

@dataclass
class Claim:
    text: str          # one-fact sentence with explicit subject
    source_span: str   # substring from the original answer this came from


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_claims(answer: str, question: str) -> list[Claim]:
    """
    Decompose *answer* into a list of atomic verifiable Claim objects.

    Uses call_llm_json() with EXTRACT_MODEL. Falls back to sentence-splitting
    if JSON parsing fails.

    Parameters
    ----------
    answer   : the LLM-generated answer to decompose
    question : the original question (provides context for better extraction)

    Returns
    -------
    List of Claim objects, capped at MAX_CLAIMS.
    """
    prompt = CLAIM_EXTRACTION_USER_PROMPT.format(question=question, answer=answer)

    try:
        data = call_llm_json(prompt, system_prompt=CLAIM_EXTRACTION_SYSTEM_PROMPT)
        raw_claims = data.get("claims", [])
        if not isinstance(raw_claims, list):
            raise ValueError("'claims' is not a list")
    except Exception as exc:
        print(f"  [claim_extractor] JSON extraction failed ({exc}); using sentence fallback.")
        return _sentence_fallback(answer)

    claims: list[Claim] = []
    for item in raw_claims[:MAX_CLAIMS]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        span = str(item.get("source_span", "")).strip()
        if len(text.split()) >= MIN_CLAIM_WORDS:
            claims.append(Claim(text=text, source_span=span or text))

    # If the LLM returned nothing useful, fall back
    return claims if claims else _sentence_fallback(answer)


# ── Fallback ───────────────────────────────────────────────────────────────────

def _sentence_fallback(answer: str) -> list[Claim]:
    """
    Split the answer on sentence boundaries and treat each sentence as a claim.
    Used when the LLM fails to return valid JSON.
    """
    # Normalize common sentence breaks
    normalized = answer.replace(".\n", ". ").replace("!\n", ". ").replace("?\n", ". ")
    parts = [s.strip() for s in normalized.split(". ")]
    claims: list[Claim] = []
    for part in parts:
        text = part.strip().rstrip(".!?")
        if len(text.split()) >= MIN_CLAIM_WORDS:
            claims.append(Claim(text=text, source_span=text))
    return claims[:MAX_CLAIMS]


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_answer = (
        "The CEO of NovaAI is Dr. Mara Chen, who co-founded the company. "
        "NovaAI was founded in 2021 and is headquartered in San Francisco. "
        "The company has raised $210M in total funding and reached a $1.4B valuation "
        "through its Series C round in November 2025."
    )
    test_question = "Tell me about NovaAI's leadership and funding."

    print("[smoke test] Extracting claims from test answer...")
    claims = extract_claims(test_answer, test_question)
    print(f"  Extracted {len(claims)} claims:")
    for i, c in enumerate(claims, 1):
        print(f"  {i:2d}. {c.text}")
