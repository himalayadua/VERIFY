"""
verifier.py
-----------
Verifies individual claims against the KG by:
  1. Retrieving relevant triples via entity_linker.find_relevant_triples_multihop()
  2. Calling an LLM to produce a SUPPORTED / CONTRADICTED / UNVERIFIABLE verdict
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, field
from enum import Enum

from claim_extractor import Claim
from entity_linker import find_relevant_triples_multihop
import llm_client as _lc


# ── Types ──────────────────────────────────────────────────────────────────────

class Verdict(str, Enum):
    SUPPORTED     = "SUPPORTED"
    CONTRADICTED  = "CONTRADICTED"
    UNVERIFIABLE  = "UNVERIFIABLE"


@dataclass
class VerificationResult:
    claim: str
    verdict: Verdict
    confidence: float                              # 0.0 – 1.0
    evidence_triples: list[dict] = field(default_factory=list)
    reasoning: str = ""


# ── Prompts ────────────────────────────────────────────────────────────────────

VERIFICATION_SYSTEM_PROMPT = """You are a rigorous fact-checking assistant for a company knowledge base.

You will be given a factual claim and relevant knowledge graph triples from NovaAI's internal database.
Evaluate whether the claim is supported, contradicted, or unverifiable based ONLY on the provided triples.

Definitions:
- SUPPORTED: The claim is directly and unambiguously confirmed by one or more triples.
- CONTRADICTED: The claim conflicts with information in one or more triples.
- UNVERIFIABLE: The triples do not touch the subject/attribute the claim is about.

Fire CONTRADICTED whenever ANY of these patterns applies (do NOT retreat to \
UNVERIFIABLE just because the claim mentions a different name than the triple):

1) Wrong filler for a unique role/attribute.
   Claim: "NovaAI's General Counsel is Priya Mehta."
   Triple: "Derek Shin | title | General Counsel"
   → CONTRADICTED — the General-Counsel slot is filled by someone else.

2) Wrong filler for a team-lead / ownership / flagship-product slot.
   Claim: "The flagship product is NovaCore, with PM owner Sarah Lin."
   Triples: "NovaPilot | is_flagship_product_of | NovaAI", \
"NovaPilot | pm_owner | Fatima Al-Rashid"
   → CONTRADICTED — the flagship and the PM owner are different.

3) Wrong numeric / money / date / percentage / tier value.
   Claim: "The Starter plan costs $5,000/month."
   Triple: "NovaPilot Starter | price | $2,500/month"
   → CONTRADICTED ($2,500 ≠ $5,000).

4) Claim denies a policy / platform / product that the KG records as \
existing.
   Claim: "NovaAI does not have an airfare booking policy."
   Triple: "NovaAI | airfare_booking_platform | TravelPerk"
   → CONTRADICTED — the platform (and therefore the policy) exists.

5) Claim denies the existence of an entity/product that appears in the KG.
   Claim: "NovaAI does not offer a product called NovaPilot."
   Triple: "NovaPilot | is_flagship_product_of | NovaAI"
   → CONTRADICTED — NovaPilot is recorded as NovaAI's product.

Keep UNVERIFIABLE only for true gaps — the triples do not cover the \
subject, the relation, or the attribute the claim is actually about. A \
different person/number/product in the SAME slot is NOT a gap; it is a \
contradiction.

Other rules:
- Do NOT use any external knowledge. Judge ONLY on the provided triples.
- If the claim bundles several facts and at least one conflicts with a \
triple, return CONTRADICTED (name the conflict in reasoning).
- If all named facts are consistent but the triples confirm only some and \
say nothing about the rest, return UNVERIFIABLE.
- Be strict on numbers/dates: even small differences ($2,500 vs $5,000, \
2019 vs 2021) are CONTRADICTED.

Return ONLY valid JSON with this exact shape:
{"verdict": "SUPPORTED|CONTRADICTED|UNVERIFIABLE", "confidence": <float 0.0-1.0>, \
"reasoning": "<one sentence explanation>"}"""

# Appended to the system prompt only when verify_claim(..., v2p_epistemic_rules=True).
# V1 / V1RAG / V2 do not set this flag.
V2P_EPISTEMIC_VERIFIER_SUFFIX = """
Additional rules for this run (V2P mode):
- A claim is NOT CONTRADICTED just because the provided triples are incomplete or
  the phrasing differs. CONTRADICTED requires an explicit conflict with a triple
  (wrong number, wrong person, denial of a recorded fact, etc.).
- If the claim only refuses, declines, or says the information is not in the KB
  without asserting a specific atomic fact (e.g. a price, name, or date) that
  a triple can contradict, return UNVERIFIABLE — not CONTRADICTED.
- Refusal- or process-only meta-claims (e.g. "I will not", "I cannot list X")
  should be UNVERIFIABLE if no triple refutes a concrete asserted fact; use
  CONTRADICTED only when a triple actually conflicts with a stated value."""

VERIFICATION_USER_PROMPT = """Claim: {claim_text}

Relevant Knowledge Graph Triples:
{triples_formatted}

Evaluate the claim based ONLY on these triples. Return JSON."""

_VALID_VERDICTS = {v.value for v in Verdict}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_triples(triples: list[dict]) -> str:
    """Format a list of KG triples for insertion into the verification prompt."""
    lines = []
    for t in triples:
        score = t.get("score", "?")
        lines.append(
            f"- {t['subject']} | {t['relation']} | {t['object']}  (relevance score: {score})"
        )
    return "\n".join(lines) if lines else "(none)"


# ── Public API ─────────────────────────────────────────────────────────────────

def verify_claim(
    claim: Claim,
    triples: list[dict],
    top_n: int = 10,
    method: str = "llm",
    reflections_block: str = "",
    v2p_epistemic_rules: bool = False,
    seed_entities: list[str] | None = None,
) -> VerificationResult:
    """
    Verify a single claim against the KG using multi-hop retrieval.

    Steps:
      1. Retrieve relevant triples via multi-hop entity linking.
      2. If no triples found: return UNVERIFIABLE without an LLM call.
      3. Otherwise, call the LLM to produce a verdict.

    Parameters
    ----------
    claim               : Claim object with .text attribute
    triples             : full list of KG triples loaded from kg_triples.json
    top_n               : max triples to retrieve per claim
    method              : "llm" (Anthropic entity linking + fuzzy fallback) or "fuzzy"
    reflections_block   : optional text prepended to the verification system prompt (default: no change)
    v2p_epistemic_rules : if True, append stricter epistemic rules (V2P/V2P-RAG only; default: off)
    seed_entities       : optional list of entity name hints (from RGR reflections) to boost retrieval

    Returns
    -------
    VerificationResult
    """
    evidence = find_relevant_triples_multihop(
        claim.text, triples, top_n=top_n, method=method, max_hops=2,
        seed_entities=seed_entities,
    )

    # Short-circuit: no relevant triples found
    if not evidence:
        return VerificationResult(
            claim=claim.text,
            verdict=Verdict.UNVERIFIABLE,
            confidence=0.0,
            evidence_triples=[],
            reasoning="No relevant KG triples found for this claim.",
        )

    prompt = VERIFICATION_USER_PROMPT.format(
        claim_text=claim.text,
        triples_formatted=_format_triples(evidence),
    )

    sys_prompt = VERIFICATION_SYSTEM_PROMPT
    if reflections_block and str(reflections_block).strip():
        sys_prompt = f"{reflections_block.strip()}\n\n{VERIFICATION_SYSTEM_PROMPT}"
    if v2p_epistemic_rules:
        sys_prompt = f"{sys_prompt}\n\n{V2P_EPISTEMIC_VERIFIER_SUFFIX.strip()}"

    try:
        data = _lc.call_llm_json(prompt, model=_lc.ACTION_MODEL, system_prompt=sys_prompt)
        raw_verdict = str(data.get("verdict", "UNVERIFIABLE")).strip().upper()
        verdict = Verdict(raw_verdict) if raw_verdict in _VALID_VERDICTS else Verdict.UNVERIFIABLE
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0.5))))
        reasoning = str(data.get("reasoning", "")).strip()
    except Exception as exc:
        verdict = Verdict.UNVERIFIABLE
        confidence = 0.0
        reasoning = f"Verification call failed: {exc}"

    return VerificationResult(
        claim=claim.text,
        verdict=verdict,
        confidence=confidence,
        evidence_triples=evidence,
        reasoning=reasoning,
    )


def verify_claims(
    claims: list[Claim],
    triples: list[dict],
    reflections_block: str = "",
    v2p_epistemic_rules: bool = False,
    seed_entities: list[str] | None = None,
) -> list[VerificationResult]:
    """Verify a list of claims sequentially. Returns one result per claim."""
    results = []
    for i, claim in enumerate(claims, 1):
        print(f"    Verifying claim {i}/{len(claims)}: \"{claim.text[:60]}...\"" if len(claim.text) > 60
              else f"    Verifying claim {i}/{len(claims)}: \"{claim.text}\"")
        results.append(
            verify_claim(
                claim,
                triples,
                reflections_block=reflections_block,
                v2p_epistemic_rules=v2p_epistemic_rules,
                seed_entities=seed_entities,
            )
        )
    return results


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from entity_linker import load_kg

    kg = load_kg()
    print(f"[smoke test] Loaded {len(kg)} KG triples\n")

    test_cases = [
        ("The CEO of NovaAI is Dr. Mara Chen.",           "SUPPORTED"),
        ("NovaAI was founded in 2019.",                    "CONTRADICTED"),
        ("The NovaPilot Starter plan costs $5,000/month.", "CONTRADICTED"),
        ("NovaPilot Growth is priced at $8,000/month.",    "SUPPORTED"),
        ("NovaAI has a partnership with Microsoft.",       "UNVERIFIABLE"),
    ]

    print(f"{'CLAIM':<55} {'EXPECTED':<15} {'GOT':<15} {'MATCH'}")
    print("-" * 100)
    for text, expected in test_cases:
        claim = Claim(text=text, source_span=text)
        result = verify_claim(claim, kg)
        match = "✓" if result.verdict.value == expected else "✗"
        print(f"{text:<55} {expected:<15} {result.verdict.value:<15} {match}")
        print(f"  Reasoning: {result.reasoning}")