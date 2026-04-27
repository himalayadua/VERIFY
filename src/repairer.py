"""
repairer.py
-----------
Rewrites an LLM answer to fix any CONTRADICTED claims using correct KG evidence.
SUPPORTED claims are kept verbatim. UNVERIFIABLE claims are softened.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, field

from verifier import VerificationResult, Verdict
from llm_client import call_llm, ACTION_MODEL


# ── Types ──────────────────────────────────────────────────────────────────────

@dataclass
class RepairResult:
    repaired_answer: str
    changes_made: list[str] = field(default_factory=list)   # contradicted claim texts
    had_hallucinations: bool = False


# ── Prompts ────────────────────────────────────────────────────────────────────

REPAIR_SYSTEM_PROMPT = """You are an answer correction assistant for an internal company AI assistant.

You will be given an original answer and fact-checking results for each of its claims.
Rewrite the answer to be factually accurate.

Rules:
- Keep all SUPPORTED claims exactly as written — do not paraphrase them.
- For CONTRADICTED claims: replace the wrong value with the CORRECT VALUE from the \
[KG EVIDENCE] block. The object field of the most relevant triple is the correct value \
to use. NEVER say "not specified" or "unknown" if a KG triple is provided — use its value.
- For UNVERIFIABLE claims: keep them but soften with "Based on available information, ...".
- Maintain the original answer's tone, structure, and approximate length.
- Output only the corrected answer text — no preamble, no explanation, no meta-commentary."""

# Appended when repair_answer(..., v2p_meta_repair_rules=True) — V2P / V2P-RAG only.
V2P_META_REPAIR_SUFFIX = """
- For claims marked UNVERIFIABLE that are refusals or process-only (no concrete wrong
  number/name/date), do not invent corrections or replace them with made-up values;
  keep a brief uncertainty phrase if needed."""

REPAIR_USER_PROMPT = """Question: {question}

Original Answer:
{original_answer}

Fact-Checking Results:
{verification_summary}

Rewrite the answer to be accurate using the correct information from the fact-checking results. \
Output only the corrected answer."""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_verification_summary(results: list[VerificationResult]) -> str:
    """Render all verification results into a structured string for the repair prompt.

    For CONTRADICTED claims, all evidence triples are shown with an explicit
    [KG EVIDENCE] block so the repair model knows exactly what value to substitute.
    For SUPPORTED/UNVERIFIABLE, only the top triple is shown for brevity.
    """
    lines = []
    for r in results:
        if r.verdict == Verdict.CONTRADICTED:
            # Show ALL triples for contradicted claims — repair model must pick correct value
            if r.evidence_triples:
                triples_str = "\n".join(
                    f"    {t['subject']} | {t['relation']} | {t['object']}  (score: {t.get('score','?')})"
                    for t in r.evidence_triples
                )
                kg_block = f"  [KG EVIDENCE — use these values to correct the claim]:\n{triples_str}"
            else:
                kg_block = "  [KG EVIDENCE]: none found — remove or mark this claim as uncertain"

            lines.append(
                f'Claim: "{r.claim}"\n'
                f"  Verdict  : CONTRADICTED\n"
                f"{kg_block}\n"
                f"  Reasoning: {r.reasoning}"
            )
        else:
            # SUPPORTED / UNVERIFIABLE — top triple is sufficient context
            if r.evidence_triples:
                t = r.evidence_triples[0]
                evidence_str = f"{t['subject']} | {t['relation']} | {t['object']}"
            else:
                evidence_str = "none found"

            lines.append(
                f'Claim: "{r.claim}"\n'
                f"  Verdict  : {r.verdict.value}\n"
                f"  Evidence : {evidence_str}\n"
                f"  Reasoning: {r.reasoning}"
            )
    return "\n\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────────

def repair_answer(
    question: str,
    original_answer: str,
    verification_results: list[VerificationResult],
    reflections_block: str = "",
    v2p_meta_repair_rules: bool = False,
) -> RepairResult:
    """
    Produce a corrected version of *original_answer* using verification results.

    Early exit: if no CONTRADICTED claims exist, returns the original answer unchanged.

    Parameters
    ----------
    question             : the original question (gives the rewriter context)
    original_answer      : the raw LLM answer before verification
    verification_results : list of VerificationResult from verify_claims()
    reflections_block    : optional text prepended to the repair system prompt (default: no change)
    v2p_meta_repair_rules : if True, append V2P-only repair heuristics (default: off for V1/V1RAG/V2)

    Returns
    -------
    RepairResult with the corrected answer, list of changed claims, and a hallucination flag.
    """
    contradicted = [r for r in verification_results if r.verdict == Verdict.CONTRADICTED]

    # Nothing to fix
    if not contradicted:
        return RepairResult(
            repaired_answer=original_answer,
            changes_made=[],
            had_hallucinations=False,
        )

    summary = _format_verification_summary(verification_results)
    prompt = REPAIR_USER_PROMPT.format(
        question=question,
        original_answer=original_answer,
        verification_summary=summary,
    )

    sys_prompt = REPAIR_SYSTEM_PROMPT
    if reflections_block and str(reflections_block).strip():
        sys_prompt = f"{reflections_block.strip()}\n\n{REPAIR_SYSTEM_PROMPT}"
    if v2p_meta_repair_rules:
        sys_prompt = f"{sys_prompt}\n\n{V2P_META_REPAIR_SUFFIX.strip()}"

    repaired = call_llm(
        prompt,
        system_prompt=sys_prompt,
        temperature=0.3,   # lower temperature for more faithful correction
        max_tokens=1024,
    )

    return RepairResult(
        repaired_answer=repaired,
        changes_made=[r.claim for r in contradicted],
        had_hallucinations=True,
    )


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from verifier import VerificationResult, Verdict

    # Simulate verification results: one wrong fact, one correct fact
    fake_results = [
        VerificationResult(
            claim="NovaAI was founded in 2019.",
            verdict=Verdict.CONTRADICTED,
            confidence=0.95,
            evidence_triples=[{
                "subject": "NovaAI", "relation": "founded_in_year",
                "object": "2021", "score": 90,
            }],
            reasoning="The KG states NovaAI was founded in 2021, not 2019.",
        ),
        VerificationResult(
            claim="The CEO of NovaAI is Dr. Mara Chen.",
            verdict=Verdict.SUPPORTED,
            confidence=0.99,
            evidence_triples=[{
                "subject": "NovaAI", "relation": "has_ceo",
                "object": "Dr. Mara Chen", "score": 95,
            }],
            reasoning="Directly confirmed by the has_ceo triple.",
        ),
    ]

    result = repair_answer(
        question="Tell me about NovaAI's founding.",
        original_answer="NovaAI was founded in 2019. The CEO of NovaAI is Dr. Mara Chen.",
        verification_results=fake_results,
    )
    print("[smoke test] Repaired answer:")
    print(f"  {result.repaired_answer}")
    print(f"\nChanges made: {result.changes_made}")
    print(f"Had hallucinations: {result.had_hallucinations}")
