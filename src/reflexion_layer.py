"""
reflexion_layer.py
------------------
Reflexion-style verbal memory for the V2 pipeline.

Adapted from the Reflexion NeurIPS 2023 paper (Shinn et al.) using patterns from:
  - hotpotqa_runs/agents.py  → format_reflections(), CoTAgent.reflect()
  - alfworld_runs/generate_reflections.py → memory[-3:] sliding window
  - hotpotqa_runs/prompts.py → COT_SIMPLE_REFLECT_INSTRUCTION, REFLECTION_HEADER

Key concepts:
  - ReflexionMemory: a sliding-window list of up to max_reflections verbal lessons
  - generate_reflection(): asks the LLM to diagnose why an answer was wrong
  - Reflections are injected into subsequent answer prompts to guide better behaviour
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, field

from verifier import VerificationResult, Verdict
import llm_client as _lc
from v2p_text import sanitize_reflection_text


# ── Prompt constants ───────────────────────────────────────────────────────────
# Adapted from hotpotqa_runs/prompts.py REFLECTION_HEADER and COT_SIMPLE_REFLECT_INSTRUCTION

REFLECTION_HEADER = (
    "You have previously made factual errors about NovaAI. "
    "Apply these lessons to improve your accuracy on this question.\n"
)

REFLECTION_SYSTEM_PROMPT = """You are an advanced reasoning agent that improves based on self-reflection.

You will be given a question, an original answer that contained factual errors, \
the specific errors found (with the correct facts from the KG), and the corrected answer.

In 2-3 sentences, diagnose the root cause of the errors and devise a concise, \
high-level strategy to avoid the same mistakes. Focus on:
- What category of information was hallucinated \
(e.g., pricing, names, dates, headcounts, policy thresholds, percentages)
- Why the model likely made the error \
(e.g., confusing similar entities, guessing specific numbers, mixing up tiers)
- A concrete strategy to be more careful next time

Use complete sentences. Be specific — avoid generic advice like "be more careful".

IMPORTANT: End your reflection with exactly one of these tagged lines:
  EXACT CORRECT VALUES: [comma-separated concrete values, names, or dates from the KG evidence above]
  LOOK FOR: [comma-separated description of what facts are still missing or uncertain]
Use "EXACT CORRECT VALUES" when the KG evidence clearly reveals what the right answer should be.
Use "LOOK FOR" when you know what information is missing but its exact value is unclear."""

REFLECTION_USER_PROMPT = """Question: {question}

Original Answer (contained errors): {original_answer}

Errors Found (claim vs. correct KG fact):
{error_summary}

Corrected Answer: {repaired_answer}

{existing_reflections_block}

Reflection:"""

# V2P paper-aligned Me-fail path (no CONTRADICTED required)
REFLECTION_ME_FAIL_SYSTEM_PROMPT = """You are an advanced reasoning agent that improves based on self-reflection.

The previous answer did not satisfy the task evaluator: required key facts were missing \
from the final text, or the answer was incomplete relative to the question.

You will see the question, the answers, a list of key facts that were still missing, \
and a short summary of claim-level verification (SUPPORTED / CONTRADICTED / UNVERIFIABLE).

In 2-3 sentences, diagnose why the answer failed the task (e.g. omission, refusal when \
facts were available, wrong focus) and give a concrete strategy for the next attempt — \
e.g. explicitly enumerate required entities, use only stated KG-backed facts, avoid \
claiming the knowledge base is empty when verification shows supporting triples.

Use complete sentences. Be specific — avoid generic advice like "be more careful".

IMPORTANT: End your reflection with exactly one of these tagged lines:
  EXACT CORRECT VALUES: [comma-separated concrete values, names, or dates that should appear in the answer]
  LOOK FOR: [comma-separated description of what facts need to be retrieved or stated]
Use "EXACT CORRECT VALUES" when you can infer the correct answer from the verification summary.
Use "LOOK FOR" when you know what information is missing but not its exact value."""

REFLECTION_ME_FAIL_USER_PROMPT = """Question: {question}

Original Answer: {original_answer}

Final Answer (after any repair): {repaired_answer}

Key facts still missing from the final answer (evaluator signal):
{missing_facts_block}

Key fact recall (fraction satisfied): {kfr:.3f}

Claim verification summary:
{verification_summary}

{existing_reflections_block}

Reflection:"""


# ── ReflexionMemory ────────────────────────────────────────────────────────────

@dataclass
class ReflexionMemory:
    """
    Sliding window of verbal reflections (up to max_reflections entries).

    Mirrors the memory[-3:] pattern from alfworld_runs/generate_reflections.py.
    Used by V2Pipeline to inject past lessons into answer generation prompts.
    """
    max_reflections: int = 3
    reflections: list[str] = field(default_factory=list)

    def add_reflection(self, reflection: str) -> None:
        """
        Append a new reflection and enforce the sliding window cap.
        Silently ignores empty strings.
        """
        if self.max_reflections <= 0:
            return
        cleaned = reflection.strip()
        if cleaned:
            self.reflections.append(cleaned)
            # Sliding window: keep only the most recent max_reflections entries
            # Mirrors: memory[-3:] from alfworld_runs/generate_reflections.py
            self.reflections = self.reflections[-self.max_reflections:]

    def format_for_prompt(self) -> str:
        """
        Return a formatted string for injection into the answer system prompt.
        Returns empty string when no reflections exist.

        Adapted from format_reflections() in hotpotqa_runs/agents.py.
        """
        if not self.reflections:
            return ""
        bullets = "\n".join(f"- {r}" for r in self.reflections)
        return (
            "[PRIOR SESSION LEARNING — APPLY THESE LESSONS]\n"
            f"{bullets}\n"
            "[END PRIOR SESSION LEARNING]"
        )

    def has_reflections(self) -> bool:
        """Return True if at least one reflection has been stored."""
        return bool(self.reflections)

    def clear(self) -> None:
        """Reset memory. Call between experiments, not between questions."""
        self.reflections = []


# ── Reflection generation ──────────────────────────────────────────────────────

def generate_reflection(
    question: str,
    original_answer: str,
    verification_results: list[VerificationResult],
    repaired_answer: str,
    existing_reflections: list[str],
    few_shot_examples: list[str] | None = None,
    sanitize_output: bool = False,
) -> str:
    """
    Generate a verbal reflection explaining why the original answer was wrong
    and what strategy to use next time.

    Returns an empty string if there are no CONTRADICTED claims (nothing to reflect on).

    Parameters
    ----------
    question               : the original question asked
    original_answer        : the LLM's raw answer (before repair)
    verification_results   : results from verify_claims()
    repaired_answer        : the corrected answer from repair_answer()
    existing_reflections   : current session memory (for context)
    few_shot_examples      : optional in-domain reflection exemplars (prepended to system prompt)
    sanitize_output        : if True, strip model junk from output (V2P/V2P-RAG; default: off for V2)

    Returns
    -------
    Reflection string, or "" if no contradictions found.
    """
    contradicted = [r for r in verification_results if r.verdict == Verdict.CONTRADICTED]
    if not contradicted:
        return ""

    # Build error summary: claimed fact vs. correct KG fact
    error_lines = []
    for r in contradicted:
        if r.evidence_triples:
            t = r.evidence_triples[0]
            correct = f"{t['subject']} | {t['relation']} | {t['object']}"
        else:
            correct = "no KG triple found (unknown correct value)"
        error_lines.append(
            f'- Claimed : "{r.claim}"\n'
            f"  Actual   : {correct}"
        )
    error_summary = "\n".join(error_lines)

    # Build existing-reflections context block
    if existing_reflections:
        bullets = "\n".join(f"- {r}" for r in existing_reflections)
        existing_reflections_block = (
            f"Previous reflections from this session (for context — focus on the NEW errors above):\n"
            f"{bullets}"
        )
    else:
        existing_reflections_block = ""

    prompt = REFLECTION_USER_PROMPT.format(
        question=question,
        original_answer=original_answer,
        error_summary=error_summary,
        repaired_answer=repaired_answer,
        existing_reflections_block=existing_reflections_block,
    )

    few_block = ""
    if few_shot_examples:
        lines = [s.strip() for s in few_shot_examples if s and str(s).strip()]
        if lines:
            few_block = (
                "Here are examples of concise, actionable reflections on NovaAI factual errors.\n"
                "(END OF EXAMPLES)\n\n"
            )
            few_block += "\n".join(f"Example reflection: {line}" for line in lines) + "\n\n"

    system_prompt = (
        f"{few_block}{REFLECTION_SYSTEM_PROMPT}" if few_block else REFLECTION_SYSTEM_PROMPT
    )

    reflection = _lc.call_llm(
        prompt,
        model=_lc.REFLECT_MODEL,
        system_prompt=system_prompt,
        temperature=0.4,
        max_tokens=384,
    )
    out = reflection.strip()
    if sanitize_output:
        out = sanitize_reflection_text(out)
    return out


def _format_verification_summary_brief(results: list[VerificationResult], max_claims: int = 12) -> str:
    lines: list[str] = []
    for r in results[:max_claims]:
        lines.append(
            f"- [{r.verdict.value}] \"{r.claim[:120]}{'…' if len(r.claim) > 120 else ''}\""
            f" — {r.reasoning[:100]}{'…' if len(r.reasoning) > 100 else ''}"
        )
    if len(results) > max_claims:
        lines.append(f"... ({len(results) - max_claims} more claims omitted)")
    return "\n".join(lines) if lines else "(no claims)"


def generate_reflection_me_fail(
    question: str,
    original_answer: str,
    repaired_answer: str,
    verification_results: list[VerificationResult],
    missing_key_facts: list[str],
    key_fact_recall: float,
    existing_reflections: list[str],
    few_shot_examples: list[str] | None = None,
    sanitize_output: bool = False,
) -> str:
    """
    Verbal reflection when the task evaluator (Me) failed but there may be no CONTRADICTED claims.

    Used by V2PaperPipeline paper-Me mode only; V2 continues to use generate_reflection().
    """
    if not missing_key_facts:
        return ""

    missing_block = "\n".join(f"- {f}" for f in missing_key_facts)
    ver_summary = _format_verification_summary_brief(verification_results)

    if existing_reflections:
        bullets = "\n".join(f"- {r}" for r in existing_reflections)
        existing_reflections_block = (
            "Previous reflections from this session (for context — focus on the NEW gap above):\n"
            f"{bullets}"
        )
    else:
        existing_reflections_block = ""

    prompt = REFLECTION_ME_FAIL_USER_PROMPT.format(
        question=question,
        original_answer=original_answer,
        repaired_answer=repaired_answer,
        missing_facts_block=missing_block,
        kfr=key_fact_recall,
        verification_summary=ver_summary,
        existing_reflections_block=existing_reflections_block,
    )

    few_block = ""
    if few_shot_examples:
        lines = [s.strip() for s in few_shot_examples if s and str(s).strip()]
        if lines:
            few_block = (
                "Here are examples of concise, actionable reflections.\n"
                "(END OF EXAMPLES)\n\n"
            )
            few_block += "\n".join(f"Example reflection: {line}" for line in lines) + "\n\n"

    system_prompt = (
        f"{few_block}{REFLECTION_ME_FAIL_SYSTEM_PROMPT}"
        if few_block
        else REFLECTION_ME_FAIL_SYSTEM_PROMPT
    )

    reflection = _lc.call_llm(
        prompt,
        model=_lc.REFLECT_MODEL,
        system_prompt=system_prompt,
        temperature=0.4,
        max_tokens=384,
    )
    out = reflection.strip()
    if sanitize_output:
        out = sanitize_reflection_text(out)
    return out


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[smoke test] Testing ReflexionMemory sliding window...")
    memory = ReflexionMemory(max_reflections=3)

    for i in range(1, 5):
        memory.add_reflection(f"Lesson {i}: Be careful about fact category {i}.")
        print(f"  After adding lesson {i}: {len(memory.reflections)} stored "
              f"(oldest: '{memory.reflections[0][:30]}...')")

    print(f"\n  Final count: {len(memory.reflections)} (expected 3)")
    assert len(memory.reflections) == 3, "Sliding window failed!"
    assert "Lesson 4" in memory.reflections[-1], "Wrong entry kept!"

    print("\n[smoke test] format_for_prompt() output:")
    print(memory.format_for_prompt())
    print("\n✓ ReflexionMemory smoke test passed.")
