"""
reflexion_strategies.py
-----------------------
ReflexionStrategy enum and prompt-formatting helpers aligned with the
Reflexion paper / hotpotqa_runs/agents.py (Shinn et al., NeurIPS 2023).
"""
from __future__ import annotations

from enum import Enum


class ReflexionStrategy(str, Enum):
    """
    NONE: no distilled reflection memory between trials
    LAST_ATTEMPT: inject prior trial scratchpad (short-term trajectory)
    REFLEXION: inject verbal reflections from long-term memory
    LAST_ATTEMPT_AND_REFLEXION: both scratchpad and reflections
    """

    NONE = "none"
    LAST_ATTEMPT = "last_attempt"
    REFLEXION = "reflexion"
    LAST_ATTEMPT_AND_REFLEXION = "last_attempt_and_reflexion"


# Adapted from misc/sample/reflexion-main/hotpotqa_runs/prompts.py
REFLECTION_HEADER = (
    "You have attempted to answer the following question before and failed. "
    "The following reflection(s) give a plan to avoid failing in the same way. "
    "Use them to improve your strategy.\n"
)

REFLECTION_AFTER_LAST_TRIAL_HEADER = (
    "The following reflection(s) give a plan to avoid failing in the same way. "
    "Use them to improve your strategy.\n"
)

LAST_TRIAL_HEADER = (
    "You have attempted to answer the following question before and failed. "
    "Below is the last trial you attempted.\n"
)


def format_reflections(reflections: list[str], header: str = REFLECTION_HEADER) -> str:
    """Format distilled reflections for injection into the actor prompt."""
    if not reflections:
        return ""
    cleaned = [r.strip() for r in reflections if r and str(r).strip()]
    if not cleaned:
        return ""
    return header + "Reflections:\n- " + "\n- ".join(cleaned) + "\n"


def format_last_attempt(question: str, scratchpad: str, header: str = LAST_TRIAL_HEADER) -> str:
    """Format the previous trial scratchpad (short-term memory block)."""
    sp = (scratchpad or "").strip()
    if not sp:
        return ""
    return (
        f"{header}"
        f"Question: {question}\n"
        f"{sp}\n"
        "(END PREVIOUS TRIAL)\n"
    )


def parse_inject_into(values: list[str] | None) -> set[str]:
    """
    Parse CLI-style inject list: answer, verifier, repairer, all.
    'all' expands to all three stages.
    """
    if not values:
        return {"answer"}
    out: set[str] = set()
    for v in values:
        v = (v or "").strip().lower()
        if v == "all":
            out.update({"answer", "verifier", "repairer"})
        elif v in ("answer", "verifier", "repairer"):
            out.add(v)
    return out if out else {"answer"}
