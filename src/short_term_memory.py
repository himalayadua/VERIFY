"""
short_term_memory.py
--------------------
Short-term (per-question, multi-trial) scratchpad analogous to the ReAct
scratchpad in Reflexion: recent trajectory text for LAST_ATTEMPT strategies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Scratchpad:
    """Accumulates per-trial snapshots for one question."""

    question: str
    trials: list[dict[str, Any]] = field(default_factory=list)
    _current_trial: dict[str, Any] | None = None

    def start_trial(self, trial_idx: int) -> None:
        self._current_trial = {
            "trial_idx": trial_idx,
            "raw_answer": "",
            "claims": [],
            "verification_summary": [],
            "repaired_answer": "",
            "had_hallucinations": False,
            "stop_check_passed": False,
        }

    def record_raw_answer(self, text: str) -> None:
        if self._current_trial is not None:
            self._current_trial["raw_answer"] = text

    def record_claims(self, claims: list[str]) -> None:
        if self._current_trial is not None:
            self._current_trial["claims"] = list(claims)

    def record_verifications(self, rows: list[dict[str, Any]]) -> None:
        if self._current_trial is not None:
            self._current_trial["verification_summary"] = list(rows)

    def record_repair(self, repaired: str, had_hallucinations: bool) -> None:
        if self._current_trial is not None:
            self._current_trial["repaired_answer"] = repaired
            self._current_trial["had_hallucinations"] = had_hallucinations

    def record_stop_check(self, passed: bool) -> None:
        if self._current_trial is not None:
            self._current_trial["stop_check_passed"] = passed

    def end_trial(self) -> None:
        if self._current_trial is not None:
            self.trials.append(self._current_trial)
            self._current_trial = None

    def format_for_prompt(self, last_n_trials: int = 1) -> str:
        """Compact text of the last N completed trials for LAST_ATTEMPT injection."""
        if not self.trials:
            return ""
        chunk = self.trials[-max(1, last_n_trials) :]
        lines: list[str] = []
        for t in chunk:
            idx = t.get("trial_idx", "?")
            lines.append(f"--- Trial {idx} ---")
            lines.append(f"Raw answer:\n{t.get('raw_answer', '')}")
            claims = t.get("claims") or []
            if claims:
                lines.append("Claims:")
                for i, c in enumerate(claims, 1):
                    lines.append(f"  {i}. {c}")
            vrows = t.get("verification_summary") or []
            if vrows:
                lines.append("Verifications:")
                for vr in vrows:
                    lines.append(
                        f"  [{vr.get('verdict', '?')}] \"{vr.get('claim', '')[:120]}\""
                    )
            lines.append(f"Had contradictions: {t.get('had_hallucinations', False)}")
            lines.append(f"Stop check passed: {t.get('stop_check_passed', False)}")
            lines.append(f"Final trial output:\n{t.get('repaired_answer', '')}")
        return "\n".join(lines).strip()
