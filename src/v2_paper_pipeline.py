"""
v2_paper_pipeline.py
--------------------
V2P: Reflexion-paper-faithful variant — multi-trial per question, optional
LAST_ATTEMPT / REFLEXION / combined strategies, configurable reflection
injection into answer / verifier / repairer, and short-term scratchpad.
"""
from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from claim_extractor import Claim, extract_claims
from llm_client import call_llm, ACTION_MODEL
from metrics import missing_key_facts, task_passes_kfr
from pipeline import ANSWER_SYSTEM_PROMPT_BASE, PipelineResult, _serialise_vr
from reflexion_layer import ReflexionMemory, generate_reflection, generate_reflection_me_fail
from reflexion_strategies import (
    REFLECTION_AFTER_LAST_TRIAL_HEADER,
    ReflexionStrategy,
    format_last_attempt,
    format_reflections,
    parse_inject_into,
)
from repairer import RepairResult, repair_answer
from short_term_memory import Scratchpad
from verifier import VerificationResult, Verdict, verify_claims


def _default_stop_check(vr_list: list[VerificationResult]) -> bool:
    return not any(r.verdict == Verdict.CONTRADICTED for r in vr_list)


def _effective_me_mode(me_context: dict[str, Any] | None) -> str:
    if not me_context:
        return "legacy"
    mode = str(me_context.get("me_mode") or "legacy").lower().strip()
    key_facts = me_context.get("key_facts") or []
    if mode in ("kfr", "kfr_and_no_contradiction") and not key_facts:
        return "legacy"
    return mode


def _eval_paper_me(
    repaired_answer: str,
    vr_list: list[VerificationResult],
    me_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Evaluator (Me) state: kfr, me_pass, effective mode, has_contradicted.
    legacy → me_pass means KG-consistent (no CONTRADICTED), same as _default_stop_check.
    """
    has_c = any(r.verdict == Verdict.CONTRADICTED for r in vr_list)
    mode = _effective_me_mode(me_context)
    key_facts: list[str] = list(me_context.get("key_facts") or []) if me_context else []
    thresh = float((me_context or {}).get("kfr_threshold", 0.8))

    kfr, kfr_ok = task_passes_kfr(repaired_answer, key_facts, thresh)

    if mode == "legacy":
        return {
            "me_mode": "legacy",
            "kfr": kfr,
            "me_pass": not has_c,
            "has_contradicted": has_c,
        }
    if mode == "kfr":
        return {"me_mode": "kfr", "kfr": kfr, "me_pass": kfr_ok, "has_contradicted": has_c}
    # kfr_and_no_contradiction
    return {
        "me_mode": "kfr_and_no_contradiction",
        "kfr": kfr,
        "me_pass": kfr_ok and not has_c,
        "has_contradicted": has_c,
    }


class V2PaperPipeline:
    """
    Paper-style Reflexion over the static KG verifier (generate → verify → repair),
    with multiple trials per question and selectable memory strategies.
    """

    def __init__(
        self,
        triples: list[dict],
        model: str = ACTION_MODEL,
        strategy: ReflexionStrategy | str = ReflexionStrategy.REFLEXION,
        max_trials: int = 3,
        window_size: int = 3,
        inject_into: set[str] | None = None,
        with_short_memory: bool = False,
        with_long_memory: bool = True,
        with_all_memory: bool = False,
        long_memory: ReflexionMemory | None = None,
        few_shot_examples: list[str] | None = None,
        result_variant: str = "v2p",
        use_rgr: bool = False,
    ) -> None:
        self.triples = triples
        self.result_variant = str(result_variant or "v2p")
        self.model = model
        self.use_rgr = bool(use_rgr)
        if isinstance(strategy, str):
            self.strategy = ReflexionStrategy(strategy.lower().strip())
        else:
            self.strategy = strategy
        self.max_trials = max(1, int(max_trials))
        self.window_size = max(0, int(window_size))
        self.inject_into = inject_into if inject_into is not None else {"answer"}
        if with_all_memory:
            self.with_short_memory = True
            self.with_long_memory = True
        else:
            self.with_short_memory = bool(with_short_memory)
            self.with_long_memory = bool(with_long_memory)
        self.few_shot_examples = few_shot_examples
        if long_memory is not None:
            self.memory = long_memory
        else:
            self.memory = ReflexionMemory(max_reflections=max(self.window_size, 0))
        if long_memory is None and self.window_size > 0:
            self.memory.max_reflections = self.window_size

    def reset_memory(self) -> None:
        self.memory.clear()

    def _extract_entity_hints_from_reflections(self) -> list[str]:
        """
        Parse entity/value hints from the most recent reflections for RGR.

        Strategy:
          1. Look for "EXACT CORRECT VALUES: ..." or "LOOK FOR: ..." tagged lines
             appended by the updated reflection prompts. Split by comma.
          2. Fall back to LLM extraction on the last 2 reflection texts.

        Returns a flat list of entity/value strings to inject as seed_entities
        into the next call to verify_claims().
        """
        if not self.memory.reflections:
            return []

        hints: list[str] = []
        for refl in self.memory.reflections[-2:]:
            for tag in ("EXACT CORRECT VALUES:", "LOOK FOR:"):
                if tag in refl:
                    line = refl.split(tag, 1)[1].split("\n")[0]
                    hints.extend(v.strip() for v in line.split(",") if v.strip())
                    break  # only one tag per reflection

        if hints:
            return hints

        # Fallback: LLM extraction over the two most recent reflection texts
        try:
            from entity_linker import _extract_entities_llm
            recent = " ".join(self.memory.reflections[-2:])
            extracted = _extract_entities_llm(recent)
            return extracted.get("entities", []) + extracted.get("values", [])
        except Exception:
            return []

    def _resolve_answer_base(self, question: str) -> str:
        """RAG and other V2P variants can override. Default: same as V1 un-augmented answer."""
        return ANSWER_SYSTEM_PROMPT_BASE

    def _long_reflection_block(self) -> str:
        if not self.with_long_memory or self.window_size <= 0:
            return ""
        if not self.memory.has_reflections():
            return ""
        return self.memory.format_for_prompt()

    def _verifier_repair_reflection_block(self) -> str:
        if self.strategy == ReflexionStrategy.NONE:
            return ""
        if self.strategy == ReflexionStrategy.LAST_ATTEMPT:
            return ""
        return self._long_reflection_block()

    def _build_answer_system_prompt(self, question: str, trial_idx: int, scratchpad: Scratchpad) -> str:
        if self.strategy == ReflexionStrategy.NONE:
            return self._current_answer_base

        blocks: list[str] = []

        # Short-term (previous trial scratchpad)
        if (
            self.with_short_memory
            and trial_idx > 0
            and scratchpad.trials
            and self.strategy
            in (ReflexionStrategy.LAST_ATTEMPT, ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION)
        ):
            prev = format_last_attempt(question, scratchpad.format_for_prompt(last_n_trials=1))
            if prev.strip():
                blocks.append(prev.strip())

        # Long-term distilled reflections
        if self.with_long_memory and self.window_size > 0 and self.memory.has_reflections():
            if self.strategy == ReflexionStrategy.REFLEXION:
                fb = self.memory.format_for_prompt()
                if fb.strip():
                    blocks.append(fb.strip())
            elif self.strategy == ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION:
                rf = format_reflections(
                    self.memory.reflections.copy(),
                    header=REFLECTION_AFTER_LAST_TRIAL_HEADER,
                )
                if rf.strip():
                    blocks.append(rf.strip())

        blocks.append(self._current_answer_base)
        return "\n\n".join(blocks)

    def run(
        self,
        question: str,
        stop_check: Callable[..., bool] | None = None,
        me_context: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """
        Parameters
        ----------
        me_context : optional paper-Me (evaluator) config, v2p/v2p_rag only:
            - me_mode: "legacy" | "kfr" | "kfr_and_no_contradiction" (default legacy if absent)
            - key_facts: list[str] — required for kfr / kfr_and_no_contradiction
            - kfr_threshold: float, default 0.8
            - abandon_after_consecutive_me_fails: int | None — stop after N consecutive Me fails
        """
        start = time.time()
        scratchpad = Scratchpad(question=question)
        trials_out: list[dict[str, Any]] = []
        stopped_reason = "max_trials"

        final_raw = ""
        final_claims: list[str] = []
        final_vr: list[VerificationResult] = []
        final_repair: RepairResult | None = None

        self._current_answer_base = self._resolve_answer_base(question)
        consecutive_me_fails = 0
        abandon_n = (me_context or {}).get("abandon_after_consecutive_me_fails")
        abandon_n = int(abandon_n) if abandon_n is not None else None

        for trial_idx in range(self.max_trials):
            scratchpad.start_trial(trial_idx)

            answer_sys = self._build_answer_system_prompt(question, trial_idx, scratchpad)
            if "answer" not in self.inject_into:
                answer_sys = self._current_answer_base

            raw_answer = call_llm(
                question,
                model=self.model,
                system_prompt=answer_sys,
            )
            claims: list[Claim] = extract_claims(raw_answer, question)
            scratchpad.record_raw_answer(raw_answer)
            scratchpad.record_claims([c.text for c in claims])

            v_ref = ""
            if "verifier" in self.inject_into:
                v_ref = self._verifier_repair_reflection_block()

            # RGR: extract entity hints from reflections on trial > 0
            seed_ents: list[str] | None = None
            if self.use_rgr and trial_idx > 0:
                seed_ents = self._extract_entity_hints_from_reflections() or None

            vr_list: list[VerificationResult] = verify_claims(
                claims, self.triples, reflections_block=v_ref, v2p_epistemic_rules=True,
                seed_entities=seed_ents,
            )
            scratchpad.record_verifications(
                [
                    {
                        "claim": r.claim,
                        "verdict": r.verdict.value,
                        "reasoning": r.reasoning[:200],
                    }
                    for r in vr_list
                ]
            )

            r_ref = ""
            if "repairer" in self.inject_into:
                r_ref = self._verifier_repair_reflection_block()

            repair: RepairResult = repair_answer(
                question,
                raw_answer,
                vr_list,
                reflections_block=r_ref,
                v2p_meta_repair_rules=True,
            )
            scratchpad.record_repair(repair.repaired_answer, repair.had_hallucinations)

            me_state = _eval_paper_me(repair.repaired_answer, vr_list, me_context)
            ok_default = bool(me_state["me_pass"])

            ok_custom = True
            if stop_check is not None:
                try:
                    ok_custom = bool(
                        stop_check(
                            question=question,
                            raw_answer=raw_answer,
                            repaired_answer=repair.repaired_answer,
                            verification_results=vr_list,
                            had_hallucinations=repair.had_hallucinations,
                        )
                    )
                except TypeError:
                    ok_custom = bool(stop_check(vr_list))
            stop_ok = ok_default and ok_custom
            scratchpad.record_stop_check(stop_ok)

            if _effective_me_mode(me_context) != "legacy":
                if not ok_default:
                    consecutive_me_fails += 1
                else:
                    consecutive_me_fails = 0

            trials_out.append(
                {
                    "trial_idx": trial_idx,
                    "raw_answer": raw_answer,
                    "repaired_answer": repair.repaired_answer,
                    "claims": [c.text for c in claims],
                    "verification_results": [_serialise_vr(r) for r in vr_list],
                    "had_hallucinations": repair.had_hallucinations,
                    "stop_check_passed": stop_ok,
                    "kfr": me_state["kfr"],
                    "me_pass": me_state["me_pass"],
                    "me_mode": me_state["me_mode"],
                    "consecutive_me_fails": consecutive_me_fails,
                }
            )

            final_raw = raw_answer
            final_claims = [c.text for c in claims]
            final_vr = vr_list
            final_repair = repair

            scratchpad.end_trial()

            if stop_ok:
                stopped_reason = "stop_check_passed"
                break

            if abandon_n is not None and consecutive_me_fails >= abandon_n:
                stopped_reason = "abandon_consecutive_me_fails"
                break

            # Failed trial: optionally distill a verbal reflection before the next trial only
            if trial_idx >= self.max_trials - 1:
                stopped_reason = "max_trials_exhausted"
                break

            contradicted = [r for r in vr_list if r.verdict == Verdict.CONTRADICTED]
            key_facts_cf: list[str] = list((me_context or {}).get("key_facts") or [])
            miss = missing_key_facts(repair.repaired_answer, key_facts_cf)

            can_long_reflect = (
                self.with_long_memory
                and self.window_size > 0
                and self.strategy
                in (ReflexionStrategy.REFLEXION, ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION)
            )

            if contradicted and can_long_reflect:
                reflection = generate_reflection(
                    question=question,
                    original_answer=raw_answer,
                    verification_results=vr_list,
                    repaired_answer=repair.repaired_answer,
                    existing_reflections=self.memory.reflections.copy(),
                    few_shot_examples=self.few_shot_examples,
                    sanitize_output=True,
                )
                if reflection:
                    self.memory.add_reflection(reflection)
            elif (
                not contradicted
                and not me_state["me_pass"]
                and can_long_reflect
                and miss
                and _effective_me_mode(me_context) != "legacy"
            ):
                reflection = generate_reflection_me_fail(
                    question=question,
                    original_answer=raw_answer,
                    repaired_answer=repair.repaired_answer,
                    verification_results=vr_list,
                    missing_key_facts=miss,
                    key_fact_recall=float(me_state["kfr"]),
                    existing_reflections=self.memory.reflections.copy(),
                    few_shot_examples=self.few_shot_examples,
                    sanitize_output=True,
                )
                if reflection:
                    self.memory.add_reflection(reflection)

        elapsed = time.time() - start
        assert final_repair is not None

        return PipelineResult(
            question=question,
            variant=self.result_variant,
            raw_answer=final_raw,
            final_answer=final_repair.repaired_answer,
            claims=final_claims,
            verification_results=[_serialise_vr(r) for r in final_vr],
            had_hallucinations=final_repair.had_hallucinations,
            reflections_used=self.memory.reflections.copy(),
            latency_seconds=elapsed,
            trials=trials_out,
            stopped_reason=stopped_reason,
        )


def load_reflection_fewshots(path: str) -> list[str]:
    """Load list of reflection example strings from JSON (list or {examples: [...]} )."""
    import json

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]
    ex = data.get("examples") or data.get("reflections") or []
    out: list[str] = []
    for item in ex:
        if isinstance(item, str):
            out.append(item.strip())
        elif isinstance(item, dict):
            r = item.get("reflection") or item.get("text") or ""
            if r:
                out.append(str(r).strip())
    return [x for x in out if x]


def parse_strategy(s: str) -> ReflexionStrategy:
    s = (s or "reflexion").strip().lower().replace("-", "_")
    mapping = {
        "none": ReflexionStrategy.NONE,
        "last_attempt": ReflexionStrategy.LAST_ATTEMPT,
        "reflexion": ReflexionStrategy.REFLEXION,
        "last_attempt_and_reflexion": ReflexionStrategy.LAST_ATTEMPT_AND_REFLEXION,
    }
    if s not in mapping:
        raise ValueError(f"Unknown strategy: {s}")
    return mapping[s]


__all__ = [
    "V2PaperPipeline",
    "load_reflection_fewshots",
    "parse_strategy",
    "parse_inject_into",
]
