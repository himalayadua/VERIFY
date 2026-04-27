"""
run_experiments.py
------------------
Evaluation harness for the KG-Backed Claim Verifier experiments.

For each model + variant combination this script produces:
  results/<model-slug>/v0_results.json        ← aggregate metrics + per-question records
  results/<model-slug>/v1_results.json
  results/<model-slug>/v1rag_results.json
  results/<model-slug>/v2_results.json
  results/<model-slug>/logs/v0_<ts>.log       ← full execution trace (same session as JSON)
  results/<model-slug>/logs/v1_<ts>.log
  results/<model-slug>/logs/v1rag_<ts>.log
  results/<model-slug>/logs/v2_<ts>.log
  results/<model-slug>/logs/combined_<ts>.log

The logs and JSON are produced from the SAME run, so every metric in the JSON
is directly traceable to the corresponding log entry.

Usage (run from the project root):
  python src/run_experiments.py
  python src/run_experiments.py --model deepseek-ai/deepseek-v3.2
  python src/run_experiments.py --variants v1 v1rag
  python src/run_experiments.py --limit 5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entity_linker import load_kg
from metrics import missing_key_facts, task_passes_kfr
from pipeline import V0Pipeline, V1Pipeline, V1RAGPipeline, V2Pipeline, PipelineResult
from reflexion_layer import ReflexionMemory
from model_registry import model_to_slug

# ── Paths ──────────────────────────────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR    = os.path.normpath(os.path.join(_HERE, "..", "data"))
_RESULTS_DIR = os.path.normpath(os.path.join(_HERE, "..", "results"))

DEFAULT_QUESTIONS_PATH = os.path.join(_DATA_DIR, "questions.json")
DEFAULT_KG_PATH        = os.path.join(_DATA_DIR, "kg_triples.json")


# ── Logging infrastructure ─────────────────────────────────────────────────────
# Mirrors test_integration.py so log formats are identical and both are readable
# the same way.  Logs go to results/<model-slug>/logs/ alongside the JSON files.

def _build_run_logger(
    variant: str,
    timestamp: str,
    logs_dir: str,
    combined_handler: logging.FileHandler,
) -> tuple[logging.Logger, str]:
    """
    Create a named logger that writes to:
      1. <logs_dir>/<variant>_<timestamp>.log  (DEBUG — every detail)
      2. The shared combined log file           (DEBUG)
      3. stdout                                 (INFO — progress only)
    """
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{variant}_{timestamp}.log")

    logger = logging.getLogger(f"kg_exp.{logs_dir}.{variant}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d] [%(levelname)-5s] %(message)s",
        datefmt="%H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    combined_handler.setFormatter(fmt)
    combined_handler.setLevel(logging.DEBUG)
    logger.addHandler(combined_handler)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger, log_path


def _trunc(text: str, max_chars: int = 600) -> str:
    if text is None:
        return "(None)"
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n  ... [+{len(text) - max_chars} chars truncated]"


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in str(text).splitlines())


def _banner(logger: logging.Logger, text: str, width: int = 72, char: str = "=") -> None:
    logger.info(char * width)
    logger.info(f"  {text}")
    logger.info(char * width)


def _divider(logger: logging.Logger, width: int = 72, char: str = "─") -> None:
    logger.info(char * width)


# ── LLM call interceptor ───────────────────────────────────────────────────────

class _LLMInterceptor:
    """
    Wraps llm_client.call_llm and call_llm_json to log every call.
    Restores originals on context-manager exit.
    """

    def __init__(self, logger: logging.Logger, label: str):
        self.logger = logger
        self.label  = label
        self._orig_call_llm      = None
        self._orig_call_llm_json = None

    def __enter__(self):
        import llm_client as _lc
        self._orig_call_llm      = _lc.call_llm
        self._orig_call_llm_json = _lc.call_llm_json
        _lc.call_llm      = self._wrap_call_llm(_lc.call_llm)
        _lc.call_llm_json = self._wrap_call_llm_json(_lc.call_llm_json)
        return self

    def __exit__(self, *_):
        import llm_client as _lc
        _lc.call_llm      = self._orig_call_llm
        _lc.call_llm_json = self._orig_call_llm_json

    def _wrap_call_llm(self, original):
        log   = self.logger
        label = self.label

        def wrapped(prompt, model=None, system_prompt=None, **kwargs):
            import llm_client as _lc
            model = model or _lc.ACTION_MODEL
            log.debug(
                f"[{label}] LLM CALL  model={model}\n"
                f"  System : {_trunc(system_prompt, 400) if system_prompt else '(none)'}\n"
                f"  User   : {_trunc(prompt, 600)}"
            )
            t0     = time.time()
            result = original(prompt, model=model, system_prompt=system_prompt, **kwargs)
            log.debug(
                f"[{label}] LLM DONE  ({time.time() - t0:.2f}s)\n"
                f"  Response:\n{_indent(_trunc(result, 800))}"
            )
            return result

        return wrapped

    def _wrap_call_llm_json(self, original):
        log   = self.logger
        label = self.label

        def wrapped(prompt, model=None, system_prompt=None, **kwargs):
            import llm_client as _lc
            model = model or _lc.EXTRACT_MODEL
            log.debug(
                f"[{label}] LLM JSON CALL  model={model}\n"
                f"  System : {_trunc(system_prompt, 300) if system_prompt else '(none)'}\n"
                f"  User   : {_trunc(prompt, 500)}"
            )
            t0     = time.time()
            result = original(prompt, model=model, system_prompt=system_prompt, **kwargs)
            log.debug(
                f"[{label}] LLM JSON DONE  ({time.time() - t0:.2f}s)\n"
                f"  Parsed : {json.dumps(result, indent=2)[:600]}"
            )
            return result

        return wrapped


# ── Entity linker interceptor ──────────────────────────────────────────────────

class _EntityLinkerInterceptor:
    """Wraps entity_linker.find_relevant_triples to log every KG lookup."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._orig  = None

    def __enter__(self):
        import entity_linker as _el
        self._orig = _el.find_relevant_triples
        _el.find_relevant_triples = self._wrap(_el.find_relevant_triples)
        return self

    def __exit__(self, *_):
        import entity_linker as _el
        _el.find_relevant_triples = self._orig

    def _wrap(self, original):
        log = self.logger

        def wrapped(claim, triples, top_n=5, min_score=55, method="llm", seed_entities=None):
            seed_note = f"  seeds={seed_entities}" if seed_entities else ""
            log.debug(f"[ENTITY_LINKER] Lookup  claim=\"{_trunc(claim, 100)}\"  method={method}{seed_note}")
            t0      = time.time()
            results = original(claim, triples, top_n=top_n, min_score=min_score, method=method,
                               seed_entities=seed_entities)
            elapsed = time.time() - t0
            if results:
                triple_lines = "\n".join(
                    f"    [{r.get('score','?'):3}]  {r['subject']} | {r['relation']} | {r['object']}"
                    for r in results
                )
                log.debug(f"[ENTITY_LINKER] Found {len(results)} triple(s) ({elapsed:.2f}s):\n{triple_lines}")
            else:
                log.debug(f"[ENTITY_LINKER] No triples found ({elapsed:.2f}s)")
            return results

        return wrapped


# ── Normalisation ──────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Per-question evaluation ────────────────────────────────────────────────────

def evaluate_answer(
    predicted: str,
    ground_truth: str,
    key_facts: list[str],
) -> dict:
    """
    Compute per-question evaluation metrics.

    Metrics
    -------
    exact_match      : bool — normalised strings are identical
    key_fact_recall  : float 0–1 — fraction of key_facts present in predicted
    is_correct       : bool — key_fact_recall >= 0.8 threshold
    """
    pred_norm = _normalise(predicted)
    miss = missing_key_facts(predicted, key_facts)
    kfr, kfr_ok = task_passes_kfr(predicted, key_facts, 0.8)
    found = len(key_facts) - len(miss) if key_facts else 0

    return {
        "exact_match":     _normalise(ground_truth) == pred_norm,
        "key_fact_recall": round(kfr, 3),
        "is_correct":      kfr_ok,
        "key_facts_found": found,
        "key_facts_total": len(key_facts),
        "facts_missing":   miss,
    }


# ── Aggregate metrics ──────────────────────────────────────────────────────────

def compute_aggregate_metrics(results: list[dict]) -> dict:
    """Aggregate per-question results into summary statistics."""
    n = len(results)
    if n == 0:
        return {}

    correct  = sum(1 for r in results if r.get("is_correct", False))
    accuracy = correct / n
    avg_kfr  = sum(r.get("key_fact_recall", 0.0) for r in results) / n

    hally = [r for r in results if r.get("had_hallucinations") is not None]
    hallucination_rate = (
        sum(1 for r in hally if r["had_hallucinations"]) / len(hally)
        if hally else None
    )

    latencies  = sorted(r.get("latency_seconds", 0.0) for r in results)
    avg_lat    = sum(latencies) / n
    median_lat = latencies[n // 2]
    p95_idx    = min(int(0.95 * n), n - 1)
    p95_lat    = latencies[p95_idx]

    claim_counts = [len(r["claims"]) for r in results if r.get("claims") is not None]
    avg_claims   = sum(claim_counts) / len(claim_counts) if claim_counts else None

    all_verdicts: list[str] = []
    for r in results:
        for vr in (r.get("verification_results") or []):
            all_verdicts.append(vr.get("verdict", "UNVERIFIABLE"))
    if all_verdicts:
        total_v = len(all_verdicts)
        verdict_dist = {
            "SUPPORTED":    round(all_verdicts.count("SUPPORTED")    / total_v, 3),
            "CONTRADICTED": round(all_verdicts.count("CONTRADICTED") / total_v, 3),
            "UNVERIFIABLE": round(all_verdicts.count("UNVERIFIABLE") / total_v, 3),
        }
    else:
        verdict_dist = None

    by_category: dict[str, dict] = {}
    for r in results:
        cat = r.get("category", "unknown")
        e   = by_category.setdefault(cat, {"correct": 0, "kfr_sum": 0.0, "n": 0})
        e["n"] += 1; e["kfr_sum"] += r.get("key_fact_recall", 0.0)
        if r.get("is_correct"): e["correct"] += 1
    by_cat_out = {
        cat: {"accuracy": round(v["correct"]/v["n"], 3), "avg_kfr": round(v["kfr_sum"]/v["n"], 3), "n": v["n"]}
        for cat, v in by_category.items()
    }

    by_diff: dict[str, dict] = {}
    for r in results:
        diff = r.get("difficulty", "unknown")
        e    = by_diff.setdefault(diff, {"correct": 0, "kfr_sum": 0.0, "n": 0})
        e["n"] += 1; e["kfr_sum"] += r.get("key_fact_recall", 0.0)
        if r.get("is_correct"): e["correct"] += 1
    by_diff_out = {
        diff: {"accuracy": round(v["correct"]/v["n"], 3), "avg_kfr": round(v["kfr_sum"]/v["n"], 3), "n": v["n"]}
        for diff, v in by_diff.items()
    }

    out = {
        "total_questions":        n,
        "accuracy":               round(accuracy, 3),
        "avg_kfr":                round(avg_kfr, 3),
        "hallucination_rate":     round(hallucination_rate, 3) if hallucination_rate is not None else None,
        "avg_claims_per_answer":  round(avg_claims, 2) if avg_claims is not None else None,
        "verdict_distribution":   verdict_dist,
        "avg_latency_seconds":    round(avg_lat, 2),
        "median_latency_seconds": round(median_lat, 2),
        "p95_latency_seconds":    round(p95_lat, 2),
        "by_category":            by_cat_out,
        "by_difficulty":          by_diff_out,
    }

    em_rows = [r for r in results if r.get("extended_metrics")]
    if em_rows:
        from metrics import aggregate_extended_metrics

        out["extended_metrics_aggregate"] = aggregate_extended_metrics(em_rows)

    return out


# ── Save results ───────────────────────────────────────────────────────────────

def save_results(output: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {path}")


# ── Summary table ──────────────────────────────────────────────────────────────

def _print_summary_table(agg: dict[str, dict], variants: list[str]) -> None:
    rows = []
    for v in variants:
        m     = agg.get(v, {})
        hrate = m.get("hallucination_rate")
        rows.append([
            v.upper(),
            f"{m.get('accuracy', 0.0):.1%}",
            f"{m.get('avg_kfr', 0.0):.3f}",
            f"{hrate * 100:.1f}%" if hrate is not None else "N/A",
            f"{m.get('avg_latency_seconds', 0.0):.1f}s",
        ])

    headers = ["Variant", "Accuracy", "Avg KFR", "Hallucination%", "Avg Latency"]

    try:
        from tabulate import tabulate
        print("\n" + tabulate(rows, headers=headers, tablefmt="github"))
    except ImportError:
        col_w = [12, 10, 10, 16, 12]
        header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
        print("\n" + "-" * len(header_line))
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print("  ".join(str(c).ljust(w) for c, w in zip(row, col_w)))
        print("-" * len(header_line))


# ── Per-variant runner (variant-outer loop + full logging) ─────────────────────


def _v2p_kfr_stop_check(gt_answer: str, key_facts: list, kfr_threshold: float):
    """
    Optional eval hook for v2p / v2p_rag: stop trials when KFR ≥ threshold
    (uses ground truth; measurement-only, not for deployment).
    """
    def stop_check(
        question: str = "",
        raw_answer: str = "",
        repaired_answer: str = "",
        verification_results=None,
        had_hallucinations=None,
    ) -> bool:
        m = evaluate_answer(repaired_answer, gt_answer, key_facts)
        return float(m["key_fact_recall"]) >= kfr_threshold

    return stop_check


def _run_variant(
    variant: str,
    questions: list[dict],
    triples: list[dict],
    selected_model: str,
    logger: logging.Logger,
    gold_by_id: dict[str, dict] | None = None,
    v2p_config: dict | None = None,
) -> list[dict]:
    """
    Run all questions for a single variant with full logging.
    Returns a list of per-question result dicts (same shape as before).
    """
    v2p_config = v2p_config or {}

    # Instantiate pipeline
    if variant == "v0":
        pipeline = V0Pipeline(model=selected_model)
    elif variant == "v1":
        pipeline = V1Pipeline(triples, model=selected_model)
    elif variant == "v1rag":
        pipeline = V1RAGPipeline(triples, model=selected_model)
    elif variant in ("v2p", "v2p_rag", "v2p_rgr", "v2p_rag_rgr"):
        from v2_paper_pipeline import V2PaperPipeline, load_reflection_fewshots, parse_strategy, parse_inject_into

        if variant in ("v2p_rag", "v2p_rag_rgr"):
            from v2p_rag_pipeline import V2pRagPipeline

            V2PCls = V2pRagPipeline
        else:
            V2PCls = V2PaperPipeline

        use_rgr = variant in ("v2p_rgr", "v2p_rag_rgr")
        ws = int(v2p_config.get("window", 3))
        shared = ReflexionMemory(max_reflections=max(ws, 0))
        if ws > 0:
            shared.max_reflections = ws
        few_path = v2p_config.get("few_shots")
        few_list = load_reflection_fewshots(few_path) if few_path and os.path.isfile(few_path) else None
        pipeline = V2PCls(
            triples,
            model=selected_model,
            strategy=parse_strategy(str(v2p_config.get("strategy", "reflexion"))),
            max_trials=int(v2p_config.get("max_trials", 3)),
            window_size=ws,
            inject_into=parse_inject_into(list(v2p_config.get("inject") or ["answer"])),
            with_short_memory=bool(v2p_config.get("short_memory", False)),
            with_long_memory=bool(v2p_config.get("long_memory", True)),
            with_all_memory=bool(v2p_config.get("all_memory", False)),
            long_memory=shared,
            few_shot_examples=few_list,
            use_rgr=use_rgr,
        )
    elif variant == "v2":
        pipeline = V2Pipeline(
            triples,
            model=selected_model,
            memory=ReflexionMemory(max_reflections=3),
        )
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    _banner(logger, f"VARIANT: {variant.upper()}  |  {len(questions)} questions  |  model: {selected_model}", char="═")
    logger.info(f"Pipeline : {pipeline.__class__.__name__}")
    logger.info("")

    all_results: list[dict] = []

    with _LLMInterceptor(logger, variant.upper()) as _, _EntityLinkerInterceptor(logger) as __:

        for qi, q in enumerate(questions, 1):
            qid        = q["id"]
            question   = q["question"]
            key_facts  = q["key_facts"]
            gt_answer  = q["ground_truth_answer"]
            category   = q.get("category", "?")
            difficulty = q.get("difficulty", "?")

            # ── Question header ────────────────────────────────────────────────
            _divider(logger, char="━")
            logger.info(
                f"Question {qi}/{len(questions)}  [{qid}]  "
                f"category={category}  difficulty={difficulty}"
            )
            logger.info(f"  Q: {question}")
            _divider(logger, char="━")

            t_start = time.time()

            # ── V2 / V2P: log memory state before this question ────────────────
            if variant in ("v2", "v2p", "v2p_rag", "v2p_rgr", "v2p_rag_rgr"):
                mem = pipeline.memory
                _v2p_tag = {
                    "v2": "V2", "v2p": "V2P", "v2p_rag": "V2P-RAG",
                    "v2p_rgr": "V2P-RGR", "v2p_rag_rgr": "V2P-RAG-RGR",
                }[variant]
                if mem.has_reflections():
                    logger.info(f"[{_v2p_tag} MEMORY] {len(mem.reflections)} reflection(s) in session memory:")
                    for i, r in enumerate(mem.reflections, 1):
                        logger.info(f"  [{i}] {r}")
                else:
                    logger.info(f"[{_v2p_tag} MEMORY] Session memory is empty (first question or after reset)")
                logger.info("")

            v2p_me_ctx: dict | None = None
            if variant.startswith("v2p"):
                # Always build Me context for all v2p* variants.
                # Previously this was gated on --v2p-paper-me which meant the default
                # "legacy" mode rewarded refusals (no CONTRADICTED = pass), hiding KFR=0%.
                ab = v2p_config.get("paper_me_abandon")
                if ab is None:
                    ab = 3
                ab_int = int(ab) if int(ab) > 0 else None
                v2p_me_ctx = {
                    "me_mode": str(v2p_config.get("paper_me_mode", "kfr_and_no_contradiction")).lower(),
                    "key_facts": list(key_facts),
                    "kfr_threshold": float(v2p_config.get("paper_me_threshold", 0.8)),
                    "ground_truth_answer": gt_answer,
                    "abandon_after_consecutive_me_fails": ab_int,
                }

            v2p_kfr_stop = None
            sk = v2p_config.get("stop_kfr")
            if variant.startswith("v2p") and sk is not None:
                paper_t = float(v2p_config.get("paper_me_threshold", 0.8))
                if float(sk) != paper_t:
                    v2p_kfr_stop = _v2p_kfr_stop_check(gt_answer, key_facts, float(sk))
                else:
                    logger.debug(
                        "[V2P] Skipping --v2p-stop-kfr duplicate: Me already uses same KFR threshold."
                    )

            # ── Run pipeline ───────────────────────────────────────────────────
            logger.info(f"[{variant.upper()}] Running pipeline...")
            if v2p_me_ctx is not None:
                logger.info(
                    f"[V2P] paper_me mode={v2p_me_ctx['me_mode']} kfr_threshold={v2p_me_ctx['kfr_threshold']} "
                    f"abandon={v2p_me_ctx['abandon_after_consecutive_me_fails']}"
                )
            try:
                run_kw: dict = {}
                if v2p_me_ctx is not None:
                    run_kw["me_context"] = v2p_me_ctx
                if v2p_kfr_stop is not None:
                    result: PipelineResult = pipeline.run(question, stop_check=v2p_kfr_stop, **run_kw)
                else:
                    result = pipeline.run(question, **run_kw)
            except Exception as exc:
                logger.error(f"[{variant.upper()}] Pipeline FAILED: {exc}", exc_info=True)
                all_results.append({
                    "id": qid, "category": category, "difficulty": difficulty,
                    "question": question, "ground_truth_answer": gt_answer,
                    "key_facts": key_facts, "error": str(exc),
                    "raw_answer": "", "final_answer": "",
                    "exact_match": False, "key_fact_recall": 0.0,
                    "is_correct": False, "key_facts_found": 0, "key_facts_total": len(key_facts),
                    "facts_missing": key_facts,
                    "claims": None, "verification_results": None,
                    "had_hallucinations": None, "reflections_used": None,
                    "latency_seconds": 0.0,
                })
                continue

            t_elapsed = time.time() - t_start

            # ── Log raw answer ─────────────────────────────────────────────────
            logger.info("")
            logger.info("── RAW ANSWER ──")
            logger.info(_indent(result.raw_answer))

            # ── Claims ────────────────────────────────────────────────────────
            if result.claims is not None:
                logger.info("")
                logger.info(f"── CLAIMS EXTRACTED  ({len(result.claims)}) ──")
                for i, c in enumerate(result.claims, 1):
                    logger.info(f"  {i:2d}. {c}")

            # ── Verification results ───────────────────────────────────────────
            if result.verification_results is not None:
                logger.info("")
                logger.info("── VERIFICATION RESULTS ──")
                for vr in result.verification_results:
                    verdict   = vr["verdict"]
                    conf      = vr.get("confidence", 0.0)
                    reasoning = vr.get("reasoning", "")
                    evidence  = vr.get("evidence_triples", [])

                    level = logging.WARNING if verdict == "CONTRADICTED" else (
                        logging.DEBUG if verdict == "UNVERIFIABLE" else logging.INFO
                    )
                    logger.log(level, f"  [{verdict:<13}] (conf={conf:.2f})  \"{_trunc(vr['claim'], 80)}\"")
                    logger.debug(f"    Reasoning : {reasoning}")
                    if evidence:
                        logger.debug(
                            f"    Evidence  : {evidence[0]['subject']} | "
                            f"{evidence[0]['relation']} | {evidence[0]['object']}"
                        )

            # ── Repair ────────────────────────────────────────────────────────
            if result.had_hallucinations is not None:
                logger.info("")
                if result.had_hallucinations:
                    n_contradicted = len([vr for vr in (result.verification_results or []) if vr["verdict"] == "CONTRADICTED"])
                    logger.warning("── REPAIR (hallucinations detected) ──")
                    logger.warning(f"  Claims repaired: {n_contradicted}")
                    logger.info("  Repaired answer:")
                    logger.info(_indent(result.final_answer))
                else:
                    logger.info("── REPAIR: not needed (no contradictions) ──")

            # ── V2 / V2P: reflection + multi-trial trace ───────────────────────
            if variant == "v2":
                logger.info("")
                current_refs = result.reflections_used or []
                if result.had_hallucinations and current_refs:
                    logger.info("── V2 REFLECTION GENERATED ──")
                    logger.info(f"  {current_refs[-1]}")
                    logger.info(f"\n[V2 MEMORY] Updated — now {len(current_refs)} reflection(s)")
                else:
                    logger.info(f"[V2 MEMORY] No new reflection. Memory: {len(current_refs)} reflection(s)")
            if variant.startswith("v2p"):
                _log_tag = {
                    "v2p": "V2P", "v2p_rag": "V2P-RAG",
                    "v2p_rgr": "V2P-RGR", "v2p_rag_rgr": "V2P-RAG-RGR",
                }.get(variant, variant.upper())
                logger.info("")
                current_refs = result.reflections_used or []
                logger.info(
                    f"[{_log_tag}] Stopped: {getattr(result, 'stopped_reason', '')}  |  "
                    f"reflections: {len(current_refs)}"
                )
                if getattr(result, "trials", None):
                    logger.info(f"[{_log_tag}] Multi-trial trace: {len(result.trials)} trial(s)")
                    for tr in result.trials or []:
                        logger.debug(f"  trial {tr.get('trial_idx')}: stop_ok={tr.get('stop_check_passed')}")
                if current_refs:
                    logger.info(f"[{_log_tag} MEMORY] Latest reflection: {current_refs[-1][:200]}...")

            # ── Final answer (V0 path) ─────────────────────────────────────────
            if result.claims is None:
                logger.info("")
                logger.info("── FINAL ANSWER (V0 — no verification) ──")
                logger.info(_indent(result.final_answer))

            # ── Evaluation ────────────────────────────────────────────────────
            metrics       = evaluate_answer(result.final_answer, gt_answer, key_facts)
            correct_marker = "✓" if metrics["is_correct"] else "✗"

            logger.info("")
            logger.info("── EVALUATION ──")
            logger.info(f"  Ground truth  : {gt_answer}")
            logger.info(f"  Key facts     : {key_facts}")
            logger.info(
                f"  KFR           : {metrics['key_fact_recall']:.3f} "
                f"({metrics['key_facts_found']}/{metrics['key_facts_total']} facts found) {correct_marker}"
            )
            if metrics["facts_missing"]:
                logger.warning(f"  Missing facts : {metrics['facts_missing']}")
            logger.info(f"  Exact match   : {metrics['exact_match']}")
            logger.info(f"  Latency       : {t_elapsed:.2f}s")
            logger.info("")

            record: dict = {
                "id": qid, "category": category, "difficulty": difficulty,
                "question": question, "ground_truth_answer": gt_answer,
                "key_facts": key_facts,
                **result.to_dict(),
                **metrics,
            }
            if gold_by_id and qid in gold_by_id:
                from metrics import compute_question_extended_metrics

                record["extended_metrics"] = compute_question_extended_metrics(
                    record, gold_by_id[qid]
                )
            all_results.append(record)

    # ── Variant summary ────────────────────────────────────────────────────────
    _banner(logger, f"SUMMARY — {variant.upper()}", char="═")
    n           = len(all_results)
    n_correct   = sum(1 for r in all_results if r.get("is_correct", False))
    n_halluc    = sum(1 for r in all_results if r.get("had_hallucinations") is True)
    avg_kfr     = sum(r.get("key_fact_recall", 0.0) for r in all_results) / max(n, 1)
    avg_latency = sum(r.get("latency_seconds", 0.0) for r in all_results) / max(n, 1)

    logger.info(f"  Questions     : {n}")
    logger.info(f"  Correct       : {n_correct}/{n}  ({n_correct/max(n,1):.1%})")
    logger.info(f"  Avg KFR       : {avg_kfr:.3f}")
    if any(r.get("had_hallucinations") is not None for r in all_results):
        logger.info(f"  Hallucinations: {n_halluc}/{n} questions had contradicted claims")
    logger.info(f"  Avg latency   : {avg_latency:.2f}s")
    logger.info("")
    logger.info(f"  {'ID':<6} {'Category':<20} {'Diff':<8} {'KFR':<7} {'OK?':<5} {'Hallucs':<8} {'Latency'}")
    logger.info("  " + "-" * 62)
    for r in all_results:
        ok  = "✓" if r.get("is_correct") else "✗"
        h   = "yes" if r.get("had_hallucinations") is True else ("no" if r.get("had_hallucinations") is False else "N/A")
        lat = f"{r.get('latency_seconds', 0.0):.1f}s"
        logger.info(f"  {r['id']:<6} {r['category']:<20} {r['difficulty']:<8} {r.get('key_fact_recall',0.0):<7.3f} {ok:<5} {h:<8} {lat}")
    logger.info("")

    return all_results


# ── Main experiment runner ─────────────────────────────────────────────────────

def run_all_experiments(
    questions_path: str = DEFAULT_QUESTIONS_PATH,
    kg_path: str = DEFAULT_KG_PATH,
    output_dir: str = _RESULTS_DIR,
    variants: list[str] | None = None,
    limit: int | None = None,
    model: str | None = None,
    gold_claims_path: str | None = None,
    v2p_config: dict | None = None,
) -> dict[str, list[dict]]:
    """
    Run all specified variants on all questions, logging every step and saving
    JSON results.  Logs and JSON are from the SAME session.

    Parameters
    ----------
    questions_path : path to questions.json
    kg_path        : path to kg_triples.json
    output_dir     : base directory; <model-slug>/ is appended automatically
    variants          : subset of ["v0","v1","v1rag","v2","v2p","v2p_rag","v2p_rgr","v2p_rag_rgr"] (default: v0–v2)
    limit             : only run the first N questions (quick testing)
    model             : NIM model name; overrides ACTION/EXTRACT/REFLECT_MODEL
    gold_claims_path: optional JSON for extended metrics (per-question + aggregate)
    v2p_config        : optional dict for v2p / v2p_rag (strategy, window, stop_kfr,
                        paper_me, paper_me_mode, paper_me_threshold, paper_me_abandon, etc.)

    Returns
    -------
    dict mapping variant → list of per-question result dicts
    """
    if variants is None:
        variants = ["v0", "v1", "v1rag", "v2"]

    gold_by_id: dict[str, dict] | None = None
    if gold_claims_path and os.path.isfile(gold_claims_path):
        with open(gold_claims_path, encoding="utf-8") as gf:
            gold_by_id = json.load(gf)

    # ── Apply model override ───────────────────────────────────────────────────
    import llm_client as _lc
    if model:
        _lc.ACTION_MODEL  = model
        _lc.REFLECT_MODEL = model
        _lc.EXTRACT_MODEL = model
    selected_model = _lc.ACTION_MODEL

    # ── Per-model output + log directories ────────────────────────────────────
    model_slug     = model_to_slug(selected_model)
    run_output_dir = os.path.join(output_dir, model_slug)
    logs_dir       = os.path.join(run_output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 65)
    print("  KG-BACKED CLAIM VERIFIER — EXPERIMENT RUN")
    print(f"  Model    : {selected_model}")
    print(f"  Variants : {', '.join(v.upper() for v in variants)}")
    print(f"  Output   : {run_output_dir}/")
    print(f"  Logs     : {logs_dir}/")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Load data ──────────────────────────────────────────────────────────────
    with open(questions_path, encoding="utf-8") as f:
        question_data = json.load(f)
    questions = question_data["questions"]
    if limit:
        questions = questions[:limit]
        print(f"\n[INFO] Limiting to first {limit} questions for testing.")

    triples = load_kg(kg_path)
    print(f"\nLoaded {len(questions)} questions  |  {len(triples)} KG triples\n")

    # ── Shared combined log ────────────────────────────────────────────────────
    combined_path = os.path.join(logs_dir, f"combined_{ts}.log")
    combined_fh   = logging.FileHandler(combined_path, encoding="utf-8")

    # ── Run each variant (variant-outer loop) ─────────────────────────────────
    # Variant-outer ensures each variant gets an uninterrupted log and its own
    # session context (V2 memory accumulates cleanly across all questions).
    all_results: dict[str, list[dict]] = {}
    agg_by_variant: dict[str, dict]    = {}
    timestamp = datetime.now().isoformat(timespec="seconds")

    for variant in variants:
        logger, log_path = _build_run_logger(variant, ts, logs_dir, combined_fh)

        _banner(logger, "KG-BACKED CLAIM VERIFIER — EXPERIMENT RUN", char="═")
        logger.info(f"  Model     : {selected_model}")
        logger.info(f"  Variant   : {variant.upper()}")
        logger.info(f"  Questions : {len(questions)}")
        logger.info(f"  KG triples: {len(triples)}")
        logger.info(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Log file  : {log_path}")
        logger.info("")

        results = _run_variant(
            variant,
            questions,
            triples,
            selected_model,
            logger,
            gold_by_id=gold_by_id,
            v2p_config=v2p_config,
        )
        all_results[variant] = results

        # ── Compute metrics + save JSON immediately after this variant ─────────
        agg = compute_aggregate_metrics(results)
        agg_by_variant[variant] = agg
        output_payload = {
            "variant":           variant,
            "model":             selected_model,
            "run_timestamp":     timestamp,
            "aggregate_metrics": agg,
            "questions":         results,
        }
        json_path = os.path.join(run_output_dir, f"{variant}_results.json")
        save_results(output_payload, json_path)
        logger.info(f"Results saved → {json_path}")

        for h in logger.handlers:
            h.flush()

    # ── Cross-variant comparison in combined log ───────────────────────────────
    if len(variants) > 1:
        combined_logger = logging.getLogger(f"kg_exp.{logs_dir}.combined")
        combined_logger.setLevel(logging.DEBUG)
        combined_logger.propagate = False
        if not combined_logger.handlers:
            fmt = logging.Formatter(
                fmt="[%(asctime)s.%(msecs)03d] [%(levelname)-5s] %(message)s",
                datefmt="%H:%M:%S",
            )
            combined_fh.setFormatter(fmt)
            combined_logger.addHandler(combined_fh)

        _banner(combined_logger, "COMPARISON ACROSS VARIANTS", char="═")
        headers  = ["Variant", "Correct", "Avg KFR", "Hallucs", "Avg Latency"]
        col_w    = [10, 16, 10, 10, 12]
        combined_logger.info("  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_w)))
        combined_logger.info("  " + "-" * (sum(col_w) + 2 * len(col_w)))
        for v in variants:
            m       = agg_by_variant.get(v, {})
            n       = m.get("total_questions", 0)
            correct = round((m.get("accuracy") or 0) * n)
            hrate   = m.get("hallucination_rate")
            h_str   = f"{round((hrate or 0) * n)}/{n}" if hrate is not None else "N/A"
            row = [
                v.upper(),
                f"{correct}/{n} ({m.get('accuracy',0):.1%})",
                f"{m.get('avg_kfr',0):.3f}",
                h_str,
                f"{m.get('avg_latency_seconds',0):.1f}s",
            ]
            combined_logger.info("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, col_w)))
        combined_logger.info("")
        combined_fh.flush()

    # ── Print console summary ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    _print_summary_table(agg_by_variant, variants)

    print("\n" + "=" * 65)
    print("  OUTPUT FILES")
    print("=" * 65)
    for v in variants:
        print(f"  {v.upper()} JSON → {os.path.join(run_output_dir, v + '_results.json')}")
        print(f"  {v.upper()} log  → {os.path.join(logs_dir, v + '_' + ts + '.log')}")
    if len(variants) > 1:
        print(f"  Combined → {combined_path}")
    print("=" * 65 + "\n")

    return all_results


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run KG-backed claim verifier experiments (V0 / V1 / V1RAG / V2)."
    )
    parser.add_argument(
        "--variants", nargs="+",
        choices=["v0", "v1", "v1rag", "v2", "v2p", "v2p_rag", "v2p_rgr", "v2p_rag_rgr"],
        default=["v0", "v1", "v1rag", "v2"],
        help="Which pipeline variants to run (default: v0–v2; v2p / v2p_rag = paper Reflexion; v2p_rgr / v2p_rag_rgr add Reflexion-Guided Retrieval).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N questions (useful for quick tests).",
    )
    parser.add_argument(
        "--questions", type=str, default=DEFAULT_QUESTIONS_PATH,
        help="Path to questions.json.",
    )
    parser.add_argument(
        "--kg", type=str, default=DEFAULT_KG_PATH,
        help="Path to kg_triples.json.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=_RESULTS_DIR,
        help="Base directory for result JSON files (model slug is appended automatically).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=(
            "NVIDIA NIM model to use for all pipeline tasks. "
            "Overrides ACTION_MODEL / EXTRACT_MODEL / REFLECT_MODEL. "
            "Example: --model deepseek-ai/deepseek-v3.2"
        ),
    )
    parser.add_argument(
        "--gold-claims", type=str, default=None,
        help="Path to gold_claims.json (enables extended_metrics per question + aggregate).",
    )
    # ── V2P (paper Reflexion) options ─────────────────────────────────────────
    parser.add_argument(
        "--v2p-strategy", type=str, default="reflexion",
        choices=["none", "last_attempt", "reflexion", "last_attempt_and_reflexion"],
        help="ReflexionStrategy for v2p (default: reflexion).",
    )
    parser.add_argument(
        "--v2p-inject", nargs="+", default=["answer"],
        help="Where to inject long-term reflections: answer verifier repairer all (default: answer).",
    )
    parser.add_argument("--v2p-window", type=int, default=3, help="Long-term reflection window (0 disables storage).")
    parser.add_argument("--v2p-max-trials", type=int, default=3, help="Max trials per question for v2p.")
    parser.add_argument("--v2p-short-memory", action="store_true", help="Enable short-term scratchpad for v2p.")
    parser.add_argument("--v2p-no-short-memory", action="store_true", help="Disable short-term scratchpad for v2p.")
    parser.add_argument("--v2p-long-memory", action="store_true", help="Enable long-term reflections for v2p.")
    parser.add_argument("--v2p-no-long-memory", action="store_true", help="Disable long-term reflections for v2p.")
    parser.add_argument("--v2p-all-memory", action="store_true", help="Enable both short- and long-term memory for v2p.")
    parser.add_argument(
        "--v2p-few-shots", type=str, default=None,
        help="Path to reflection_fewshot.json for reflection few-shot examples.",
    )
    parser.add_argument(
        "--v2p-stop-kfr", type=float, default=None,
        help="Optional (eval only): for v2p / v2p_rag, AND an extra stop when KFR ≥ this (oracle). Skipped when same threshold as --v2p-paper-me-threshold with --v2p-paper-me (Me already enforces KFR).",
    )
    parser.add_argument(
        "--v2p-paper-me",
        action="store_true",
        help="Paper-aligned Me inside v2p: KFR-based stop/retry + me-fail reflections (oracle key_facts from questions).",
    )
    parser.add_argument(
        "--v2p-paper-me-mode",
        type=str,
        default="kfr_and_no_contradiction",
        choices=["legacy", "kfr", "kfr_and_no_contradiction"],
        help="Me mode when --v2p-paper-me is set (default: kfr_and_no_contradiction).",
    )
    parser.add_argument(
        "--v2p-paper-me-threshold",
        type=float,
        default=0.8,
        help="KFR threshold for paper Me (default: 0.8).",
    )
    parser.add_argument(
        "--v2p-paper-me-abandon",
        type=int,
        default=3,
        help="Stop after N consecutive Me fails (0 = no cap; default: 3).",
    )
    args = parser.parse_args()

    short_mem = bool(args.v2p_short_memory)
    long_mem = True
    if args.v2p_no_short_memory:
        short_mem = False
    if args.v2p_long_memory:
        long_mem = True
    if args.v2p_no_long_memory:
        long_mem = False

    v2p_cfg = {
        "strategy": args.v2p_strategy,
        "inject": args.v2p_inject,
        "window": args.v2p_window,
        "max_trials": args.v2p_max_trials,
        "short_memory": short_mem,
        "long_memory": long_mem,
        "all_memory": bool(args.v2p_all_memory),
        "few_shots": args.v2p_few_shots,
        "stop_kfr": args.v2p_stop_kfr,
        "paper_me": bool(args.v2p_paper_me),
        "paper_me_mode": args.v2p_paper_me_mode,
        "paper_me_threshold": float(args.v2p_paper_me_threshold),
        "paper_me_abandon": int(args.v2p_paper_me_abandon),
    }

    run_all_experiments(
        questions_path=args.questions,
        kg_path=args.kg,
        output_dir=args.output_dir,
        variants=args.variants,
        limit=args.limit,
        model=args.model,
        gold_claims_path=args.gold_claims,
        v2p_config=v2p_cfg,
    )
