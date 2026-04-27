#!/usr/bin/env python3
"""
test_integration.py
-------------------
End-to-end integration test for the KG-Backed Claim Verifier (V0 / V1 / V2).

For each variant a dedicated, timestamped log file is written to test/logs/:
  test/logs/v0_YYYYMMDD_HHMMSS.log
  test/logs/v1_YYYYMMDD_HHMMSS.log
  test/logs/v2_YYYYMMDD_HHMMSS.log
  test/logs/combined_YYYYMMDD_HHMMSS.log   (all three together)

Every step is logged:
  - Prompts sent to the LLM (full system prompt + first 600 chars of user prompt)
  - Full LLM responses
  - Every claim extracted from an answer
  - KG triples retrieved for each claim
  - Verdict + reasoning for each claim
  - Repair prompt and repaired answer (V1/V2)
  - Reflection generated and memory state (V2 only)
  - Per-question evaluation metrics
  - Aggregate summary at the end

Usage (run from the project root kg-claim-verifier/):
    python test/test_integration.py
    python test/test_integration.py --variants v0 v1
    python test/test_integration.py --variants v2 --questions 3
    python test/test_integration.py --all-questions     # run all 25
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
import time
from datetime import datetime
from typing import Any

# ── Path setup ─────────────────────────────────────────────────────────────────
_TEST_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_TEST_DIR)
_SRC_DIR     = os.path.join(_PROJECT_DIR, "src")
_DATA_DIR    = os.path.join(_PROJECT_DIR, "data")
_LOGS_DIR    = os.path.join(_TEST_DIR, "logs")   # base; model slug appended at runtime

sys.path.insert(0, _SRC_DIR)

# ── Test questions ─────────────────────────────────────────────────────────────
# Five representative questions chosen to maximise hallucination exposure:
#   • Specific numbers (prices, dates) LLMs commonly guess wrong
#   • Different categories: product pricing, company overview, HR policy, leadership
#   • Easy + medium + hard difficulty coverage
#
# Sourced from data/questions.json — key_facts verified against kg_triples.json.

DEFAULT_TEST_QUESTIONS = [
    {
        "id": "q09",
        "category": "products_pricing",
        "difficulty": "easy",
        "question": "What is the price of the NovaPilot Starter plan and what does it include?",
        "ground_truth_answer": (
            "The NovaPilot Starter plan costs $2,500/month. "
            "It includes up to 10 users and 50,000 automation runs per month."
        ),
        "key_facts": ["$2,500/month", "10 users", "50,000"],
    },
    {
        "id": "q02",
        "category": "company_overview",
        "difficulty": "medium",
        "question": "What is NovaAI's current valuation, and how was it achieved?",
        "ground_truth_answer": (
            "NovaAI reached a $1.4B valuation through its Series C round, "
            "which was completed in November 2025 and led by Horizon Ventures. "
            "Total funding stands at $210M."
        ),
        "key_facts": ["$1.4B", "Series C", "November 2025", "Horizon Ventures", "$210M"],
    },
    {
        "id": "q16",
        "category": "products_pricing",
        "difficulty": "easy",
        "question": "What is the price and model limit for NovaGuard Standard?",
        "ground_truth_answer": (
            "NovaGuard Standard costs $3,000/month and can monitor up to 3 AI models. "
            "It includes weekly reports and email alerts, and is available via self-serve."
        ),
        "key_facts": ["$3,000/month", "3", "weekly", "email alerts", "self-serve"],
    },
    {
        "id": "q21",
        "category": "hr_policy",
        "difficulty": "hard",
        "question": "How does NovaAI's performance review cycle work?",
        "ground_truth_answer": (
            "NovaAI runs a bi-annual performance review cycle. "
            "The mid-year review calibration happens in July with results by July 31. "
            "The annual review calibration is in January with results by January 31. "
            "Ratings use a 5-point scale. A rating of Developing or below triggers a PIP within 30 days."
        ),
        "key_facts": ["bi-annual", "July", "July 31", "January", "January 31", "5-point", "30 days"],
    },
    {
        "id": "q08",
        "category": "leadership",
        "difficulty": "medium",
        "question": "What is Marcus Webb's role, location, and FY2026 quota?",
        "ground_truth_answer": (
            "Marcus Webb is the Enterprise Sales team lead. "
            "He is based in New York with an approved fully remote work arrangement. "
            "His FY2026 quota is $18M."
        ),
        "key_facts": ["Marcus Webb", "Enterprise Sales", "New York", "fully remote", "$18M"],
    },
]


# ── Logging setup ──────────────────────────────────────────────────────────────

class _SectionFilter(logging.Filter):
    """Pass-through filter — lets us attach section markers to log records."""
    pass


def _build_logger(
    variant: str,
    timestamp: str,
    combined_handler: logging.FileHandler,
    logs_dir: str = _LOGS_DIR,
) -> logging.Logger:
    """
    Create a named logger that writes to:
      1. <logs_dir>/<variant>_<timestamp>.log
      2. The shared combined log file
      3. stdout (INFO+ level only)
    """
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{variant}_{timestamp}.log")

    logger = logging.getLogger(f"kg_test.{logs_dir}.{variant}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d] [%(levelname)-5s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Per-variant file handler (DEBUG level — captures everything)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Shared combined handler
    combined_handler.setFormatter(fmt)
    combined_handler.setLevel(logging.DEBUG)
    logger.addHandler(combined_handler)

    # Console handler (INFO only — less noisy)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger, log_path


# ── Prompt truncation helper ───────────────────────────────────────────────────

def _trunc(text: str, max_chars: int = 600) -> str:
    """Truncate long strings for log readability."""
    if text is None:
        return "(None)"
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n  ... [+{len(text) - max_chars} chars truncated]"


def _indent(text: str, prefix: str = "    ") -> str:
    """Indent a multi-line string for structured log output."""
    return "\n".join(prefix + line for line in str(text).splitlines())


# ── LLM call interceptor ───────────────────────────────────────────────────────

class _LLMInterceptor:
    """
    Wraps llm_client.call_llm and call_llm_json to log every call.
    Restores originals when used as a context manager.
    """

    def __init__(self, logger: logging.Logger, label: str):
        self.logger = logger
        self.label = label
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
        log = self.logger
        label = self.label

        def wrapped(prompt, model=None, system_prompt=None, **kwargs):
            import llm_client as _lc
            model = model or _lc.ACTION_MODEL
            log.debug(
                f"[{label}] LLM CALL  model={model}\n"
                f"  System : {_trunc(system_prompt, 400) if system_prompt else '(none)'}\n"
                f"  User   : {_trunc(prompt, 600)}"
            )
            t0 = time.time()
            result = original(prompt, model=model, system_prompt=system_prompt, **kwargs)
            elapsed = time.time() - t0
            log.debug(
                f"[{label}] LLM DONE  ({elapsed:.2f}s)\n"
                f"  Response:\n{_indent(_trunc(result, 800))}"
            )
            return result

        return wrapped

    def _wrap_call_llm_json(self, original):
        log = self.logger
        label = self.label

        def wrapped(prompt, model=None, system_prompt=None, **kwargs):
            import llm_client as _lc
            model = model or _lc.EXTRACT_MODEL
            log.debug(
                f"[{label}] LLM JSON CALL  model={model}\n"
                f"  System : {_trunc(system_prompt, 300) if system_prompt else '(none)'}\n"
                f"  User   : {_trunc(prompt, 500)}"
            )
            t0 = time.time()
            result = original(prompt, model=model, system_prompt=system_prompt, **kwargs)
            elapsed = time.time() - t0
            log.debug(
                f"[{label}] LLM JSON DONE  ({elapsed:.2f}s)\n"
                f"  Parsed : {json.dumps(result, indent=2)[:600]}"
            )
            return result

        return wrapped


# ── Entity linker interceptor ──────────────────────────────────────────────────

class _EntityLinkerInterceptor:
    """
    Wraps entity_linker.find_relevant_triples to log every KG lookup.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._orig = None

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

        def wrapped(claim, triples, top_n=5, min_score=55, method="llm"):
            log.debug(f"[ENTITY_LINKER] Lookup  claim=\"{_trunc(claim, 100)}\"  method={method}")
            t0 = time.time()
            results = original(claim, triples, top_n=top_n, min_score=min_score, method=method)
            elapsed = time.time() - t0
            if results:
                triple_lines = "\n".join(
                    f"    [{r.get('score','?'):3}]  {r['subject']} | {r['relation']} | {r['object']}"
                    for r in results
                )
                log.debug(
                    f"[ENTITY_LINKER] Found {len(results)} triple(s) ({elapsed:.2f}s):\n{triple_lines}"
                )
            else:
                log.debug(f"[ENTITY_LINKER] No triples found ({elapsed:.2f}s)")
            return results

        return wrapped


# ── Evaluation helper ──────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _evaluate(predicted: str, ground_truth: str, key_facts: list[str]) -> dict:
    pred_norm  = _normalise(predicted)
    pred_words = set(pred_norm.split())

    def _fact_present(fact: str) -> bool:
        # Word-level match: all significant words (len>2) of the fact must appear
        # in the prediction's word set. Avoids false negatives from punctuation
        # differences like "$2,500/month" vs "$2,500 per month".
        fact_words = [w for w in _normalise(fact).split() if len(w) > 2]
        return bool(fact_words) and all(w in pred_words for w in fact_words)

    found = sum(1 for f in key_facts if _fact_present(f))
    kfr   = found / len(key_facts) if key_facts else 0.0
    return {
        "exact_match":     _normalise(ground_truth) == pred_norm,
        "key_fact_recall": round(kfr, 3),
        "is_correct":      kfr >= 0.8,
        "facts_found":     found,
        "facts_total":     len(key_facts),
        "facts_missing":   [f for f in key_facts if not _fact_present(f)],
    }


# ── Section header helpers ─────────────────────────────────────────────────────

def _banner(logger: logging.Logger, text: str, width: int = 72, char: str = "=") -> None:
    logger.info(char * width)
    logger.info(f"  {text}")
    logger.info(char * width)


def _divider(logger: logging.Logger, width: int = 72, char: str = "─") -> None:
    logger.info(char * width)


# ── Per-variant test runner ────────────────────────────────────────────────────

def run_variant_test(
    variant: str,
    questions: list[dict],
    triples: list[dict],
    logger: logging.Logger,
    model: str | None = None,
) -> list[dict]:
    """
    Run a single variant (v0 / v1 / v1rag / v2) over all test questions.
    If *model* is given it overrides the default ACTION/EXTRACT/REFLECT models.
    Returns a list of per-question result dicts.
    """
    from pipeline import V0Pipeline, V1Pipeline, V2Pipeline
    from reflexion_layer import ReflexionMemory
    import llm_client as _lc

    effective_model = model or _lc.ACTION_MODEL

    _banner(logger, f"VARIANT: {variant.upper()}  |  {len(questions)} questions  |  model: {effective_model}", char="═")

    # Instantiate pipeline with the selected model
    if variant == "v0":
        pipeline = V0Pipeline(model=effective_model)
    elif variant == "v1":
        pipeline = V1Pipeline(triples, model=effective_model)
    elif variant == "v1rag":
        from pipeline import V1RAGPipeline
        pipeline = V1RAGPipeline(triples, model=effective_model)
    else:  # v2
        shared_memory = ReflexionMemory(max_reflections=3)
        pipeline = V2Pipeline(triples, model=effective_model, memory=shared_memory)

    logger.info(f"Pipeline: {pipeline.__class__.__name__} initialised")
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

            # ── Question header ───────────────────────────────────────────────
            _divider(logger, char="━")
            logger.info(
                f"Question {qi}/{len(questions)}  [{qid}]  "
                f"category={category}  difficulty={difficulty}"
            )
            logger.info(f"  Q: {question}")
            _divider(logger, char="━")

            t_start = time.time()

            # ── V2: log memory state before this question ─────────────────────
            if variant == "v2":
                mem = pipeline.memory
                if mem.has_reflections():
                    logger.info(
                        f"[V2 MEMORY] {len(mem.reflections)} reflection(s) in session memory:"
                    )
                    for i, r in enumerate(mem.reflections, 1):
                        logger.info(f"  [{i}] {r}")
                else:
                    logger.info("[V2 MEMORY] Session memory is empty (first question or after reset)")
                logger.info("")

            # ── Run pipeline ──────────────────────────────────────────────────
            logger.info(f"[{variant.upper()}] Running pipeline...")
            try:
                result = pipeline.run(question)
            except Exception as exc:
                logger.error(f"[{variant.upper()}] Pipeline FAILED: {exc}", exc_info=True)
                all_results.append({
                    "id": qid, "category": category, "difficulty": difficulty,
                    "question": question, "error": str(exc),
                    "is_correct": False, "key_fact_recall": 0.0,
                })
                continue

            t_elapsed = time.time() - t_start

            # ── Log raw answer ────────────────────────────────────────────────
            logger.info("")
            logger.info("── RAW ANSWER ──")
            logger.info(_indent(result.raw_answer))

            # ── V1/V2: log claims ─────────────────────────────────────────────
            if result.claims is not None:
                logger.info("")
                logger.info(f"── CLAIMS EXTRACTED  ({len(result.claims)}) ──")
                for i, c in enumerate(result.claims, 1):
                    logger.info(f"  {i:2d}. {c}")

            # ── V1/V2: log verification results ──────────────────────────────
            if result.verification_results is not None:
                logger.info("")
                logger.info("── VERIFICATION RESULTS ──")
                for vr in result.verification_results:
                    verdict   = vr["verdict"]
                    conf      = vr.get("confidence", 0.0)
                    reasoning = vr.get("reasoning", "")
                    evidence  = vr.get("evidence_triples", [])

                    # Choose log level based on verdict
                    level = logging.INFO
                    if verdict == "CONTRADICTED":
                        level = logging.WARNING
                    elif verdict == "UNVERIFIABLE":
                        level = logging.DEBUG

                    logger.log(
                        level,
                        f"  [{verdict:<13}] (conf={conf:.2f})  \"{_trunc(vr['claim'], 80)}\""
                    )
                    logger.debug(f"    Reasoning : {reasoning}")
                    if evidence:
                        logger.debug(
                            f"    Evidence  : {evidence[0]['subject']} | "
                            f"{evidence[0]['relation']} | {evidence[0]['object']}"
                        )

            # ── V1/V2: log repair ─────────────────────────────────────────────
            if result.had_hallucinations is not None:
                logger.info("")
                if result.had_hallucinations:
                    logger.warning("── REPAIR (hallucinations detected) ──")
                    logger.warning(f"  Claims repaired: {len([vr for vr in (result.verification_results or []) if vr['verdict'] == 'CONTRADICTED'])}")
                    logger.info("  Repaired answer:")
                    logger.info(_indent(result.final_answer))
                else:
                    logger.info("── REPAIR: not needed (no contradictions) ──")

            # ── V2: log reflection and updated memory ─────────────────────────
            if variant == "v2":
                logger.info("")
                mem = pipeline.memory
                # result.reflections_used = pre-run snapshot (what was injected)
                # pipeline.memory.reflections = post-run state (may have new entry)
                injected_count = len(result.reflections_used or [])
                post_run_count = len(mem.reflections)
                if result.had_hallucinations:
                    # A new reflection was appended if post-run count grew
                    if post_run_count > injected_count and mem.reflections:
                        newest = mem.reflections[-1]
                        logger.info("── V2 REFLECTION GENERATED ──")
                        logger.info(f"  {newest}")
                    logger.info(
                        f"\n[V2 MEMORY] Updated — now {post_run_count} "
                        f"reflection(s) in session memory"
                    )
                else:
                    logger.info(
                        f"[V2 MEMORY] No new reflection (no contradictions). "
                        f"Memory unchanged: {post_run_count} reflection(s)"
                    )

            # ── Final answer ──────────────────────────────────────────────────
            if result.had_hallucinations is not None and result.had_hallucinations:
                pass  # Already logged in repair section
            elif result.claims is None:
                # V0: no repair, raw = final
                logger.info("")
                logger.info("── FINAL ANSWER (V0 — no verification) ──")
                logger.info(_indent(result.final_answer))

            # ── Evaluation ────────────────────────────────────────────────────
            metrics = _evaluate(result.final_answer, gt_answer, key_facts)
            correct_marker = "✓" if metrics["is_correct"] else "✗"

            logger.info("")
            logger.info("── EVALUATION ──")
            logger.info(f"  Ground truth  : {gt_answer}")
            logger.info(f"  Key facts     : {key_facts}")
            logger.info(f"  KFR           : {metrics['key_fact_recall']:.3f} "
                        f"({metrics['facts_found']}/{metrics['facts_total']} facts found) {correct_marker}")
            if metrics["facts_missing"]:
                logger.warning(f"  Missing facts : {metrics['facts_missing']}")
            logger.info(f"  Exact match   : {metrics['exact_match']}")
            logger.info(f"  Latency       : {t_elapsed:.2f}s")
            logger.info("")

            all_results.append({
                "id": qid,
                "category": category,
                "difficulty": difficulty,
                "question": question,
                "raw_answer": result.raw_answer,
                "final_answer": result.final_answer,
                "claims": result.claims,
                "had_hallucinations": result.had_hallucinations,
                "reflections_used": result.reflections_used,
                "latency_seconds": round(t_elapsed, 2),
                **metrics,
            })

    # ── Variant summary ───────────────────────────────────────────────────────
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

    logger.info("Per-question results:")
    logger.info(f"  {'ID':<6} {'Category':<20} {'Diff':<8} {'KFR':<7} {'OK?':<5} {'Hallucs':<8} {'Latency'}")
    logger.info("  " + "-" * 62)
    for r in all_results:
        ok      = "✓" if r.get("is_correct") else "✗"
        h       = "yes" if r.get("had_hallucinations") is True else ("no" if r.get("had_hallucinations") is False else "N/A")
        latency = f"{r.get('latency_seconds', 0.0):.1f}s"
        logger.info(
            f"  {r['id']:<6} {r['category']:<20} {r['difficulty']:<8} "
            f"{r.get('key_fact_recall', 0.0):<7.3f} {ok:<5} {h:<8} {latency}"
        )
    logger.info("")

    return all_results


# ── Comparative summary ────────────────────────────────────────────────────────

def _log_comparison(
    combined_logger: logging.Logger,
    all_variant_results: dict[str, list[dict]],
    variants: list[str],
) -> None:
    _banner(combined_logger, "COMPARISON ACROSS VARIANTS", char="═")

    rows: list[list[str]] = []
    for v in variants:
        results = all_variant_results.get(v, [])
        n       = len(results)
        correct = sum(1 for r in results if r.get("is_correct", False))
        avg_kfr = sum(r.get("key_fact_recall", 0.0) for r in results) / max(n, 1)
        n_h     = sum(1 for r in results if r.get("had_hallucinations") is True)
        avg_lat = sum(r.get("latency_seconds", 0.0) for r in results) / max(n, 1)
        h_str   = f"{n_h}/{n}" if any(r.get("had_hallucinations") is not None for r in results) else "N/A"
        rows.append([
            v.upper(),
            f"{correct}/{n} ({correct/max(n,1):.1%})",
            f"{avg_kfr:.3f}",
            h_str,
            f"{avg_lat:.1f}s",
        ])

    headers = ["Variant", "Correct", "Avg KFR", "Hallucs", "Avg Latency"]
    col_w   = [10, 16, 10, 10, 12]
    header_line = "  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    sep_line    = "  " + "-" * (sum(col_w) + 2 * len(col_w))
    combined_logger.info(header_line)
    combined_logger.info(sep_line)
    for row in rows:
        combined_logger.info("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, col_w)))
    combined_logger.info("")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integration test for KG-Backed Claim Verifier (V0 / V1 / V2)."
    )
    parser.add_argument(
        "--variants", nargs="+", choices=["v0", "v1", "v1rag", "v2"],
        default=["v0", "v1", "v1rag", "v2"],
        help="Which variants to test (default: all four).",
    )
    parser.add_argument(
        "--questions", type=int, default=5,
        help="Number of test questions to use (1–5, default: 5).",
    )
    parser.add_argument(
        "--all-questions", action="store_true",
        help="Run against all 25 questions from data/questions.json instead of the default 5.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=(
            "NVIDIA NIM model to use for all pipeline tasks. "
            "Overrides ACTION_MODEL / EXTRACT_MODEL / REFLECT_MODEL. "
            "Example: --model google/gemma-4-31b-it"
        ),
    )
    args = parser.parse_args()

    # ── Apply model override ───────────────────────────────────────────────────
    # Patch the three llm_client module-level constants before any pipeline
    # modules are imported so every downstream call uses the selected model.
    import llm_client as _lc
    if args.model:
        _lc.ACTION_MODEL  = args.model
        _lc.REFLECT_MODEL = args.model
        _lc.EXTRACT_MODEL = args.model
        print(f"[INFO] Model override: {args.model}")
    selected_model = _lc.ACTION_MODEL

    # ── Resolve test questions ─────────────────────────────────────────────────
    if args.all_questions:
        questions_path = os.path.join(_DATA_DIR, "questions.json")
        with open(questions_path, encoding="utf-8") as f:
            test_questions = json.load(f)["questions"]
        print(f"[INFO] Using all {len(test_questions)} questions from questions.json")
    else:
        n_q = max(1, min(args.questions, len(DEFAULT_TEST_QUESTIONS)))
        test_questions = DEFAULT_TEST_QUESTIONS[:n_q]
        print(f"[INFO] Using {n_q} default test question(s)")

    # ── Timestamp for this run ─────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    variants = args.variants

    # ── Per-model log directory ────────────────────────────────────────────────
    from model_registry import model_to_slug
    model_slug = model_to_slug(selected_model)
    run_logs_dir = os.path.join(_LOGS_DIR, model_slug)
    os.makedirs(run_logs_dir, exist_ok=True)

    # ── Shared combined log ────────────────────────────────────────────────────
    combined_path = os.path.join(run_logs_dir, f"combined_{ts}.log")
    combined_fh   = logging.FileHandler(combined_path, encoding="utf-8")

    # ── Load KG triples ────────────────────────────────────────────────────────
    kg_path = os.path.join(_DATA_DIR, "kg_triples.json")
    with open(kg_path, encoding="utf-8") as f:
        triples = json.load(f)

    # ── Run each variant ───────────────────────────────────────────────────────
    all_variant_results: dict[str, list[dict]] = {}

    for variant in variants:
        logger, log_path = _build_logger(variant, ts, combined_fh, logs_dir=run_logs_dir)

        # Run-header
        _banner(logger, f"KG-BACKED CLAIM VERIFIER — INTEGRATION TEST", char="═")
        logger.info(f"  Variant   : {variant.upper()}")
        logger.info(f"  Model     : {selected_model}")
        logger.info(f"  Questions : {len(test_questions)}")
        logger.info(f"  KG triples: {len(triples)}")
        logger.info(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Log file  : {log_path}")
        logger.info("")

        results = run_variant_test(variant, test_questions, triples, logger, model=selected_model)
        all_variant_results[variant] = results

        logger.info(f"Log written to: {log_path}")
        # Flush per-variant handlers before moving to next variant
        for h in logger.handlers:
            h.flush()

    # ── Write comparison to combined log ───────────────────────────────────────
    if len(variants) > 1:
        combined_logger = logging.getLogger("kg_test.combined")
        combined_logger.setLevel(logging.DEBUG)
        combined_logger.propagate = False
        if not combined_logger.handlers:
            combined_fh.setFormatter(
                logging.Formatter(
                    fmt="[%(asctime)s.%(msecs)03d] [%(levelname)-5s] %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            combined_logger.addHandler(combined_fh)
        _log_comparison(combined_logger, all_variant_results, variants)
        combined_fh.flush()

    # ── Print log file paths to stdout ─────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  INTEGRATION TEST COMPLETE")
    print(f"  Model: {selected_model}")
    print("=" * 65)
    for variant in variants:
        log_path = os.path.join(run_logs_dir, f"{variant}_{ts}.log")
        print(f"  {variant.upper()} log  → {log_path}")
    if len(variants) > 1:
        print(f"  Combined  → {combined_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
