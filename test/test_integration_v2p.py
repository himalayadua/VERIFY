#!/usr/bin/env python3
"""
test_integration_v2p.py
-----------------------
Integration harness for V2PaperPipeline (paper-style Reflexion) only.
Does not modify or invoke test_integration.py or V0/V1/V1RAG/V2 pipelines.
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

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_TEST_DIR)
_SRC_DIR = os.path.join(_PROJECT_DIR, "src")
_DATA_DIR = os.path.join(_PROJECT_DIR, "data")
_LOGS_DIR = os.path.join(_TEST_DIR, "logs")

sys.path.insert(0, _SRC_DIR)

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
]


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _evaluate(predicted: str, ground_truth: str, key_facts: list[str]) -> dict:
    pred_norm = _normalise(predicted)
    pred_words = set(pred_norm.split())

    def _fact_present(fact: str) -> bool:
        fact_words = [w for w in _normalise(fact).split() if len(w) > 2]
        return bool(fact_words) and all(w in pred_words for w in fact_words)

    found = sum(1 for f in key_facts if _fact_present(f))
    kfr = found / len(key_facts) if key_facts else 0.0
    return {
        "exact_match": _normalise(ground_truth) == pred_norm,
        "key_fact_recall": round(kfr, 3),
        "is_correct": kfr >= 0.8,
        "key_facts_found": found,
        "key_facts_total": len(key_facts),
        "facts_missing": [f for f in key_facts if not _fact_present(f)],
    }


def _build_logger(variant: str, timestamp: str, combined_handler: logging.FileHandler, logs_dir: str):
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{variant}_{timestamp}.log")
    logger = logging.getLogger(f"kg_test_v2p.{logs_dir}.{variant}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    fmt = logging.Formatter(fmt="[%(asctime)s.%(msecs)03d] [%(levelname)-5s] %(message)s", datefmt="%H:%M:%S")
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
    return text if len(text) <= max_chars else text[:max_chars] + f"\n  ... [+{len(text) - max_chars} chars]"


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in str(text).splitlines())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V2P / V2P-RAG (paper Reflexion) integration harness — does not run V0/V1/V1RAG/V2."
    )
    parser.add_argument(
        "--pipeline",
        default="v2p",
        choices=["v2p", "v2p_rag"],
        help="v2p = standard paper Reflexion; v2p_rag = RAG in answer step + V2P verify/repair/reflection",
    )
    parser.add_argument(
        "--strategy",
        default="reflexion",
        choices=["none", "last_attempt", "reflexion", "last_attempt_and_reflexion"],
    )
    parser.add_argument("--inject", nargs="+", default=["answer"], help="answer verifier repairer all")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument("--with-short-memory", action="store_true")
    parser.add_argument("--no-short-memory", action="store_true")
    parser.add_argument("--with-long-memory", action="store_true")
    parser.add_argument("--no-long-memory", action="store_true")
    parser.add_argument("--with-all-memory", action="store_true")
    parser.add_argument("--few-shots", type=str, default=None)
    parser.add_argument("--gold-claims", type=str, default=None)
    parser.add_argument("--questions", type=int, default=1)
    parser.add_argument("--all-questions", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument(
        "--paper-me",
        action="store_true",
        help="Paper-aligned Me: pass me_context from question key_facts (oracle KFR stop + me-fail reflections).",
    )
    parser.add_argument(
        "--paper-me-mode",
        type=str,
        default="kfr_and_no_contradiction",
        choices=["legacy", "kfr", "kfr_and_no_contradiction"],
    )
    parser.add_argument("--paper-me-threshold", type=float, default=0.8)
    parser.add_argument(
        "--paper-me-abandon",
        type=int,
        default=3,
        help="Consecutive Me-fail cap (0 = no cap).",
    )
    args = parser.parse_args()

    import llm_client as _lc

    if args.model:
        _lc.ACTION_MODEL = _lc.REFLECT_MODEL = _lc.EXTRACT_MODEL = args.model

    from model_registry import model_to_slug
    from entity_linker import load_kg
    from v2_paper_pipeline import V2PaperPipeline, load_reflection_fewshots, parse_strategy, parse_inject_into
    from reflexion_layer import ReflexionMemory
    if args.pipeline == "v2p_rag":
        from v2p_rag_pipeline import V2pRagPipeline as V2PCls
    else:
        V2PCls = V2PaperPipeline

    short = bool(args.with_short_memory)
    long_m = True
    if args.no_short_memory:
        short = False
    if args.with_long_memory:
        long_m = True
    if args.no_long_memory:
        long_m = False

    few_list = None
    if args.few_shots and os.path.isfile(args.few_shots):
        few_list = load_reflection_fewshots(args.few_shots)

    gold_by_id = None
    if args.gold_claims and os.path.isfile(args.gold_claims):
        with open(args.gold_claims, encoding="utf-8") as gf:
            gold_by_id = json.load(gf)

    if args.all_questions:
        with open(os.path.join(_DATA_DIR, "questions.json"), encoding="utf-8") as f:
            questions = json.load(f)["questions"]
    else:
        n = max(1, min(args.questions, len(DEFAULT_TEST_QUESTIONS)))
        questions = DEFAULT_TEST_QUESTIONS[:n]

    kg_path = os.path.join(_DATA_DIR, "kg_triples.json")
    triples = load_kg(kg_path)

    ws = max(0, int(args.window))
    shared = ReflexionMemory(max_reflections=max(ws, 0))
    if ws > 0:
        shared.max_reflections = ws

    pl_tag = "v2p_rag" if args.pipeline == "v2p_rag" else "v2p"
    pipeline = V2PCls(
        triples,
        model=_lc.ACTION_MODEL,
        strategy=parse_strategy(args.strategy),
        max_trials=int(args.max_trials),
        window_size=ws,
        inject_into=parse_inject_into(list(args.inject)),
        with_short_memory=short,
        with_long_memory=long_m,
        with_all_memory=bool(args.with_all_memory),
        long_memory=shared,
        few_shot_examples=few_list,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_to_slug(_lc.ACTION_MODEL)
    run_logs_dir = os.path.join(_LOGS_DIR, model_slug)
    combined_path = os.path.join(run_logs_dir, f"{pl_tag}_combined_{ts}.log")
    combined_fh = logging.FileHandler(combined_path, encoding="utf-8")
    logger, log_path = _build_logger(pl_tag, ts, combined_fh, run_logs_dir)

    logger.info(f"{'V2P-RAG' if pl_tag == 'v2p_rag' else 'V2P'} INTEGRATION TEST")
    logger.info(f"  strategy={args.strategy} inject={args.inject} window={ws} max_trials={args.max_trials}")
    logger.info(f"  short_mem={short} long_mem={long_m} all_mem={args.with_all_memory}")
    logger.info(f"  paper_me={args.paper_me} mode={args.paper_me_mode}")
    logger.info(f"  questions={len(questions)} model={_lc.ACTION_MODEL}")

    from metrics import compute_question_extended_metrics

    for qi, q in enumerate(questions, 1):
        qid = q["id"]
        logger.info(f"\n--- Question {qi}/{len(questions)} [{qid}] ---\n{q['question']}")
        t0 = time.time()
        try:
            me_ctx = None
            if args.paper_me:
                ab = int(args.paper_me_abandon)
                me_ctx = {
                    "me_mode": args.paper_me_mode,
                    "key_facts": list(q["key_facts"]),
                    "kfr_threshold": float(args.paper_me_threshold),
                    "ground_truth_answer": q["ground_truth_answer"],
                    "abandon_after_consecutive_me_fails": ab if ab > 0 else None,
                }
            if me_ctx is not None:
                result = pipeline.run(q["question"], me_context=me_ctx)
            else:
                result = pipeline.run(q["question"])
        except Exception as exc:
            logger.error(f"FAILED: {exc}", exc_info=True)
            continue
        elapsed = time.time() - t0
        metrics = _evaluate(result.final_answer, q["ground_truth_answer"], q["key_facts"])
        rec = {**result.to_dict(), **metrics, "id": qid, "question": q["question"], "key_facts": q["key_facts"]}
        if gold_by_id and qid in gold_by_id:
            rec["extended_metrics"] = compute_question_extended_metrics(rec, gold_by_id[qid])
            logger.info(f"extended_metrics: {json.dumps(rec.get('extended_metrics'), indent=2)[:1200]}")
        logger.info(f"KFR={metrics['key_fact_recall']} correct={metrics['is_correct']} latency={elapsed:.1f}s")
        logger.info(_trunc(result.final_answer, 800))
        if result.trials:
            logger.info(f"trials: {len(result.trials)} stopped={result.stopped_reason}")

    logger.info(f"\nLog: {log_path}\nCombined: {combined_path}")
    print(f"Done. Log: {log_path}")


def test_v2p_paper_me_exhausts_max_trials_when_kfr_fails() -> None:
    """Paper Me (kfr_and_no_contradiction): Me stays false until max_trials (mocked LLM stack)."""
    from unittest.mock import patch

    from claim_extractor import Claim
    from entity_linker import load_kg
    from repairer import RepairResult
    from reflexion_layer import ReflexionMemory
    from verifier import VerificationResult, Verdict
    from v2_paper_pipeline import V2PaperPipeline

    kg_path = os.path.join(_DATA_DIR, "kg_triples.json")
    triples = load_kg(kg_path)
    mem = ReflexionMemory(max_reflections=3)
    pipeline = V2PaperPipeline(
        triples,
        model="stub-model",
        max_trials=3,
        window_size=3,
        with_long_memory=True,
        long_memory=mem,
    )
    me_ctx = {
        "me_mode": "kfr_and_no_contradiction",
        "key_facts": ["$2,500/month", "10 users"],
        "kfr_threshold": 0.8,
        "abandon_after_consecutive_me_fails": None,
    }
    vr_ok = VerificationResult(claim="c", verdict=Verdict.SUPPORTED, confidence=1.0)

    with patch("v2_paper_pipeline.call_llm", return_value="no numbers"):
        with patch(
            "v2_paper_pipeline.extract_claims",
            return_value=[Claim(text="t", source_span="no numbers")],
        ):
            with patch("v2_paper_pipeline.verify_claims", return_value=[vr_ok]):
                with patch(
                    "v2_paper_pipeline.repair_answer",
                    return_value=RepairResult(repaired_answer="no numbers", had_hallucinations=False),
                ):
                    with patch(
                        "v2_paper_pipeline.generate_reflection_me_fail", return_value="reflect"
                    ):
                        result = pipeline.run("q?", me_context=me_ctx)

    assert len(result.trials) == 3
    assert result.stopped_reason == "max_trials_exhausted"
    assert all("kfr" in t and "me_pass" in t for t in result.trials)
    assert result.trials[0]["me_mode"] == "kfr_and_no_contradiction"
    assert result.trials[0]["me_pass"] is False


if __name__ == "__main__":
    main()
