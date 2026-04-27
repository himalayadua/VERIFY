"""
rescore_results.py
------------------
Re-apply the fixed KFR metric to all existing *_results.json files without
re-running any LLM calls.  Useful after patching src/metrics.py to correct
the numeric-tokenisation bug (len>2 filter → stop-word filter, currency
collapse, digit-comma-digit collapse).

Writes a new file with _rescored.json suffix alongside the original.

Usage (run from project root):
    python scripts/rescore_results.py                        # all results/
    python scripts/rescore_results.py --dir results/gpt-oss-120b
    python scripts/rescore_results.py --file results/gpt-oss-120b/v2p_results.json
    python scripts/rescore_results.py --in-place             # overwrite originals
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow importing from src/
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from metrics import key_fact_recall, task_passes_kfr  # noqa: E402


def rescore_file(path: str | Path, in_place: bool = False) -> Path:
    """
    Re-compute key_fact_recall and is_correct for every question record in a
    *_results.json file, then recompute aggregate accuracy / avg_kfr.

    Parameters
    ----------
    path     : path to an existing *_results.json
    in_place : if True, overwrite the original; otherwise write *_rescored.json

    Returns
    -------
    Path of the written output file.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    questions = data.get("questions", [])
    if not questions:
        print(f"  [SKIP] {path.name} — no 'questions' key")
        return path

    n_changed = 0
    for q in questions:
        kf = q.get("key_facts") or []
        if not kf:
            continue

        # Use final_answer (post-repair) for scoring, same as evaluate_answer()
        answer = str(q.get("final_answer") or q.get("raw_answer") or "")
        kfr, kfr_ok = task_passes_kfr(answer, kf, threshold=0.8)
        kfr = round(kfr, 3)

        old_kfr = q.get("key_fact_recall")
        old_ok = q.get("is_correct")
        if old_kfr != kfr or old_ok != kfr_ok:
            n_changed += 1

        q["key_fact_recall"] = kfr
        q["is_correct"] = kfr_ok

        # Also rescore trial-level kfr if present (v2p multi-trial trace)
        for trial in q.get("trials") or []:
            t_answer = str(trial.get("repaired_answer") or trial.get("raw_answer") or "")
            t_kfr, _ = task_passes_kfr(t_answer, kf, threshold=0.8)
            trial["kfr"] = round(t_kfr, 3)

    # Recompute aggregate metrics
    n = len(questions)
    correct = sum(1 for q in questions if q.get("is_correct", False))
    avg_kfr = sum(q.get("key_fact_recall", 0.0) for q in questions) / max(n, 1)

    agg = data.get("aggregate_metrics", {})
    agg["accuracy"] = round(correct / max(n, 1), 3)
    agg["avg_kfr"] = round(avg_kfr, 3)
    agg["total_questions"] = n
    data["aggregate_metrics"] = agg

    if in_place:
        out_path = path
    else:
        out_path = path.with_name(path.stem + "_rescored.json")

    out_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"  [{n_changed:3d} changed]  {path.name}  →  {out_path.name}"
        f"  (accuracy={agg['accuracy']:.1%}  avg_kfr={agg['avg_kfr']:.3f})"
    )
    return out_path


def rescore_dir(directory: str | Path, in_place: bool = False) -> list[Path]:
    """Rescore all *_results.json files in *directory* (non-recursive)."""
    directory = Path(directory)
    files = sorted(directory.glob("*_results.json"))
    if not files:
        # Try subdirs (model-slug level)
        files = sorted(directory.glob("*/*_results.json"))
    if not files:
        print(f"  [WARN] No *_results.json found under {directory}")
        return []
    out: list[Path] = []
    for f in files:
        if "_rescored" in f.stem:
            continue
        out.append(rescore_file(f, in_place=in_place))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Re-score existing *_results.json with the fixed KFR metric."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--file", type=str, default=None,
        help="Rescore a single *_results.json file.",
    )
    group.add_argument(
        "--dir", type=str, default=None,
        help="Rescore all *_results.json files in this directory.",
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Overwrite original files instead of writing *_rescored.json.",
    )
    args = parser.parse_args()

    default_results_dir = _HERE.parent / "results"

    print("=" * 65)
    print("  KG-BACKED CLAIM VERIFIER — RESCORE (fixed KFR metric)")
    print("=" * 65)

    if args.file:
        rescore_file(args.file, in_place=args.in_place)
    else:
        target = Path(args.dir) if args.dir else default_results_dir
        rescore_dir(target, in_place=args.in_place)

    print("=" * 65)
    print("  Done.")
    print("=" * 65)
