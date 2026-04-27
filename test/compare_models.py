#!/usr/bin/env python3
"""
compare_models.py
-----------------
Read results from results/<model-slug>/<variant>_results.json and print a
cross-model, cross-variant comparison table.

Usage (from the project root):
    python test/compare_models.py
    python test/compare_models.py --results-dir results
    python test/compare_models.py --variants v0 v1
    python test/compare_models.py --sort kfr      # sort by avg KFR descending
    python test/compare_models.py --sort accuracy  # sort by accuracy descending
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE        = Path(__file__).parent
_PROJECT_DIR = _HERE.parent
_RESULTS_DIR = _PROJECT_DIR / "results"

sys.path.insert(0, str(_PROJECT_DIR / "src"))


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_results(results_dir: Path, variants: list[str]) -> dict[str, dict[str, dict]]:
    """
    Scan results/<model-slug>/<variant>_results.json.

    Returns:
        { model_slug: { variant: aggregate_metrics_dict } }
    """
    data: dict[str, dict[str, dict]] = {}

    if not results_dir.exists():
        print(f"[WARN] Results directory not found: {results_dir}")
        return data

    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_slug = model_dir.name
        for variant in variants:
            result_file = model_dir / f"{variant}_results.json"
            if not result_file.exists():
                continue
            try:
                with open(result_file, encoding="utf-8") as f:
                    payload = json.load(f)
                agg = payload.get("aggregate_metrics", {})
                if agg:
                    data.setdefault(model_slug, {})[variant] = agg
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[WARN] Could not read {result_file}: {exc}")

    return data


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _f(value: float | None, decimals: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def _lat(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}s"


# ── Table printing ─────────────────────────────────────────────────────────────

def _print_table(rows: list[list[str]], headers: list[str]) -> None:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"

    print(sep)
    print(header_line)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths)) + " |")
    print(sep)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-model comparison of KG claim verifier experiment results."
    )
    parser.add_argument(
        "--results-dir", type=str, default=str(_RESULTS_DIR),
        help="Base results directory containing <model-slug>/ subdirectories.",
    )
    parser.add_argument(
        "--variants", nargs="+", default=["v0", "v1", "v1rag", "v2"],
        help="Variants to include in the comparison (default: all four).",
    )
    parser.add_argument(
        "--sort", choices=["kfr", "accuracy", "hallucination", "latency", "model"],
        default="model",
        help="Sort rows by metric (default: model name).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    variants    = args.variants

    all_data = _load_results(results_dir, variants)

    if not all_data:
        print("No results found. Run the integration test or experiments first.")
        print(f"  Expected path: {results_dir}/<model-slug>/<variant>_results.json")
        sys.exit(0)

    # ── Per-variant comparison tables ──────────────────────────────────────────
    for variant in variants:
        rows: list[tuple] = []
        for model_slug, variant_map in all_data.items():
            m = variant_map.get(variant)
            if m is None:
                continue
            rows.append((
                model_slug,
                m.get("accuracy"),
                m.get("avg_kfr"),
                m.get("hallucination_rate"),
                m.get("avg_latency_seconds"),
                m.get("total_questions", 0),
            ))

        if not rows:
            continue

        # Sort
        sort_key = {
            "model":        lambda r: r[0],
            "accuracy":     lambda r: -(r[1] or 0),
            "kfr":          lambda r: -(r[2] or 0),
            "hallucination":lambda r: (r[3] or 0),
            "latency":      lambda r: (r[4] or 0),
        }[args.sort]
        rows.sort(key=sort_key)

        headers = ["Model", "Accuracy", "Avg KFR", "Hallucination%", "Avg Latency", "N"]
        table_rows = [
            [slug, _pct(acc), _f(kfr), _pct(hall), _lat(lat), str(n)]
            for slug, acc, kfr, hall, lat, n in rows
        ]

        print(f"\n{'=' * 72}")
        print(f"  VARIANT: {variant.upper()}")
        print(f"{'=' * 72}")
        _print_table(table_rows, headers)

    # ── Cross-variant summary per model ────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  CROSS-VARIANT SUMMARY  (v0 → v1 → v1rag → v2 improvement)")
    print(f"{'=' * 72}")

    for model_slug in sorted(all_data):
        variant_map = all_data[model_slug]
        if len(variant_map) < 2:
            continue

        print(f"\n  Model: {model_slug}")
        row_headers = ["Variant", "Accuracy", "Avg KFR", "Hallucination%", "Avg Latency"]
        table_rows = []
        for v in variants:
            m = variant_map.get(v)
            if m is None:
                continue
            table_rows.append([
                v.upper(),
                _pct(m.get("accuracy")),
                _f(m.get("avg_kfr")),
                _pct(m.get("hallucination_rate")),
                _lat(m.get("avg_latency_seconds")),
            ])
        if table_rows:
            _print_table(table_rows, row_headers)

    # ── Best model per variant ──────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  BEST MODEL PER VARIANT  (by Avg KFR)")
    print(f"{'=' * 72}")

    for variant in variants:
        best_slug, best_kfr = None, -1.0
        for model_slug, variant_map in all_data.items():
            m = variant_map.get(variant)
            if m and (m.get("avg_kfr") or 0) > best_kfr:
                best_kfr  = m["avg_kfr"]
                best_slug = model_slug
        if best_slug:
            print(f"  {variant.upper():<8} → {best_slug}  (KFR={best_kfr:.3f})")

    print()


if __name__ == "__main__":
    main()
