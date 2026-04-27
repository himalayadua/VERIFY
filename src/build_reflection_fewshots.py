#!/usr/bin/env python3
"""
build_reflection_fewshots.py
----------------------------
Scan results JSON for hallucinated runs and emit a scaffold JSON for
hand-curated reflection few-shot examples.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))


def _find_default_results() -> str | None:
    base = os.path.join(_ROOT, "results")
    if not os.path.isdir(base):
        return None
    for slug in sorted(os.listdir(base)):
        p = os.path.join(base, slug, "v1_results.json")
        if os.path.isfile(p):
            return p
    legacy = os.path.join(_ROOT, "results", "v1_results.json")
    return legacy if os.path.isfile(legacy) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results",
        default=_find_default_results() or "",
        help="Path to v1 (or v2) results JSON with per-question records.",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(_ROOT, "data", "reflection_fewshot_scaffold.json"),
        help="Output scaffold JSON path.",
    )
    args = ap.parse_args()

    if not args.results or not os.path.isfile(args.results):
        print(f"[WARN] No results file at {args.results!r}; writing empty scaffold.")
        rows: list[dict] = []
    else:
        with open(args.results, encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for q in data.get("questions", []):
            if not q.get("had_hallucinations"):
                continue
            vrs = q.get("verification_results") or []
            contrad = [v for v in vrs if v.get("verdict") == "CONTRADICTED"]
            if not contrad:
                continue
            err_parts = []
            for v in contrad[:3]:
                ev = (v.get("evidence_triples") or [{}])[0]
                err_parts.append(
                    f'Claimed: "{v.get("claim", "")}" '
                    f'vs KG: {ev.get("subject")}|{ev.get("relation")}|{ev.get("object")}'
                )
            rows.append(
                {
                    "id": q.get("id"),
                    "question": q.get("question"),
                    "original_answer": q.get("raw_answer", "")[:800],
                    "error_summary": "\n".join(err_parts),
                    "suggested_reflection": "",
                }
            )

    out = {"examples": rows[:20]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(rows)} scaffold row(s) → {args.out}")


if __name__ == "__main__":
    main()
