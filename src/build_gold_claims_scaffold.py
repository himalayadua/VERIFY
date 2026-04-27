#!/usr/bin/env python3
"""
build_gold_claims_scaffold.py
-----------------------------
Build a starter gold_claims.json from questions.json + kg_triples.json
using ground_truth_answer sentences and fuzzy triple matching.
Hand-review before treating as authoritative gold.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

try:
    from thefuzz import fuzz
except ImportError:
    fuzz = None  # type: ignore


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _sentences(text: str) -> list[str]:
    t = text.replace(".\n", ". ").strip()
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
    return [p.rstrip(".!?") + "." if not p.endswith((".", "!", "?")) else p for p in parts]


def _best_triples_for_sentence(sent: str, triples: list[dict], top: int = 3) -> list[dict]:
    if not fuzz:
        return []
    nq = _norm(sent)
    scored: list[tuple[int, dict]] = []
    for t in triples:
        blob = _norm(f"{t.get('subject','')} {t.get('relation','')} {t.get('object','')}")
        score = fuzz.token_set_ratio(nq, blob)
        if score >= 40:
            scored.append((score, dict(t)))
    scored.sort(key=lambda x: -x[0])
    return [t[1] for t in scored[:top]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=os.path.join(_ROOT, "data", "questions.json"))
    ap.add_argument("--kg", default=os.path.join(_ROOT, "data", "kg_triples.json"))
    ap.add_argument(
        "--out",
        default=os.path.join(_ROOT, "data", "gold_claims_scaffold.json"),
    )
    args = ap.parse_args()

    with open(args.questions, encoding="utf-8") as f:
        qdata = json.load(f)["questions"]
    with open(args.kg, encoding="utf-8") as f:
        triples = json.load(f)

    out: dict[str, dict] = {}
    for q in qdata:
        qid = q["id"]
        gt = q.get("ground_truth_answer", "")
        kf = q.get("key_facts", [])
        gold_claims = []
        for sent in _sentences(gt):
            ev = _best_triples_for_sentence(sent, triples)
            gold_evidence = [
                {"subject": e["subject"], "relation": e["relation"], "object": e["object"]}
                for e in ev
                if isinstance(e, dict) and "subject" in e
            ]
            gold_claims.append(
                {
                    "text": sent,
                    "label": "SUPPORTED",
                    "gold_evidence": gold_evidence,
                }
            )
        ent_links: dict[str, str] = {}
        for fact in kf:
            ent_links[str(fact)] = str(fact)
        for gc in gold_claims:
            for ev in gc.get("gold_evidence") or []:
                if isinstance(ev, dict):
                    ent_links[str(ev.get("subject", ""))] = str(ev.get("subject", ""))
                    ent_links[str(ev.get("object", ""))] = str(ev.get("object", ""))
        out[qid] = {
            "gold_claims": gold_claims,
            "gold_entity_links": ent_links,
            "must_not_contradict": [],
            "key_facts": kf,
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote scaffold for {len(out)} questions → {args.out}")


if __name__ == "__main__":
    main()
