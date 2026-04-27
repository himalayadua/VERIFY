"""
prior_knowledge_probe.py
------------------------
Probe the LLM's prior-training knowledge of the evaluation dataset.

For a sample of high-specificity triples from data/kg_triples.json, ask
the LLM — under a strict "answer or UNKNOWN" system prompt — to produce
the object value given only the subject and relation. Each response is
classified as:

  UNKNOWN  : LLM explicitly declines.
  CORRECT  : LLM's response matches the KG object value.
  WRONG    : LLM produces a confident but non-matching answer.

A high UNKNOWN rate and a near-zero CORRECT rate is evidence that the
evaluation set is absent from the LLM's training distribution and the
KG therefore provides a meaningful benchmark.

Sampling strategy
-----------------
Draws only from "high-specificity" triples — those whose object appears
exactly once in the KG AND whose object is not a widely-known
industry-standard value (AWS, Carta, etc., kept in GENERIC_OBJECTS).

Usage
-----
  python3 src/prior_knowledge_probe.py                       # 30 samples, seed 42
  python3 src/prior_knowledge_probe.py --n 50 --seed 7
  python3 src/prior_knowledge_probe.py --n 30 --model claude-haiku-4-5-20251001
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anthropic
from thefuzz import fuzz

from llm_client import call_llm, ACTION_MODEL


_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))
_RESULTS_DIR = os.path.normpath(os.path.join(_HERE, "..", "results"))
_KG_PATH = os.path.join(_DATA_DIR, "kg_triples.json")
_DEFAULT_OUT = os.path.join(_RESULTS_DIR, "prior_knowledge_probe.json")


# Objects that are widely-known industry-standard values. Even if they
# appear exactly once in the KG, the LLM might correctly guess them
# from generic knowledge ("most startups use Carta for cap tables"),
# which is NOT evidence that the LLM has prior knowledge of this KG.
GENERIC_OBJECTS = {
    "aws", "gcp", "azure",
    "carta", "anthem blue cross", "anthem",
    "workday", "greenhouse",
    "apache 2.0", "mit license",
    "github", "gitlab", "slack",
    "privately held", "publicly traded",
    "san francisco", "new york", "london", "tokyo",
}


PROBE_SYSTEM_PROMPT = """You are being tested for prior-training knowledge of a company called NovaAI.

Rules:
- Answer ONLY if you have confident, verified recall of the exact fact from your training data.
- If you do NOT have confident prior knowledge of the exact answer, respond with only the single word UNKNOWN.
- Do NOT guess. Do NOT approximate. Do NOT reason from generic industry defaults. Do NOT produce placeholder or illustrative values.
- Your response must be either (a) the exact factual value you recall, or (b) the single word UNKNOWN.
- Keep your response to at most one short line. No explanations, no caveats, no preambles."""


def _humanize_relation(rel: str) -> str:
    """Turn snake_case relation names into readable phrases."""
    return rel.replace("_", " ").strip()


def _build_probe_prompt(subject: str, relation: str) -> str:
    return (
        f"Subject: {subject}\n"
        f"Attribute: {_humanize_relation(relation)}\n\n"
        f"What is the {_humanize_relation(relation)} of \"{subject}\" at NovaAI? "
        f"Respond with the exact value or UNKNOWN."
    )


def _is_high_specificity(obj: str) -> bool:
    """Keep triples whose object is distinctive (not a generic industry default)."""
    if not obj or len(obj.strip()) < 2:
        return False
    if obj.strip().lower() in GENERIC_OBJECTS:
        return False
    return True


def _sample_triples(kg: list[dict], n: int, seed: int) -> list[dict]:
    """Draw n high-specificity triples whose object appears only once in the KG."""
    obj_counts = Counter(t["object"] for t in kg)
    candidates = [
        t for t in kg
        if obj_counts[t["object"]] == 1 and _is_high_specificity(t["object"])
    ]
    rng = random.Random(seed)
    n = min(n, len(candidates))
    return rng.sample(candidates, n)


# ── Response classification ──────────────────────────────────────────────────

_UNKNOWN_TOKEN_RE = re.compile(r"\bunknown\b", re.IGNORECASE)


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _classify(response: str, expected_object: str) -> str:
    """
    Label the response as UNKNOWN / CORRECT / WRONG.

    - UNKNOWN  : bare 'UNKNOWN' (possibly with punctuation / whitespace).
    - CORRECT  : fuzzy-matches the expected value (token_set_ratio ≥ 70
                 OR normalised expected is a substring of the response).
    - WRONG    : anything else.
    """
    resp = response.strip()
    # Strict: must be essentially just "UNKNOWN" — not a sentence containing it.
    if len(resp) <= 20 and _UNKNOWN_TOKEN_RE.fullmatch(resp.strip(".!? \n\t")):
        return "UNKNOWN"
    # Also accept very short "I don't know" style fallbacks
    low = resp.lower()
    if len(resp) <= 40 and any(
        phrase in low for phrase in ("i don't know", "i do not know", "not sure", "no information", "cannot answer")
    ):
        return "UNKNOWN"

    n_resp = _normalize(resp)
    n_exp = _normalize(expected_object)
    if not n_exp:
        return "WRONG"
    if n_exp in n_resp:
        return "CORRECT"
    if fuzz.token_set_ratio(n_exp, n_resp) >= 70:
        return "CORRECT"
    return "WRONG"


# ── LLM call ─────────────────────────────────────────────────────────────────

def _call_llm(prompt: str, model: str, use_anthropic: bool) -> str:
    """Ask the model to answer the probe. Uses Anthropic if requested + key present; else NVIDIA NIM via llm_client."""
    if use_anthropic:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key or key == "your-anthropic-key-here":
            raise RuntimeError("ANTHROPIC_API_KEY not set; rerun without --anthropic or set the key.")
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=model,
            max_tokens=80,
            temperature=0.0,
            system=PROBE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    return call_llm(prompt, system_prompt=PROBE_SYSTEM_PROMPT, model=model, temperature=0.0, max_tokens=80).strip()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="Number of triples to sample (default 30).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default 42).")
    ap.add_argument("--model", type=str, default=None,
                    help="Model name. Default: ACTION_MODEL (via NVIDIA NIM). "
                         "Pass --anthropic for a Claude model (e.g. claude-haiku-4-5-20251001).")
    ap.add_argument("--anthropic", action="store_true",
                    help="Use Anthropic Claude instead of NVIDIA NIM.")
    ap.add_argument("--out", type=str, default=_DEFAULT_OUT, help="Output JSON path.")
    args = ap.parse_args()

    if args.anthropic:
        model = args.model or "claude-haiku-4-5-20251001"
    else:
        model = args.model or ACTION_MODEL

    with open(_KG_PATH, encoding="utf-8") as f:
        kg = json.load(f)
    print(f"Loaded {len(kg)} KG triples.")

    sampled = _sample_triples(kg, args.n, args.seed)
    print(f"Sampled {len(sampled)} high-specificity triples (seed={args.seed}).")
    print(f"Model: {model}  |  backend: {'Anthropic' if args.anthropic else 'NVIDIA NIM'}\n")

    probes: list[dict] = []
    counts = Counter()
    for i, t in enumerate(sampled, 1):
        prompt = _build_probe_prompt(t["subject"], t["relation"])
        try:
            response = _call_llm(prompt, model=model, use_anthropic=args.anthropic)
        except Exception as exc:
            print(f"  [{i}/{len(sampled)}] ERROR calling LLM: {exc}")
            probes.append({
                "subject": t["subject"], "relation": t["relation"],
                "expected": t["object"], "response": f"<ERROR: {exc}>", "label": "ERROR",
            })
            counts["ERROR"] += 1
            continue

        label = _classify(response, t["object"])
        counts[label] += 1
        print(f"  [{i}/{len(sampled)}] {label:<8}  {t['subject']} | {t['relation']}")
        print(f"              expected: {t['object']}")
        print(f"              got     : {response[:120]}")
        probes.append({
            "subject": t["subject"],
            "relation": t["relation"],
            "expected": t["object"],
            "response": response,
            "label": label,
        })

    total = len(probes)
    out = {
        "model": model,
        "backend": "anthropic" if args.anthropic else "nvidia_nim",
        "n_sampled": total,
        "seed": args.seed,
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "counts": dict(counts),
        "rates": {
            label: round(counts[label] / total, 3) if total else 0.0
            for label in ("UNKNOWN", "CORRECT", "WRONG", "ERROR")
        },
        "probes": probes,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  model      : {model}")
    print(f"  n_sampled  : {total}")
    for label in ("UNKNOWN", "CORRECT", "WRONG", "ERROR"):
        c = counts.get(label, 0)
        pct = (100 * c / total) if total else 0.0
        print(f"  {label:<8}  : {c:3d}  ({pct:.1f}%)")
    print(f"\n  saved → {args.out}")

    if counts.get("CORRECT", 0) > 0:
        print("\n  ⚠ CORRECT responses indicate possible prior-knowledge hits. "
              "Inspect the 'probes' list in the output file and consider replacing those facts.")


if __name__ == "__main__":
    main()
