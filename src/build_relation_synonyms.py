"""
build_relation_synonyms.py
--------------------------
One-shot script that asks the LLM to generate 5-8 natural-language synonyms /
paraphrases for every unique KG relation. Output is cached to
data/relation_synonyms.json so the entity linker can use it at runtime without
paying any API cost.

Usage:
  cd kg-claim-verifier/src
  python build_relation_synonyms.py          # fills in any missing relations
  python build_relation_synonyms.py --force  # regenerate for all relations
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_client import call_llm_json, EXTRACT_MODEL

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "data"))
_KG_PATH = os.path.join(_DATA_DIR, "kg_triples.json")
_OUT_PATH = os.path.join(_DATA_DIR, "relation_synonyms.json")

_BATCH_SIZE = 10

_SYSTEM_PROMPT = (
    "You are a data annotation assistant. For each relation name in the list, "
    "produce 5-8 short natural-language phrasings that a person might use to "
    "describe that concept. Phrasings should be concise (2-5 words each), "
    "distinct, and cover both formal and informal wording.\n\n"
    "Examples:\n"
    "  'funded_in_year' -> ['founded in', 'established in', 'started in', "
    "'launched in']\n"
    "  'series_c_date'  -> ['series C funding', 'raised Series C', 'Series C "
    "round', 'Series C closed on']\n"
    "  'sla_uptime'     -> ['SLA uptime', 'service level agreement uptime', "
    "'uptime guarantee', 'uptime SLA']\n\n"
    "Return ONLY valid JSON of this exact shape:\n"
    '{"relation_name_1": ["phrase1", "phrase2", ...], '
    '"relation_name_2": [...]}'
)


def _load_existing() -> dict[str, list[str]]:
    if os.path.exists(_OUT_PATH):
        with open(_OUT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(mapping: dict[str, list[str]]) -> None:
    os.makedirs(os.path.dirname(_OUT_PATH), exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False, sort_keys=True)


def _generate_batch(relations: list[str]) -> dict[str, list[str]]:
    user_prompt = "Relations:\n" + "\n".join(f"  - {r}" for r in relations)
    result = call_llm_json(
        user_prompt,
        system_prompt=_SYSTEM_PROMPT,
        model=EXTRACT_MODEL,
        temperature=0.0,
        max_tokens=2048,
    )
    # Normalise: strip, ensure list, skip empty
    cleaned: dict[str, list[str]] = {}
    for rel, synonyms in result.items():
        if not isinstance(synonyms, list):
            continue
        uniq = []
        seen = set()
        for s in synonyms:
            if not isinstance(s, str):
                continue
            s = s.strip()
            if s and s.lower() not in seen:
                seen.add(s.lower())
                uniq.append(s)
        if uniq:
            cleaned[rel] = uniq
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Regenerate synonyms for all relations, ignoring cache.")
    args = parser.parse_args()

    with open(_KG_PATH, encoding="utf-8") as f:
        kg = json.load(f)
    all_relations = sorted({t["relation"] for t in kg})
    print(f"KG has {len(all_relations)} unique relations.")

    existing = {} if args.force else _load_existing()
    missing = [r for r in all_relations if r not in existing]
    print(f"Already cached: {len(existing)}  |  to generate: {len(missing)}")

    if not missing:
        print("Nothing to do.")
        return

    for i in range(0, len(missing), _BATCH_SIZE):
        batch = missing[i:i + _BATCH_SIZE]
        print(f"\nBatch {i // _BATCH_SIZE + 1}: requesting synonyms for {len(batch)} relations...")
        try:
            generated = _generate_batch(batch)
        except Exception as exc:
            print(f"  ERROR in batch: {exc}")
            continue
        print(f"  Got {len(generated)}/{len(batch)} successfully.")
        existing.update(generated)
        _save(existing)  # save after each batch for resumability

    print(f"\nSaved {len(existing)} relations → {_OUT_PATH}")
    # Sanity-print a few
    print("\nSample entries:")
    for r in list(existing.keys())[:3]:
        print(f"  {r}: {existing[r]}")


if __name__ == "__main__":
    main()
