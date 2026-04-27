"""
entity_linker.py
----------------
Loads KG triples from data/kg_triples.json and, given a natural-language
claim, returns the most relevant triples ranked by relevance.

Two matching strategies
-----------------------
  "llm"   : Primary path. Calls the Anthropic API (claude-haiku-4-5-20251001)
             to extract named entities and relation phrases from the claim.
             Fuzzy-matches those entities against KG entity names, retrieves
             candidate triples via the entity index, and optionally boosts the
             score when the extracted relation phrase also matches the triple's
             relation.  Falls back to "fuzzy" if the API fails or returns no
             entity matches.

  "fuzzy" : Fallback / standalone path. Scores every triple against the raw
             claim string using three text views of the triple (unchanged from
             the original implementation).

Public API
----------
    load_kg(path)                               → list of triple dicts
    find_relevant_triples(claim, triples,       → list of triple dicts + "score"
                          top_n=5, min_score=55,
                          method="llm")
    find_relevant_triples_multihop(...)          → multi-hop iterative retrieval
"""

import json
import os
import re
import sys
from typing import Any

import anthropic
from thefuzz import fuzz

# Allow importing llm_client from the same src/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_KG_PATH = os.path.normpath(
    os.path.join(_HERE, "..", "data", "kg_triples.json")
)
_RELATION_SYNONYMS_PATH = os.path.normpath(
    os.path.join(_HERE, "..", "data", "relation_synonyms.json")
)


def _load_relation_synonyms() -> dict[str, list[str]]:
    """Load relation → [synonyms] map if present. Missing file is non-fatal."""
    if not os.path.exists(_RELATION_SYNONYMS_PATH):
        return {}
    try:
        with open(_RELATION_SYNONYMS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


_RELATION_SYNONYMS: dict[str, list[str]] = _load_relation_synonyms()


# ── normalisation ──────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lower-case and replace underscores with spaces."""
    return text.lower().replace("_", " ")


def _normalise_full(text: str) -> str:
    """Lower-case, replace underscores, strip punctuation, collapse whitespace (for LLM path).

    Underscores are replaced first so KG relation names like 'co_founded_by'
    become 'co founded by' and fuzzy-match correctly against extracted phrases
    like 'co-founded'. Consistent with _normalise() which also does .replace("_"," ").
    """
    text = text.lower()
    text = text.replace("_", " ")          # treat underscores as word separators
    text = re.sub(r"[^\w\s]", " ", text)   # strip remaining punctuation (hyphens, slashes, …)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── fuzzy-path helpers (unchanged) ────────────────────────────────────────────

def _triple_views(triple: dict[str, str]) -> list[str]:
    """Return the text representations used for fuzzy scoring."""
    s = _normalise(triple["subject"])
    r = _normalise(triple["relation"])
    o = _normalise(triple["object"])
    return [
        f"{s} {r} {o}",   # full triple
        f"{s} {r}",        # subject + relation
        f"{r} {o}",        # relation + object
    ]


def _score(claim: str, triple: dict[str, str]) -> int:
    """Return the best fuzzy match score between *claim* and this triple."""
    normalised_claim = _normalise(claim)
    return max(
        fuzz.token_set_ratio(normalised_claim, view)
        for view in _triple_views(triple)
    )


# ── entity index ──────────────────────────────────────────────────────────────

def _is_named_entity_object(obj: str) -> bool:
    """Return True if obj looks like a named entity (not purely numeric/punctuation)."""
    if len(obj) <= 3:
        return False
    # Strip punctuation and spaces; if what remains is purely digits → not an entity
    stripped = re.sub(r"[\W_]", "", obj)
    return len(stripped) > 0 and not stripped.isdigit()


def _build_entity_index(triples: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """
    Build an O(1) lookup mapping entity names → list of their triples.

    Indexes:
      - every subject
      - every object that looks like a named entity (len > 3 and not purely numeric)
    """
    index: dict[str, list[dict[str, str]]] = {}
    for triple in triples:
        for entity in (triple["subject"],):
            index.setdefault(entity, []).append(triple)
        obj = triple["object"]
        if _is_named_entity_object(obj):
            index.setdefault(obj, []).append(triple)
    return index


# ── LLM extraction ────────────────────────────────────────────────────────────

_ENTITY_EXTRACTION_SYSTEM = (
    "You are an information extraction assistant. "
    "Extract from the claim:\n"
    "  - entities:  all named entities (people, products, companies, teams)\n"
    "  - relations: short relation phrases (e.g. 'founded in', 'priced at')\n"
    "  - values:    concrete NUMERIC or DATE-LIKE values that a KG triple "
    "might store as its object. Only include:\n"
    "                 * numbers and counts (e.g. '2500', '15 days')\n"
    "                 * money / prices (e.g. '$30,000/month', '1.4B')\n"
    "                 * percentages (e.g. '99.9%')\n"
    "                 * dates, years, quarters (e.g. '2021', 'Q3 2022', "
    "'March 2024')\n"
    "                 * version numbers (e.g. '2.0', 'v4.1')\n"
    "               Include multiple surface forms where reasonable "
    "(e.g. both '$30,000' and '30000'). Do NOT include titles, roles, "
    "categories, product variants, or descriptive phrases — those lead to "
    "noisy matches. Do NOT repeat the entity names here.\n"
    "Return ONLY valid JSON with this exact shape:\n"
    '{"entities": ["..."], "relations": ["..."], "values": ["..."]}'
)


def _parse_entity_json(raw: str) -> dict:
    """Strip code fences and parse JSON; return empty result on failure."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"entities": [], "relations": [], "values": []}
    # Back-compat: older responses may not include "values"
    parsed.setdefault("values", [])
    return parsed


def _extract_entities_llm(claim: str) -> dict:
    """
    Extract named entities and relation phrases from *claim* via LLM.

    Strategy:
      1. Try Anthropic claude-haiku if ANTHROPIC_API_KEY is set and non-placeholder.
      2. Fall back to NVIDIA NIM (via llm_client.call_llm_json) if Anthropic is
         unavailable or returns an auth error.
      3. Return {"entities": [], "relations": []} only if both fail.

    Returns {"entities": [...], "relations": [...]} on success.
    """
    user_content = f'Claim: "{claim}"'
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # ── 1. Try Anthropic ──────────────────────────────────────────────────────
    if anthropic_key and anthropic_key not in ("your-anthropic-key-here", ""):
        try:
            client = anthropic.Anthropic(api_key=anthropic_key)
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                temperature=0,
                system=_ENTITY_EXTRACTION_SYSTEM,
                messages=[{"role": "user", "content": user_content}],
            )
            return _parse_entity_json(message.content[0].text)
        except Exception as exc:
            print(f"  [entity_linker] Anthropic unavailable ({exc}); falling back to NVIDIA NIM.")

    # ── 2. Fall back to NVIDIA NIM ────────────────────────────────────────────
    try:
        from llm_client import call_llm_json, EXTRACT_MODEL
        return call_llm_json(
            user_content,
            system_prompt=_ENTITY_EXTRACTION_SYSTEM,
            model=EXTRACT_MODEL,
            temperature=0.0,
        )
    except Exception as exc:
        print(f"  [entity_linker] NVIDIA NIM extraction also failed ({exc}).")
        return {"entities": [], "relations": [], "values": []}


# ── LLM linking path ──────────────────────────────────────────────────────────

def _link_llm(
    claim: str,
    triples: list[dict[str, str]],
    entity_index: dict[str, list[dict[str, str]]],
    top_n: int,
    entity_threshold: int = 70,
    relation_threshold: int = 55,
    seed_entities: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Primary linking strategy.

    1. Extract entities/relations from *claim* via LLM.
    2. Fuzzy-match each extracted entity against all KG entity names.
    3. Optionally boost KG matching for *seed_entities* (from RGR reflections).
    4. Retrieve candidate triples via entity_index.
    5. Optionally boost composite score using extracted relation phrases.
    6. If no entity matches found, signal fallback by returning None.
    """
    extracted = _extract_entities_llm(claim)
    kg_entity_names = list(entity_index.keys())

    # --- entity fuzzy matching ---
    best_entity_score: dict[str, int] = {}
    for entity in extracted.get("entities", []):
        norm_q = _normalise_full(entity)
        for kg_entity in kg_entity_names:
            score = fuzz.token_sort_ratio(norm_q, _normalise_full(kg_entity))
            if score >= entity_threshold:
                if kg_entity not in best_entity_score or score > best_entity_score[kg_entity]:
                    best_entity_score[kg_entity] = score

    # --- seed entity injection (Reflexion-Guided Retrieval) ---
    # Hints extracted from prior reflections ("EXACT CORRECT VALUES: ..." lines)
    # are injected here so that verified-correct entity names can boost retrieval
    # even when the LLM's own extraction misses them.
    if seed_entities:
        for entity in seed_entities:
            norm_seed = _normalise_full(entity)
            if not norm_seed:
                continue
            for kg_entity in kg_entity_names:
                score = fuzz.token_set_ratio(norm_seed, _normalise_full(kg_entity))
                if score >= entity_threshold:
                    boosted = min(score + 10, 100)  # +10 rank boost for seed hints
                    if kg_entity not in best_entity_score or boosted > best_entity_score[kg_entity]:
                        best_entity_score[kg_entity] = boosted

    # --- retrieve candidate triples (deduplicated) ---
    seen_keys: set[tuple] = set()
    candidates: list[tuple[dict[str, str], float]] = []

    for entity, raw_score in best_entity_score.items():
        entity_score_norm = raw_score / 100.0
        for triple in entity_index.get(entity, []):
            key = (triple["subject"], triple["relation"], triple["object"])
            if key not in seen_keys:
                seen_keys.add(key)
                candidates.append((triple, entity_score_norm))

    # --- value-side retrieval ---
    # For concrete values in the claim ("$30,000", "2021", "Q3 2022"), scan KG
    # object fields directly. This surfaces triples whose subject wasn't caught
    # by entity matching (e.g., claim is 'NovaPilot Enterprise costs $30,000'
    # but the linker only matched 'NovaPilot', missing the Enterprise variant).
    values = [v for v in extracted.get("values", []) if v and len(v.strip()) >= 2]
    if values:
        norm_values = [(_normalise_full(v), v) for v in values]
        norm_values = [(nv, raw) for nv, raw in norm_values if len(nv) >= 2]
        for triple in triples:
            key = (triple["subject"], triple["relation"], triple["object"])
            if key in seen_keys:
                continue
            obj_norm = _normalise_full(triple["object"])
            if not obj_norm:
                continue
            for nv, _ in norm_values:
                if nv in obj_norm or obj_norm in nv:
                    candidates.append((triple, 0.85))
                    seen_keys.add(key)
                    break
                if fuzz.partial_ratio(nv, obj_norm) >= 90:
                    candidates.append((triple, 0.80))
                    seen_keys.add(key)
                    break

    # If neither entity matching nor value matching produced anything, signal
    # fallback to pure-fuzzy scoring.
    if not candidates:
        return None  # type: ignore[return-value]

    # --- relation boosting ---
    # Additive bonus: triples whose relation matches the extracted phrase score
    # HIGHER than non-matching ones. The previous weighted-average formula
    # (0.7*entity + 0.3*rel) capped at 1.0 and actually penalised matching
    # triples when entity_score_norm was already 1.0 (perfect entity match).
    relation_phrases = extracted.get("relations", [])
    if relation_phrases:
        norm_phrases = [_normalise_full(p) for p in relation_phrases]
        norm_phrases = [p for p in norm_phrases if p]
        boosted = []
        for triple, entity_score_norm in candidates:
            # Compare claim relations against the KG relation name AND its
            # curated synonyms (see data/relation_synonyms.json). Takes the max
            # score across all comparisons — so "Series C funding" can match
            # relation `series_c_date` via its synonym "Series C funding".
            rel_name = triple["relation"]
            targets = [_normalise_full(rel_name)]
            for syn in _RELATION_SYNONYMS.get(rel_name, []):
                n = _normalise_full(syn)
                if n:
                    targets.append(n)
            best_rel = 0
            for p in norm_phrases:
                for t in targets:
                    score = fuzz.token_sort_ratio(p, t)
                    if score > best_rel:
                        best_rel = score
            if best_rel >= relation_threshold:
                # Additive bonus: relation match raises score above entity-only baseline
                composite = entity_score_norm + 0.3 * (best_rel / 100.0)
            else:
                composite = entity_score_norm
            boosted.append((triple, composite))
        candidates = boosted

    candidates.sort(key=lambda x: x[1], reverse=True)

    # Diversify by relation: when a high-degree entity has many triples, avoid
    # returning top_n triples that are all the same relation (e.g., 10x
    # co_founded_by). Cap each (subject, relation) pair to max_per_relation;
    # fill any remaining slots with the next-best candidates regardless of cap.
    diversified = _diversify_by_relation(candidates, top_n, max_per_relation=2)

    results = []
    for triple, score_norm in diversified:
        results.append({**triple, "score": int(round(score_norm * 100))})
    return results


def _diversify_by_relation(
    candidates: list[tuple[dict[str, str], float]],
    top_n: int,
    max_per_relation: int = 2,
) -> list[tuple[dict[str, str], float]]:
    """
    Pick up to top_n triples from a score-sorted candidate list while limiting
    how many share the same (subject, relation) pair. Falls back to filling
    remaining slots by raw score if the cap would leave us short.
    """
    picked: list[tuple[dict[str, str], float]] = []
    seen_triple: set[tuple[str, str, str]] = set()
    per_sr_count: dict[tuple[str, str], int] = {}

    # Pass 1: respect the cap
    for triple, score in candidates:
        key3 = (triple["subject"], triple["relation"], triple["object"])
        sr_key = (triple["subject"], triple["relation"])
        if per_sr_count.get(sr_key, 0) >= max_per_relation:
            continue
        picked.append((triple, score))
        seen_triple.add(key3)
        per_sr_count[sr_key] = per_sr_count.get(sr_key, 0) + 1
        if len(picked) >= top_n:
            return picked

    # Pass 2: if cap left us short, fill from remaining by score
    for triple, score in candidates:
        if len(picked) >= top_n:
            break
        key3 = (triple["subject"], triple["relation"], triple["object"])
        if key3 in seen_triple:
            continue
        picked.append((triple, score))
        seen_triple.add(key3)

    return picked


# ── public API ────────────────────────────────────────────────────────────────

def load_kg(path: str = _DEFAULT_KG_PATH) -> list[dict[str, str]]:
    """Load and return the list of triples from a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_relevant_triples(
    claim: str,
    triples: list[dict[str, str]],
    top_n: int = 5,
    min_score: int = 55,
    method: str = "llm",
    seed_entities: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Find KG triples relevant to *claim*.

    Parameters
    ----------
    claim         : natural-language sentence to check
    triples       : list of {"subject", "relation", "object"} dicts
    top_n         : maximum number of results to return
    min_score     : triples with a score below this threshold are excluded
                    (applied to fuzzy path; LLM path uses its own thresholds)
    method        : "llm"  – LLM entity extraction + fuzzy linking (default);
                              falls back to "fuzzy" on any exception or no-match.
                    "fuzzy" – pure triple-view fuzzy scoring (no API call)
    seed_entities : optional list of entity name hints (from RGR reflections)
                    boosted during fuzzy matching on the LLM path.

    Returns
    -------
    List of dicts, each containing the original triple fields plus a
    "score" key (0–100 int), sorted by score descending.
    """
    if method == "llm":
        try:
            entity_index = _build_entity_index(triples)
            results = _link_llm(claim, triples, entity_index, top_n, seed_entities=seed_entities)
            if results is not None:
                return results
            # No entity matches found — fall through to fuzzy
            print("  [entity_linker] LLM found no entity matches; falling back to fuzzy.")
        except Exception as exc:
            print(f"  [entity_linker] LLM path failed ({exc}); falling back to fuzzy.")
        # Fuzzy fallback
        return _find_relevant_fuzzy(claim, triples, top_n, min_score)

    elif method == "fuzzy":
        return _find_relevant_fuzzy(claim, triples, top_n, min_score)

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'llm' or 'fuzzy'.")


def _find_relevant_fuzzy(
    claim: str,
    triples: list[dict[str, str]],
    top_n: int,
    min_score: int,
) -> list[dict[str, Any]]:
    """Pure triple-view fuzzy scoring (original implementation)."""
    scored = []
    for triple in triples:
        s = _score(claim, triple)
        if s >= min_score:
            scored.append({**triple, "score": s})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]


# ── Multi-hop retrieval ───────────────────────────────────────────────────────

def find_relevant_triples_multihop(
    claim: str,
    triples: list[dict[str, str]],
    top_n: int = 5,
    min_score: int = 55,
    method: str = "llm",
    max_hops: int = 2,
    seed_entities: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Iterative retrieval: run find_relevant_triples, then use discovered
    entities to retrieve additional triples in a second pass.
    Falls back to single-hop if no new entities are found.

    Parameters
    ----------
    claim         : natural-language sentence to check
    triples       : full KG triple list
    top_n         : max results per hop
    min_score     : minimum fuzzy score threshold
    method        : "llm" or "fuzzy" (passed to find_relevant_triples)
    max_hops      : number of retrieval hops (1 = standard, 2+ = iterative)
    seed_entities : optional list of entity name hints (from RGR reflections)
                    passed through to the LLM linking path for ranking boost.

    Returns
    -------
    Combined list of triples from all hops, deduplicated.
    """
    # Hop 1: standard retrieval on the claim text
    hop1 = find_relevant_triples(
        claim, triples, top_n=top_n, min_score=min_score, method=method,
        seed_entities=seed_entities,
    )

    if max_hops <= 1:
        return hop1

    # Hop 2: use entities discovered in hop1 to find additional triples
    seen = {(t["subject"], t["relation"], t["object"]) for t in hop1}
    hop2 = []

    new_entities = set()
    for t in hop1:
        new_entities.add(t["subject"])
        new_entities.add(t["object"])

    for entity in new_entities:
        for t in _find_relevant_fuzzy(entity, triples, top_n=3, min_score=min_score):
            key = (t["subject"], t["relation"], t["object"])
            if key not in seen:
                seen.add(key)
                hop2.append(t)

    return hop1 + hop2


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    kg = load_kg()
    print(f"Loaded {len(kg)} triples from KG\n")

    test_claims = [
        "The CEO of NovaAI is Bob Smith",
        "NovaAI was founded in 2019",
        "The NovaPilot Starter plan costs $5,000 per month",
        "NovaAI has offices in San Francisco and Tokyo",
    ]

    for claim in test_claims:
        print("=" * 70)
        print(f"CLAIM : {claim}")
        for method in ("llm", "fuzzy"):
            print(f"\n  -- method={method} --")
            results = find_relevant_triples(claim, kg, top_n=5, method=method)
            if not results:
                print("    (no matches above threshold)")
            else:
                for r in results:
                    print(
                        f"    [{r['score']:3d}]  {r['subject']}  |  "
                        f"{r['relation']}  |  {r['object']}"
                    )
        print()