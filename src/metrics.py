"""
metrics.py
----------
Extended evaluation metrics for claim verification and retrieval
(requires gold_claims.json produced from the scaffold + manual review).
"""
from __future__ import annotations

import re
from typing import Any

# Stop-words excluded from key-fact presence checks.
# Replaces the old `len(w) > 2` filter which silently dropped short but
# semantically critical tokens like "10", "3", "6M", "US", etc.
_STOP_WORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
    'be', 'been', 'has', 'have', 'had', 'it', 'its', 'this', 'that',
    'as', 'up', 'per', 'not', 'no', 'so', 'do', 'did', 'can', 'may',
    'will', 'would', 'could', 'should', 'than', 'then', 'when', 'if',
    'about', 'into', 'out', 'also',
})


def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r'[$€£¥]', '', s)                 # strip currency symbols before collapse
    s = re.sub(r'(?<=\d)[,.](?=\d)', '', s)       # 2,500 → 2500; collapse thousands sep
    s = re.sub(r'[^\w\s]', ' ', s)                # remaining punctuation → space
    return re.sub(r'\s+', ' ', s).strip()


def _token_overlap(a: str, b: str) -> float:
    ta, tb = set(_norm_text(a).split()), set(_norm_text(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _match_claims_to_gold(
    pred_claims: list[str],
    gold_claims: list[dict[str, Any]],
    min_score: float = 0.35,
) -> list[int | None]:
    """Greedy match each predicted claim to best unmatched gold index."""
    used: set[int] = set()
    mapping: list[int | None] = []
    for pc in pred_claims:
        best_j: int | None = None
        best_s = 0.0
        for j, gc in enumerate(gold_claims):
            if j in used:
                continue
            gt = str(gc.get("text", ""))
            s = _token_overlap(pc, gt)
            if s > best_s:
                best_s = s
                best_j = j
        if best_j is not None and best_s >= min_score:
            used.add(best_j)
            mapping.append(best_j)
        else:
            mapping.append(None)
    return mapping


def claim_prf1(
    pred_claims: list[str],
    pred_verdicts: list[str],
    gold_claims: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Per-label micro precision/recall/F1 after greedy claim alignment.
    pred_verdicts aligned 1:1 with pred_claims.
    """
    labels = ("SUPPORTED", "CONTRADICTED", "UNVERIFIABLE")
    match_idx = _match_claims_to_gold(pred_claims, gold_claims)

    tp = {l: 0 for l in labels}
    fp = {l: 0 for l in labels}
    fn_d = {l: 0 for l in labels}

    matched_gold: set[int] = set()
    for i, pc in enumerate(pred_claims):
        pv = (pred_verdicts[i] if i < len(pred_verdicts) else "UNVERIFIABLE").upper()
        if pv not in labels:
            pv = "UNVERIFIABLE"
        mj = match_idx[i] if i < len(match_idx) else None
        if mj is None:
            fp[pv] += 1
            continue
        matched_gold.add(mj)
        gv = str(gold_claims[mj].get("label", "SUPPORTED")).upper()
        if gv not in labels:
            gv = "SUPPORTED"
        if pv == gv:
            tp[pv] += 1
        else:
            fp[pv] += 1
            fn_d[gv] += 1

    for j, gc in enumerate(gold_claims):
        if j not in matched_gold:
            gv = str(gc.get("label", "SUPPORTED")).upper()
            if gv not in labels:
                gv = "SUPPORTED"
            fn_d[gv] += 1

    per_label: dict[str, dict[str, float]] = {}
    micro_tp = micro_fp = micro_fn = 0
    for l in labels:
        micro_tp += tp[l]
        micro_fp += fp[l]
        micro_fn += fn_d[l]
        p = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) else 0.0
        r = tp[l] / (tp[l] + fn_d[l]) if (tp[l] + fn_d[l]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_label[l] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}

    p_micro = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    r_micro = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro) if (p_micro + r_micro) else 0.0

    return {
        "micro_precision": round(p_micro, 3),
        "micro_recall": round(r_micro, 3),
        "micro_f1": round(f1_micro, 3),
        "per_label": per_label,
    }


def contradiction_precision(
    pred_claims: list[str],
    pred_verdicts: list[str],
    gold_claims: list[dict[str, Any]],
) -> dict[str, float]:
    """Among predicted CONTRADICTED, fraction where matched gold is also CONTRADICTED."""
    match_idx = _match_claims_to_gold(pred_claims, gold_claims)
    pred_contra = [
        (i, match_idx[i])
        for i, v in enumerate(pred_verdicts)
        if str(v).upper() == "CONTRADICTED"
    ]
    if not pred_contra:
        return {"contradiction_precision": 1.0, "n_predicted_contradicted": 0, "n_true_contradicted": 0}
    ok = 0
    for i, mj in pred_contra:
        if mj is None:
            continue
        gv = str(gold_claims[mj].get("label", "SUPPORTED")).upper()
        if gv == "CONTRADICTED":
            ok += 1
    prec = ok / len(pred_contra)
    return {
        "contradiction_precision": round(prec, 3),
        "n_predicted_contradicted": len(pred_contra),
        "n_true_contradicted": ok,
    }


def unverifiable_calibration(
    pred_claims: list[str],
    pred_verdicts: list[str],
    gold_claims: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Among predicted UNVERIFIABLE, fraction where gold says UNVERIFIABLE or
    matched gold claim has empty gold_evidence (no KG support expected).
    """
    match_idx = _match_claims_to_gold(pred_claims, gold_claims)
    idxs = [i for i, v in enumerate(pred_verdicts) if str(v).upper() == "UNVERIFIABLE"]
    if not idxs:
        return {"unverifiable_calibration": 1.0, "n_predicted_unverifiable": 0, "n_calibrated_ok": 0}
    ok = 0
    for i in idxs:
        mj = match_idx[i] if i < len(match_idx) else None
        if mj is None:
            ok += 1
            continue
        gc = gold_claims[mj]
        gv = str(gc.get("label", "SUPPORTED")).upper()
        ev = gc.get("gold_evidence") or []
        if gv == "UNVERIFIABLE" or not ev:
            ok += 1
    return {
        "unverifiable_calibration": round(ok / len(idxs), 3),
        "n_predicted_unverifiable": len(idxs),
        "n_calibrated_ok": ok,
    }


def _fact_words_present(answer: str, fact: str) -> bool:
    pred_norm = _norm_text(answer)
    pred_words = set(pred_norm.split())
    fact_words = [w for w in _norm_text(fact).split() if w not in _STOP_WORDS]
    return bool(fact_words) and all(w in pred_words for w in fact_words)


def key_fact_recall(answer: str, key_facts: list[str]) -> float:
    if not key_facts:
        return 0.0
    return sum(1 for f in key_facts if _fact_words_present(answer, f)) / len(key_facts)


def missing_key_facts(answer: str, key_facts: list[str]) -> list[str]:
    """Key facts whose lexical check fails against *answer* (same rule as KFR)."""
    return [f for f in key_facts if not _fact_words_present(answer, f)]


def task_passes_kfr(answer: str, key_facts: list[str], threshold: float = 0.8) -> tuple[float, bool]:
    """
    Benchmark-style Me for eval: KFR on *answer* vs key_facts, pass if KFR >= threshold.
    If *key_facts* is empty, returns (0.0, True) so the harness does not block (no oracle).
    """
    if not key_facts:
        return (0.0, True)
    kfr = key_fact_recall(answer, key_facts)
    return (round(kfr, 3), kfr >= threshold)


def repair_success_rate(raw_answer: str, final_answer: str, key_facts: list[str]) -> dict[str, Any]:
    raw_k = key_fact_recall(raw_answer, key_facts)
    fin_k = key_fact_recall(final_answer, key_facts)
    return {
        "raw_key_fact_recall": round(raw_k, 3),
        "final_key_fact_recall": round(fin_k, 3),
        "repair_success": fin_k > raw_k,
        "repair_delta_kfr": round(fin_k - raw_k, 3),
    }


def over_repair_rate(
    raw_answer: str,
    final_answer: str,
    gold_claims: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Heuristic: gold SUPPORTED claims whose text substantially appears in raw
    but disappear in final count as over-repairs.
    """
    raw_n, fin_n = _norm_text(raw_answer), _norm_text(final_answer)
    over = 0
    checked = 0
    for gc in gold_claims:
        if str(gc.get("label", "SUPPORTED")).upper() != "SUPPORTED":
            continue
        t = _norm_text(str(gc.get("text", "")))
        if len(t) < 12:
            continue
        checked += 1
        if t in raw_n or all(w in set(raw_n.split()) for w in t.split() if len(w) > 3):
            if t not in fin_n and not all(w in set(fin_n.split()) for w in t.split() if len(w) > 3):
                over += 1
    rate = over / checked if checked else 0.0
    return {"over_repair_count": over, "over_repair_rate": round(rate, 3), "supported_gold_checked": checked}


def _triple_key(t: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(t.get("subject", "")),
        str(t.get("relation", "")),
        str(t.get("object", "")),
    )


def evidence_hit_at_k(
    verification_results: list[dict[str, Any]],
    gold_evidence: list[dict[str, Any]],
    k: int = 5,
) -> dict[str, Any]:
    """Fraction of gold triples found in top-k union of retrieved evidence per claim."""
    if not gold_evidence:
        return {"evidence_hit_at_k": 1.0, "gold_triples": 0, "hits": 0}
    gold_keys = {_triple_key(t) for t in gold_evidence if "subject" in t}
    if not gold_keys:
        return {"evidence_hit_at_k": 1.0, "gold_triples": 0, "hits": 0}
    retrieved: set[tuple[str, str, str]] = set()
    for vr in verification_results:
        evs = vr.get("evidence_triples") or []
        for t in evs[:k]:
            if isinstance(t, dict) and "subject" in t:
                retrieved.add(_triple_key(t))
    hits = len(gold_keys & retrieved)
    return {
        "evidence_hit_at_k": round(hits / len(gold_keys), 3),
        "gold_triples": len(gold_keys),
        "hits": hits,
        "k": k,
    }


def entity_linking_accuracy(
    verification_results: list[dict[str, Any]],
    gold_entity_links: dict[str, str],
) -> dict[str, Any]:
    """
    Compare subject strings from top evidence triple to gold_entity_links keys.
    A hit is when predicted subject matches any gold link key/value canonically.
    """
    if not gold_entity_links:
        return {"entity_linking_accuracy": 1.0, "n_checked": 0, "n_hits": 0}
    canon = {_norm_text(k): _norm_text(v) for k, v in gold_entity_links.items()}
    checked = 0
    hits = 0
    for vr in verification_results:
        evs = vr.get("evidence_triples") or []
        if not evs:
            continue
        sub = str(evs[0].get("subject", ""))
        if not sub:
            continue
        checked += 1
        ns = _norm_text(sub)
        if ns in canon or any(ns == _norm_text(k) for k in gold_entity_links.keys()):
            hits += 1
    acc = hits / checked if checked else 1.0
    return {"entity_linking_accuracy": round(acc, 3), "n_checked": checked, "n_hits": hits}


def compute_question_extended_metrics(
    record: dict[str, Any],
    gold_pack: dict[str, Any],
    evidence_k: int = 5,
) -> dict[str, Any]:
    """Single-question bundle of extended metrics."""
    pred_claims = list(record.get("claims") or [])
    vrs = record.get("verification_results") or []
    pred_verdicts = [str(v.get("verdict", "UNVERIFIABLE")) for v in vrs]
    gold_claims = list(gold_pack.get("gold_claims") or [])
    gel = gold_pack.get("gold_entity_links") or {}
    gold_ev_union: list[dict[str, Any]] = []
    for gc in gold_claims:
        gold_ev_union.extend(gc.get("gold_evidence") or [])

    out: dict[str, Any] = {}
    out["claim_prf1"] = claim_prf1(pred_claims, pred_verdicts, gold_claims)
    out["contradiction_precision"] = contradiction_precision(pred_claims, pred_verdicts, gold_claims)
    out["unverifiable_calibration"] = unverifiable_calibration(pred_claims, pred_verdicts, gold_claims)
    kf = list(record.get("key_facts") or gold_pack.get("key_facts") or [])
    out["repair_success"] = repair_success_rate(
        str(record.get("raw_answer", "")),
        str(record.get("final_answer", "")),
        kf,
    )
    out["over_repair"] = over_repair_rate(
        str(record.get("raw_answer", "")),
        str(record.get("final_answer", "")),
        gold_claims,
    )
    out["evidence_hit"] = evidence_hit_at_k(vrs, gold_ev_union, k=evidence_k)
    out["entity_linking"] = entity_linking_accuracy(vrs, gel)
    return out


def aggregate_extended_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean-aggregate numeric leaves from per-question extended dicts."""
    if not rows:
        return {}
    keys = [
        ("claim_prf1", "micro_f1"),
        ("claim_prf1", "micro_precision"),
        ("claim_prf1", "micro_recall"),
        ("contradiction_precision", "contradiction_precision"),
        ("unverifiable_calibration", "unverifiable_calibration"),
        ("repair_success", "repair_delta_kfr"),
        ("over_repair", "over_repair_rate"),
        ("evidence_hit", "evidence_hit_at_k"),
        ("entity_linking", "entity_linking_accuracy"),
    ]
    out: dict[str, Any] = {}
    for top, leaf in keys:
        vals = []
        for r in rows:
            d = r.get("extended_metrics") or {}
            v = d.get(top) or {}
            if isinstance(v, dict) and leaf in v:
                vals.append(float(v[leaf]))
        if vals:
            out[f"avg_{top}_{leaf}"] = round(sum(vals) / len(vals), 3)
    ok = sum(
        1
        for r in rows
        if bool((r.get("extended_metrics") or {}).get("repair_success", {}).get("repair_success"))
    )
    if rows:
        out["avg_repair_success_rate"] = round(ok / len(rows), 3)
    return out
