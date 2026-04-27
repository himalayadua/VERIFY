#!/usr/bin/env python3
"""Regenerate results-common.md from results_v0-v2/*_results.json."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VARIANTS = ["v0", "v1", "v1rag", "v2"]

sys.path.insert(0, str(ROOT.parent / "src"))
from model_registry import MODEL_REGISTRY, model_to_slug  # noqa: E402

slug_to_model = {model_to_slug(m): m for m in MODEL_REGISTRY}


def load_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def folder_score(dpath: Path) -> tuple[int, int]:
    s, nvar = 0, 0
    for v in VARIANTS:
        fp = dpath / f"{v}_results.json"
        if not fp.exists():
            continue
        nvar += 1
        data = load_json(fp)
        qs = data.get("questions") or []
        ok = sum(1 for q in qs if not q.get("error"))
        s += ok
    return nvar, s


def error_fraction(data: dict) -> float:
    qs = data.get("questions") or []
    if not qs:
        return 1.0
    return sum(1 for q in qs if q.get("error")) / len(qs)


def discover_model_paths() -> dict[str, Path]:
    by_model: dict[str, list[Path]] = {}
    for d in ROOT.iterdir():
        if not d.is_dir() or d.name.startswith("_"):
            continue
        v0 = d / "v0_results.json"
        if not v0.exists():
            continue
        mid = load_json(v0).get("model") or slug_to_model.get(d.name)
        if not mid:
            continue
        by_model.setdefault(mid, []).append(d)
    chosen = {}
    for mid, paths in by_model.items():
        paths = sorted(paths, key=lambda p: folder_score(p), reverse=True)
        chosen[mid] = paths[0]
    return chosen


MODEL_PATH = discover_model_paths()


def load_variant(mid: str, v: str) -> dict | None:
    fp = MODEL_PATH[mid] / f"{v}_results.json"
    if not fp.exists():
        return None
    return load_json(fp)


def short_name(mid: str) -> str:
    return mid.split("/")[-1]


def pct(x) -> str:
    if x is None:
        return "—"
    return f"{100 * float(x):.1f}%"


def num(x, nd: int = 3) -> str:
    if x is None:
        return "—"
    return f"{float(x):.{nd}f}"


def dagger(data: dict | None) -> str:
    if data is None:
        return ""
    ef = error_fraction(data)
    if ef >= 0.10:
        return "†"
    return ""


def cell_acc(data: dict | None) -> str:
    if data is None:
        return "—"
    a = (data.get("aggregate_metrics") or {}).get("accuracy")
    return pct(a) + dagger(data)


def cell_kfr(data: dict | None) -> str:
    if data is None:
        return "—"
    k = (data.get("aggregate_metrics") or {}).get("avg_kfr")
    return num(k) + dagger(data)


def cell_hall(data: dict | None) -> str:
    if data is None:
        return "—"
    h = (data.get("aggregate_metrics") or {}).get("hallucination_rate")
    if h is None:
        return "—"
    return pct(h) + dagger(data)


def cell_lat(data: dict | None) -> str:
    if data is None:
        return "—"
    sec = (data.get("aggregate_metrics") or {}).get("avg_latency_seconds")
    return f"{float(sec or 0):.1f}s" + dagger(data)


def rag_suppress(d1: dict | None, dr: dict | None) -> str:
    if d1 is None or dr is None:
        return "—"
    x1 = (d1.get("aggregate_metrics") or {}).get("hallucination_rate")
    xr = (dr.get("aggregate_metrics") or {}).get("hallucination_rate")
    if x1 is None or xr is None:
        return "—"
    pp = (float(x1) - float(xr)) * 100
    if pp > 30:
        tag = "✅ Strong"
    elif pp > 10:
        tag = "✅ Moderate"
    elif pp > 0:
        tag = "✓ Mild"
    else:
        tag = "flat/−"
    return f"{tag} ({pp:+.0f}pp)"


def lift_v0_v1r(d0: dict | None, dr: dict | None) -> str:
    if d0 is None or dr is None:
        return "—"
    a0 = (d0.get("aggregate_metrics") or {}).get("avg_kfr") or 0
    a1 = (dr.get("aggregate_metrics") or {}).get("avg_kfr") or 0
    a0, a1 = float(a0), float(a1)
    if a0 == 0:
        return "+∞" if a1 > 0 else "0%"
    return f"{(a1 - a0) / a0 * 100:+.0f}%"


def best_variant(mid: str, row_data: dict[str, dict | None]) -> str:
    best, bestv = -1.0, None
    for v in VARIANTS:
        d = row_data.get(v)
        if d is None:
            continue
        acc = (d.get("aggregate_metrics") or {}).get("accuracy")
        if acc is None:
            continue
        acc = float(acc)
        if acc > best:
            best, bestv = acc, v
    if bestv is None:
        return "—"
    return {"v0": "V0", "v1": "V1", "v1rag": "V1RAG", "v2": "V2"}[bestv]


def sep_row(n_cols: int) -> str:
    return "|" + "|".join(["---"] * n_cols) + "|"


def classify_cell(info: dict) -> str:
    if info.get("error"):
        return "—"
    if info.get("had_hallucinations"):
        return "!"
    ans = (info.get("final_answer") or "").lower()
    refuse_tokens = (
        "i don't have",
        "i do not have",
        "i'm sorry, but i don't",
        "not contain",
        "no information",
        "don't have that information",
        "does not contain",
        "no details about",
        "i cannot answer",
    )
    refused = any(t in ans for t in refuse_tokens) and (info.get("key_fact_recall") or 0) < 0.5
    if refused:
        return "R"
    if info.get("is_correct"):
        return "✓"
    if (info.get("key_fact_recall") or 0) > 0:
        return "P"
    return "R"


def cell_md_q(info: dict | None) -> str:
    if info is None or info.get("error"):
        return "— _err_"
    marker = classify_cell(info)
    vc = Counter(
        c.get("verdict") if isinstance(c, dict) else None
        for c in (info.get("verification_results") or [])
    )
    s, c, u = vc.get("SUPPORTED", 0), vc.get("CONTRADICTED", 0), vc.get("UNVERIFIABLE", 0)
    kfr = float(info.get("key_fact_recall") or 0)
    return f"{kfr:.2f} /{marker} {s}/{c}/{u}"


def main() -> None:
    rows: list[dict] = []
    for mid in MODEL_PATH:
        rd = {v: load_variant(mid, v) for v in VARIANTS}
        rows.append({"mid": mid, "short": short_name(mid), "folder": MODEL_PATH[mid].name, "d": rd})

    def v1r_kfr(row):
        dr = row["d"].get("v1rag")
        if dr is None:
            return -1.0
        return float((dr.get("aggregate_metrics") or {}).get("avg_kfr") or 0)

    rows.sort(key=lambda r: (-v1r_kfr(r), r["mid"].lower()))
    n_models = len(rows)
    n_cols = 1 + n_models

    lines: list[str] = []

    def L(s: str = "") -> None:
        lines.append(s)

    L("# KG Claim Verifier — Cross-Model Evaluation Results")
    L("")
    L("**Results root:** `results_v0-v2/` (this document lives alongside per-model JSON + logs).")
    L("")
    L("**Benchmark:** 45 questions across 5 categories (company overview, leadership, products & pricing, HR policy, benefits)  ")
    L("**Difficulty split:** Easy (n=21) · Medium (n=13) · Hard (n=11)  ")
    L("**Variants:** V0 baseline → V1 verify+repair → V1RAG RAG+verify+repair → V2 reflexion memory  ")
    L(f"**Models in this report:** {n_models} (every registry model with `v0_results.json` under `results_v0-v2/`).  ")
    L("**Folder deduplication:** If two directories share the same `model` id, the directory with more successful `(variant × question)` cells wins (`google-gemma-3n-e4b-it_run1`, `openai-gpt-oss-20b_run1`, `z-ai-glm-5_1_run1` beat their non-`_run1` twins).")
    L("")
    L("**† marker:** In master tables, a trailing **†** means ≥10% of questions in that variant hit a pipeline error (e.g. LLM retries exhausted); aggregates still reflect `aggregate_metrics` in the JSON (completed questions).")
    L("")
    L("**Incomplete export:** `minimaxai/minimax-m2.7` — only **V0** JSON is present in this folder; other variants show **—**.")
    L("")
    L("---")
    L("")
    L("## Metric Definitions")
    L("")
    L("| Metric | Description |")
    L("|---|---|")
    L('| **Accuracy** | % questions where Key Fact Recall ≥ 0.8 (considered "correct") |')
    L("| **Avg KFR** | Average Key Fact Recall — fraction of ground-truth key facts present in the final answer |")
    L("| **Hallucination %** | % questions where at least one claim was CONTRADICTED by the KG (V1/V1RAG/V2 only) |")
    L("| **Avg Latency** | Mean wall-clock time per question (includes all LLM calls in the pipeline) |")
    L("| **Avg Claims** | Average number of atomic claims extracted per answer (V1/V1RAG/V2 only) |")
    L("| **SUPPORTED / CONTRADICTED / UNVERIFIABLE** | Fraction of all claims across all questions falling into each verdict bucket |")
    L("")
    L("---")
    L("")
    L("## Master Comparison Table — All Models × All Variants")
    L("")
    L("**Columns** = Hugging Face model id suffix (after `/`). See **Per-Model Analysis** for full ids and source folder names.")
    L("")

    hdr = "| Variant | " + " | ".join(r["short"] for r in rows) + " |"
    sep = sep_row(n_cols)

    L("### Accuracy (% questions correct, KFR ≥ 0.8)")
    L("")
    L(hdr)
    L(sep)
    vlab = {"v0": "V0", "v1": "V1", "v1rag": "V1RAG", "v2": "V2"}
    for v in VARIANTS:
        cells = [cell_acc(row["d"][v]) for row in rows]
        L(f"| **{vlab[v]}** | " + " | ".join(cells) + " |")

    L("")
    L("### Average Key Fact Recall (KFR, 0–1)")
    L("")
    L(hdr)
    L(sep)
    for v in VARIANTS:
        lab = {"v0": "V0", "v1": "V1", "v1rag": "V1RAG", "v2": "V2"}[v]
        cells = [cell_kfr(row["d"][v]) for row in rows]
        L(f"| **{lab}** | " + " | ".join(cells) + " |")

    L("")
    L("### Hallucination rate (V1 / V1RAG / V2 only)")
    L("")
    L(hdr)
    L(sep)
    for v in ["v1", "v1rag", "v2"]:
        lab = {"v1": "V1", "v1rag": "V1RAG", "v2": "V2"}[v]
        cells = [cell_hall(row["d"][v]) for row in rows]
        L(f"| **{lab}** | " + " | ".join(cells) + " |")

    L("")
    L("### RAG hallucination suppression (V1 → V1RAG, Δ percentage points)")
    L("")
    L(hdr)
    L(sep)
    cells = [rag_suppress(row["d"]["v1"], row["d"]["v1rag"]) for row in rows]
    L("| **Δ (V1−V1RAG)** | " + " | ".join(cells) + " |")

    L("")
    L("### Average latency per question")
    L("")
    L(hdr)
    L(sep)
    for v in VARIANTS:
        lab = {"v0": "V0", "v1": "V1", "v1rag": "V1RAG", "v2": "V2"}[v]
        cells = [cell_lat(row["d"][v]) for row in rows]
        L(f"| **{lab}** | " + " | ".join(cells) + " |")

    L("")
    L("### Best variant (by accuracy) per model")
    L("")
    L("| Model | Best | V0 acc | V1 acc | V1RAG acc | V2 acc | Lift V0→V1RAG (KFR) |")
    L("|---|---|---|---|---|---|---|")
    for row in rows:
        mid = row["mid"]
        d = row["d"]
        L(
            f"| `{mid}` | **{best_variant(mid, d)}** | {cell_acc(d['v0'])} | {cell_acc(d['v1'])} | "
            f"{cell_acc(d['v1rag'])} | {cell_acc(d['v2'])} | {lift_v0_v1r(d['v0'], d['v1rag'])} |"
        )

    L("")
    L("### Verdict distribution — V1RAG (share of all claims)")
    L("")
    L("| Verdict | " + " | ".join(r["short"] for r in rows) + " |")
    L(sep_row(1 + n_models))
    for verdict in ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]:
        cells = []
        for row in rows:
            dr = row["d"]["v1rag"]
            if dr is None:
                cells.append("—")
            else:
                vd = (dr.get("aggregate_metrics") or {}).get("verdict_distribution") or {}
                val = vd.get(verdict)
                cells.append(pct(val) if val is not None else "—")
        L(f"| **{verdict}** | " + " | ".join(cells) + " |")

    cat_order = [
        ("company_overview", "Company Overview (n=6)"),
        ("leadership", "Leadership (n=13)"),
        ("products_pricing", "Products & Pricing (n=13)"),
        ("hr_policy", "HR Policy (n=5)"),
        ("benefits", "Benefits (n=8)"),
    ]
    diff_order = [("easy", "Easy (n=21)"), ("medium", "Medium (n=13)"), ("hard", "Hard (n=11)")]

    L("")
    L("---")
    L("")
    L("## Accuracy by Category — V1RAG")
    L("")
    L("| Category | " + " | ".join(r["short"] for r in rows) + " |")
    L(sep_row(1 + n_models))
    for key, title in cat_order:
        cells = []
        for row in rows:
            dr = row["d"]["v1rag"]
            if dr is None:
                cells.append("—")
            else:
                bc = (dr.get("aggregate_metrics") or {}).get("by_category") or {}
                cells.append(pct((bc.get(key) or {}).get("accuracy")))
        L(f"| {title} | " + " | ".join(cells) + " |")

    L("")
    L("## KFR by Category — V1RAG")
    L("")
    L("| Category | " + " | ".join(r["short"] for r in rows) + " |")
    L(sep_row(1 + n_models))
    for key, title in cat_order:
        cells = []
        for row in rows:
            dr = row["d"]["v1rag"]
            if dr is None:
                cells.append("—")
            else:
                bc = (dr.get("aggregate_metrics") or {}).get("by_category") or {}
                cells.append(num((bc.get(key) or {}).get("avg_kfr")))
        L(f"| {title} | " + " | ".join(cells) + " |")

    L("")
    L("## Accuracy by Difficulty — V1RAG")
    L("")
    L("| Difficulty | " + " | ".join(r["short"] for r in rows) + " |")
    L(sep_row(1 + n_models))
    for key, title in diff_order:
        cells = []
        for row in rows:
            dr = row["d"]["v1rag"]
            if dr is None:
                cells.append("—")
            else:
                bd = (dr.get("aggregate_metrics") or {}).get("by_difficulty") or {}
                cells.append(pct((bd.get(key) or {}).get("accuracy")))
        L(f"| {title} | " + " | ".join(cells) + " |")

    # --- Questions analysis (all models, same 6 questions) ---
    SELECTED = ["q01", "q09", "q18", "q25", "q38", "q36"]
    q_meta: dict[str, dict] = {}

    def qload(qid: str) -> None:
        for row in rows:
            mid = row["mid"]
            for v in VARIANTS:
                d = row["d"][v]
                if d is None:
                    continue
                for q in d.get("questions") or []:
                    if q.get("id") != qid:
                        continue
                    if qid not in q_meta:
                        q_meta[qid] = {
                            "question": q.get("question"),
                            "category": q.get("category"),
                            "difficulty": q.get("difficulty"),
                            "ground_truth": q.get("ground_truth_answer"),
                            "key_facts": q.get("key_facts"),
                        }
                    break

    for qid in SELECTED:
        qload(qid)

    L("")
    L("---")
    L("")
    L("## Questions Analysis — How Variants Behave on the Same Question")
    L("")
    L(
        "Micro-case studies on **six questions** (easy / medium / hard across categories). "
        f"**All {n_models} models** appear as columns (same order as master tables). "
        "Cells use the legend from the earlier 5-model write-up:"
    )
    L("")
    L("- `KFR /marker S/C/U` — Key-fact recall; marker `✓` correct, `!` hallucination flagged, `R` refuse/partial low recall, `P` partial; `S/C/U` = supported/contradicted/unverifiable claim counts. V0 has no verification → `0/0/0`.")
    L("- `— _err_` = question-level pipeline error for that model × variant.")
    L("")

    for qid in SELECTED:
        meta = q_meta.get(qid) or {}
        L(f"### {qid.upper()} — {meta.get('question', '')}")
        L(f"`{meta.get('category')}` / `{meta.get('difficulty')}`")
        L("")
        L(f"**Ground truth:** {meta.get('ground_truth', '')}  ")
        kf = meta.get("key_facts") or []
        L(f"**Key facts ({len(kf)}):** " + ", ".join(f"`{k}`" for k in kf))
        L("")
        qh = "| Variant | " + " | ".join(r["short"] for r in rows) + " |"
        L(qh)
        L(sep_row(1 + n_models))
        for v in VARIANTS:
            lab = {"v0": "V0", "v1": "V1", "v1rag": "V1RAG", "v2": "V2"}[v]
            cells = []
            for row in rows:
                d = row["d"][v]
                info = None
                if d:
                    for q in d.get("questions") or []:
                        if q.get("id") == qid:
                            info = q
                            break
                cells.append(cell_md_q(info))
            L(f"| **{lab}** | " + " | ".join(cells) + " |")
        L("")

    L("### Cross-question synthesis (variant roles)")
    L("")
    L(
        "- **V0** — Fast, unverified; often refuses (e.g. gpt-oss) or confidently confabulates names/prices when it does answer (see Q01/Q09 in the grids above)."
    )
    L(
        "- **V1** — Verifier surfaces unsupported claims; you may see high KFR with `!` when the model is right but the KG did not ground the answer."
    )
    L(
        "- **V1RAG** — Strongest *verified* behavior when retrieval hits; also a **refusal mode** when the retriever returns nothing (safe but incomplete)."
    )
    L(
        "- **V2** — Mixed: can break refusal habits on some models, but often **regresses** vs V1RAG on the same item because verbal reflections do not replace missing KG triples. Several models show **high error rates†** on V2 in the master table — treat those cells as noisy."
    )
    L("")

    L("---")
    L("")
    L("## Key Findings (full cohort)")
    L("")
    L("### 1. V1RAG remains the default “production” variant")
    L(
        "Across models with stable runs, **V1RAG** delivers the best tradeoff of **accuracy, KFR, and low contradiction rate** once retrieval fires. "
        "It is not universal: HR policy stays hard, and some models (e.g. parts of the Qwen family) still produce mostly UNVERIFIABLE claims even with RAG."
    )
    L("")
    L("### 2. Large instruction models top the leaderboard under V1RAG")
    top3 = sorted(rows, key=lambda r: -v1r_kfr(r))[:3]
    L(
        "By **average KFR under V1RAG**, the top of this export are: "
        + ", ".join(f"`{r['mid']}` ({num((r['d']['v1rag'].get('aggregate_metrics') or {}).get('avg_kfr'))})" for r in top3 if r["d"].get("v1rag"))
        + ". **Accuracy** leaders cluster in the 40% range on this 45-question suite (see Best variant table)."
    )
    L("")
    L("### 3. Qwen 3.5 × pipeline instability")
    L(
        "`qwen3.5-122b` and `qwen3.5-397b` show **heavy per-question errors†** on V1/V1RAG/V2 in this export; interpret their aggregates cautiously. "
        "`qwen3-next-80b-a3b-thinking` also shows a **fragile V2** run. Registry `enable_thinking: False` may interact badly with extraction/verification — worth an ablation."
    )
    L("")
    L("### 4. `qwen3-coder-480b` is mostly broken on V1+")
    L(
        "This checkpoint hits **many errors** on V1/V1RAG/V2; V0 partially completes. Treat as **out-of-distribution** for this JSON/claim workflow unless retried with different decoding or prompts."
    )
    L("")
    L("### 5. MiniMax M2.7 incomplete")
    L("Only **V0** results exist in `results_v0-v2/minimaxai-minimax-m2_7/`; no cross-variant comparison yet.")
    L("")
    L("### 6. HR policy is still the ceiling")
    L(
        "Even strong models rarely get HR-policy questions to **correct** (KFR ≥ 0.8); the bottleneck is likely **multi-condition retrieval + verifier coverage**, not raw LM scale alone."
    )
    L("")

    L("---")
    L("")
    L("## Per-Model Analysis")
    L("")
    L("One subsection per model (alphabetical). **Source folder** after each title is the directory under `results_v0-v2/` used after deduplication.")
    L("")

    for row in sorted(rows, key=lambda r: r["mid"].lower()):
        mid = row["mid"]
        d = row["d"]
        folder = row["folder"]
        L(f"### `{mid}`")
        L(f"*Folder:* `{folder}`")
        L("")
        L("| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U | err q |")
        L("|---|---|---|---|---|---|---|---|---|---|")
        for v in VARIANTS:
            lab = {"v0": "V0", "v1": "V1", "v1rag": "V1RAG", "v2": "V2"}[v]
            dv = d[v]
            if dv is None:
                L(f"| {lab} | — | — | — | — | — | — | — | — | — |")
                continue
            agg = dv.get("aggregate_metrics") or {}
            vd = agg.get("verdict_distribution") or {}
            nq = agg.get("total_questions") or 0
            eq = sum(1 for q in dv.get("questions") or [] if q.get("error"))
            L(
                f"| {lab} | {pct(agg.get('accuracy'))} | {num(agg.get('avg_kfr'))} | "
                f"{pct(agg.get('hallucination_rate')) if agg.get('hallucination_rate') is not None else '—'} | "
                f"{num(agg.get('avg_latency_seconds'), 1)}s | "
                f"{num(agg.get('avg_claims_per_answer'), 1) if agg.get('avg_claims_per_answer') is not None else '—'} | "
                f"{pct(vd.get('SUPPORTED')) if vd.get('SUPPORTED') is not None else '—'} | "
                f"{pct(vd.get('CONTRADICTED')) if vd.get('CONTRADICTED') is not None else '—'} | "
                f"{pct(vd.get('UNVERIFIABLE')) if vd.get('UNVERIFIABLE') is not None else '—'} | {eq}/{nq} |"
            )
        L("")
        L("| Category | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |")
        L("|---|---|---|---|---|")
        for key, title in cat_order:
            cells = [title]
            for v in VARIANTS:
                dv = d[v]
                if dv is None:
                    cells.append("—")
                    continue
                bc = (dv.get("aggregate_metrics") or {}).get("by_category") or {}
                b = bc.get(key) or {}
                cells.append(f"{num(b.get('avg_kfr'))} / {pct(b.get('accuracy'))}")
            L("| " + " | ".join(cells) + " |")
        L("")
        L("| Difficulty | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |")
        L("|---|---|---|---|---|")
        for key, title in diff_order:
            cells = [title]
            for v in VARIANTS:
                dv = d[v]
                if dv is None:
                    cells.append("—")
                    continue
                bd = (dv.get("aggregate_metrics") or {}).get("by_difficulty") or {}
                b = bd.get(key) or {}
                cells.append(f"{num(b.get('avg_kfr'))} / {pct(b.get('accuracy'))}")
            L("| " + " | ".join(cells) + " |")
        L("")

    L("---")
    L("")
    L("## Cross-Model Summary: What Works")
    L("")
    L("| Finding | Evidence in this export |")
    L("|---|---|")
    L("| **V1RAG is the primary quality win** | Highest KFR/accuracy for most stable models; largest drops in hallucination vs V1 (Δ row). |")
    L("| **Scale helps but does not solve HR** | Large instruct models lead V1RAG KFR; HR row in category tables stays near zero accuracy for many. |")
    L("| **V2 is high-variance** | Several models show heavy question errors or worse KFR vs V1RAG; a few (e.g. some Mistral/DeepSeek runs) peak on V2 accuracy — check per-model tables. |")
    L("| **Treat Qwen 3.5 + coder runs as experimental** | High `err q` counts and UNVERIFIABLE-heavy verdicts under RAG for some checkpoints. |")
    L("| **MiniMax M2.7** | V0-only folder — no variant comparison yet. |")
    L("")
    L("---")
    L("")
    L("*Generated by `results_v0-v2/_build_results_common.py` · All tables read from `aggregate_metrics` in each `*_results.json`.*")

    out = ROOT / "results-common.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
