# KG Claim Verifier — Cross-Model Evaluation Results

**Results root:** `results_v0-v2/` (this document lives alongside per-model JSON + logs).

**Benchmark:** 45 questions across 5 categories (company overview, leadership, products & pricing, HR policy, benefits)  
**Difficulty split:** Easy (n=21) · Medium (n=13) · Hard (n=11)  
**Variants:** V0 baseline → V1 verify+repair → V1RAG RAG+verify+repair → V2 reflexion memory  
**Models in this report:** 18 (every registry model with `v0_results.json` under `results_v0-v2/`).
**Folder deduplication:** If two directories share the same `model` id, the directory with more successful `(variant × question)` cells wins (`google-gemma-3n-e4b-it_run1`, `openai-gpt-oss-20b_run1`, `z-ai-glm-5_1_run1` beat their non-`_run1` twins).

**V2P variants:** V2P / V2P_RAG / V2P_RGR / V2P_RAG_RGR results are available for `openai/gpt-oss-120b` and `openai/gpt-oss-20b` only; see the dedicated section below.
**Rescore note:** V0–V2 numbers for `openai/gpt-oss-120b` and `openai/gpt-oss-20b` reflect the fixed KFR metric (rescored JSONs); changes vs. original run are noted inline in the per-model sections.

**† marker:** In master tables, a trailing **†** means ≥10% of questions in that variant hit a pipeline error (e.g. LLM retries exhausted); aggregates still reflect `aggregate_metrics` in the JSON (completed questions).

**Incomplete export:** `minimaxai/minimax-m2.7` — only **V0** JSON is present in this folder; other variants show **—**.

---

## Metric Definitions

| Metric | Description |
|---|---|
| **Accuracy** | % questions where Key Fact Recall ≥ 0.8 (considered "correct") |
| **Avg KFR** | Average Key Fact Recall — fraction of ground-truth key facts present in the final answer |
| **Hallucination %** | % questions where at least one claim was CONTRADICTED by the KG (V1/V1RAG/V2 only) |
| **Avg Latency** | Mean wall-clock time per question (includes all LLM calls in the pipeline) |
| **Avg Claims** | Average number of atomic claims extracted per answer (V1/V1RAG/V2 only) |
| **SUPPORTED / CONTRADICTED / UNVERIFIABLE** | Fraction of all claims across all questions falling into each verdict bucket |

---

## Master Comparison Table — All Models × All Variants

**Columns** = Hugging Face model id suffix (after `/`). See **Per-Model Analysis** for full ids and source folder names.

### Accuracy (% questions correct, KFR ≥ 0.8)

| Variant | deepseek-v3.2 | mistral-large-3-675b-instruct-2512 | glm-5.1 | seed-oss-36b-instruct | deepseek-v3.1-terminus | kimi-k2-thinking | gemma-3n-e2b-it | gemma-4-31b-it | kimi-k2-instruct-0905 | gpt-oss-120b | gemma-3n-e4b-it | gpt-oss-20b | minimax-m2.5 | qwen3-next-80b-a3b-thinking | qwen3.5-397b-a17b | qwen3.5-122b-a10b | qwen3-coder-480b-a35b-instruct | minimax-m2.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V0** | 0.0% | 8.9% | 0.0% | 6.7% | 8.9% | 8.9% | 4.4% | 0.0%† | 6.7% | 0.0% | 8.9% | 0.0% | 2.2% | 8.9% | 0.0% | 4.4% | 0.0%† | 0.0% |
| **V1** | 20.0% | 46.7% | 31.1% | 13.3% | 31.1% | 11.1% | 13.3% | 31.1% | 22.2% | 15.6% | 24.4% | 2.2% | 11.1%† | 8.9%† | 2.2%† | 2.2%† | 0.0%† | — |
| **V1RAG** | 20.0% | 44.4% | 42.2% | 37.8% | 40.0% | 40.0% | 37.8% | 35.6% | 37.8% | 31.1% | 28.9% | 33.3% | 28.9%† | 11.1%† | 2.2%† | 2.2%† | 0.0%† | — |
| **V2** | 20.0% | 48.9% | 40.0% | 17.8% | 26.7% | 4.4% | 2.2% | 28.9% | 17.8% | 20.0% | 24.4% | 15.6% | 4.4%† | 0.0%† | 0.0%† | 0.0%† | 0.0%† | — |

### Average Key Fact Recall (KFR, 0–1)

| Variant | deepseek-v3.2 | mistral-large-3-675b-instruct-2512 | glm-5.1 | seed-oss-36b-instruct | deepseek-v3.1-terminus | kimi-k2-thinking | gemma-3n-e2b-it | gemma-4-31b-it | kimi-k2-instruct-0905 | gpt-oss-120b | gemma-3n-e4b-it | gpt-oss-20b | minimax-m2.5 | qwen3-next-80b-a3b-thinking | qwen3.5-397b-a17b | qwen3.5-122b-a10b | qwen3-coder-480b-a35b-instruct | minimax-m2.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V0** | 0.317 | 0.414 | 0.144 | 0.221 | 0.239 | 0.342 | 0.297 | 0.124† | 0.182 | 0.088 | 0.348 | 0.007 | 0.193 | 0.315 | 0.221 | 0.200 | 0.030† | 0.137 |
| **V1** | 0.567 | 0.727 | 0.588 | 0.292 | 0.507 | 0.273 | 0.410 | 0.453 | 0.420 | 0.308 | 0.543 | 0.040 | 0.209† | 0.177† | 0.033† | 0.022† | 0.000† | — |
| **V1RAG** | 0.737 | 0.670 | 0.667 | 0.619 | 0.617 | 0.590 | 0.570 | 0.565 | 0.558 | 0.565 | 0.553 | 0.570 | 0.389† | 0.153† | 0.037† | 0.030† | 0.000† | — |
| **V2** | 0.617 | 0.762 | 0.615 | 0.321 | 0.496 | 0.161 | 0.249 | 0.494 | 0.365 | 0.350 | 0.523 | 0.264 | 0.076† | 0.026† | 0.011† | 0.000† | 0.011† | — |

### Hallucination rate (V1 / V1RAG / V2 only)

| Variant | deepseek-v3.2 | mistral-large-3-675b-instruct-2512 | glm-5.1 | seed-oss-36b-instruct | deepseek-v3.1-terminus | kimi-k2-thinking | gemma-3n-e2b-it | gemma-4-31b-it | kimi-k2-instruct-0905 | gpt-oss-120b | gemma-3n-e4b-it | gpt-oss-20b | minimax-m2.5 | qwen3-next-80b-a3b-thinking | qwen3.5-397b-a17b | qwen3.5-122b-a10b | qwen3-coder-480b-a35b-instruct | minimax-m2.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V1** | 60.0% | 95.6% | 84.4% | 42.2% | 57.8% | 51.1% | 31.1% | 68.9% | 55.6% | 37.8% | 68.9% | 6.7% | 65.0%† | 76.9%† | 25.0%† | 66.7%† | — | — |
| **V1RAG** | 0.0% | 22.2% | 15.6% | 6.7% | 17.8% | 6.7% | 4.4% | 4.4% | 16.3% | 2.2% | 6.7% | 4.4% | 17.4%† | 0.0%† | 0.0%† | 0.0%† | — | — |
| **V2** | 80.0% | 95.6% | 75.6% | 55.6% | 62.2% | 42.2% | 6.7% | 71.1% | 50.0% | 44.4% | 64.4% | 33.3% | 66.7%† | 0.0%† | 0.0%† | — | 0.0%† | — |

### RAG hallucination suppression (V1 → V1RAG, Δ percentage points)

| Variant | deepseek-v3.2 | mistral-large-3-675b-instruct-2512 | glm-5.1 | seed-oss-36b-instruct | deepseek-v3.1-terminus | kimi-k2-thinking | gemma-3n-e2b-it | gemma-4-31b-it | kimi-k2-instruct-0905 | gpt-oss-120b | gemma-3n-e4b-it | gpt-oss-20b | minimax-m2.5 | qwen3-next-80b-a3b-thinking | qwen3.5-397b-a17b | qwen3.5-122b-a10b | qwen3-coder-480b-a35b-instruct | minimax-m2.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Δ (V1−V1RAG)** | ✅ Strong (+60pp) | ✅ Strong (+73pp) | ✅ Strong (+69pp) | ✅ Strong (+36pp) | ✅ Strong (+40pp) | ✅ Strong (+44pp) | ✅ Moderate (+27pp) | ✅ Strong (+64pp) | ✅ Strong (+39pp) | ✅ Strong (+36pp) | ✅ Strong (+62pp) | ✓ Mild (+2pp) | ✅ Strong (+48pp) | ✅ Strong (+77pp) | ✅ Moderate (+25pp) | ✅ Strong (+67pp) | — | — |

### Average latency per question

| Variant | deepseek-v3.2 | mistral-large-3-675b-instruct-2512 | glm-5.1 | seed-oss-36b-instruct | deepseek-v3.1-terminus | kimi-k2-thinking | gemma-3n-e2b-it | gemma-4-31b-it | kimi-k2-instruct-0905 | gpt-oss-120b | gemma-3n-e4b-it | gpt-oss-20b | minimax-m2.5 | qwen3-next-80b-a3b-thinking | qwen3.5-397b-a17b | qwen3.5-122b-a10b | qwen3-coder-480b-a35b-instruct | minimax-m2.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V0** | 99.5s | 91.7s | 33.2s | 16.3s | 98.7s | 20.2s | 24.8s | 79.8s† | 1.6s | 1.7s | 10.4s | 0.9s | 28.1s | 12.0s | 76.8s | 7.6s | 1.1s† | 54.6s |
| **V1** | 1693.5s | 689.9s | 254.8s | 366.3s | 459.9s | 735.4s | 98.5s | 281.2s | 11.4s | 25.4s | 77.8s | 4.9s | 238.9s† | 137.5s† | 6.5s† | 4.0s† | 0.0s† | — |
| **V1RAG** | 848.9s | 297.7s | 456.2s | 186.3s | 28.5s | 577.4s | 34.0s | 136.6s | 12.3s | 42.1s | 35.6s | 14.5s | 256.4s† | 29.3s† | 3.2s† | 6.0s† | 0.0s† | — |
| **V2** | 590.9s | 531.7s | 379.8s | 235.8s | 45.7s | 886.8s | 67.3s | 336.3s | 11.7s | 80.8s | 54.0s | 12.5s | 49.7s† | 4.0s† | 4.7s† | 0.0s† | 1.2s† | — |

### Best variant (by accuracy) per model

| Model | Best | V0 acc | V1 acc | V1RAG acc | V2 acc | Lift V0→V1RAG (KFR) |
|---|---|---|---|---|---|---|
| `deepseek-ai/deepseek-v3.2` | **V1** | 0.0% | 20.0% | 20.0% | 20.0% | +132% |
| `mistralai/mistral-large-3-675b-instruct-2512` | **V2** | 8.9% | 46.7% | 44.4% | 48.9% | +62% |
| `z-ai/glm-5.1` | **V1RAG** | 0.0% | 31.1% | 42.2% | 40.0% | +363% |
| `bytedance/seed-oss-36b-instruct` | **V1RAG** | 6.7% | 13.3% | 37.8% | 17.8% | +180% |
| `deepseek-ai/deepseek-v3.1-terminus` | **V1RAG** | 8.9% | 31.1% | 40.0% | 26.7% | +158% |
| `moonshotai/kimi-k2-thinking` | **V1RAG** | 8.9% | 11.1% | 40.0% | 4.4% | +73% |
| `google/gemma-3n-e2b-it` | **V1RAG** | 4.4% | 13.3% | 37.8% | 2.2% | +92% |
| `google/gemma-4-31b-it` | **V1RAG** | 0.0%† | 31.1% | 35.6% | 28.9% | +356% |
| `moonshotai/kimi-k2-instruct-0905` | **V1RAG** | 6.7% | 22.2% | 37.8% | 17.8% | +207% |
| `openai/gpt-oss-120b` | **V1RAG** | 0.0% | 15.6% | 31.1% | 20.0% | +542% |
| `google/gemma-3n-e4b-it` | **V1RAG** | 8.9% | 24.4% | 28.9% | 24.4% | +59% |
| `openai/gpt-oss-20b` | **V1RAG** | 0.0% | 2.2% | 33.3% | 15.6% | +8043% |
| `minimaxai/minimax-m2.5` | **V1RAG** | 2.2% | 11.1%† | 28.9%† | 4.4%† | +102% |
| `qwen/qwen3-next-80b-a3b-thinking` | **V1RAG** | 8.9% | 8.9%† | 11.1%† | 0.0%† | -51% |
| `qwen/qwen3.5-397b-a17b` | **V1** | 0.0% | 2.2%† | 2.2%† | 0.0%† | -83% |
| `qwen/qwen3.5-122b-a10b` | **V0** | 4.4% | 2.2%† | 2.2%† | 0.0%† | -85% |
| `qwen/qwen3-coder-480b-a35b-instruct` | **V0** | 0.0%† | 0.0%† | 0.0%† | 0.0%† | -100% |
| `minimaxai/minimax-m2.7` | **V0** | 0.0% | — | — | — | — |

### Verdict distribution — V1RAG (share of all claims)

| Verdict | deepseek-v3.2 | mistral-large-3-675b-instruct-2512 | glm-5.1 | seed-oss-36b-instruct | deepseek-v3.1-terminus | kimi-k2-thinking | gemma-3n-e2b-it | gemma-4-31b-it | kimi-k2-instruct-0905 | gpt-oss-120b | gemma-3n-e4b-it | gpt-oss-20b | minimax-m2.5 | qwen3-next-80b-a3b-thinking | qwen3.5-397b-a17b | qwen3.5-122b-a10b | qwen3-coder-480b-a35b-instruct | minimax-m2.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **SUPPORTED** | 100.0% | 67.6% | 78.7% | 73.6% | 65.2% | 46.1% | 48.1% | 90.0% | 78.0% | 54.1% | 74.1% | 68.3% | 79.7% | 39.1% | 0.0% | 0.0% | — | — |
| **CONTRADICTED** | 0.0% | 9.3% | 7.4% | 3.4% | 8.5% | 3.9% | 1.5% | 2.5% | 9.9% | 1.4% | 2.1% | 2.9% | 8.1% | 0.0% | 0.0% | 0.0% | — | — |
| **UNVERIFIABLE** | 0.0% | 23.1% | 13.9% | 23.0% | 26.2% | 50.0% | 50.4% | 7.5% | 12.1% | 44.5% | 23.8% | 28.8% | 12.2% | 60.9% | 100.0% | 100.0% | — | — |

---

## Accuracy by Category — V1RAG

| Category | deepseek-v3.2 | mistral-large-3-675b-instruct-2512 | glm-5.1 | seed-oss-36b-instruct | deepseek-v3.1-terminus | kimi-k2-thinking | gemma-3n-e2b-it | gemma-4-31b-it | kimi-k2-instruct-0905 | gpt-oss-120b | gemma-3n-e4b-it | gpt-oss-20b | minimax-m2.5 | qwen3-next-80b-a3b-thinking | qwen3.5-397b-a17b | qwen3.5-122b-a10b | qwen3-coder-480b-a35b-instruct | minimax-m2.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Company Overview (n=6) | 25.0% | 50.0% | 50.0% | 50.0% | 33.3% | 50.0% | 66.7% | 50.0% | 50.0% | 50.0% | 50.0% | 50.0% | 33.3% | 0.0% | 16.7% | 0.0% | 0.0% | — |
| Leadership (n=13) | 0.0% | 53.8% | 46.2% | 30.8% | 53.8% | 38.5% | 30.8% | 38.5% | 53.8% | 30.8% | 23.1% | 23.1% | 23.1% | 7.7% | 0.0% | 0.0% | 0.0% | — |
| Products & Pricing (n=13) | — | 38.5% | 38.5% | 38.5% | 38.5% | 53.8% | 38.5% | 38.5% | 30.8% | 38.5% | 38.5% | 30.8% | 23.1% | 30.8% | 0.0% | 7.7% | 0.0% | — |
| HR Policy (n=5) | — | 0.0% | 0.0% | 0.0% | 0.0% | 20.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | — |
| Benefits (n=8) | — | 62.5% | 62.5% | 62.5% | 50.0% | 25.0% | 50.0% | 37.5% | 37.5% | 25.0% | 25.0% | 50.0% | 62.5% | 0.0% | 0.0% | 0.0% | 0.0% | — |

## KFR by Category — V1RAG

| Category | deepseek-v3.2 | mistral-large-3-675b-instruct-2512 | glm-5.1 | seed-oss-36b-instruct | deepseek-v3.1-terminus | kimi-k2-thinking | gemma-3n-e2b-it | gemma-4-31b-it | kimi-k2-instruct-0905 | gpt-oss-120b | gemma-3n-e4b-it | gpt-oss-20b | minimax-m2.5 | qwen3-next-80b-a3b-thinking | qwen3.5-397b-a17b | qwen3.5-122b-a10b | qwen3-coder-480b-a35b-instruct | minimax-m2.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Company Overview (n=6) | 0.734 | 0.711 | 0.795 | 0.728 | 0.561 | 0.795 | 0.761 | 0.700 | 0.583 | 0.711 | 0.711 | 0.756 | 0.333 | 0.000 | 0.167 | 0.000 | 0.000 | — |
| Leadership (n=13) | 0.750 | 0.735 | 0.688 | 0.605 | 0.709 | 0.650 | 0.590 | 0.609 | 0.747 | 0.547 | 0.658 | 0.509 | 0.494 | 0.108 | 0.000 | 0.000 | 0.000 | — |
| Products & Pricing (n=13) | — | 0.610 | 0.631 | 0.610 | 0.610 | 0.657 | 0.597 | 0.567 | 0.533 | 0.610 | 0.648 | 0.524 | 0.231 | 0.412 | 0.000 | 0.092 | 0.000 | — |
| HR Policy (n=5) | — | 0.467 | 0.400 | 0.320 | 0.438 | 0.247 | 0.129 | 0.234 | 0.180 | 0.280 | 0.095 | 0.334 | 0.000 | 0.029 | 0.133 | 0.029 | 0.000 | — |
| Benefits (n=8) | — | 0.760 | 0.760 | 0.760 | 0.635 | 0.448 | 0.625 | 0.594 | 0.510 | 0.531 | 0.396 | 0.573 | 0.760 | 0.000 | 0.000 | 0.000 | 0.000 | — |

## Accuracy by Difficulty — V1RAG

| Difficulty | deepseek-v3.2 | mistral-large-3-675b-instruct-2512 | glm-5.1 | seed-oss-36b-instruct | deepseek-v3.1-terminus | kimi-k2-thinking | gemma-3n-e2b-it | gemma-4-31b-it | kimi-k2-instruct-0905 | gpt-oss-120b | gemma-3n-e4b-it | gpt-oss-20b | minimax-m2.5 | qwen3-next-80b-a3b-thinking | qwen3.5-397b-a17b | qwen3.5-122b-a10b | qwen3-coder-480b-a35b-instruct | minimax-m2.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Easy (n=21) | 33.3% | 57.1% | 61.9% | 57.1% | 47.6% | 47.6% | 57.1% | 47.6% | 47.6% | 42.9% | 42.9% | 52.4% | 47.6% | 14.3% | 0.0% | 4.8% | 0.0% | — |
| Medium (n=13) | 0.0% | 15.4% | 15.4% | 15.4% | 7.7% | 30.8% | 15.4% | 15.4% | 7.7% | 15.4% | 7.7% | 7.7% | 0.0% | 15.4% | 7.7% | 0.0% | 0.0% | — |
| Hard (n=11) | — | 54.5% | 36.4% | 27.3% | 63.6% | 36.4% | 27.3% | 36.4% | 54.5% | 27.3% | 27.3% | 18.2% | 27.3% | 0.0% | 0.0% | 0.0% | 0.0% | — |

---

## Questions Analysis — How Variants Behave on the Same Question

Micro-case studies on **six questions** (easy / medium / hard across categories). **All 18 models** appear as columns (same order as master tables). Cells use the legend from the earlier 5-model write-up:

- `KFR /marker S/C/U` — Key-fact recall; marker `✓` correct, `!` hallucination flagged, `R` refuse/partial low recall, `P` partial; `S/C/U` = supported/contradicted/unverifiable claim counts. V0 has no verification → `0/0/0`.
- `— _err_` = question-level pipeline error for that model × variant.

### Q01 — When was NovaAI founded and who co-founded it?
`company_overview` / `easy`

**Ground truth:** NovaAI was founded in 2021 by Dr. Mara Chen and Lucas Ferreira.  
**Key facts (3):** `2021`, `Dr. Mara Chen`, `Lucas Ferreira`

| Variant | deepseek-v3.2 | mistral-large-3-675b-instruct-2512 | glm-5.1 | seed-oss-36b-instruct | deepseek-v3.1-terminus | kimi-k2-thinking | gemma-3n-e2b-it | gemma-4-31b-it | kimi-k2-instruct-0905 | gpt-oss-120b | gemma-3n-e4b-it | gpt-oss-20b | minimax-m2.5 | qwen3-next-80b-a3b-thinking | qwen3.5-397b-a17b | qwen3.5-122b-a10b | qwen3-coder-480b-a35b-instruct | minimax-m2.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V0** | 0.33 /P 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.33 /P 0/0/0 | 0.33 /P 0/0/0 | — _err_ | 0.33 /P 0/0/0 | 0.00 /R 0/0/0 | 0.33 /P 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.33 /P 0/0/0 | 0.33 /P 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 |
| **V1** | 1.00 /! 0/4/1 | 1.00 /! 1/4/5 | 1.00 /! 0/1/2 | 0.00 /! 0/1/2 | 1.00 /! 1/4/0 | 0.00 /! 1/1/0 | 0.33 /P 1/0/0 | 1.00 /! 0/1/0 | 1.00 /! 0/1/1 | 1.00 /! 0/1/0 | 0.33 /P 1/0/0 | 0.00 /R 0/0/1 | 1.00 /! 0/1/1 | 1.00 /! 0/2/3 | 1.00 /! 0/2/1 | 1.00 /! 0/1/2 | — _err_ | — _err_ |
| **V1RAG** | 1.00 /✓ 3/0/0 | 1.00 /✓ 3/0/0 | 1.00 /✓ 2/0/0 | 1.00 /✓ 1/0/1 | 1.00 /✓ 2/0/0 | 1.00 /✓ 1/0/1 | 1.00 /✓ 2/0/1 | 1.00 /✓ 2/0/0 | 1.00 /✓ 3/0/0 | 1.00 /✓ 2/0/0 | 1.00 /✓ 3/0/0 | 1.00 /✓ 4/0/0 | — _err_ | — _err_ | — _err_ | — _err_ | — _err_ | — _err_ |
| **V2** | 1.00 /! 1/2/0 | 1.00 /! 0/5/9 | 1.00 /! 0/1/1 | 0.00 /! 0/1/4 | 1.00 /! 0/2/1 | 0.00 /! 0/1/3 | 0.33 /P 1/0/0 | 1.00 /! 0/1/0 | 0.33 /P 1/0/0 | 1.00 /! 0/2/1 | 0.33 /P 1/0/0 | 0.00 /R 0/0/1 | 0.00 /R 0/0/3 | — _err_ | — _err_ | — _err_ | — _err_ | — _err_ |

### Q09 — What is the price of the NovaPilot Starter plan and what does it include?
`products_pricing` / `easy`

**Ground truth:** The NovaPilot Starter plan costs $2,500/month. It includes up to 10 users and 50,000 automation runs per month.  
**Key facts (3):** `$2,500/month`, `10 users`, `50,000`

| Variant | deepseek-v3.2 | mistral-large-3-675b-instruct-2512 | glm-5.1 | seed-oss-36b-instruct | deepseek-v3.1-terminus | kimi-k2-thinking | gemma-3n-e2b-it | gemma-4-31b-it | kimi-k2-instruct-0905 | gpt-oss-120b | gemma-3n-e4b-it | gpt-oss-20b | minimax-m2.5 | qwen3-next-80b-a3b-thinking | qwen3.5-397b-a17b | qwen3.5-122b-a10b | qwen3-coder-480b-a35b-instruct | minimax-m2.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V0** | — _err_ | 0.33 /P 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.33 /P 0/0/0 | 0.33 /P 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.33 /P 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.33 /P 0/0/0 | 0.33 /P 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 |
| **V1** | — _err_ | 1.00 /! 0/3/12 | 1.00 /! 0/2/1 | 0.00 /R 0/0/3 | 1.00 /! 0/1/1 | 0.00 /! 0/1/1 | 1.00 /! 1/2/12 | 1.00 /! 0/1/1 | 1.00 /! 0/1/1 | 1.00 /! 0/1/4 | 1.00 /! 0/1/6 | 0.00 /R 0/0/1 | 1.00 /! 1/1/4 | 0.67 /! 0/1/3 | 0.00 /R 0/0/3 | — _err_ | — _err_ | — _err_ |
| **V1RAG** | — _err_ | 1.00 /✓ 4/0/0 | 1.00 /✓ 3/0/0 | 1.00 /✓ 3/0/0 | 1.00 /✓ 4/0/0 | 1.00 /✓ 0/0/1 | 0.33 /! 1/1/0 | 1.00 /✓ 3/0/0 | 1.00 /✓ 3/0/0 | 1.00 /✓ 3/0/1 | 1.00 /✓ 2/0/1 | 1.00 /✓ 3/0/0 | — _err_ | 1.00 /✓ 1/0/0 | — _err_ | 1.00 /✓ 0/0/4 | — _err_ | — _err_ |
| **V2** | — _err_ | 1.00 /! 0/3/12 | 1.00 /! 0/1/2 | 0.00 /! 0/2/1 | 1.00 /! 0/2/3 | 0.33 /P 0/0/1 | 0.00 /R 0/0/6 | 1.00 /! 0/1/0 | 0.00 /R 1/0/0 | 1.00 /! 0/2/2 | 0.67 /! 1/2/3 | 0.00 /R 0/0/1 | — _err_ | — _err_ | — _err_ | — _err_ | — _err_ | — _err_ |

### Q18 — What is NovaAI's PTO policy?
`hr_policy` / `medium`

**Ground truth:** NovaAI has a flexible/unlimited PTO policy. Employees must take a minimum of 15 days per year. PTO requests of 3 or more days must be submitted in Workday at least 5 business days in advance. Sick leave is tracked separately from PTO. NovaAI also provides 11 paid company holidays per year.  
**Key facts (6):** `flexible/unlimited`, `15 days`, `5 business days`, `Workday`, `separately`, `11`

| Variant | deepseek-v3.2 | mistral-large-3-675b-instruct-2512 | glm-5.1 | seed-oss-36b-instruct | deepseek-v3.1-terminus | kimi-k2-thinking | gemma-3n-e2b-it | gemma-4-31b-it | kimi-k2-instruct-0905 | gpt-oss-120b | gemma-3n-e4b-it | gpt-oss-20b | minimax-m2.5 | qwen3-next-80b-a3b-thinking | qwen3.5-397b-a17b | qwen3.5-122b-a10b | qwen3-coder-480b-a35b-instruct | minimax-m2.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V0** | — _err_ | 0.17 /P 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.17 /P 0/0/0 | 0.17 /R 0/0/0 | 0.50 /P 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.33 /P 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.33 /R 0/0/0 | 0.17 /R 0/0/0 | — _err_ | 0.00 /R 0/0/0 |
| **V1** | — _err_ | 0.33 /! 0/1/14 | 0.17 /R 0/0/3 | 0.17 /! 0/1/1 | 0.17 /P 1/0/3 | 0.00 /R 0/0/5 | 0.50 /! 0/1/14 | 0.00 /R 0/0/2 | 0.00 /R 1/0/1 | 0.00 /R 0/0/4 | 0.50 /P 3/0/10 | 0.00 /R 0/0/1 | 0.00 /R 1/0/4 | — _err_ | — _err_ | — _err_ | — _err_ | — _err_ |
| **V1RAG** | — _err_ | 0.50 /! 2/1/0 | 0.50 /P 2/0/0 | 0.50 /P 1/0/1 | 0.50 /P 3/0/1 | 0.00 /R 0/0/2 | 0.17 /P 0/0/1 | 0.50 /P 1/0/1 | 0.50 /! 0/1/1 | 0.50 /P 1/0/1 | 0.00 /R 0/0/2 | 0.50 /P 2/0/0 | — _err_ | — _err_ | 0.67 /P 0/0/4 | — _err_ | — _err_ | — _err_ |
| **V2** | — _err_ | 0.67 /! 0/1/14 | 0.17 /R 0/0/3 | 0.00 /R 0/0/1 | 0.00 /R 0/0/2 | 0.00 /R 0/0/2 | 0.17 /P 0/0/10 | 0.50 /! 0/1/0 | 0.17 /! 0/1/1 | 0.00 /R 0/0/3 | 0.17 /P 1/0/6 | 0.50 /! 0/1/0 | — _err_ | — _err_ | 0.17 /R 0/0/8 | — _err_ | — _err_ | — _err_ |

### Q25 — What is the annual L&D budget per employee and how does it work?
`benefits` / `medium`

**Ground truth:** Each employee receives a $2,500 annual L&D budget, refreshed on January 1. Unused budget does not roll over. Requests must be submitted via the Workday L&D portal.  
**Key facts (4):** `$2,500`, `January 1`, `does not roll over`, `Workday`

| Variant | deepseek-v3.2 | mistral-large-3-675b-instruct-2512 | glm-5.1 | seed-oss-36b-instruct | deepseek-v3.1-terminus | kimi-k2-thinking | gemma-3n-e2b-it | gemma-4-31b-it | kimi-k2-instruct-0905 | gpt-oss-120b | gemma-3n-e4b-it | gpt-oss-20b | minimax-m2.5 | qwen3-next-80b-a3b-thinking | qwen3.5-397b-a17b | qwen3.5-122b-a10b | qwen3-coder-480b-a35b-instruct | minimax-m2.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V0** | — _err_ | 0.25 /P 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.75 /P 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | — _err_ | 0.00 /R 0/0/0 |
| **V1** | — _err_ | 0.75 /! 1/4/10 | 0.75 /! 0/1/1 | 0.25 /! 0/1/8 | 0.00 /R 0/0/7 | 0.00 /R 0/0/3 | 0.50 /! 0/1/14 | 0.25 /! 0/1/1 | 0.50 /! 1/1/0 | 0.25 /! 0/1/4 | 0.25 /! 2/1/12 | 0.00 /R 0/0/1 | 0.00 /R 0/0/6 | — _err_ | — _err_ | — _err_ | — _err_ | — _err_ |
| **V1RAG** | — _err_ | 0.25 /P 4/0/11 | 0.25 /R 1/0/0 | 0.25 /P 1/0/0 | 0.25 /R 1/0/0 | 0.25 /R 1/0/0 | 0.00 /R 0/0/2 | 0.25 /R 1/0/0 | 0.25 /P 2/0/0 | 0.25 /R 1/0/6 | 0.00 /R 0/0/2 | 0.25 /P 1/0/1 | 0.25 /R 1/0/0 | — _err_ | — _err_ | — _err_ | — _err_ | — _err_ |
| **V2** | — _err_ | 0.75 /! 3/2/10 | 0.25 /! 0/1/9 | 0.00 /R 0/0/3 | 0.00 /R 0/0/3 | 0.00 /R 0/0/8 | 0.25 /! 0/1/10 | 0.25 /! 0/1/0 | 0.25 /! 0/1/0 | 0.25 /! 0/1/3 | 0.25 /! 3/1/5 | 0.50 /! 0/2/5 | — _err_ | — _err_ | — _err_ | — _err_ | — _err_ | — _err_ |

### Q38 — Who leads the SMB Sales team, and what is their FY2026 quota?
`leadership` / `hard`

**Ground truth:** Tanya Osei leads the SMB Sales team, and her FY2026 quota is $6M.  
**Key facts (3):** `Tanya Osei`, `SMB Sales`, `$6M`

| Variant | deepseek-v3.2 | mistral-large-3-675b-instruct-2512 | glm-5.1 | seed-oss-36b-instruct | deepseek-v3.1-terminus | kimi-k2-thinking | gemma-3n-e2b-it | gemma-4-31b-it | kimi-k2-instruct-0905 | gpt-oss-120b | gemma-3n-e4b-it | gpt-oss-20b | minimax-m2.5 | qwen3-next-80b-a3b-thinking | qwen3.5-397b-a17b | qwen3.5-122b-a10b | qwen3-coder-480b-a35b-instruct | minimax-m2.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V0** | — _err_ | 0.33 /P 0/0/0 | 0.33 /R 0/0/0 | 0.33 /R 0/0/0 | 0.33 /R 0/0/0 | 0.33 /R 0/0/0 | 0.33 /P 0/0/0 | 0.00 /R 0/0/0 | 0.33 /P 0/0/0 | 0.00 /R 0/0/0 | 0.33 /P 0/0/0 | 0.00 /R 0/0/0 | 0.33 /R 0/0/0 | 0.33 /P 0/0/0 | 0.33 /R 0/0/0 | 0.33 /R 0/0/0 | — _err_ | 0.00 /R 0/0/0 |
| **V1** | — _err_ | 0.67 /! 0/4/7 | 0.67 /! 0/1/1 | 0.33 /R 0/0/2 | 0.00 /R 0/0/3 | 0.00 /! 0/1/2 | 0.67 /! 0/2/1 | 0.00 /R 0/0/2 | 0.67 /! 0/1/0 | 0.33 /P 0/0/4 | 0.67 /! 0/3/1 | 0.00 /R 0/0/1 | — _err_ | — _err_ | — _err_ | — _err_ | — _err_ | — _err_ |
| **V1RAG** | — _err_ | 0.67 /P 3/0/1 | 0.67 /P 2/0/0 | 0.67 /P 2/0/0 | 0.67 /P 2/0/0 | 0.67 /P 1/0/0 | 0.67 /P 2/0/1 | 0.67 /P 2/0/0 | 0.67 /P 2/0/0 | 0.67 /P 2/0/0 | 0.67 /P 3/0/1 | 0.67 /P 2/0/0 | 0.67 /P 2/0/0 | — _err_ | — _err_ | — _err_ | — _err_ | — _err_ |
| **V2** | — _err_ | 0.67 /! 0/3/4 | 0.67 /! 0/1/1 | 0.67 /! 0/1/1 | 0.67 /! 0/2/1 | 0.33 /R 0/0/2 | 0.33 /R 0/0/6 | 0.67 /! 0/1/1 | 0.67 /! 0/1/0 | 0.00 /R 0/0/1 | 0.67 /! 0/2/0 | 0.67 /! 1/1/0 | — _err_ | — _err_ | 0.33 /R 0/0/6 | — _err_ | — _err_ | — _err_ |

### Q36 — Who is the PM owner of NovaAI's flagship product?
`products_pricing` / `hard`

**Ground truth:** Fatima Al-Rashid is the PM owner of NovaPilot, NovaAI's flagship product.  
**Key facts (2):** `Fatima Al-Rashid`, `NovaPilot`

| Variant | deepseek-v3.2 | mistral-large-3-675b-instruct-2512 | glm-5.1 | seed-oss-36b-instruct | deepseek-v3.1-terminus | kimi-k2-thinking | gemma-3n-e2b-it | gemma-4-31b-it | kimi-k2-instruct-0905 | gpt-oss-120b | gemma-3n-e4b-it | gpt-oss-20b | minimax-m2.5 | qwen3-next-80b-a3b-thinking | qwen3.5-397b-a17b | qwen3.5-122b-a10b | qwen3-coder-480b-a35b-instruct | minimax-m2.7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V0** | — _err_ | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | 0.00 /R 0/0/0 | — _err_ | 0.00 /R 0/0/0 |
| **V1** | — _err_ | 0.50 /! 0/3/5 | 0.00 /R 0/0/2 | 0.00 /R 0/0/1 | 0.00 /R 0/0/3 | 0.00 /! 0/1/2 | 0.00 /R 0/0/1 | 0.00 /R 0/0/2 | 0.00 /R 0/0/2 | 0.00 /R 0/0/1 | 0.50 /! 0/1/4 | 0.00 /R 0/0/1 | — _err_ | — _err_ | — _err_ | — _err_ | — _err_ | — _err_ |
| **V1RAG** | — _err_ | 0.00 /R 0/0/2 | 0.50 /P 2/0/0 | 0.00 /R 0/0/1 | 0.00 /R 0/0/2 | 0.50 /! 0/1/0 | 0.00 /R 0/0/2 | 0.00 /R 1/0/0 | 0.00 /R 0/0/1 | 0.00 /R 0/0/2 | 0.50 /P 2/0/1 | 0.00 /R 0/0/1 | 0.00 /R 1/0/1 | — _err_ | — _err_ | 0.00 /R 0/0/3 | — _err_ | — _err_ |
| **V2** | — _err_ | 0.50 /! 0/2/2 | 0.00 /! 0/1/1 | 0.00 /R 0/0/2 | 0.50 /! 0/1/2 | 0.00 /R 0/0/2 | 0.00 /R 0/0/7 | 0.00 /R 1/0/1 | 0.00 /R 0/0/1 | 0.00 /R 0/0/3 | 0.00 /R 0/0/1 | 0.00 /R 0/0/1 | — _err_ | — _err_ | — _err_ | — _err_ | — _err_ | — _err_ |

### Cross-question synthesis (variant roles)

- **V0** — Fast, unverified; often refuses (e.g. gpt-oss) or confidently confabulates names/prices when it does answer (see Q01/Q09 in the grids above).
- **V1** — Verifier surfaces unsupported claims; you may see high KFR with `!` when the model is right but the KG did not ground the answer.
- **V1RAG** — Strongest *verified* behavior when retrieval hits; also a **refusal mode** when the retriever returns nothing (safe but incomplete).
- **V2** — Mixed: can break refusal habits on some models, but often **regresses** vs V1RAG on the same item because verbal reflections do not replace missing KG triples. Several models show **high error rates†** on V2 in the master table — treat those cells as noisy.

---

## Key Findings (full cohort)

### 1. V1RAG remains the default “production” variant
Across models with stable runs, **V1RAG** delivers the best tradeoff of **accuracy, KFR, and low contradiction rate** once retrieval fires. It is not universal: HR policy stays hard, and some models (e.g. parts of the Qwen family) still produce mostly UNVERIFIABLE claims even with RAG.

### 2. Large instruction models top the leaderboard under V1RAG
By **average KFR under V1RAG**, the top of this export are: `deepseek-ai/deepseek-v3.2` (0.737), `mistralai/mistral-large-3-675b-instruct-2512` (0.670), `z-ai/glm-5.1` (0.667). **Accuracy** leaders cluster in the 40% range on this 45-question suite (see Best variant table).

### 3. Qwen 3.5 × pipeline instability
`qwen3.5-122b` and `qwen3.5-397b` show **heavy per-question errors†** on V1/V1RAG/V2 in this export; interpret their aggregates cautiously. `qwen3-next-80b-a3b-thinking` also shows a **fragile V2** run. Registry `enable_thinking: False` may interact badly with extraction/verification — worth an ablation.

### 4. `qwen3-coder-480b` is mostly broken on V1+
This checkpoint hits **many errors** on V1/V1RAG/V2; V0 partially completes. Treat as **out-of-distribution** for this JSON/claim workflow unless retried with different decoding or prompts.

### 5. MiniMax M2.7 incomplete
Only **V0** results exist in `results_v0-v2/minimaxai-minimax-m2_7/`; no cross-variant comparison yet.

### 6. HR policy is still the ceiling
Even strong models rarely get HR-policy questions to **correct** (KFR ≥ 0.8); the bottleneck is likely **multi-condition retrieval + verifier coverage**, not raw LM scale alone.

---

## Per-Model Analysis

One subsection per model (alphabetical). **Source folder** after each title is the directory under `results_v0-v2/` used after deduplication.

### `bytedance/seed-oss-36b-instruct`
*Folder:* `bytedance-seed-oss-36b-instruct`

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U | err q |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 6.7% | 0.221 | — | 16.3s | — | — | — | — | 0/45 |
| V1 | 13.3% | 0.292 | 42.2% | 366.3s | 3.8 | 0.6% | 11.2% | 88.2% | 0/45 |
| V1RAG | 37.8% | 0.619 | 6.7% | 186.3s | 1.9 | 73.6% | 3.4% | 23.0% | 0/45 |
| V2 | 17.8% | 0.321 | 55.6% | 235.8s | 2.6 | 16.2% | 29.1% | 54.7% | 0/45 |

| Category | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.111 / 0.0% | 0.250 / 16.7% | 0.728 / 50.0% | 0.333 / 33.3% |
| Leadership (n=13) | 0.242 / 0.0% | 0.329 / 7.7% | 0.605 / 30.8% | 0.429 / 15.4% |
| Products & Pricing (n=13) | 0.252 / 15.4% | 0.304 / 15.4% | 0.610 / 38.5% | 0.374 / 15.4% |
| HR Policy (n=5) | 0.067 / 0.0% | 0.033 / 0.0% | 0.320 / 0.0% | 0.000 / 0.0% |
| Benefits (n=8) | 0.312 / 12.5% | 0.406 / 25.0% | 0.760 / 62.5% | 0.250 / 25.0% |

| Difficulty | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.274 / 14.3% | 0.438 / 23.8% | 0.763 / 57.1% | 0.397 / 28.6% |
| Medium (n=13) | 0.103 / 0.0% | 0.060 / 0.0% | 0.444 / 15.4% | 0.188 / 7.7% |
| Hard (n=11) | 0.257 / 0.0% | 0.288 / 9.1% | 0.549 / 27.3% | 0.333 / 9.1% |

### `deepseek-ai/deepseek-v3.1-terminus`
*Folder:* `deepseek-ai-deepseek-v3_1-terminus`

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U | err q |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 8.9% | 0.239 | — | 98.7s | — | — | — | — | 0/45 |
| V1 | 31.1% | 0.507 | 57.8% | 459.9s | 4.8 | 7.5% | 17.3% | 75.2% | 0/45 |
| V1RAG | 40.0% | 0.617 | 17.8% | 28.5s | 3.1 | 65.2% | 8.5% | 26.2% | 0/45 |
| V2 | 26.7% | 0.496 | 62.2% | 45.7s | 3.0 | 2.2% | 29.9% | 67.9% | 0/45 |

| Category | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.333 / 33.3% | 0.639 / 33.3% | 0.561 / 33.3% | 0.639 / 33.3% |
| Leadership (n=13) | 0.258 / 0.0% | 0.519 / 30.8% | 0.709 / 53.8% | 0.513 / 15.4% |
| Products & Pricing (n=13) | 0.214 / 7.7% | 0.553 / 38.5% | 0.610 / 38.5% | 0.565 / 30.8% |
| HR Policy (n=5) | 0.225 / 0.0% | 0.443 / 20.0% | 0.438 / 0.0% | 0.029 / 0.0% |
| Benefits (n=8) | 0.188 / 12.5% | 0.354 / 25.0% | 0.635 / 50.0% | 0.542 / 50.0% |

| Difficulty | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.282 / 19.0% | 0.577 / 38.1% | 0.636 / 47.6% | 0.659 / 47.6% |
| Medium (n=13) | 0.106 / 0.0% | 0.433 / 23.1% | 0.471 / 7.7% | 0.249 / 7.7% |
| Hard (n=11) | 0.314 / 0.0% | 0.462 / 27.3% | 0.754 / 63.6% | 0.477 / 9.1% |

### `deepseek-ai/deepseek-v3.2`
*Folder:* `deepseek-ai-deepseek-v3_2`

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U | err q |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 0.0% | 0.317 | — | 99.5s | — | — | — | — | 0/5 |
| V1 | 20.0% | 0.567 | 60.0% | 1693.5s | 6.2 | 3.2% | 25.8% | 71.0% | 0/5 |
| V1RAG | 20.0% | 0.737 | 0.0% | 848.9s | 3.2 | 100.0% | 0.0% | 0.0% | 0/5 |
| V2 | 20.0% | 0.617 | 80.0% | 590.9s | 4.2 | 4.8% | 33.3% | 61.9% | 0/5 |

| Category | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.333 / 0.0% | 0.584 / 25.0% | 0.734 / 25.0% | 0.584 / 25.0% |
| Leadership (n=13) | 0.250 / 0.0% | 0.500 / 0.0% | 0.750 / 0.0% | 0.750 / 0.0% |
| Products & Pricing (n=13) | — / — | — / — | — / — | — / — |
| HR Policy (n=5) | — / — | — / — | — / — | — / — |
| Benefits (n=8) | — / — | — / — | — / — | — / — |

| Difficulty | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.417 / 0.0% | 0.722 / 33.3% | 0.806 / 33.3% | 0.806 / 33.3% |
| Medium (n=13) | 0.167 / 0.0% | 0.334 / 0.0% | 0.633 / 0.0% | 0.334 / 0.0% |
| Hard (n=11) | — / — | — / — | — / — | — / — |

### `google/gemma-3n-e2b-it`
*Folder:* `google-gemma-3n-e2b-it`

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U | err q |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 4.4% | 0.297 | — | 24.8s | — | — | — | — | 0/45 |
| V1 | 13.3% | 0.410 | 31.1% | 98.5s | 7.1 | 5.3% | 6.0% | 88.7% | 0/45 |
| V1RAG | 37.8% | 0.570 | 4.4% | 34.0s | 2.9 | 48.1% | 1.5% | 50.4% | 0/45 |
| V2 | 2.2% | 0.249 | 6.7% | 67.3s | 6.7 | 4.0% | 1.7% | 94.4% | 0/45 |

| Category | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.395 / 0.0% | 0.572 / 33.3% | 0.761 / 66.7% | 0.433 / 0.0% |
| Leadership (n=13) | 0.296 / 0.0% | 0.458 / 7.7% | 0.590 / 30.8% | 0.232 / 0.0% |
| Products & Pricing (n=13) | 0.273 / 7.7% | 0.372 / 15.4% | 0.597 / 38.5% | 0.259 / 7.7% |
| HR Policy (n=5) | 0.224 / 0.0% | 0.249 / 0.0% | 0.129 / 0.0% | 0.092 / 0.0% |
| Benefits (n=8) | 0.312 / 12.5% | 0.375 / 12.5% | 0.625 / 50.0% | 0.219 / 0.0% |

| Difficulty | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.354 / 9.5% | 0.455 / 19.0% | 0.732 / 57.1% | 0.257 / 4.8% |
| Medium (n=13) | 0.240 / 0.0% | 0.381 / 7.7% | 0.373 / 15.4% | 0.244 / 0.0% |
| Hard (n=11) | 0.257 / 0.0% | 0.360 / 9.1% | 0.492 / 27.3% | 0.238 / 0.0% |

### `google/gemma-3n-e4b-it`
*Folder:* `google-gemma-3n-e4b-it_run1`

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U | err q |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 8.9% | 0.348 | — | 10.4s | — | — | — | — | 0/45 |
| V1 | 24.4% | 0.543 | 68.9% | 77.8s | 7.6 | 12.9% | 14.1% | 73.0% | 0/45 |
| V1RAG | 28.9% | 0.553 | 6.7% | 35.6s | 4.2 | 74.1% | 2.1% | 23.8% | 0/45 |
| V2 | 24.4% | 0.523 | 64.4% | 54.0s | 5.2 | 20.0% | 16.6% | 63.4% | 0/45 |

| Category | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.478 / 16.7% | 0.745 / 50.0% | 0.711 / 50.0% | 0.655 / 33.3% |
| Leadership (n=13) | 0.299 / 0.0% | 0.597 / 23.1% | 0.658 / 23.1% | 0.577 / 30.8% |
| Products & Pricing (n=13) | 0.308 / 15.4% | 0.464 / 23.1% | 0.648 / 38.5% | 0.401 / 15.4% |
| HR Policy (n=5) | 0.361 / 0.0% | 0.337 / 0.0% | 0.095 / 0.0% | 0.456 / 0.0% |
| Benefits (n=8) | 0.385 / 12.5% | 0.562 / 25.0% | 0.396 / 25.0% | 0.573 / 37.5% |

| Difficulty | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.410 / 19.0% | 0.605 / 33.3% | 0.605 / 42.9% | 0.588 / 33.3% |
| Medium (n=13) | 0.268 / 0.0% | 0.414 / 15.4% | 0.399 / 7.7% | 0.372 / 7.7% |
| Hard (n=11) | 0.322 / 0.0% | 0.580 / 18.2% | 0.636 / 27.3% | 0.576 / 27.3% |

### `google/gemma-4-31b-it`
*Folder:* `google-gemma-4-31b-it`

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U | err q |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 0.0% | 0.124 | — | 79.8s | — | — | — | — | 5/45 |
| V1 | 31.1% | 0.453 | 68.9% | 281.2s | 1.6 | 1.4% | 45.8% | 52.8% | 0/45 |
| V1RAG | 35.6% | 0.565 | 4.4% | 136.6s | 1.8 | 90.0% | 2.5% | 7.5% | 0/45 |
| V2 | 28.9% | 0.494 | 71.1% | 336.3s | 1.8 | 13.6% | 40.7% | 45.7% | 0/45 |

| Category | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.083 / 0.0% | 0.728 / 50.0% | 0.700 / 50.0% | 0.700 / 50.0% |
| Leadership (n=13) | 0.197 / 0.0% | 0.417 / 30.8% | 0.609 / 38.5% | 0.497 / 15.4% |
| Products & Pricing (n=13) | 0.077 / 0.0% | 0.507 / 30.8% | 0.567 / 38.5% | 0.473 / 30.8% |
| HR Policy (n=5) | 0.000 / 0.0% | 0.054 / 0.0% | 0.234 / 0.0% | 0.234 / 0.0% |
| Benefits (n=8) | 0.188 / 0.0% | 0.469 / 37.5% | 0.594 / 37.5% | 0.531 / 50.0% |

| Difficulty | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.143 / 0.0% | 0.593 / 47.6% | 0.700 / 47.6% | 0.664 / 52.4% |
| Medium (n=13) | 0.031 / 0.0% | 0.262 / 7.7% | 0.378 / 15.4% | 0.269 / 0.0% |
| Hard (n=11) | 0.197 / 0.0% | 0.413 / 27.3% | 0.526 / 36.4% | 0.436 / 18.2% |

### `minimaxai/minimax-m2.5`
*Folder:* `minimaxai-minimax-m2_5`

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U | err q |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 2.2% | 0.193 | — | 28.1s | — | — | — | — | 0/45 |
| V1 | 11.1% | 0.209 | 65.0% | 238.9s | 5.4 | 8.3% | 13.0% | 78.7% | 25/45 |
| V1RAG | 28.9% | 0.389 | 17.4% | 256.4s | 3.2 | 79.7% | 8.1% | 12.2% | 22/45 |
| V2 | 4.4% | 0.076 | 66.7% | 49.7s | 4.8 | 13.8% | 24.1% | 62.1% | 39/45 |

| Category | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.167 / 0.0% | 0.422 / 16.7% | 0.333 / 33.3% | 0.278 / 16.7% |
| Leadership (n=13) | 0.322 / 0.0% | 0.150 / 7.7% | 0.494 / 23.1% | 0.135 / 7.7% |
| Products & Pricing (n=13) | 0.141 / 7.7% | 0.377 / 23.1% | 0.231 / 23.1% | 0.000 / 0.0% |
| HR Policy (n=5) | 0.033 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% |
| Benefits (n=8) | 0.188 / 0.0% | 0.000 / 0.0% | 0.760 / 62.5% | 0.000 / 0.0% |

| Difficulty | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.202 / 4.8% | 0.307 / 19.0% | 0.516 / 47.6% | 0.067 / 0.0% |
| Medium (n=13) | 0.097 / 0.0% | 0.226 / 7.7% | 0.019 / 0.0% | 0.154 / 15.4% |
| Hard (n=11) | 0.288 / 0.0% | 0.000 / 0.0% | 0.583 / 27.3% | 0.000 / 0.0% |

### `minimaxai/minimax-m2.7`
*Folder:* `minimaxai-minimax-m2_7`

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U | err q |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 0.0% | 0.137 | — | 54.6s | — | — | — | — | 0/45 |
| V1 | — | — | — | — | — | — | — | — | — |
| V1RAG | — | — | — | — | — | — | — | — | — |
| V2 | — | — | — | — | — | — | — | — | — |

| Category | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.083 / 0.0% | — | — | — |
| Leadership (n=13) | 0.242 / 0.0% | — | — | — |
| Products & Pricing (n=13) | 0.077 / 0.0% | — | — | — |
| HR Policy (n=5) | 0.000 / 0.0% | — | — | — |
| Benefits (n=8) | 0.188 / 0.0% | — | — | — |

| Difficulty | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.155 / 0.0% | — | — | — |
| Medium (n=13) | 0.031 / 0.0% | — | — | — |
| Hard (n=11) | 0.227 / 0.0% | — | — | — |

### `mistralai/mistral-large-3-675b-instruct-2512`
*Folder:* `mistralai-mistral-large-3-675b-instruct-2512`

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U | err q |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 8.9% | 0.414 | — | 91.7s | — | — | — | — | 0/45 |
| V1 | 46.7% | 0.727 | 95.6% | 689.9s | 10.6 | 7.1% | 23.7% | 69.1% | 0/45 |
| V1RAG | 44.4% | 0.670 | 22.2% | 297.7s | 4.8 | 67.6% | 9.3% | 23.1% | 0/45 |
| V2 | 48.9% | 0.762 | 95.6% | 531.7s | 9.9 | 9.8% | 27.7% | 62.4% | 0/45 |

| Category | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.339 / 0.0% | 0.795 / 50.0% | 0.711 / 50.0% | 0.767 / 50.0% |
| Leadership (n=13) | 0.394 / 0.0% | 0.787 / 46.2% | 0.735 / 53.8% | 0.806 / 53.8% |
| Products & Pricing (n=13) | 0.374 / 15.4% | 0.690 / 53.8% | 0.610 / 38.5% | 0.706 / 46.2% |
| HR Policy (n=5) | 0.493 / 0.0% | 0.563 / 20.0% | 0.467 / 0.0% | 0.656 / 20.0% |
| Benefits (n=8) | 0.521 / 25.0% | 0.740 / 50.0% | 0.760 / 62.5% | 0.844 / 62.5% |

| Difficulty | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.446 / 19.0% | 0.779 / 61.9% | 0.732 / 57.1% | 0.831 / 66.7% |
| Medium (n=13) | 0.397 / 0.0% | 0.597 / 23.1% | 0.513 / 15.4% | 0.635 / 23.1% |
| Hard (n=11) | 0.375 / 0.0% | 0.780 / 45.5% | 0.739 / 54.5% | 0.780 / 45.5% |

### `moonshotai/kimi-k2-instruct-0905`
*Folder:* `moonshotai-kimi-k2-instruct-0905`

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U | err q |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 6.7% | 0.182 | — | 1.6s | — | — | — | — | 0/45 |
| V1 | 22.2% | 0.420 | 55.6% | 11.4s | 2.3 | 10.6% | 26.0% | 63.5% | 0/45 |
| V1RAG | 37.8% | 0.558 | 16.3% | 12.3s | 2.1 | 78.0% | 9.9% | 12.1% | 2/45 |
| V2 | 17.8% | 0.365 | 50.0% | 11.7s | 2.1 | 18.1% | 34.0% | 47.9% | 1/45 |

| Category | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.139 / 0.0% | 0.395 / 16.7% | 0.583 / 50.0% | 0.517 / 16.7% |
| Leadership (n=13) | 0.197 / 0.0% | 0.288 / 7.7% | 0.747 / 53.8% | 0.491 / 23.1% |
| Products & Pricing (n=13) | 0.203 / 15.4% | 0.460 / 30.8% | 0.533 / 30.8% | 0.349 / 15.4% |
| HR Policy (n=5) | 0.000 / 0.0% | 0.292 / 20.0% | 0.180 / 0.0% | 0.033 / 0.0% |
| Benefits (n=8) | 0.271 / 12.5% | 0.667 / 37.5% | 0.510 / 37.5% | 0.281 / 25.0% |

| Difficulty | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.264 / 14.3% | 0.589 / 38.1% | 0.636 / 47.6% | 0.398 / 23.8% |
| Medium (n=13) | 0.031 / 0.0% | 0.223 / 7.7% | 0.392 / 7.7% | 0.339 / 7.7% |
| Hard (n=11) | 0.204 / 0.0% | 0.329 / 9.1% | 0.606 / 54.5% | 0.333 / 18.2% |

### `moonshotai/kimi-k2-thinking`
*Folder:* `moonshotai-kimi-k2-thinking`

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U | err q |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 8.9% | 0.342 | — | 20.2s | — | — | — | — | 0/45 |
| V1 | 11.1% | 0.273 | 51.1% | 735.4s | 3.8 | 3.5% | 17.1% | 79.4% | 0/45 |
| V1RAG | 40.0% | 0.590 | 6.7% | 577.4s | 1.7 | 46.1% | 3.9% | 50.0% | 0/45 |
| V2 | 4.4% | 0.161 | 42.2% | 886.8s | 3.3 | 0.7% | 15.6% | 83.7% | 0/45 |

| Category | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.339 / 0.0% | 0.144 / 0.0% | 0.795 / 50.0% | 0.278 / 16.7% |
| Leadership (n=13) | 0.311 / 0.0% | 0.274 / 15.4% | 0.650 / 38.5% | 0.211 / 0.0% |
| Products & Pricing (n=13) | 0.340 / 15.4% | 0.279 / 15.4% | 0.657 / 53.8% | 0.064 / 0.0% |
| HR Policy (n=5) | 0.245 / 0.0% | 0.347 / 0.0% | 0.247 / 20.0% | 0.000 / 0.0% |
| Benefits (n=8) | 0.458 / 25.0% | 0.312 / 12.5% | 0.448 / 25.0% | 0.250 / 12.5% |

| Difficulty | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.417 / 19.0% | 0.256 / 9.5% | 0.668 / 47.6% | 0.234 / 9.5% |
| Medium (n=13) | 0.274 / 0.0% | 0.193 / 0.0% | 0.464 / 30.8% | 0.026 / 0.0% |
| Hard (n=11) | 0.280 / 0.0% | 0.401 / 27.3% | 0.591 / 36.4% | 0.182 / 0.0% |

### `openai/gpt-oss-120b`
*Folder:* `openai-gpt-oss-120b`

**Note:** V0–V2 numbers are from `*_results_rescored.json` (fixed KFR metric). Changes vs. original: V1 acc 13.3%→15.6%, KFR 0.296→0.308; V1RAG KFR 0.555→0.565; V2 acc 17.8%→20.0%, KFR 0.327→0.350.

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U | err q |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 0.0% | 0.088 | — | 1.7s | — | — | — | — | 0/45 |
| V1 | 15.6% | 0.308 | 37.8% | 25.4s | 3.1 | 2.1% | 14.3% | 83.6% | 0/45 |
| V1RAG | 31.1% | 0.565 | 2.2% | 42.1s | 3.2 | 54.1% | 1.4% | 44.5% | 0/45 |
| V2 | 20.0% | 0.350 | 44.4% | 80.8s | 3.4 | 4.0% | 19.2% | 76.8% | 0/45 |

| Category | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.083 / 0.0% | 0.450 / 33.3% | 0.711 / 50.0% | 0.283 / 16.7% |
| Leadership (n=13) | 0.118 / 0.0% | 0.105 / 0.0% | 0.547 / 30.8% | 0.241 / 7.7% |
| Products & Pricing (n=13) | 0.111 / 0.0% | 0.436 / 23.1% | 0.610 / 38.5% | 0.471 / 30.8% |
| HR Policy (n=5) | 0.000 / 0.0% | 0.100 / 0.0% | 0.280 / 0.0% | 0.100 / 0.0% |
| Benefits (n=8) | 0.062 / 0.0% | 0.385 / 12.5% | 0.531 / 25.0% | 0.406 / 25.0% |

| Difficulty | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.095 / 0.0% | 0.448 / 28.6% | 0.621 / 42.9% | 0.448 / 33.3% |
| Medium (n=13) | 0.024 / 0.0% | 0.211 / 0.0% | 0.520 / 15.4% | 0.240 / 0.0% |
| Hard (n=11) | 0.151 / 0.0% | 0.106 / 0.0% | 0.470 / 27.3% | 0.197 / 9.1% |

#### V2P variants (`results/openai-gpt-oss-120b/`)

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U |
|---|---|---|---|---|---|---|---|---|
| V2P | 80.0% | 0.890 | 6.7% | 50.2s | 3.69 | 60.8% | 1.8% | 37.3% |
| V2P_RAG | 73.3% | 0.839 | 2.2% | 38.9s | 3.09 | 66.2% | 0.7% | 33.1% |
| V2P_RGR | 71.1% | 0.829 | 4.4% | 308.0s | 3.11 | 62.1% | 2.9% | 35.0% |
| V2P_RAG_RGR | 77.8% | 0.865 | 4.4% | 38.2s | 3.33 | 63.3% | 1.3% | 35.3% |

| Category | V2P KFR / acc | V2P_RAG KFR / acc | V2P_RGR KFR / acc | V2P_RAG_RGR KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.822 / 66.7% | 0.811 / 66.7% | 0.783 / 66.7% | 0.933 / 83.3% |
| Leadership (n=13) | 0.965 / 92.3% | 0.940 / 84.6% | 0.904 / 84.6% | 0.940 / 84.6% |
| Products & Pricing (n=13) | 0.815 / 69.2% | 0.717 / 53.8% | 0.752 / 61.5% | 0.747 / 61.5% |
| HR Policy (n=5) | 0.946 / 100.0% | 0.770 / 80.0% | 0.766 / 60.0% | 0.680 / 60.0% |
| Benefits (n=8) | 0.906 / 75.0% | 0.938 / 87.5% | 0.906 / 75.0% | 1.000 / 100.0% |

| Difficulty | V2P KFR / acc | V2P_RAG KFR / acc | V2P_RGR KFR / acc | V2P_RAG_RGR KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.873 / 76.2% | 0.829 / 66.7% | 0.829 / 66.7% | 0.868 / 76.2% |
| Medium (n=13) | 0.835 / 69.2% | 0.755 / 69.2% | 0.685 / 53.8% | 0.849 / 76.9% |
| Hard (n=11) | 0.989 / 100.0% | 0.958 / 90.9% | 1.000 / 100.0% | 0.879 / 81.8% |

### `openai/gpt-oss-20b`
*Folder:* `openai-gpt-oss-20b_run1`

**Note:** V0–V2 numbers are from `*_results_rescored.json` (fixed KFR metric). Changes vs. original: V1RAG acc 31.1%→33.3%, KFR 0.538→0.570; V2 acc 13.3%→15.6%, KFR 0.251→0.264.

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U | err q |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 0.0% | 0.007 | — | 0.9s | — | — | — | — | 0/45 |
| V1 | 2.2% | 0.040 | 6.7% | 4.9s | 1.1 | 0.0% | 5.9% | 94.1% | 0/45 |
| V1RAG | 33.3% | 0.570 | 4.4% | 14.5s | 2.3 | 68.3% | 2.9% | 28.8% | 0/45 |
| V2 | 15.6% | 0.264 | 33.3% | 12.5s | 1.6 | 12.5% | 23.6% | 63.9% | 0/45 |

| Category | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.000 / 0.0% | 0.000 / 0.0% | 0.756 / 50.0% | 0.167 / 16.7% |
| Leadership (n=13) | 0.000 / 0.0% | 0.000 / 0.0% | 0.509 / 23.1% | 0.192 / 0.0% |
| Products & Pricing (n=13) | 0.026 / 0.0% | 0.094 / 7.7% | 0.524 / 30.8% | 0.186 / 15.4% |
| HR Policy (n=5) | 0.000 / 0.0% | 0.133 / 0.0% | 0.334 / 0.0% | 0.272 / 0.0% |
| Benefits (n=8) | 0.000 / 0.0% | 0.000 / 0.0% | 0.573 / 50.0% | 0.500 / 37.5% |

| Difficulty | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.000 / 0.0% | 0.000 / 0.0% | 0.668 / 52.4% | 0.271 / 23.8% |
| Medium (n=13) | 0.000 / 0.0% | 0.068 / 0.0% | 0.402 / 7.7% | 0.150 / 0.0% |
| Hard (n=11) | 0.030 / 0.0% | 0.091 / 9.1% | 0.451 / 18.2% | 0.330 / 9.1% |

#### V2P variants (`results/openai-gpt-oss-20b/`)

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U |
|---|---|---|---|---|---|---|---|---|
| V2P | 66.7% | 0.807 | 6.7% | 34.0s | 3.24 | 66.4% | 2.1% | 31.5% |
| V2P_RAG | 53.3% | 0.658 | 4.4% | 32.6s | 2.42 | 75.2% | 3.7% | 21.1% |
| V2P_RGR | 68.9% | 0.809 | 11.1% | 47.5s | 2.96 | 49.6% | 4.5% | 45.9% |
| V2P_RAG_RGR | 60.0% | 0.743 | 13.3% | 47.8s | 2.62 | 62.7% | 7.6% | 29.7% |

| Category | V2P KFR / acc | V2P_RAG KFR / acc | V2P_RGR KFR / acc | V2P_RAG_RGR KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.878 / 66.7% | 0.783 / 66.7% | 0.711 / 50.0% | 0.783 / 66.7% |
| Leadership (n=13) | 0.856 / 69.2% | 0.712 / 61.5% | 0.863 / 76.9% | 0.788 / 69.2% |
| Products & Pricing (n=13) | 0.819 / 76.9% | 0.670 / 53.8% | 0.739 / 61.5% | 0.709 / 46.2% |
| HR Policy (n=5) | 0.472 / 40.0% | 0.438 / 20.0% | 0.909 / 80.0% | 0.451 / 20.0% |
| Benefits (n=8) | 0.865 / 62.5% | 0.594 / 50.0% | 0.844 / 75.0% | 0.875 / 87.5% |

| Difficulty | V2P KFR / acc | V2P_RAG KFR / acc | V2P_RGR KFR / acc | V2P_RAG_RGR KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.814 / 66.7% | 0.725 / 57.1% | 0.749 / 61.9% | 0.765 / 57.1% |
| Medium (n=13) | 0.753 / 61.5% | 0.519 / 38.5% | 0.768 / 61.5% | 0.643 / 46.2% |
| Hard (n=11) | 0.860 / 72.7% | 0.693 / 63.6% | 0.970 / 90.9% | 0.818 / 81.8% |

### `qwen/qwen3-coder-480b-a35b-instruct`
*Folder:* `qwen-qwen3-coder-480b-a35b-instruct`

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U | err q |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 0.0% | 0.030 | — | 1.1s | — | — | — | — | 30/45 |
| V1 | 0.0% | 0.000 | — | 0.0s | — | — | — | — | 45/45 |
| V1RAG | 0.0% | 0.000 | — | 0.0s | — | — | — | — | 45/45 |
| V2 | 0.0% | 0.011 | 0.0% | 1.2s | 3.0 | 0.0% | 0.0% | 100.0% | 44/45 |

| Category | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.000 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% |
| Leadership (n=13) | 0.050 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% | 0.038 / 0.0% |
| Products & Pricing (n=13) | 0.017 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% |
| HR Policy (n=5) | 0.000 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% |
| Benefits (n=8) | 0.062 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% |

| Difficulty | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.036 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% |
| Medium (n=13) | 0.048 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% |
| Hard (n=11) | 0.000 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% | 0.045 / 0.0% |

### `qwen/qwen3-next-80b-a3b-thinking`
*Folder:* `qwen-qwen3-next-80b-a3b-thinking`

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U | err q |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 8.9% | 0.315 | — | 12.0s | — | — | — | — | 0/45 |
| V1 | 8.9% | 0.177 | 76.9% | 137.5s | 6.6 | 0.0% | 19.8% | 80.2% | 32/45 |
| V1RAG | 11.1% | 0.153 | 0.0% | 29.3s | 2.3 | 39.1% | 0.0% | 60.9% | 35/45 |
| V2 | 0.0% | 0.026 | 0.0% | 4.0s | 5.0 | 0.0% | 0.0% | 100.0% | 43/45 |

| Category | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.250 / 0.0% | 0.422 / 16.7% | 0.000 / 0.0% | 0.000 / 0.0% |
| Leadership (n=13) | 0.269 / 0.0% | 0.181 / 7.7% | 0.108 / 7.7% | 0.000 / 0.0% |
| Products & Pricing (n=13) | 0.294 / 15.4% | 0.238 / 15.4% | 0.412 / 30.8% | 0.038 / 0.0% |
| HR Policy (n=5) | 0.133 / 0.0% | 0.000 / 0.0% | 0.029 / 0.0% | 0.000 / 0.0% |
| Benefits (n=8) | 0.583 / 25.0% | 0.000 / 0.0% | 0.000 / 0.0% | 0.083 / 0.0% |

| Difficulty | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.403 / 19.0% | 0.223 / 9.5% | 0.179 / 14.3% | 0.056 / 0.0% |
| Medium (n=13) | 0.182 / 0.0% | 0.254 / 15.4% | 0.243 / 15.4% | 0.000 / 0.0% |
| Hard (n=11) | 0.303 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% |

### `qwen/qwen3.5-122b-a10b`
*Folder:* `qwen-qwen3_5-122b-a10b`

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U | err q |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 4.4% | 0.200 | — | 7.6s | — | — | — | — | 0/45 |
| V1 | 2.2% | 0.022 | 66.7% | 4.0s | 5.0 | 0.0% | 13.3% | 86.7% | 42/45 |
| V1RAG | 2.2% | 0.030 | 0.0% | 6.0s | 3.0 | 0.0% | 0.0% | 100.0% | 41/45 |
| V2 | 0.0% | 0.000 | — | 0.0s | — | — | — | — | 45/45 |

| Category | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.083 / 0.0% | 0.167 / 16.7% | 0.000 / 0.0% | 0.000 / 0.0% |
| Leadership (n=13) | 0.268 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% |
| Products & Pricing (n=13) | 0.122 / 0.0% | 0.000 / 0.0% | 0.092 / 7.7% | 0.000 / 0.0% |
| HR Policy (n=5) | 0.033 / 0.0% | 0.000 / 0.0% | 0.029 / 0.0% | 0.000 / 0.0% |
| Benefits (n=8) | 0.406 / 25.0% | 0.000 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% |

| Difficulty | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.250 / 9.5% | 0.048 / 4.8% | 0.057 / 4.8% | 0.000 / 0.0% |
| Medium (n=13) | 0.044 / 0.0% | 0.000 / 0.0% | 0.011 / 0.0% | 0.000 / 0.0% |
| Hard (n=11) | 0.288 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% |

### `qwen/qwen3.5-397b-a17b`
*Folder:* `qwen-qwen3_5-397b-a17b`

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U | err q |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 0.0% | 0.221 | — | 76.8s | — | — | — | — | 0/45 |
| V1 | 2.2% | 0.033 | 25.0% | 6.5s | 4.2 | 0.0% | 11.8% | 88.2% | 41/45 |
| V1RAG | 2.2% | 0.037 | 0.0% | 3.2s | 3.0 | 0.0% | 0.0% | 100.0% | 43/45 |
| V2 | 0.0% | 0.011 | 0.0% | 4.7s | 7.0 | 0.0% | 0.0% | 100.0% | 43/45 |

| Category | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.250 / 0.0% | 0.167 / 16.7% | 0.167 / 16.7% | 0.000 / 0.0% |
| Leadership (n=13) | 0.281 / 0.0% | 0.038 / 0.0% | 0.000 / 0.0% | 0.026 / 0.0% |
| Products & Pricing (n=13) | 0.159 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% |
| HR Policy (n=5) | 0.100 / 0.0% | 0.000 / 0.0% | 0.133 / 0.0% | 0.033 / 0.0% |
| Benefits (n=8) | 0.281 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% | 0.000 / 0.0% |

| Difficulty | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.281 / 0.0% | 0.048 / 4.8% | 0.000 / 0.0% | 0.000 / 0.0% |
| Medium (n=13) | 0.095 / 0.0% | 0.000 / 0.0% | 0.128 / 7.7% | 0.013 / 0.0% |
| Hard (n=11) | 0.257 / 0.0% | 0.045 / 0.0% | 0.000 / 0.0% | 0.030 / 0.0% |

### `z-ai/glm-5.1`
*Folder:* `z-ai-glm-5_1_run1`

| Variant | Accuracy | Avg KFR | Halluc. % | Avg latency | Avg claims | S | C | U | err q |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 0.0% | 0.144 | — | 33.2s | — | — | — | — | 0/45 |
| V1 | 31.1% | 0.588 | 84.4% | 254.8s | 2.4 | 0.0% | 42.6% | 57.4% | 0/45 |
| V1RAG | 42.2% | 0.667 | 15.6% | 456.2s | 2.7 | 78.7% | 7.4% | 13.9% | 0/45 |
| V2 | 40.0% | 0.615 | 75.6% | 379.8s | 2.9 | 15.2% | 39.4% | 45.5% | 0/45 |

| Category | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Company Overview (n=6) | 0.083 / 0.0% | 0.761 / 50.0% | 0.795 / 50.0% | 0.761 / 50.0% |
| Leadership (n=13) | 0.242 / 0.0% | 0.522 / 23.1% | 0.688 / 46.2% | 0.641 / 38.5% |
| Products & Pricing (n=13) | 0.103 / 0.0% | 0.541 / 30.8% | 0.631 / 38.5% | 0.509 / 30.8% |
| HR Policy (n=5) | 0.000 / 0.0% | 0.334 / 0.0% | 0.400 / 0.0% | 0.394 / 20.0% |
| Benefits (n=8) | 0.188 / 0.0% | 0.802 / 50.0% | 0.760 / 62.5% | 0.771 / 62.5% |

| Difficulty | V0 KFR / acc | V1 KFR / acc | V1RAG KFR / acc | V2 KFR / acc |
|---|---|---|---|---|
| Easy (n=21) | 0.155 / 0.0% | 0.756 / 52.4% | 0.787 / 61.9% | 0.762 / 57.1% |
| Medium (n=13) | 0.031 / 0.0% | 0.383 / 7.7% | 0.488 / 15.4% | 0.461 / 23.1% |
| Hard (n=11) | 0.257 / 0.0% | 0.511 / 18.2% | 0.648 / 36.4% | 0.515 / 27.3% |

---

## V2P Variants — openai/gpt-oss-120b and openai/gpt-oss-20b

**Variants:** V2P (Reflexion multi-trial actor–evaluator loop) · V2P_RAG (+ RAG context on trial 0) · V2P_RGR (+ Reflexion-Guided Retrieval from trial 1 onward) · V2P_RAG_RGR (V2P_RAG + RGR)
**Me evaluator mode:** `kfr_and_no_contradiction` — loop stops when KFR ≥ 0.8 and no CONTRADICTED claims; max 3 trials.
**Results directory:** `results/openai-gpt-oss-{120b,20b}/` (separate from `results_v0-v2/`). Per-model breakdowns are also in the per-model sections above.

### Accuracy and Avg KFR

| Variant | gpt-oss-120b Acc | gpt-oss-120b KFR | gpt-oss-20b Acc | gpt-oss-20b KFR |
|---|---|---|---|---|
| **V2P** | **80.0%** | 0.890 | 66.7% | 0.807 |
| **V2P_RAG** | 73.3% | 0.839 | 53.3% | 0.658 |
| **V2P_RGR** | 71.1% | 0.829 | **68.9%** | 0.809 |
| **V2P_RAG_RGR** | 77.8% | 0.865 | 60.0% | 0.743 |

### Hallucination Rate and Avg Latency

| Variant | gpt-oss-120b Halluc % | gpt-oss-120b Latency | gpt-oss-20b Halluc % | gpt-oss-20b Latency |
|---|---|---|---|---|
| **V2P** | 6.7% | 50.2s | 6.7% | 34.0s |
| **V2P_RAG** | 2.2% | 38.9s | 4.4% | 32.6s |
| **V2P_RGR** | 4.4% | 308.0s | 11.1% | 47.5s |
| **V2P_RAG_RGR** | 4.4% | 38.2s | 13.3% | 47.8s |

*Note: V2P_RGR 120b latency is high (308s avg) due to additional KG retrieval passes on reflections; median is 40s.*

### Verdict Distribution

| Variant | Model | SUPPORTED | CONTRADICTED | UNVERIFIABLE |
|---|---|---|---|---|
| **V2P** | gpt-oss-120b | 60.8% | 1.8% | 37.3% |
| **V2P** | gpt-oss-20b | 66.4% | 2.1% | 31.5% |
| **V2P_RAG** | gpt-oss-120b | 66.2% | 0.7% | 33.1% |
| **V2P_RAG** | gpt-oss-20b | 75.2% | 3.7% | 21.1% |
| **V2P_RGR** | gpt-oss-120b | 62.1% | 2.9% | 35.0% |
| **V2P_RGR** | gpt-oss-20b | 49.6% | 4.5% | 45.9% |
| **V2P_RAG_RGR** | gpt-oss-120b | 63.3% | 1.3% | 35.3% |
| **V2P_RAG_RGR** | gpt-oss-20b | 62.7% | 7.6% | 29.7% |

### V2P Accuracy by Category

| Category | 120b V2P | 120b V2P_RAG | 120b V2P_RGR | 120b V2P_RAG_RGR | 20b V2P | 20b V2P_RAG | 20b V2P_RGR | 20b V2P_RAG_RGR |
|---|---|---|---|---|---|---|---|---|
| Company Overview (n=6) | 66.7% | 66.7% | 66.7% | 83.3% | 66.7% | 66.7% | 50.0% | 66.7% |
| Leadership (n=13) | 92.3% | 84.6% | 84.6% | 84.6% | 69.2% | 61.5% | 76.9% | 69.2% |
| Products & Pricing (n=13) | 69.2% | 53.8% | 61.5% | 61.5% | 76.9% | 53.8% | 61.5% | 46.2% |
| HR Policy (n=5) | 100.0% | 80.0% | 60.0% | 60.0% | 40.0% | 20.0% | 80.0% | 20.0% |
| Benefits (n=8) | 75.0% | 87.5% | 75.0% | 100.0% | 62.5% | 50.0% | 75.0% | 87.5% |

### V2P Accuracy by Difficulty

| Difficulty | 120b V2P | 120b V2P_RAG | 120b V2P_RGR | 120b V2P_RAG_RGR | 20b V2P | 20b V2P_RAG | 20b V2P_RGR | 20b V2P_RAG_RGR |
|---|---|---|---|---|---|---|---|---|
| Easy (n=21) | 76.2% | 66.7% | 66.7% | 76.2% | 66.7% | 57.1% | 61.9% | 57.1% |
| Medium (n=13) | 69.2% | 69.2% | 53.8% | 76.9% | 61.5% | 38.5% | 61.5% | 46.2% |
| Hard (n=11) | 100.0% | 90.9% | 100.0% | 81.8% | 72.7% | 63.6% | 90.9% | 81.8% |

### V2P vs best V0-V2 variant (V1RAG)

| Model | Best V0-V2 (V1RAG) Acc | Best V2P Acc | Best V2P Variant | Lift |
|---|---|---|---|---|
| `openai/gpt-oss-120b` | 31.1% | **80.0%** | V2P | **+157%** |
| `openai/gpt-oss-20b` | 33.3% | **68.9%** | V2P_RGR | **+107%** |

---

## Cross-Model Summary: What Works

| Finding | Evidence in this export |
|---|---|
| **V1RAG is the primary quality win** | Highest KFR/accuracy for most stable models; largest drops in hallucination vs V1 (Δ row). |
| **Scale helps but does not solve HR** | Large instruct models lead V1RAG KFR; HR row in category tables stays near zero accuracy for many. |
| **V2 is high-variance** | Several models show heavy question errors or worse KFR vs V1RAG; a few (e.g. some Mistral/DeepSeek runs) peak on V2 accuracy — check per-model tables. |
| **Treat Qwen 3.5 + coder runs as experimental** | High `err q` counts and UNVERIFIABLE-heavy verdicts under RAG for some checkpoints. |
| **MiniMax M2.7** | V0-only folder — no variant comparison yet. |
| **V2P is a major leap for gpt-oss models** | V2P (Reflexion multi-trial) takes gpt-oss-120b from 31.1% (V1RAG) to 80.0% (+157%); V2P_RGR takes gpt-oss-20b from 33.3% to 68.9% (+107%). HR policy, a persistent ceiling for V0-V2, reaches 100% under V2P on 120b. |

---

*Generated by `results_v0-v2/_build_results_common.py` · All tables read from `aggregate_metrics` in each `*_results.json`.*
