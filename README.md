# KG-Backed Claim Verifier

A system that checks LLM-generated answers for factual correctness using a private-domain Knowledge Graph (KG). Since LLMs tend to hallucinate on private or specialized knowledge not present in their training data, we build a small KG from a fictional company (NovaAI) and evaluate how well different verification strategies can detect and correct these hallucinations.

## Experiment Results (45 Questions, qwen/qwen3-next-80b-a3b-instruct)

| Variant | Description | Accuracy | Avg KFR | Hallucination Rate | Avg Latency |
|---------|-------------|----------|---------|-------------------|-------------|
| **V0** | Raw LLM, no KG verification | 6.7% (3/45) | 0.349 | N/A | 4.2s |
| **V1 (original)** | Post-hoc verification: generate → extract claims → single-hop entity linking → verify → repair | 35.6% (16/45) | 0.579 | 53.3% | 48.1s |
| **V1 (multi-hop)** | Post-hoc verification with iterative two-hop entity linking in the verification stage | 37.8% (17/45) | 0.581 | 57.8% | 60.7s |
| **V1-RAG** | Pre-generation KG retrieval (iterative two-hop) injected into the LLM prompt, followed by post-hoc verification with multi-hop entity linking | 42.2% (19/45) | 0.662 | 6.7% | 19.0s |
| **V2 (original)** | V1 + Reflexion-style sliding-window memory (max 3 reflections) across questions | 31.1% (14/45) | 0.534 | 71.1% | 39.9s |
| **V2 (multi-hop)** | V2 with iterative two-hop entity linking in the verification stage | 35.6% (16/45) | 0.575 | 64.4% | 35.0s |

**Metrics:**
- **Accuracy**: percentage of questions where Key Fact Recall (KFR) ≥ 0.8
- **Avg KFR**: average fraction of ground-truth key facts present in the final answer (0–1)
- **Hallucination Rate**: percentage of questions where at least one claim was marked CONTRADICTED
- **Avg Latency**: average wall-clock time per question

### Key Findings

1. **V0 → V1**: Post-hoc KG verification improves accuracy from 6.7% to 35.6%, confirming that claim extraction + entity linking + verification + repair is effective at catching and correcting LLM hallucinations.

2. **V1 → V1 (multi-hop)**: Iterative two-hop retrieval provides a small improvement (35.6% → 37.8%), with notable gains on multi-hop questions (e.g., q37, q41, q42, q44 flipped from incorrect to correct).

3. **V1-RAG achieves the best overall performance**: 42.2% accuracy with near-zero hallucination rate (6.7%), and is also the fastest variant (19.0s avg). Injecting retrieved KG triples into the generation prompt prevents hallucinations at the source rather than fixing them after the fact.

4. **V2 Reflexion does not help**: V2 consistently underperforms V1 across all configurations. The sliding-window reflection memory introduces instability — the LLM either becomes overly cautious (refusing to answer) or the reflections from earlier questions do not generalize to later ones.

5. **Retrieval precision is the core bottleneck**: A separate experiment injecting the entire KG (262 triples, ~3K tokens) into the prompt achieved 80% accuracy on multi-hop questions with 0% hallucination rate, confirming that when retrieval is perfect, the pipeline works extremely well. The gap between V1-RAG (42.2%) and FullKG (80%) represents the improvement space for future retrieval strategies.

---

## Pipeline Overview

### V0 — Raw LLM (Baseline)

```
Question → LLM generates answer → Final Answer
```

No verification. Establishes the hallucination baseline. The LLM receives a system prompt identifying it as a NovaAI internal assistant, but has no access to any actual NovaAI data.

### V1 — Post-hoc KG Verification

```
Question → LLM generates answer (no KG context)
              ↓
         extract_claims() → atomic claims
              ↓
         For each claim:
              entity_linker.find_relevant_triples_multihop()
              → retrieve relevant KG triples (two-hop)
              ↓
              LLM verdict: SUPPORTED / CONTRADICTED / UNVERIFIABLE
              ↓
         repair_answer() → replace contradicted facts with KG values
              ↓
         Final Answer (corrected)
```

**Files involved:**
- `llm_client.py` → `call_llm()` generates the raw answer
- `claim_extractor.py` → `extract_claims()` decomposes the answer into atomic, single-fact claims
- `entity_linker.py` → `find_relevant_triples_multihop()` retrieves relevant KG triples for each claim
- `verifier.py` → `verify_claim()` asks the LLM to judge each claim against retrieved triples
- `repairer.py` → `repair_answer()` rewrites the answer, replacing contradicted claims with correct KG values

### V1-RAG — KG-Augmented Generation + Verification

```
Question → entity_linker retrieves relevant KG triples (two-hop)
              ↓
         Inject triples into system prompt
              ↓
         LLM generates answer (with KG context)
              ↓
         [Same V1 verification pipeline: extract → verify → repair]
              ↓
         Final Answer
```

The key difference from V1: KG retrieval happens **before** answer generation, not just during verification. The LLM sees real facts from the KG when generating its answer, which dramatically reduces hallucinations at the source.

### V2 — Reflexion-style Verification

```
[Same as V1, plus:]
         If hallucinations detected:
              generate_reflection() → verbal lesson about the error
              memory.add_reflection() → sliding window (max 3)
              ↓
         Next question's system prompt includes stored reflections
```

Adapted from the Reflexion paper (Shinn et al., NeurIPS 2023). A `ReflexionMemory` persists across questions in a session, storing up to 3 recent reflections that are injected into subsequent answer prompts.

---

## Iterative Two-Hop Entity Linking (Our Improvement)

### Problem

The original entity linker performs a single-pass search: it extracts entities from a claim via LLM, fuzzy-matches them against the KG entity index, and returns the top-N matching triples. This works well for single-hop facts (e.g., "NovaAI's CEO is X" → directly matches `NovaAI | has_ceo | Dr. Mara Chen`), but fails on multi-hop questions where the answer requires chaining across multiple triples.

For example, the question "Who is the PM owner of NovaAI's flagship product?" requires two reasoning steps:
1. `NovaAI → is_flagship_product_of → NovaPilot` (find the flagship product)
2. `NovaPilot → pm_owner → Fatima Al-Rashid` (find its PM)

A single-pass search on the claim text will likely find triples related to "NovaAI" but miss the `NovaPilot | pm_owner | Fatima Al-Rashid` triple entirely, because "NovaPilot" does not appear in the original claim.

### Solution

We implemented iterative two-hop retrieval in `entity_linker.py`:

```python
def find_relevant_triples_multihop(claim, triples, top_n=5, method="llm", max_hops=2):
    # Hop 1: standard retrieval on the claim text
    hop1 = find_relevant_triples(claim, triples, top_n=top_n, method=method)

    # Hop 2: extract new entities discovered in hop1, search again
    seen = {(t["subject"], t["relation"], t["object"]) for t in hop1}
    hop2 = []
    new_entities = set()
    for t in hop1:
        new_entities.add(t["subject"])
        new_entities.add(t["object"])
    for entity in new_entities:
        for t in fuzzy_search(entity, triples, top_n=3):
            key = (t["subject"], t["relation"], t["object"])
            if key not in seen:
                seen.add(key)
                hop2.append(t)

    # Merge both hops, deduplicated
    return hop1 + hop2
```

**Hop 1** retrieves triples directly relevant to the claim text. **Hop 2** takes all entities (subjects and objects) discovered in Hop 1 and uses them as new search queries, finding additional triples one step deeper in the graph. The two result sets are merged with deduplication and passed to the verification LLM together.

This allows the verifier to see evidence that spans two edges in the KG, enabling it to detect contradictions in multi-hop claims that single-pass retrieval would miss.

### Fallback Chain

The entity linking module uses a three-level fallback strategy:
1. **LLM path** (primary): Extract named entities from the claim via LLM, fuzzy-match against KG entity index
2. **Fuzzy path** (fallback): Score each triple against the raw claim text using character-level similarity
3. If no matches are found at any level, the claim is marked as UNVERIFIABLE

---

## V1-RAG: Pre-Generation KG Retrieval (Our Improvement)

### Problem

In V1, the LLM generates its answer with zero knowledge of the private domain, then the verifier attempts to fix errors after the fact. But when the LLM fabricates entity names that do not exist in the KG at all (e.g., inventing "NovaCore" instead of the real product "NovaPilot"), entity linking cannot find any matching triples, and the error goes undetected.

### Solution

V1-RAG moves KG retrieval **before** answer generation. The pipeline first retrieves relevant triples for the **question** (not the answer) using iterative two-hop retrieval, then injects them into the LLM's system prompt:

```
System: You are a helpful internal assistant for NovaAI employees...

Below are verified facts from the internal knowledge base.
Base your answer strictly on these facts.
If the facts do not contain enough information, say so — do NOT guess.

- NovaPilot | is flagship product of | NovaAI
- NovaPilot | pm owner | Fatima Al-Rashid
- NovaPilot | launched in | Q3 2022
- NovaAI | vp of product | Oliver Haines
...
```

With real KG facts in the prompt, the LLM generates answers grounded in actual data instead of hallucinating. The answer then still goes through the full V1 verification pipeline (extract → verify → repair) as a second safety net.

### Why This Works

- **Prevents hallucinations at the source**: The LLM sees correct facts before answering, so it does not need to guess
- **Reduces verification load**: Fewer contradicted claims means fewer repair calls, which explains the faster latency (19s vs 48s)
- **Honest about gaps**: When retrieval does not find relevant triples, the LLM says "I don't have this information" instead of fabricating — this is preferable to hallucination even though it scores 0 on KFR

### Limitation

V1-RAG's performance is bounded by retrieval recall. If the relevant triples are not retrieved in the two-hop search, the LLM cannot answer correctly. Our FullKG experiment (injecting all 262 triples) achieved 80% accuracy on multi-hop questions, confirming that retrieval precision is the primary bottleneck. Improving retrieval strategies (e.g., embedding-based semantic search, graph traversal algorithms) is the most promising direction for future work.

---

## Knowledge Graph

- **Domain**: NovaAI, a fictional AI company
- **Source document**: `data/company_doc.md` — internal wiki covering company overview, 6 departments, 4 products with pricing, 20 employees, and 9 HR policies
- **KG statistics**: 262 triples, 53 unique subjects, 146 unique relations
- **Storage**: `data/kg_triples.json` — flat JSON array of `{subject, relation, object}` dicts

## Evaluation Questions

- **Total**: 45 questions in `data/questions.json`
- **By difficulty**: 21 easy, 13 medium, 11 hard
- **By category**: leadership (13), products_pricing (13), benefits (8), company_overview (6), hr_policy (5)
- **By hops**: q01–q35 are single-hop (hops=1), q36–q45 are multi-hop (hops=2–3)
- **Each question includes**: `id`, `category`, `difficulty`, `question`, `ground_truth_answer`, `key_facts`, and optionally `hops` and `hop_chain`

## Project Structure

```
kg-claim-verifier/
├── data/
│   ├── company_doc.md          # Source document (NovaAI internal wiki)
│   ├── kg_triples.json         # 262 KG triples
│   └── questions.json          # 45 evaluation questions
├── src/
│   ├── llm_client.py           # NVIDIA NIM API gateway
│   ├── extract_triples.py      # KG construction script
│   ├── claim_extractor.py      # Answer → atomic claims decomposition
│   ├── entity_linker.py        # KG triple retrieval (LLM + fuzzy + multi-hop)
│   ├── verifier.py             # Claim verification against KG
│   ├── repairer.py             # Answer correction using KG evidence
│   ├── reflexion_layer.py      # V2 Reflexion memory
│   ├── pipeline.py             # V0, V1, V1-RAG, V2 pipeline orchestration
│   └── run_experiments.py      # Experiment runner with evaluation metrics
├── test/
│   ├── test_integration.py     # End-to-end integration test with detailed logging
│   └── logs/                   # Timestamped test logs
├── results/                    # JSON result files from run_experiments.py
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
echo "NVIDIA_API_KEY=nvapi-your-key-here" > .env
```

Get a free NVIDIA NIM API key at [build.nvidia.com](https://build.nvidia.com).

## Usage

```bash
# Run all variants on all 45 questions (with detailed logs)
cd test
python test_integration.py --variants v0 v1 v1rag v2 --all-questions

# Run specific variants on multi-hop questions only
cd src
python run_experiments.py --variants v1 v1rag --questions ../data/questions_multihop.json

# Quick smoke test (1 question)
cd test
python test_integration.py --variants v1 --questions 1
```