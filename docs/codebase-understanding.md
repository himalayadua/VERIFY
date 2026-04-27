# KG Claim Verifier: Codebase Understanding

## Purpose

This project evaluates how well LLM answers align with a knowledge graph (KG), then improves answers through claim extraction, claim verification, and repair.  
It supports four pipeline variants (`v0`, `v1`, `v1rag`, `v2`) and compares their behavior and quality.

## Repository Focus

- Runtime code: `src/`
- Integration harness: `test/test_integration.py`
- Primary data inputs: `data/questions.json`, `data/kg_triples.json`
- Primary outputs: `results/*_results.json`, `test/logs/*.log`

## `src/` Module Responsibilities

- `src/run_experiments.py`
  - Main experiment runner and CLI entrypoint.
  - Loads data, runs selected variants, evaluates answers, writes aggregate results.
- `src/pipeline.py`
  - Core orchestration for `V0Pipeline`, `V1Pipeline`, `V1RAGPipeline`, `V2Pipeline`.
  - Defines `PipelineResult`, the shared output contract for all variants.
- `src/llm_client.py`
  - Shared LLM gateway (NVIDIA NIM).
  - Handles retries, JSON parsing, and model selection constants.
- `src/claim_extractor.py`
  - Converts a generated answer into atomic claims.
  - Uses structured LLM extraction with sentence-based fallback.
- `src/verifier.py`
  - Verifies each claim against KG evidence.
  - Produces verdicts (`SUPPORTED`, `CONTRADICTED`, `UNVERIFIABLE`) with confidence/reasoning.
- `src/entity_linker.py`
  - Retrieves relevant KG triples for a query/claim.
  - Combines entity extraction + fuzzy matching + multihop retrieval.
- `src/repairer.py`
  - Rewrites answers when contradicted claims are found, grounded in KG evidence.
- `src/reflexion_layer.py`
  - Maintains short-term reflection memory (`ReflexionMemory`) for `v2`.
  - Generates lessons from hallucinations to improve later answers in the same run.
- `src/extract_triples.py`
  - Utility script to build/write canonical KG triples JSON.
- `src/build_relation_synonyms.py`
  - Utility script to generate relation paraphrases used by linker scoring.
- `src/prior_knowledge_probe.py`
  - Utility script to estimate model prior knowledge of KG facts.

## Core Data Contracts

- KG triple shape:
  - `{"subject": "...", "relation": "...", "object": "..."}`
- Claim shape:
  - Atomic text statements extracted from a generated answer.
- Verification output:
  - Verdict + confidence + reasoning + supporting/related evidence triples.
- Pipeline output (`PipelineResult`):
  - Includes generated answer(s), extracted claims, verification results, and variant-specific metadata (for example reflections in `v2`).

## High-Level Runtime Flow (Experiments)

```text
CLI (run_experiments.py)
  -> load questions + KG
  -> select pipeline variant(s)
  -> for each question:
       pipeline.run(question)
       -> final_answer
       -> evaluate_answer(final_answer, ground_truth, key_facts)
  -> compute aggregates
  -> write results JSON
```

## Variant Flows (Detailed Arrow Flowcharts)

### `v0` (Baseline)

```text
question
  -> call_llm
  -> raw_answer
  -> final_answer (no extraction/verification/repair)
```

### `v1` (Generate -> Verify -> Repair)

```text
question
  -> call_llm (initial answer)
  -> extract_claims(answer)
  -> verify_claims(claims, kg):
       each claim
         -> find_relevant_triples_multihop
         -> if evidence exists: LLM verdict
         -> else: UNVERIFIABLE
  -> repair_answer(original, verification_results, kg_evidence)
  -> final_answer
```

### `v1rag` (RAG + Verify + Repair)

```text
question
  -> retrieve relevant KG triples for question
  -> build KG-augmented prompt
  -> call_llm (grounded answer)
  -> extract_claims
  -> verify_claims
  -> repair_answer
  -> final_answer
```

### `v2` (V1 + Reflexion Memory)

```text
question
  -> include memory reflections in system prompt (if available)
  -> call_llm
  -> extract_claims
  -> verify_claims
  -> repair_answer
  -> if contradictions/hallucinations:
       generate_reflection
       -> memory.add_reflection
  -> final_answer (+ reflections_used metadata)
```

## Internal Retrieval + Verification Flow

```text
claim
  -> entity_linker.find_relevant_triples_multihop
     -> hop1 retrieval (entity/relation linking, fuzzy matching)
     -> hop2 expansion from discovered entities
     -> merged evidence triples
  -> verifier prompt with evidence
  -> LLM JSON verdict parse
  -> VerificationResult
```

## `test/test_integration.py` Run Flow

This file acts as an integration harness that executes full pipeline runs and records rich logs. It reports metrics but does not enforce strict `pytest`-style assertions.

```text
python test/test_integration.py [args]
  -> parse CLI args (variants, question count, all-questions)
  -> load questions (inline defaults or data/questions.json)
  -> load KG (data/kg_triples.json)
  -> init combined log file
  -> for each variant:
       init per-variant logger
       instantiate pipeline
       enter interceptors:
         patch llm_client.call_llm / call_llm_json for logging
         patch entity_linker.find_relevant_triples for logging
       for each question:
         pipeline.run(question)
           -> variant-specific flow (v0/v1/v1rag/v2)
         evaluate final_answer vs ground_truth/key_facts
         append per-question result
       exit interceptors (restore original functions)
       log per-variant summary
  -> if multiple variants: log comparison table
  -> flush/close logs
  -> print paths to generated log files
```

## What to Read for Fast Recall

- End-to-end orchestration: `src/run_experiments.py`, `src/pipeline.py`
- Claim lifecycle: `src/claim_extractor.py` -> `src/verifier.py` -> `src/repairer.py`
- KG retrieval internals: `src/entity_linker.py`
- Reflection behavior (`v2`): `src/reflexion_layer.py`
- API/model behavior and retry policy: `src/llm_client.py`
- Real-world run trace and logs: `test/test_integration.py`

## Environment and External Dependencies

- Required:
  - `NVIDIA_API_KEY` for runtime LLM calls.
- Optional but used in some paths:
  - `ANTHROPIC_API_KEY` for entity extraction/probing paths.
- Typical libraries:
  - `openai`, `anthropic`, `python-dotenv`, `thefuzz`, `python-Levenshtein`, `tabulate`, `numpy`.

## Failure/Fallback Behavior

- Claim extraction fallback:
  - If structured extraction fails, falls back to sentence splitting.
- Entity linking fallback:
  - If LLM/entity path fails or is sparse, falls back to fuzzy retrieval.
- Verification fallback:
  - Missing evidence or parsing failures yield `UNVERIFIABLE`.
- Repair short-circuit:
  - No contradicted claims means original answer is returned unchanged.

## Practical Extension Points

- Add a new pipeline variant in `src/pipeline.py` and wire it into `src/run_experiments.py`.
- Improve retrieval quality in `src/entity_linker.py` (entity resolution, relation scoring, multihop strategy).
- Tune verification robustness in `src/verifier.py` (prompt/schema confidence handling).
- Improve extraction quality in `src/claim_extractor.py` (claim granularity and filtering).
- Expand reflection policy in `src/reflexion_layer.py` (memory strategy/persistence).
