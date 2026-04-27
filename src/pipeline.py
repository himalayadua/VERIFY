"""
pipeline.py
-----------
Orchestrates all three experimental variants:

  V0Pipeline — Raw LLM, no verification (hallucination baseline)
  V1Pipeline — Static KG Verifier: generate → extract → verify → repair
  V2Pipeline — Reflexion-style: V1 + session memory that improves over time

All three share the same PipelineResult dataclass for uniform downstream handling.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, field

from entity_linker import load_kg
from claim_extractor import Claim, extract_claims
from verifier import VerificationResult, verify_claims
from repairer import RepairResult, repair_answer
from reflexion_layer import ReflexionMemory, generate_reflection
from llm_client import call_llm, ACTION_MODEL


# ── System prompts ─────────────────────────────────────────────────────────────

ANSWER_SYSTEM_PROMPT_BASE = (
    "You are a helpful internal assistant for NovaAI employees.\n"
    "Answer questions accurately about NovaAI's products, people, and policies.\n"
    "Be specific — include exact names, numbers, and dates when relevant.\n"
    "If you do not know the answer, say so clearly rather than guessing."
)

# V2 template: {base_prompt}\n\n{reflection_memory_block}
# The reflection_memory_block slot is filled by ReflexionMemory.format_for_prompt()
_V2_PROMPT_TEMPLATE = "{base_prompt}\n\n{reflection_memory_block}"


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Uniform result container for all three pipeline variants.
    Fields absent from V0 are set to None.
    """
    question: str
    variant: str                               # "v0", "v1", or "v2"
    raw_answer: str
    final_answer: str
    claims: list[str] | None = None            # None for V0
    verification_results: list[dict] | None = None  # serialised VerificationResults
    had_hallucinations: bool | None = None     # None for V0
    reflections_used: list[str] | None = None  # None for V0 and V1
    latency_seconds: float = 0.0
    trials: list[dict] | None = None           # V2P multi-trial trace (optional)
    stopped_reason: str | None = None           # V2P stop reason (optional)

    def to_dict(self) -> dict:
        out = {
            "question": self.question,
            "variant": self.variant,
            "raw_answer": self.raw_answer,
            "final_answer": self.final_answer,
            "claims": self.claims,
            "verification_results": self.verification_results,
            "had_hallucinations": self.had_hallucinations,
            "reflections_used": self.reflections_used,
            "latency_seconds": round(self.latency_seconds, 3),
        }
        if self.trials is not None:
            out["trials"] = self.trials
        if self.stopped_reason is not None:
            out["stopped_reason"] = self.stopped_reason
        return out


def _serialise_vr(vr: VerificationResult) -> dict:
    """Convert a VerificationResult to a JSON-serialisable dict."""
    return {
        "claim": vr.claim,
        "verdict": vr.verdict.value,
        "confidence": round(vr.confidence, 3),
        "evidence_triples": vr.evidence_triples,
        "reasoning": vr.reasoning,
    }


# ── V0 — Raw LLM ──────────────────────────────────────────────────────────────

class V0Pipeline:
    """
    Raw LLM — no claim extraction, no KG verification, no repair.
    Establishes the hallucination baseline.
    """

    def __init__(self, model: str = ACTION_MODEL):
        self.model = model

    def run(self, question: str) -> PipelineResult:
        start = time.time()
        answer = call_llm(
            question,
            model=self.model,
            system_prompt=ANSWER_SYSTEM_PROMPT_BASE,
        )
        elapsed = time.time() - start

        return PipelineResult(
            question=question,
            variant="v0",
            raw_answer=answer,
            final_answer=answer,
            claims=None,
            verification_results=None,
            had_hallucinations=None,
            reflections_used=None,
            latency_seconds=elapsed,
        )

# ── V1-RAG — KG-Augmented Generation + Verification ───────────────────

class V1RAGPipeline:
    """
    KG-Augmented Generation + Verification:
      1. Retrieve relevant KG triples for the QUESTION using iterative retrieval
      2. Inject triples into the system prompt so the LLM answers with real facts
      3. Generate answer with KG context
      4-6. Extract claims, verify, and repair (same as V1)
    """

    def __init__(self, triples: list[dict], model: str = ACTION_MODEL):
        self.triples = triples
        self.model = model

    def run(self, question: str) -> PipelineResult:
        from entity_linker import find_relevant_triples, _find_relevant_fuzzy
        start = time.time()

        # Step 1: Iterative retrieval on the question
        hop1 = find_relevant_triples(question, self.triples, top_n=10, method="llm")

        seen = {(t["subject"], t["relation"], t["object"]) for t in hop1}
        hop2 = []
        new_entities = set()
        for t in hop1:
            new_entities.add(t["subject"])
            new_entities.add(t["object"])
        for entity in new_entities:
            for t in _find_relevant_fuzzy(entity, self.triples, top_n=3, min_score=55):
                key = (t["subject"], t["relation"], t["object"])
                if key not in seen:
                    seen.add(key)
                    hop2.append(t)

        relevant = hop1 + hop2

        # Step 2: Build augmented system prompt with KG context
        if relevant:
            kg_lines = "\n".join(
                f"- {t['subject']} | {t['relation']} | {t['object']}"
                for t in relevant
            )
            system_prompt = (
                ANSWER_SYSTEM_PROMPT_BASE
                + "\n\nBelow are verified facts from the internal knowledge base. "
                "Base your answer strictly on these facts. "
                "If the facts do not contain enough information, say so — do NOT guess.\n\n"
                + kg_lines
            )
        else:
            system_prompt = ANSWER_SYSTEM_PROMPT_BASE

        # Step 3: Generate answer (now with KG context)
        raw_answer = call_llm(question, model=self.model, system_prompt=system_prompt)

        # Step 4: Extract atomic claims
        claims: list[Claim] = extract_claims(raw_answer, question)

        # Step 5: Verify claims (uses multihop retrieval via verifier.py)
        vr_list: list[VerificationResult] = verify_claims(claims, self.triples)

        # Step 6: Repair if any contradictions found
        repair: RepairResult = repair_answer(question, raw_answer, vr_list)

        elapsed = time.time() - start

        return PipelineResult(
            question=question,
            variant="v1rag",
            raw_answer=raw_answer,
            final_answer=repair.repaired_answer,
            claims=[c.text for c in claims],
            verification_results=[_serialise_vr(r) for r in vr_list],
            had_hallucinations=repair.had_hallucinations,
            reflections_used=None,
            latency_seconds=elapsed,
        )

# ── V1 — Static KG Verifier ───────────────────────────────────────────────────

class V1Pipeline:
    """
    Static KG Verifier:
      1. Generate raw answer (no KG context)
      2. Extract atomic claims
      3. Verify each claim against the KG
      4. Repair contradicted claims
    No memory between questions.
    """

    def __init__(self, triples: list[dict], model: str = ACTION_MODEL):
        self.triples = triples
        self.model = model

    def run(self, question: str) -> PipelineResult:
        start = time.time()

        # Step 1: Generate raw answer
        raw_answer = call_llm(
            question,
            model=self.model,
            system_prompt=ANSWER_SYSTEM_PROMPT_BASE,
        )

        # Step 2: Extract atomic claims
        claims: list[Claim] = extract_claims(raw_answer, question)

        # Step 3: Verify each claim against the KG
        vr_list: list[VerificationResult] = verify_claims(claims, self.triples)

        # Step 4: Repair if any contradictions found
        repair: RepairResult = repair_answer(question, raw_answer, vr_list)

        elapsed = time.time() - start

        return PipelineResult(
            question=question,
            variant="v1",
            raw_answer=raw_answer,
            final_answer=repair.repaired_answer,
            claims=[c.text for c in claims],
            verification_results=[_serialise_vr(r) for r in vr_list],
            had_hallucinations=repair.had_hallucinations,
            reflections_used=None,
            latency_seconds=elapsed,
        )


# ── V2 — Reflexion-style ──────────────────────────────────────────────────────

class V2Pipeline:
    """
    Reflexion-style verifier (cross-question memory variant):
      - Identical to V1 PLUS a ReflexionMemory that persists ACROSS questions
        in a session (sliding window of 3 reflections).
      - After each contradiction, generates a verbal reflection and stores it.
      - Subsequent questions include prior reflections in the answer prompt.

    Design note: V2 uses one trial per question with cross-question memory
    accumulation. It does NOT retry the same question multiple times. The
    multi-trial-per-question loop from the Reflexion paper is implemented in
    V2PaperPipeline (v2_paper_pipeline.py). This intentional design tests
    whether lessons from earlier questions improve accuracy on later ones.
    """

    def __init__(
        self,
        triples: list[dict],
        model: str = ACTION_MODEL,
        memory: ReflexionMemory | None = None,
    ):
        self.triples = triples
        self.model = model
        self.memory = memory if memory is not None else ReflexionMemory(max_reflections=3)

    def reset_memory(self) -> None:
        """Reset session memory. Call between experiments, NOT between questions."""
        self.memory.clear()

    def run(self, question: str) -> PipelineResult:
        start = time.time()

        # Snapshot the memory that will actually be injected into the prompt.
        # This must be captured BEFORE any new reflection is generated so that
        # PipelineResult.reflections_used accurately reports "what was used as
        # input to produce this answer" rather than the post-run memory state.
        injected_reflections: list[str] = self.memory.reflections.copy()

        # Step 1: Build system prompt — inject memory if available
        if self.memory.has_reflections():
            system_prompt = _V2_PROMPT_TEMPLATE.format(
                base_prompt=ANSWER_SYSTEM_PROMPT_BASE,
                reflection_memory_block=self.memory.format_for_prompt(),
            )
        else:
            system_prompt = ANSWER_SYSTEM_PROMPT_BASE

        # Step 2: Generate raw answer (with memory context if present)
        raw_answer = call_llm(
            question,
            model=self.model,
            system_prompt=system_prompt,
        )

        # Steps 3-5: Extract → verify → repair (identical to V1)
        claims: list[Claim] = extract_claims(raw_answer, question)
        vr_list: list[VerificationResult] = verify_claims(claims, self.triples)
        repair: RepairResult = repair_answer(question, raw_answer, vr_list)

        # Step 6 (V2 only): Generate reflection and update memory
        if repair.had_hallucinations:
            reflection = generate_reflection(
                question=question,
                original_answer=raw_answer,
                verification_results=vr_list,
                repaired_answer=repair.repaired_answer,
                existing_reflections=injected_reflections,
            )
            if reflection:
                self.memory.add_reflection(reflection)
                print(f"    [V2] New reflection stored. Memory size: {len(self.memory.reflections)}")

        elapsed = time.time() - start

        return PipelineResult(
            question=question,
            variant="v2",
            raw_answer=raw_answer,
            final_answer=repair.repaired_answer,
            claims=[c.text for c in claims],
            verification_results=[_serialise_vr(r) for r in vr_list],
            had_hallucinations=repair.had_hallucinations,
            reflections_used=injected_reflections,   # what was injected into the prompt
            latency_seconds=elapsed,
        )


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    kg = load_kg()
    print(f"[smoke test] Loaded {len(kg)} KG triples\n")

    question = "Who is the CEO of NovaAI and when was she appointed?"

    print("=" * 60)
    print("V0 — Raw LLM")
    print("=" * 60)
    v0 = V0Pipeline()
    r0 = v0.run(question)
    print(f"Answer: {r0.final_answer}")
    print(f"Latency: {r0.latency_seconds:.2f}s\n")

    print("=" * 60)
    print("V1 — Static KG Verifier")
    print("=" * 60)
    v1 = V1Pipeline(kg)
    r1 = v1.run(question)
    print(f"Raw   : {r1.raw_answer}")
    print(f"Final : {r1.final_answer}")
    print(f"Claims: {r1.claims}")
    print(f"Had hallucinations: {r1.had_hallucinations}")
    print(f"Latency: {r1.latency_seconds:.2f}s\n")

    print("=" * 60)
    print("V2 — Reflexion-style")
    print("=" * 60)
    v2 = V2Pipeline(kg)
    r2 = v2.run(question)
    print(f"Final : {r2.final_answer}")
    print(f"Reflections stored: {r2.reflections_used}")
    print(f"Latency: {r2.latency_seconds:.2f}s")
