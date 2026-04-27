"""
v2p_rag_pipeline.py
-------------------
V2P-RAG: same multi-trial Reflexion loop as V2PaperPipeline, but the base answer
system prompt is augmented with V1RAG-style KG retrieval (duplicated here so
V1RAGPipeline in pipeline.py is left unchanged).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from entity_linker import _find_relevant_fuzzy, find_relevant_triples
from pipeline import ANSWER_SYSTEM_PROMPT_BASE
from v2_paper_pipeline import V2PaperPipeline


def build_v2p_rag_answer_base(question: str, triples: list[dict]) -> str:
    """
    RAG system prompt: mirrors V1RAGPipeline.run (steps 1–2) without importing
    V1RAGPipeline, so v1rag behavior in pipeline.py is untouched.
    """
    hop1 = find_relevant_triples(question, triples, top_n=10, method="llm")

    seen = {(t["subject"], t["relation"], t["object"]) for t in hop1}
    hop2 = []
    new_entities = set()
    for t in hop1:
        new_entities.add(t["subject"])
        new_entities.add(t["object"])
    for entity in new_entities:
        for t in _find_relevant_fuzzy(entity, triples, top_n=3, min_score=55):
            key = (t["subject"], t["relation"], t["object"])
            if key not in seen:
                seen.add(key)
                hop2.append(t)

    relevant = hop1 + hop2

    if relevant:
        kg_lines = "\n".join(
            f"- {t['subject']} | {t['relation']} | {t['object']}" for t in relevant
        )
        return (
            ANSWER_SYSTEM_PROMPT_BASE
            + "\n\nBelow are verified facts from the internal knowledge base. "
            "Base your answer strictly on these facts. "
            "If the facts do not contain enough information, say so — do NOT guess.\n\n"
            + kg_lines
        )
    return ANSWER_SYSTEM_PROMPT_BASE


class V2pRagPipeline(V2PaperPipeline):
    """V2P with RAG-injected answer prompt (V2P rules still apply in verify/repair/reflection)."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("result_variant", "v2p_rag")
        super().__init__(*args, **kwargs)

    def _resolve_answer_base(self, question: str) -> str:
        return build_v2p_rag_answer_base(question, self.triples)


__all__ = ["V2pRagPipeline", "build_v2p_rag_answer_base"]
