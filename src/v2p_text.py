"""
v2p_text.py
-----------
Text cleanup helpers used only by V2P / V2P-RAG paths (not V1/V1RAG/V2).
"""
from __future__ import annotations

import re


def sanitize_reflection_text(text: str) -> str:
    """
    Strip model-specific thinking wrappers and other junk from reflection output.
    """
    if not text:
        return ""
    s = str(text)
    # Common tags from reasoning models
    s = re.sub(
        r"<\s*redacted_?thinking[^>]*>[\s\S]*?<\s*/\s*redacted_?thinking\s*>",
        "",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(r"<\s*think\s*>[\s\S]*?<\s*/\s*think\s*>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```(?:json|markdown)?\s*[\s\S]*?```", "", s, flags=re.IGNORECASE)
    s = s.strip()
    return s
