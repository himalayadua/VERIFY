"""
model_registry.py
-----------------
Central registry for all NVIDIA NIM models used in experiments.

Each entry defines:
  api_key_env  - env-var name holding this model's NVIDIA API key
  temperature  - recommended sampling temperature
  top_p        - nucleus sampling threshold
  max_tokens   - hard cap on output tokens (models vary significantly)
  extra_body   - optional dict merged into the OpenAI API request body
                 (used for chat_template_kwargs, top_k, thinking toggles, etc.)

Thinking/reasoning modes are disabled by default (enable_thinking: False where
applicable) so pipeline tasks receive clean text output suitable for JSON
parsing and claim verification.

Usage:
    from model_registry import get_model_config, get_api_key_for_model, model_to_slug, ALL_MODELS
"""
from __future__ import annotations

import os
import re

# ── Per-model configuration ─────────────────────────────────────────────────────
#
# env-var naming convention: NVIDIA_API_KEY_<UPPER_SLUG>
# where slug = model name with / → _ and . → _ (all non-alphanum except - → _).

MODEL_REGISTRY: dict[str, dict] = {

    # ── Google Gemma 4 31B ─────────────────────────────────────────────────────
    "google/gemma-4-31b-it": {
        "api_key_env": "NVIDIA_API_KEY_GOOGLE_GEMMA_4_31B_IT",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 2048,
        # Disable thinking so content field stays clean
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },

    # ── Qwen 3.5 122B ─────────────────────────────────────────────────────────
    "qwen/qwen3.5-122b-a10b": {
        "api_key_env": "NVIDIA_API_KEY_QWEN_QWEN3_5_122B_A10B",
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 2048,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },

    # ── MiniMax M2.5 ──────────────────────────────────────────────────────────
    "minimaxai/minimax-m2.5": {
        "api_key_env": "NVIDIA_API_KEY_MINIMAXAI_MINIMAX_M2_5",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 2048,
    },

    # ── Qwen 3.5 397B ─────────────────────────────────────────────────────────
    # top_k goes as a top-level body field via extra_body
    "qwen/qwen3.5-397b-a17b": {
        "api_key_env": "NVIDIA_API_KEY_QWEN_QWEN3_5_397B_A17B",
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 2048,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
            "top_k": 20,
        },
    },

    # ── DeepSeek V3.2 ─────────────────────────────────────────────────────────
    # Thinking is off by default when extra_body thinking key is absent
    "deepseek-ai/deepseek-v3.2": {
        "api_key_env": "NVIDIA_API_KEY_DEEPSEEK_AI_DEEPSEEK_V3_2",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 2048,
    },

    # ── Kimi K2 Thinking ──────────────────────────────────────────────────────
    # reasoning_content is in a separate delta field; content stays clean
    "moonshotai/kimi-k2-thinking": {
        "api_key_env": "NVIDIA_API_KEY_MOONSHOTAI_KIMI_K2_THINKING",
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 2048,
    },

    # ── Mistral Large 3 675B ──────────────────────────────────────────────────
    "mistralai/mistral-large-3-675b-instruct-2512": {
        "api_key_env": "NVIDIA_API_KEY_MISTRALAI_MISTRAL_LARGE_3",
        "temperature": 0.15,
        "top_p": 1.0,
        "max_tokens": 2048,
    },

    # ── DeepSeek V3.1 Terminus ────────────────────────────────────────────────
    "deepseek-ai/deepseek-v3.1-terminus": {
        "api_key_env": "NVIDIA_API_KEY_DEEPSEEK_AI_DEEPSEEK_V3_1_TERMINUS",
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 2048,
    },

    # ── Kimi K2 Instruct ──────────────────────────────────────────────────────
    "moonshotai/kimi-k2-instruct-0905": {
        "api_key_env": "NVIDIA_API_KEY_MOONSHOTAI_KIMI_K2_INSTRUCT_0905",
        "temperature": 0.6,
        "top_p": 0.9,
        "max_tokens": 2048,
    },

    # ── Qwen3 Next 80B Thinking ───────────────────────────────────────────────
    "qwen/qwen3-next-80b-a3b-thinking": {
        "api_key_env": "NVIDIA_API_KEY_QWEN_QWEN3_NEXT_80B_THINKING",
        "temperature": 0.6,
        "top_p": 0.7,
        "max_tokens": 2048,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },

    # ── ByteDance Seed OSS 36B ────────────────────────────────────────────────
    # Thinking is off by default when thinking_budget is absent
    "bytedance/seed-oss-36b-instruct": {
        "api_key_env": "NVIDIA_API_KEY_BYTEDANCE_SEED_OSS_36B",
        "temperature": 1.1,
        "top_p": 0.95,
        "max_tokens": 2048,
    },

    # ── Qwen3 Coder 480B ──────────────────────────────────────────────────────
    "qwen/qwen3-coder-480b-a35b-instruct": {
        "api_key_env": "NVIDIA_API_KEY_QWEN_QWEN3_CODER_480B",
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 2048,
    },

    # ── OpenAI GPT OSS 120B ───────────────────────────────────────────────────
    "openai/gpt-oss-120b": {
        "api_key_env": "NVIDIA_API_KEY_OPENAI_GPT_OSS_120B",
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 2048,
    },

    # ── OpenAI GPT OSS 20B ────────────────────────────────────────────────────
    "openai/gpt-oss-20b": {
        "api_key_env": "NVIDIA_API_KEY_OPENAI_GPT_OSS_20B",
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 2048,
    },

    # ── Google Gemma 3N E4B ───────────────────────────────────────────────────
    # Small model — strict 512-token cap
    "google/gemma-3n-e4b-it": {
        "api_key_env": "NVIDIA_API_KEY_GOOGLE_GEMMA_3N_E4B_IT",
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 512,
    },

    # ── Google Gemma 3N E2B ───────────────────────────────────────────────────
    # Smallest model — strict 512-token cap
    "google/gemma-3n-e2b-it": {
        "api_key_env": "NVIDIA_API_KEY_GOOGLE_GEMMA_3N_E2B_IT",
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 512,
    },

    # ── Z-AI GLM 5.1 ─────────────────────────────────────────────────────────
    # Explicitly disable thinking + clear any reasoning tokens
    "z-ai/glm-5.1": {
        "api_key_env": "NVIDIA_API_KEY_ZAI_GLM_5_1",
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False, "clear_thinking": True},
        },
    },

    # ── MiniMax M2.7 ──────────────────────────────────────────────────────────
    "minimaxai/minimax-m2.7": {
        "api_key_env": "NVIDIA_API_KEY_MINIMAXAI_MINIMAX_M2_7",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 2048,
    },
}

# ── Fallback config for any model not in the registry ──────────────────────────
_DEFAULT_CONFIG: dict = {
    "api_key_env": "NVIDIA_API_KEY",
    "temperature": 0.6,
    "top_p": 0.7,
    "max_tokens": 2048,
}

# ── Ordered list of all registered models ──────────────────────────────────────
ALL_MODELS: list[str] = list(MODEL_REGISTRY.keys())


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_model_config(model: str) -> dict:
    """Return the config dict for *model*, falling back to _DEFAULT_CONFIG."""
    return MODEL_REGISTRY.get(model, _DEFAULT_CONFIG)


def get_api_key_for_model(model: str) -> str | None:
    """
    Return the NVIDIA API key for *model*.

    Lookup order:
      1. The model-specific env var (e.g. NVIDIA_API_KEY_GOOGLE_GEMMA_4_31B_IT)
      2. The shared fallback NVIDIA_API_KEY
    Returns None if neither is set.
    """
    cfg = get_model_config(model)
    env_var = cfg.get("api_key_env", "NVIDIA_API_KEY")
    key = os.environ.get(env_var)
    if not key:
        key = os.environ.get("NVIDIA_API_KEY")
    return key


def model_to_slug(model: str) -> str:
    """
    Convert a model name to a filesystem-safe, human-readable slug.

    Examples:
        "google/gemma-4-31b-it"          -> "google-gemma-4-31b-it"
        "qwen/qwen3.5-122b-a10b"         -> "qwen-qwen3_5-122b-a10b"
        "mistralai/mistral-large-3-..."  -> "mistralai-mistral-large-3-..."
    """
    slug = model.replace("/", "-").replace(".", "_")
    slug = re.sub(r"[^a-zA-Z0-9\-_]", "_", slug)
    return slug
