"""
llm_client.py
-------------
Single gateway for all NVIDIA NIM LLM API calls.
All other modules import from here — never construct their own OpenAI client.

Per-model behaviour (API keys, extra_body kwargs, max_tokens caps) is driven
by model_registry.py.  Callers that do not pass an explicit model receive the
current value of ACTION_MODEL / EXTRACT_MODEL at call time (sentinel pattern),
so patching those module-level constants before a test run automatically
propagates to every downstream call without requiring any other code changes.
"""
from __future__ import annotations

import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the project root (one level above src/)
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, "..", ".env"))

_BASE_URL = "https://integrate.api.nvidia.com/v1"

# ── Model constants ────────────────────────────────────────────────────────────
# These are the defaults used when no model is specified at call time.
# Patch these module-level names (e.g. llm_client.ACTION_MODEL = "...")
# before running a test to switch all downstream calls to a different model.
ACTION_MODEL  = "qwen/qwen3-next-80b-a3b-instruct"   # main generation tasks
REFLECT_MODEL = "qwen/qwen3-next-80b-a3b-instruct"   # reflection generation
EXTRACT_MODEL = "qwen/qwen3-next-80b-a3b-instruct"   # structured JSON extraction

# Sentinel: distinguishes "caller passed None" from "caller did not pass model"
_UNSET = object()


# ── Client factory ─────────────────────────────────────────────────────────────

def get_nvidia_client(model: str | None = None) -> OpenAI:
    """
    Return an OpenAI-compatible client pointed at NVIDIA NIM.

    If *model* is provided, the model-specific API key is looked up first
    (from model_registry); falls back to the shared NVIDIA_API_KEY.
    """
    if model:
        from model_registry import get_api_key_for_model
        api_key = get_api_key_for_model(model)
    else:
        api_key = os.environ.get("NVIDIA_API_KEY")

    if not api_key:
        raise RuntimeError(
            "No NVIDIA API key found. Set NVIDIA_API_KEY (or a model-specific key) in .env."
        )
    return OpenAI(base_url=_BASE_URL, api_key=api_key)


# ── Core call functions ────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    model=_UNSET,
    system_prompt: str | None = None,
    temperature: float = 0.6,
    top_p: float = 0.7,
    max_tokens: int = 2048,
    extra_body: dict | None = None,
    retries: int = 3,
) -> str:
    """
    Call the NVIDIA NIM chat completions API.

    Parameters
    ----------
    prompt        : user message content
    model         : NIM model identifier (defaults to ACTION_MODEL if omitted)
    system_prompt : optional system message
    temperature   : sampling temperature
    top_p         : nucleus sampling threshold
    max_tokens    : maximum output tokens (capped by model registry if lower)
    extra_body    : extra fields merged into the request body (e.g. chat_template_kwargs)
                    merged with any model-registry extra_body (caller wins on conflicts)
    retries       : number of attempts before raising RuntimeError

    Returns
    -------
    Stripped response text from the first choice.
    """
    # Resolve model from sentinel
    if model is _UNSET:
        model = ACTION_MODEL

    # Apply per-model registry settings
    from model_registry import get_model_config
    cfg = get_model_config(model)

    # Respect the registry's hard max_tokens cap (small models have low limits)
    effective_max_tokens = min(max_tokens, cfg.get("max_tokens", max_tokens))

    # Merge extra_body: registry base → caller override
    merged_extra: dict | None = None
    registry_extra = cfg.get("extra_body")
    if registry_extra or extra_body:
        merged_extra = {}
        if registry_extra:
            merged_extra.update(registry_extra)
        if extra_body:
            merged_extra.update(extra_body)

    client = get_nvidia_client(model)
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(retries):
        try:
            kwargs: dict = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=effective_max_tokens,
                stream=False,
            )
            if merged_extra:
                kwargs["extra_body"] = merged_extra

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            return (content or "").strip()
        except Exception as exc:
            wait = 2 ** attempt
            print(
                f"  [llm_client] call_llm attempt {attempt + 1}/{retries} failed "
                f"({type(exc).__name__}: {exc}). "
                f"{'Retrying in ' + str(wait) + 's...' if attempt < retries - 1 else 'Giving up.'}"
            )
            if attempt < retries - 1:
                time.sleep(wait)

    raise RuntimeError(f"call_llm failed after {retries} attempts for model '{model}'")


def call_llm_json(
    prompt: str,
    model=_UNSET,
    system_prompt: str | None = None,
    temperature: float = 0.2,
    top_p: float = 0.7,
    max_tokens: int = 1024,
    extra_body: dict | None = None,
    retries: int = 3,
) -> dict:
    """
    Like call_llm but strips markdown code fences and parses the response as JSON.

    Parameters
    ----------
    model : NIM model identifier (defaults to EXTRACT_MODEL if omitted)

    Returns
    -------
    Parsed dict. Raises ValueError if JSON parsing fails after all retries.
    """
    # Resolve model from sentinel
    if model is _UNSET:
        model = EXTRACT_MODEL

    last_error: Exception | None = None

    for attempt in range(retries):
        try:
            raw = call_llm(
                prompt,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body,
                retries=1,
            )
            # Strip markdown code fences: ```json ... ``` or ``` ... ```
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
            return json.loads(cleaned)
        except (json.JSONDecodeError, RuntimeError, ValueError) as exc:
            last_error = exc
            wait = 2 ** attempt
            print(
                f"  [llm_client] call_llm_json attempt {attempt + 1}/{retries} failed "
                f"({type(exc).__name__}: {exc}). "
                f"{'Retrying in ' + str(wait) + 's...' if attempt < retries - 1 else 'Giving up.'}"
            )
            if attempt < retries - 1:
                time.sleep(wait)

    raise ValueError(
        f"call_llm_json failed to produce valid JSON after {retries} attempts. "
        f"Last error: {last_error}"
    )


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[smoke test] Testing call_llm...")
    result = call_llm("What is 2 + 2? Answer in one word.")
    print(f"  Response: {result}")

    print("\n[smoke test] Testing call_llm_json...")
    result_json = call_llm_json(
        'Return this exact JSON: {"status": "ok", "value": 42}',
        system_prompt="Return only valid JSON, no explanation.",
    )
    print(f"  Parsed JSON: {result_json}")
