"""
check_models.py
---------------
Quick connectivity and response check for all 18 registered NVIDIA NIM models.

Sends a minimal "say OK" prompt to each model via the same llm_client.call_llm()
path used in production, catching the exact errors that surface during experiments.

Usage (from project root):
    python test/check_models.py                   # all 18 models, 16 parallel workers
    python test/check_models.py --models google/gemma-4-31b-it qwen/qwen3.5-122b-a10b
    python test/check_models.py --timeout 90      # seconds per attempt (default 75)
    python test/check_models.py --retries 1       # overrides llm_client retries (default 1)
    python test/check_models.py --workers 4       # concurrent threads (default 16)
    python test/check_models.py --sequential      # run one at a time (for debugging)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Optional

# ── Add src/ to path so we can import llm_client, model_registry ──────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import llm_client                          # noqa: E402  (path set above)
from model_registry import ALL_MODELS, get_model_config, get_api_key_for_model  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
CHECK_PROMPT = (
    "Reply with exactly one word: OK"
)
CHECK_SYSTEM = "You are a helpful assistant. Follow instructions precisely."

# ANSI colours (disabled if not a tty)
_TTY = sys.stdout.isatty()
GREEN  = "\033[32m" if _TTY else ""
RED    = "\033[31m" if _TTY else ""
YELLOW = "\033[33m" if _TTY else ""
CYAN   = "\033[36m" if _TTY else ""
DIM    = "\033[2m"  if _TTY else ""
RESET  = "\033[0m"  if _TTY else ""
BOLD   = "\033[1m"  if _TTY else ""


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class ModelCheckResult:
    model: str
    status: str          # "PASS" | "FAIL" | "TIMEOUT" | "NO_KEY"
    latency_s: float = 0.0
    response_preview: str = ""
    error_type: str = ""
    error_msg: str = ""
    api_key_env: str = ""
    has_key: bool = True


# ── Per-model check ───────────────────────────────────────────────────────────
def _check_one(model: str, call_timeout: float, retries: int) -> ModelCheckResult:
    """
    Call llm_client.call_llm() for *model* with a tiny prompt.

    call_timeout  : Python-level wall-clock timeout wrapping the call (in seconds).
                    This is separate from llm_client's internal retry logic.
    retries       : passed to call_llm as the retries= argument.
    """
    cfg = get_model_config(model)
    api_key_env = cfg.get("api_key_env", "NVIDIA_API_KEY")
    api_key = get_api_key_for_model(model)

    if not api_key:
        return ModelCheckResult(
            model=model,
            status="NO_KEY",
            error_type="MissingAPIKey",
            error_msg=f"Env var '{api_key_env}' is not set (and NVIDIA_API_KEY is also unset)",
            api_key_env=api_key_env,
            has_key=False,
        )

    start = time.monotonic()

    # Run the actual call inside a tiny thread pool so we can enforce wall-clock timeout.
    # llm_client.call_llm is blocking and may hang for up to (default OpenAI timeout) per
    # attempt when the NVIDIA gateway returns a 504 slowly.
    with ThreadPoolExecutor(max_workers=1) as _ex:
        future = _ex.submit(
            llm_client.call_llm,
            CHECK_PROMPT,
            model=model,
            system_prompt=CHECK_SYSTEM,
            max_tokens=16,          # tiny — we just need one word back
            retries=retries,
        )
        try:
            response = future.result(timeout=call_timeout)
            elapsed = time.monotonic() - start
            return ModelCheckResult(
                model=model,
                status="PASS",
                latency_s=round(elapsed, 2),
                response_preview=(response or "")[:80],
                api_key_env=api_key_env,
            )
        except FuturesTimeout:
            elapsed = time.monotonic() - start
            future.cancel()
            return ModelCheckResult(
                model=model,
                status="TIMEOUT",
                latency_s=round(elapsed, 2),
                error_type="Timeout",
                error_msg=f"No response within {call_timeout}s",
                api_key_env=api_key_env,
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            return ModelCheckResult(
                model=model,
                status="FAIL",
                latency_s=round(elapsed, 2),
                error_type=type(exc).__name__,
                error_msg=str(exc)[:200],
                api_key_env=api_key_env,
            )


# ── Formatting helpers ─────────────────────────────────────────────────────────
def _status_str(status: str) -> str:
    if status == "PASS":
        return f"{GREEN}{BOLD}PASS{RESET}"
    if status == "TIMEOUT":
        return f"{YELLOW}{BOLD}TIMEOUT{RESET}"
    if status == "NO_KEY":
        return f"{CYAN}{BOLD}NO_KEY{RESET}"
    return f"{RED}{BOLD}FAIL{RESET}"


def _print_result(r: ModelCheckResult, idx: int, total: int) -> None:
    status_pad = _status_str(r.status)
    latency = f"{r.latency_s:6.1f}s"
    key_hint = f"{DIM}({r.api_key_env}){RESET}"

    print(f"  [{idx:02d}/{total}] {status_pad}  {latency}  {r.model}")
    if r.status == "PASS":
        print(f"           {DIM}→ response: {r.response_preview!r}{RESET}")
    elif r.status in ("FAIL", "TIMEOUT"):
        print(f"           {RED}→ {r.error_type}: {r.error_msg}{RESET}")
    elif r.status == "NO_KEY":
        print(f"           {CYAN}→ {r.error_msg}{RESET}")
    print(f"           {key_hint}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick health-check for all registered NVIDIA NIM models."
    )
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL",
        help="Specific model(s) to check. Default: all 18.",
    )
    parser.add_argument(
        "--timeout", type=float, default=75.0, metavar="SECONDS",
        help="Wall-clock timeout per model call (default: 75s).",
    )
    parser.add_argument(
        "--retries", type=int, default=1, metavar="N",
        help="Number of llm_client retry attempts per model (default: 1 — fail fast).",
    )
    parser.add_argument(
        "--workers", type=int, default=16, metavar="N",
        help="Concurrent threads for parallel checks (default: 16).",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run checks one at a time instead of in parallel.",
    )
    args = parser.parse_args()

    models_to_check: list[str] = args.models if args.models else ALL_MODELS
    total = len(models_to_check)
    workers = 1 if args.sequential else min(args.workers, total)

    print(f"\n{BOLD}{'=' * 70}")
    print(f"  KG CLAIM VERIFIER — MODEL HEALTH CHECK")
    print(f"{'=' * 70}{RESET}")
    print(f"  Models    : {total}")
    print(f"  Timeout   : {args.timeout}s per model")
    print(f"  Retries   : {args.retries} attempt(s)")
    print(f"  Workers   : {workers} ({'sequential' if args.sequential else 'parallel'})")
    print(f"  Prompt    : {CHECK_PROMPT!r}")
    print(f"{'=' * 70}\n")

    results: list[ModelCheckResult] = [None] * total  # type: ignore[list-item]
    completed = 0

    if args.sequential:
        for i, model in enumerate(models_to_check):
            print(f"  Checking {model} …")
            r = _check_one(model, args.timeout, args.retries)
            results[i] = r
            completed += 1
            _print_result(r, completed, total)
            print()
    else:
        # Map from future → original index so results preserve order
        idx_map: dict = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_check_one, model, args.timeout, args.retries): i
                for i, model in enumerate(models_to_check)
            }
            idx_map = futures
            for future in as_completed(futures):
                i = idx_map[future]
                try:
                    r = future.result()
                except Exception as exc:
                    r = ModelCheckResult(
                        model=models_to_check[i],
                        status="FAIL",
                        error_type=type(exc).__name__,
                        error_msg=str(exc)[:200],
                    )
                results[i] = r
                completed += 1
                _print_result(r, completed, total)
                print()

    # ── Summary ───────────────────────────────────────────────────────────────
    passed   = [r for r in results if r and r.status == "PASS"]
    failed   = [r for r in results if r and r.status == "FAIL"]
    timeouts = [r for r in results if r and r.status == "TIMEOUT"]
    no_keys  = [r for r in results if r and r.status == "NO_KEY"]

    avg_latency = (
        sum(r.latency_s for r in passed) / len(passed) if passed else 0.0
    )

    print(f"\n{BOLD}{'=' * 70}")
    print(f"  HEALTH CHECK SUMMARY")
    print(f"{'=' * 70}{RESET}")
    print(f"  {GREEN}{BOLD}PASS   {RESET}: {len(passed):3d}  (avg latency: {avg_latency:.1f}s)")
    print(f"  {RED}{BOLD}FAIL   {RESET}: {len(failed):3d}")
    print(f"  {YELLOW}{BOLD}TIMEOUT{RESET}: {len(timeouts):3d}")
    print(f"  {CYAN}{BOLD}NO_KEY {RESET}: {len(no_keys):3d}")

    if failed:
        print(f"\n  {RED}Failed models:{RESET}")
        for r in failed:
            print(f"    ✗  {r.model}")
            print(f"       {DIM}{r.error_type}: {r.error_msg[:120]}{RESET}")

    if timeouts:
        print(f"\n  {YELLOW}Timed-out models (>{args.timeout}s, likely 504 / slow gateway):{RESET}")
        for r in timeouts:
            print(f"    ⏱  {r.model}")

    if no_keys:
        print(f"\n  {CYAN}Models with missing API keys:{RESET}")
        for r in no_keys:
            print(f"    🔑  {r.model}  →  set {r.api_key_env} in .env")

    print(f"\n{'=' * 70}")

    # Exit non-zero if anything is not PASS or NO_KEY
    critical = len(failed) + len(timeouts)
    if critical > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
