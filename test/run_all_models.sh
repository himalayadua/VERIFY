#!/usr/bin/env bash
# run_all_models.sh
# -----------------
# Run experiments for all 18 registered NVIDIA NIM models with a parallel
# worker pool (default: 3 models at a time).
#
# Each model is a completely isolated Python process writing to its own
# results/<model-slug>/ directory, so parallel execution is fully safe.
#
# Default mode  : python src/run_experiments.py  (logs + JSON in one session)
# Optional mode : --integration-only → python test/test_integration.py
#
# Usage (run from the project root kg-claim-verifier/):
#
#   bash test/run_all_models.sh                              # all models, 3 parallel
#   bash test/run_all_models.sh --parallel 5                 # 5 at a time
#   bash test/run_all_models.sh --parallel 1                 # sequential (safe default)
#   bash test/run_all_models.sh --variants v0 v1             # specific variants
#   bash test/run_all_models.sh --limit 5                    # first 5 questions only
#   bash test/run_all_models.sh --models "m1 m2 m3"          # specific models only
#   bash test/run_all_models.sh --skip "m1 m2"               # skip specific models
#   bash test/run_all_models.sh --integration-only           # use test_integration.py
#   bash test/run_all_models.sh --integration-only --all-questions

set -uo pipefail

# ── All 18 registered models ───────────────────────────────────────────────────
ALL_MODELS=(
    "google/gemma-4-31b-it"
    "qwen/qwen3.5-122b-a10b"
    "minimaxai/minimax-m2.5"
    "qwen/qwen3.5-397b-a17b"
    "deepseek-ai/deepseek-v3.2"
    "moonshotai/kimi-k2-thinking"
    "mistralai/mistral-large-3-675b-instruct-2512"
    "deepseek-ai/deepseek-v3.1-terminus"
    "moonshotai/kimi-k2-instruct-0905"
    "qwen/qwen3-next-80b-a3b-thinking"
    "bytedance/seed-oss-36b-instruct"
    "qwen/qwen3-coder-480b-a35b-instruct"
    "openai/gpt-oss-120b"
    "openai/gpt-oss-20b"
    "google/gemma-3n-e4b-it"
    "google/gemma-3n-e2b-it"
    "z-ai/glm-5.1"
    "minimaxai/minimax-m2.7"
)

# ── Defaults ───────────────────────────────────────────────────────────────────
MAX_PARALLEL=3
INTEGRATION_ONLY=false
VARIANTS_ARGS=()
LIMIT_ARG=""
ALL_QUESTIONS=false
CUSTOM_MODELS=()
SKIP_MODELS=()

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --integration-only)
            INTEGRATION_ONLY=true
            shift
            ;;
        --variants)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                VARIANTS_ARGS+=("$1")
                shift
            done
            ;;
        --limit|--questions)
            # --limit N  for run_experiments; --questions N for test_integration
            # both accepted here and mapped correctly per script below
            LIMIT_ARG="$2"
            shift 2
            ;;
        --all-questions)
            ALL_QUESTIONS=true
            shift
            ;;
        --models)
            shift
            IFS=' ' read -ra CUSTOM_MODELS <<< "$1"
            shift
            ;;
        --skip)
            shift
            IFS=' ' read -ra SKIP_MODELS <<< "$1"
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# ── Determine final model list ─────────────────────────────────────────────────
if [[ ${#CUSTOM_MODELS[@]} -gt 0 ]]; then
    MODELS=("${CUSTOM_MODELS[@]}")
else
    MODELS=("${ALL_MODELS[@]}")
fi

# ── Build per-script argument arrays ──────────────────────────────────────────
#
# run_experiments.py  uses:  --variants, --limit N
# test_integration.py uses:  --variants, --questions N, --all-questions

EXPERIMENT_ARGS=()
INTEGRATION_ARGS=()

if [[ ${#VARIANTS_ARGS[@]} -gt 0 ]]; then
    EXPERIMENT_ARGS+=("--variants" "${VARIANTS_ARGS[@]}")
    INTEGRATION_ARGS+=("--variants" "${VARIANTS_ARGS[@]}")
fi

if [[ -n "$LIMIT_ARG" ]]; then
    EXPERIMENT_ARGS+=("--limit" "$LIMIT_ARG")
    INTEGRATION_ARGS+=("--questions" "$LIMIT_ARG")
fi

if $ALL_QUESTIONS; then
    INTEGRATION_ARGS+=("--all-questions")
    # run_experiments reads all 25 questions by default; no flag needed
fi

# ── Slug helper (mirrors model_registry.model_to_slug) ────────────────────────
_slug() {
    echo "$1" | sed 's|/|-|g; s|\.|_|g; s|[^a-zA-Z0-9_-]|_|g'
}

# ── Setup ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATUS_DIR="$(mktemp -d)"
trap 'rm -rf "$STATUS_DIR"' EXIT

# ── Worker pool state ──────────────────────────────────────────────────────────
RUNNING_PIDS=()
RUNNING_MODELS=()
RUNNING_LOGS=()
PASS=0
FAIL=0
FAILED_MODELS=()
TOTAL_STARTED=0

# ── Helpers ────────────────────────────────────────────────────────────────────
_ts() { date +"%H:%M:%S"; }

_log() { echo "[$(_ts)] $*"; }

# Reap any jobs that have finished and update counters.
_reap() {
    local new_pids=() new_models=() new_logs=()
    local i
    for i in "${!RUNNING_PIDS[@]}"; do
        local pid="${RUNNING_PIDS[$i]}"
        local mdl="${RUNNING_MODELS[$i]}"
        local log="${RUNNING_LOGS[$i]}"
        local slug
        slug=$(_slug "$mdl")
        local status_file="$STATUS_DIR/${slug}.exit"

        if [[ -f "$status_file" ]]; then
            # Job wrote its exit code — it has finished
            local ec
            ec=$(cat "$status_file")
            wait "$pid" 2>/dev/null || true
            if [[ "$ec" -eq 0 ]]; then
                _log "[DONE ✓] $mdl"
                PASS=$((PASS + 1))
            else
                _log "[FAIL ✗] $mdl  (exit=$ec)"
                _log "         stdout log → $log"
                FAIL=$((FAIL + 1))
                FAILED_MODELS+=("$mdl")
            fi
        elif kill -0 "$pid" 2>/dev/null; then
            # Still running — keep in pool
            new_pids+=("$pid")
            new_models+=("$mdl")
            new_logs+=("$log")
        else
            # Died without writing status file (OOM-killed, signal, etc.)
            wait "$pid" 2>/dev/null || true
            _log "[FAIL ✗] $mdl  (crashed without status)"
            _log "         stdout log → $log"
            FAIL=$((FAIL + 1))
            FAILED_MODELS+=("$mdl")
        fi
    done
    RUNNING_PIDS=("${new_pids[@]:+"${new_pids[@]}"}")
    RUNNING_MODELS=("${new_models[@]:+"${new_models[@]}"}")
    RUNNING_LOGS=("${new_logs[@]:+"${new_logs[@]}"}")
}

# Block until the pool has a free slot.
_wait_for_slot() {
    while [[ ${#RUNNING_PIDS[@]} -ge $MAX_PARALLEL ]]; do
        _reap
        if [[ ${#RUNNING_PIDS[@]} -ge $MAX_PARALLEL ]]; then
            sleep 2
        fi
    done
}

# Launch a model as a background job.
_launch() {
    local MODEL="$1"
    local SLUG
    SLUG=$(_slug "$MODEL")
    local STATUS_FILE="$STATUS_DIR/${SLUG}.exit"

    # Console output from Python goes here (real logs are inside results/<slug>/logs/)
    local OUT_DIR="$PROJECT_ROOT/results/$SLUG"
    mkdir -p "$OUT_DIR"
    local LOG_FILE="$OUT_DIR/stdout_$(date +%Y%m%d_%H%M%S).log"

    (
        set +e
        if $INTEGRATION_ONLY; then
            python "$PROJECT_ROOT/test/test_integration.py" \
                --model "$MODEL" \
                ${INTEGRATION_ARGS[@]:+"${INTEGRATION_ARGS[@]}"}
        else
            python "$PROJECT_ROOT/src/run_experiments.py" \
                --model "$MODEL" \
                ${EXPERIMENT_ARGS[@]:+"${EXPERIMENT_ARGS[@]}"}
        fi
        EC=$?
        echo "$EC" > "$STATUS_FILE"
        exit "$EC"
    ) > "$LOG_FILE" 2>&1 &

    local PID=$!
    RUNNING_PIDS+=("$PID")
    RUNNING_MODELS+=("$MODEL")
    RUNNING_LOGS+=("$LOG_FILE")
    TOTAL_STARTED=$((TOTAL_STARTED + 1))

    local RUNNING_COUNT=${#RUNNING_PIDS[@]}
    local MODE
    if $INTEGRATION_ONLY; then MODE="integration"; else MODE="experiments"; fi
    _log "[START ] [$RUNNING_COUNT/$MAX_PARALLEL running]  $MODEL  (pid=$PID  mode=$MODE)"
    _log "         stdout → $LOG_FILE"
}

# ── Print banner ───────────────────────────────────────────────────────────────
MODE_LABEL="run_experiments.py  (logs + JSON)"
if $INTEGRATION_ONLY; then MODE_LABEL="test_integration.py  (logs only)"; fi

echo "================================================================"
echo "  KG CLAIM VERIFIER — RUN ALL MODELS"
echo "  Project root : $PROJECT_ROOT"
echo "  Models       : ${#MODELS[@]}"
echo "  Parallel     : $MAX_PARALLEL at a time"
echo "  Mode         : $MODE_LABEL"
if [[ ${#VARIANTS_ARGS[@]} -gt 0 ]]; then
echo "  Variants     : ${VARIANTS_ARGS[*]}"
fi
if [[ -n "$LIMIT_ARG" ]]; then
echo "  Limit        : $LIMIT_ARG questions"
fi
echo "  Started      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo ""

# ── Main dispatch loop ─────────────────────────────────────────────────────────
for MODEL in "${MODELS[@]}"; do

    # Skip if requested
    SKIP=false
    for S in "${SKIP_MODELS[@]:+"${SKIP_MODELS[@]}"}"; do
        if [[ "$MODEL" == "$S" ]]; then SKIP=true; break; fi
    done
    if $SKIP; then
        _log "[SKIP  ] $MODEL"
        continue
    fi

    _wait_for_slot
    _launch "$MODEL"

done

# ── Drain remaining jobs ───────────────────────────────────────────────────────
_log ""
_log "All $TOTAL_STARTED models launched. Waiting for remaining jobs to finish..."
while [[ ${#RUNNING_PIDS[@]} -gt 0 ]]; do
    _reap
    if [[ ${#RUNNING_PIDS[@]} -gt 0 ]]; then
        _log "  Still running: ${RUNNING_MODELS[*]}"
        sleep 5
    fi
done

# ── Final summary ──────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  RUN ALL MODELS COMPLETE  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Total   : $TOTAL_STARTED"
echo "  Passed  : $PASS"
echo "  Failed  : $FAIL"
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    echo "  Failed models:"
    for M in "${FAILED_MODELS[@]}"; do
        echo "    ✗  $M"
    done
fi
echo "================================================================"

# ── Cross-model comparison ─────────────────────────────────────────────────────
if ! $INTEGRATION_ONLY; then
    COMPARE_SCRIPT="$PROJECT_ROOT/test/compare_models.py"
    if [[ -f "$COMPARE_SCRIPT" ]]; then
        echo ""
        echo "Running cross-model comparison..."
        python "$COMPARE_SCRIPT" --results-dir "$PROJECT_ROOT/results"
    fi
fi
