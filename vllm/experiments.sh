#!/usr/bin/env bash
# experiments.sh — Run vLLM scheduler-tuning experiment matrix for GLiNER Guard
#
# Runs the server with different scheduler configurations, waits for healthy,
# then launches Locust (locally or on a remote CPU pod via SSH).
#
# Usage:
#   ./experiments.sh                    # run all experiments
#   ./experiments.sh sched-safe         # run single experiment by name
#   ./experiments.sh --list             # show available experiments
#
# Environment variables (server — GPU pod):
#   GLINER_PREPARED   prepared model dir   (default: /tmp/gliner-guard-uni-vllm)
#   VLLM_PORT         server port          (default: 8000)
#
# Environment variables (Locust — remote CPU pod):
#   LOCUST_SSH        SSH target for remote Locust (e.g. root@10.0.0.5)
#                     If unset, Locust runs locally on the same machine.
#   GPU_POD_IP        GPU pod IP as seen from CPU pod (required when LOCUST_SSH is set)
#   REMOTE_TEST_DIR   path to test-script/ on CPU pod (default: ~/gliner-guard-serve/test-script)
#
# Environment variables (Locust tuning):
#   LOCUST_USERS      concurrent users     (default: 100)
#   LOCUST_SPAWN_RATE spawn rate           (default: 1)
#   LOCUST_RUNTIME    test duration         (default: 15m)
set -euo pipefail
cd "$(dirname "$0")"

SCRIPT_DIR="$(pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_SCRIPT_DIR="${PROJECT_ROOT}/test-script"
RESULTS_BASE="${PROJECT_ROOT}/results/vllm/gliner-guard-uni"

PREPARED_DIR="${GLINER_PREPARED:-/tmp/gliner-guard-uni-vllm}"
PORT="${VLLM_PORT:-8000}"
LOCUST_USERS="${LOCUST_USERS:-100}"
LOCUST_SPAWN_RATE="${LOCUST_SPAWN_RATE:-1}"
LOCUST_RUNTIME="${LOCUST_RUNTIME:-15m}"
HEALTH_TIMEOUT=300
WARMUP_SECS=30

LOCUST_SSH="${LOCUST_SSH:-}"
GPU_POD_IP="${GPU_POD_IP:-}"
REMOTE_TEST_DIR="${REMOTE_TEST_DIR:-~/gliner-guard-serve/test-script}"

# -------------------------------------------------------------------------
# Experiment definitions
#
# EXPERIMENTS[name]   = flags passed to vllm serve (single-instance)
#                       or vllm-factory-serve (multi-instance)
# INSTANCE_COUNT[name]= 0 → single instance (vllm serve)
#                       N → N instances (vllm-factory-serve)
# BATCH_SIZE[name]    = per-instance --max-batch-size for multi-instance
# -------------------------------------------------------------------------
declare -A EXPERIMENTS
declare -A INSTANCE_COUNT
declare -A BATCH_SIZE

# --- Single-instance scheduler tuning ---

# 1. Safe: conservative scheduler limits
EXPERIMENTS[sched-safe]="--dtype bfloat16 --enforce-eager --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 16384"
INSTANCE_COUNT[sched-safe]=0

# 2. Balanced: moderate concurrency
EXPERIMENTS[sched-balanced]="--dtype bfloat16 --enforce-eager --max-model-len 8192 --max-num-seqs 128 --max-num-batched-tokens 32768"
INSTANCE_COUNT[sched-balanced]=0

# 3. Aggressive: high concurrency, large token budget
EXPERIMENTS[sched-aggressive]="--dtype bfloat16 --enforce-eager --max-model-len 8192 --max-num-seqs 256 --max-num-batched-tokens 65536"
INSTANCE_COUNT[sched-aggressive]=0

# 4. Short texts: reduced max-model-len for shorter inputs
EXPERIMENTS[sched-short]="--dtype bfloat16 --enforce-eager --max-model-len 4096 --max-num-seqs 256 --max-num-batched-tokens 65536"
INSTANCE_COUNT[sched-short]=0

# --- Multi-instance (vllm-factory-serve) ---

# 5. 4 instances, balanced scheduler per instance
EXPERIMENTS[multi-4x]="--dtype bfloat16 --enforce-eager --max-model-len 8192 --max-num-batched-tokens 32768"
INSTANCE_COUNT[multi-4x]=4
BATCH_SIZE[multi-4x]=64

EXPERIMENT_ORDER=(
    sched-safe
    sched-balanced
    sched-aggressive
    sched-short
    multi-4x
)

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
wait_healthy() {
    local timeout=$1
    local url="http://localhost:${PORT}/health"
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            return 0
        fi
        sleep 3
        elapsed=$((elapsed + 3))
    done
    return 1
}

kill_server() {
    local pid=$1
    if kill -0 "$pid" 2>/dev/null; then
        echo "  Stopping server (pid=$pid)..."
        kill -TERM "$pid" 2>/dev/null || true
        sleep 5
        kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

run_locust() {
    local name=$1
    local csv_prefix=$2
    local results_dir=$3
    local host="http://localhost:${PORT}"

    if [ -n "$LOCUST_SSH" ]; then
        if [ -z "$GPU_POD_IP" ]; then
            echo "  ERROR: GPU_POD_IP must be set when LOCUST_SSH is used."
            return 1
        fi
        host="http://${GPU_POD_IP}:${PORT}"
        echo "  Running Locust remotely on ${LOCUST_SSH} → ${host}..."

        ssh "$LOCUST_SSH" "rm -f /tmp/${name}_stats.csv /tmp/${name}_stats_history.csv \
            /tmp/${name}_failures.csv /tmp/${name}_exceptions.csv /tmp/${name}.html"

        ssh "$LOCUST_SSH" "cd ${REMOTE_TEST_DIR} && \
            GLINER_HOST=${host} \
            python -m locust \
                -f test-gliner-vllm.py \
                --headless \
                -u ${LOCUST_USERS} \
                -r ${LOCUST_SPAWN_RATE} \
                --run-time ${LOCUST_RUNTIME} \
                --csv=/tmp/${name} \
                --html=/tmp/${name}.html \
                --csv-full-history" \
            2>&1 | tee "${results_dir}/${name}-locust.log"

        echo "  Copying results from CPU pod..."
        scp "${LOCUST_SSH}:/tmp/${name}_stats.csv"         "${csv_prefix}_stats.csv"         2>/dev/null || true
        scp "${LOCUST_SSH}:/tmp/${name}_stats_history.csv"  "${csv_prefix}_stats_history.csv"  2>/dev/null || true
        scp "${LOCUST_SSH}:/tmp/${name}_failures.csv"       "${csv_prefix}_failures.csv"       2>/dev/null || true
        scp "${LOCUST_SSH}:/tmp/${name}_exceptions.csv"     "${csv_prefix}_exceptions.csv"     2>/dev/null || true
        scp "${LOCUST_SSH}:/tmp/${name}.html"               "${csv_prefix}.html"               2>/dev/null || true
    else
        echo "  Running Locust locally → ${host}..."
        cd "$TEST_SCRIPT_DIR"
        GLINER_HOST="${host}" \
        python -m locust \
            -f test-gliner-vllm.py \
            --headless \
            -u $LOCUST_USERS \
            -r $LOCUST_SPAWN_RATE \
            --run-time $LOCUST_RUNTIME \
            --csv="${csv_prefix}" \
            --html="${csv_prefix}.html" \
            --csv-full-history \
            2>&1 | tee "${results_dir}/${name}-locust.log"
        cd "$SCRIPT_DIR"
    fi
}

start_server() {
    local name=$1
    local flags=$2
    local instances=${INSTANCE_COUNT[$name]:-0}
    local server_log=$3

    if [ "$instances" -gt 1 ]; then
        local bs=${BATCH_SIZE[$name]:-64}
        echo "  Starting vllm-factory-serve (${instances} instances, max-batch-size=${bs})..."
        vllm-factory-serve "${PREPARED_DIR}" \
            --num-instances "$instances" \
            --max-batch-size "$bs" \
            --port "${PORT}" \
            --io-processor-plugin mmbert_gliner2_io \
            ${flags} \
            -- --runner pooling --no-enable-prefix-caching --no-enable-chunked-prefill \
            > "$server_log" 2>&1 &
    else
        echo "  Starting vLLM server (single instance)..."
        vllm serve "${PREPARED_DIR}" \
            --runner pooling \
            --trust-remote-code \
            --port "${PORT}" \
            --no-enable-prefix-caching \
            --no-enable-chunked-prefill \
            --gpu-memory-utilization 0.80 \
            --io-processor-plugin mmbert_gliner2_io \
            ${flags} \
            > "$server_log" 2>&1 &
    fi
    echo $!
}

run_experiment() {
    local name=$1
    local flags=$2
    local instances=${INSTANCE_COUNT[$name]:-0}
    local results_dir="${RESULTS_BASE}"
    local csv_prefix="${results_dir}/${name}"
    local server_log="${results_dir}/${name}-server.log"

    mkdir -p "$results_dir"

    echo ""
    echo "============================================================"
    echo "  EXPERIMENT: ${name}"
    echo "  Flags:      ${flags}"
    echo "  Instances:  $((instances > 0 ? instances : 1))"
    echo "  Results:    ${csv_prefix}_stats.csv"
    echo "============================================================"

    if [ -f "${csv_prefix}_stats.csv" ]; then
        echo "  SKIP: results already exist. Delete to re-run."
        return 0
    fi

    local server_pid
    server_pid=$(start_server "$name" "$flags" "$server_log")

    trap "kill_server $server_pid" EXIT

    echo "  Waiting for server to become healthy (timeout=${HEALTH_TIMEOUT}s)..."
    if ! wait_healthy $HEALTH_TIMEOUT; then
        echo "  ERROR: Server failed to start. Log tail:"
        tail -30 "$server_log"
        kill_server $server_pid
        trap - EXIT
        return 1
    fi
    echo "  Server healthy."

    echo "  Warming up for ${WARMUP_SECS}s..."
    sleep $WARMUP_SECS

    run_locust "$name" "$csv_prefix" "$results_dir"

    kill_server $server_pid
    trap - EXIT

    echo "  Done: ${name}"
    if [ -f "${csv_prefix}_stats.csv" ]; then
        echo "  Results saved to ${csv_prefix}_stats.csv"
    fi
}

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if [ "${1:-}" = "--list" ]; then
    echo "Available experiments:"
    for name in "${EXPERIMENT_ORDER[@]}"; do
        local_instances=${INSTANCE_COUNT[$name]:-0}
        printf "  %-25s instances=%s  %s\n" "$name" "$((local_instances > 0 ? local_instances : 1))" "${EXPERIMENTS[$name]}"
    done
    exit 0
fi

echo "=================================================="
echo "  vLLM GLiNER Guard Experiment Runner (v2)"
echo "  Model:  ${PREPARED_DIR}"
echo "  Port:   ${PORT}"
echo "  Locust: ${LOCUST_SSH:-local}"
echo "=================================================="

if [ ! -f "${PREPARED_DIR}/config.json" ]; then
    echo "Model not prepared. Running preparation..."
    ./serve.sh &
    PREP_PID=$!
    sleep 30
    kill_server $PREP_PID
    if [ ! -f "${PREPARED_DIR}/config.json" ]; then
        echo "ERROR: Model preparation failed."
        exit 1
    fi
fi

if [ $# -gt 0 ]; then
    for name in "$@"; do
        if [ -z "${EXPERIMENTS[$name]+x}" ]; then
            echo "ERROR: Unknown experiment '${name}'"
            echo "Available: ${!EXPERIMENTS[*]}"
            exit 1
        fi
        run_experiment "$name" "${EXPERIMENTS[$name]}"
    done
else
    for name in "${EXPERIMENT_ORDER[@]}"; do
        run_experiment "$name" "${EXPERIMENTS[$name]}"
    done
fi

echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Results in: ${RESULTS_BASE}/"
echo "============================================================"
