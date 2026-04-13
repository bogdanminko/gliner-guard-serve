#!/usr/bin/env bash
# experiments.sh — Run vLLM optimization experiment matrix for GLiNER Guard
#
# Runs the server with different configurations, waits for healthy,
# then launches Locust from ../test-script for each config.
#
# Usage:
#   ./experiments.sh                  # run all experiments
#   ./experiments.sh bfloat16-eager   # run single experiment by name
#
# Prerequisites:
#   - GPU machine (A100 80G recommended)
#   - vllm-factory installed: pip install -e "../vllm-factory[gliner]"
#   - vLLM installed: pip install vllm
#   - Locust deps:  pip install locust pandas python-dotenv
#   - Model prepared: ./serve.sh (run once to prep, then Ctrl+C)
set -euo pipefail
cd "$(dirname "$0")"

SCRIPT_DIR="$(pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_SCRIPT_DIR="${PROJECT_ROOT}/test-script"
RESULTS_BASE="${PROJECT_ROOT}/results/vllm/gliner-guard-uni"

PREPARED_DIR="${GLINER_PREPARED:-/tmp/gliner-guard-uni-vllm}"
PORT="${VLLM_PORT:-8000}"
LOCUST_USERS=100
LOCUST_SPAWN_RATE=1
LOCUST_RUNTIME="15m"
HEALTH_TIMEOUT=300
WARMUP_SECS=30

# -------------------------------------------------------------------------
# Experiment definitions: NAME -> vllm serve flags
# Each experiment tests a specific vLLM optimization relevant to BERT models
# -------------------------------------------------------------------------
declare -A EXPERIMENTS

# 1. Baseline: bfloat16, eager mode (no CUDA graphs)
EXPERIMENTS[bfloat16-eager]="--dtype bfloat16 --enforce-eager"

# 2. float16, eager mode
EXPERIMENTS[float16-eager]="--dtype float16 --enforce-eager"

# 3. bfloat16 with CUDA graphs (remove enforce-eager)
EXPERIMENTS[bfloat16-cudagraph]="--dtype bfloat16"

# 4. float16 with CUDA graphs
EXPERIMENTS[float16-cudagraph]="--dtype float16"

# 5. Larger batch token budget
EXPERIMENTS[bfloat16-eager-batch16k]="--dtype bfloat16 --enforce-eager --max-num-batched-tokens 16384"

# 6. Smaller batch token budget (more responsive latency)
EXPERIMENTS[bfloat16-eager-batch4k]="--dtype bfloat16 --enforce-eager --max-num-batched-tokens 4096"

# 7. Higher GPU memory utilization
EXPERIMENTS[bfloat16-eager-mem90]="--dtype bfloat16 --enforce-eager --gpu-memory-utilization 0.90"

EXPERIMENT_ORDER=(
    bfloat16-eager
    float16-eager
    bfloat16-cudagraph
    float16-cudagraph
    bfloat16-eager-batch16k
    bfloat16-eager-batch4k
    bfloat16-eager-mem90
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

run_experiment() {
    local name=$1
    local flags=$2
    local results_dir="${RESULTS_BASE}"
    local csv_prefix="${results_dir}/${name}"
    local server_log="${results_dir}/${name}-server.log"

    mkdir -p "$results_dir"

    echo ""
    echo "============================================================"
    echo "  EXPERIMENT: ${name}"
    echo "  Flags: ${flags}"
    echo "  Results: ${csv_prefix}_stats.csv"
    echo "============================================================"

    if [ -f "${csv_prefix}_stats.csv" ]; then
        echo "  SKIP: results already exist. Delete to re-run."
        return 0
    fi

    # Start vLLM server in background
    echo "  Starting vLLM server..."
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
    local server_pid=$!

    trap "kill_server $server_pid" EXIT

    # Wait for healthy
    echo "  Waiting for server to become healthy (timeout=${HEALTH_TIMEOUT}s)..."
    if ! wait_healthy $HEALTH_TIMEOUT; then
        echo "  ERROR: Server failed to start. Log tail:"
        tail -30 "$server_log"
        kill_server $server_pid
        trap - EXIT
        return 1
    fi
    echo "  Server healthy."

    # Warmup
    echo "  Warming up for ${WARMUP_SECS}s..."
    sleep $WARMUP_SECS

    # Run Locust
    echo "  Running Locust (users=${LOCUST_USERS}, rate=${LOCUST_SPAWN_RATE}, runtime=${LOCUST_RUNTIME})..."
    cd "$TEST_SCRIPT_DIR"
    GLINER_HOST="http://localhost:${PORT}" \
    python -m locust \
        -f test-gliner-vllm.py \
        --headless \
        -u $LOCUST_USERS \
        -r $LOCUST_SPAWN_RATE \
        --run-time $LOCUST_RUNTIME \
        --csv="${csv_prefix}" \
        --csv-full-history \
        2>&1 | tee "${results_dir}/${name}-locust.log"
    cd "$SCRIPT_DIR"

    # Cleanup
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
echo "=================================================="
echo "  vLLM GLiNER Guard Experiment Runner"
echo "  Model: ${PREPARED_DIR}"
echo "  Port:  ${PORT}"
echo "=================================================="

# Check model is prepared
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

# Run selected or all experiments
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
