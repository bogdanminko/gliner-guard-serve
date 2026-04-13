#!/usr/bin/env bash
# serve.sh — prepare model + launch vLLM server for GLiNER Guard
#
# Usage:
#   ./serve.sh                          # defaults: bfloat16, enforce-eager
#   ./serve.sh --dtype float16          # override dtype
#   VLLM_EXTRA_FLAGS="--no-enforce-eager" ./serve.sh  # CUDA graphs
#
# Environment variables:
#   GLINER_MODEL_ID    HuggingFace model ID (default: hivetrace/gliner-guard-uniencoder)
#   GLINER_PLUGIN      vllm-factory plugin  (default: deberta_gliner2)
#   GLINER_PREPARED    prepared model dir   (default: /tmp/gliner-guard-uni-vllm)
#   VLLM_PORT          server port          (default: 8000)
#   VLLM_DTYPE         dtype                (default: bfloat16)
#   VLLM_GPU_MEM       gpu-memory-util      (default: 0.80)
#   VLLM_EXTRA_FLAGS   additional flags passed to vllm serve
set -euo pipefail
cd "$(dirname "$0")"

MODEL_ID="${GLINER_MODEL_ID:-hivetrace/gliner-guard-uniencoder}"
PLUGIN="${GLINER_PLUGIN:-mmbert_gliner2}"
PREPARED_DIR="${GLINER_PREPARED:-/tmp/gliner-guard-uni-vllm}"
PORT="${VLLM_PORT:-8000}"
DTYPE="${VLLM_DTYPE:-bfloat16}"
GPU_MEM="${VLLM_GPU_MEM:-0.80}"
EXTRA_FLAGS="${VLLM_EXTRA_FLAGS:-}"

IO_PLUGIN="${PLUGIN}_io"

echo "============================================="
echo "  vLLM GLiNER Guard Server"
echo "  Model:      ${MODEL_ID}"
echo "  Plugin:     ${PLUGIN}"
echo "  IO Plugin:  ${IO_PLUGIN}"
echo "  Prepared:   ${PREPARED_DIR}"
echo "  Port:       ${PORT}"
echo "  Dtype:      ${DTYPE}"
echo "  GPU Mem:    ${GPU_MEM}"
echo "  Extra:      ${EXTRA_FLAGS}"
echo "============================================="

# --- Step 1: Prepare model directory ---
if [ ! -f "${PREPARED_DIR}/config.json" ]; then
    echo "[1/2] Preparing model directory..."
    vllm-factory-prep \
        --model "${MODEL_ID}" \
        --plugin "${PLUGIN}" \
        --output "${PREPARED_DIR}"
else
    echo "[1/2] Model already prepared at ${PREPARED_DIR}"
fi

# --- Step 2: Launch vLLM server ---
echo "[2/2] Starting vLLM server on port ${PORT}..."
exec vllm serve "${PREPARED_DIR}" \
    --runner pooling \
    --trust-remote-code \
    --dtype "${DTYPE}" \
    --port "${PORT}" \
    --enforce-eager \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --gpu-memory-utilization "${GPU_MEM}" \
    --io-processor-plugin "${IO_PLUGIN}" \
    ${EXTRA_FLAGS}
