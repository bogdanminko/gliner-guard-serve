#!/usr/bin/env bash
# setup.sh — Install all dependencies for vLLM GLiNER Guard experiments
#
# Usage:
#   ./setup.sh           # full install (GPU machine)
#   ./setup.sh --no-vllm # skip vLLM install (for local testing without GPU)
set -euo pipefail
cd "$(dirname "$0")"

NO_VLLM=false
if [[ "${1:-}" == "--no-vllm" ]]; then
    NO_VLLM=true
fi

PROJECT_ROOT="$(dirname "$(pwd)")"

echo "=== Installing vllm-factory ==="
pip install -e "${PROJECT_ROOT}/vllm-factory[gliner]"

echo ""
echo "=== Installing experiment dependencies ==="
pip install python-dotenv aiohttp requests

echo ""
echo "=== Installing Locust test dependencies ==="
pip install locust pandas python-dotenv

if [ "$NO_VLLM" = false ]; then
    echo ""
    echo "=== Installing vLLM (LAST — pins shared deps) ==="
    pip install vllm
fi

echo ""
echo "=== Done ==="
pip show vllm-factory | head -3
if [ "$NO_VLLM" = false ]; then
    pip show vllm | head -3
fi
