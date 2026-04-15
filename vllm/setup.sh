#!/usr/bin/env bash
# setup.sh — Install all dependencies for vLLM GLiNER Guard experiments
#
# Usage:
#   ./setup.sh           # full install (GPU machine)
#   ./setup.sh --no-vllm # Locust-only install (CPU pod / local testing)
set -euo pipefail
cd "$(dirname "$0")"

NO_VLLM=false
if [[ "${1:-}" == "--no-vllm" ]]; then
    NO_VLLM=true
fi

PROJECT_ROOT="$(dirname "$(pwd)")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${PROJECT_ROOT}/.venv"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "=== Creating / using virtualenv: ${VENV_DIR} ==="
    if [[ ! -d "${VENV_DIR}" ]]; then
        "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    fi
    export VIRTUAL_ENV="${VENV_DIR}"
    export PATH="${VENV_DIR}/bin:${PATH}"
    hash -r
fi

echo "=== Python environment ==="
echo "python: $(command -v python)"
echo "pip:    $(command -v pip)"
python -m pip install --upgrade pip setuptools wheel

echo ""
echo "=== Installing Locust test dependencies ==="
python -m pip install locust pandas python-dotenv

if [ "$NO_VLLM" = false ]; then
    echo ""
    echo "=== Installing vllm-factory ==="
    python -m pip install -e "${PROJECT_ROOT}/vllm-factory[gliner]"

    echo ""
    echo "=== Installing experiment dependencies ==="
    python -m pip install python-dotenv aiohttp requests

    echo ""
    echo "=== Installing vLLM (LAST — pins shared deps) ==="
    python -m pip install vllm
fi

echo ""
echo "=== Done ==="
python -m pip show locust | sed -n '1,3p'
if [ "$NO_VLLM" = false ]; then
    python -m pip show vllm-factory | sed -n '1,3p'
    python -m pip show vllm | sed -n '1,3p'
fi
