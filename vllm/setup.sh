#!/usr/bin/env bash
# setup.sh — Install all dependencies for vLLM GLiNER Guard experiments
#
# Usage:
#   ./setup.sh           # full install (GPU machine)
#   ./setup.sh --no-vllm # Locust-only install (CPU pod / local testing)
#   VLLM_FACTORY_GIT_REF=<branch-or-commit> ./setup.sh
#   VLLM_FACTORY_GIT_URL=<repo-url> ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

NO_VLLM=false
if [[ "${1:-}" == "--no-vllm" ]]; then
    NO_VLLM=true
fi

PROJECT_ROOT="$(dirname "$(pwd)")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${PROJECT_ROOT}/.venv"
VLLM_SPEC="${VLLM_SPEC:-vllm==0.19.1}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-transformers>=4.56,<5.0}"
VLLM_FACTORY_GIT_URL="${VLLM_FACTORY_GIT_URL:-https://github.com/Reterno12/vllm-factory.git}"
VLLM_FACTORY_GIT_REF="${VLLM_FACTORY_GIT_REF:-feat/gliner2-modernbert-plugin}"

if [[ -n "${VLLM_FACTORY_GIT_REF}" ]]; then
    VLLM_FACTORY_SPEC="vllm-factory[gliner] @ git+${VLLM_FACTORY_GIT_URL}@${VLLM_FACTORY_GIT_REF}"
else
    VLLM_FACTORY_SPEC="vllm-factory[gliner] @ git+${VLLM_FACTORY_GIT_URL}"
fi

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
    echo "=== Installing vllm-factory from fork ==="
    echo "repo: ${VLLM_FACTORY_GIT_URL}"
    echo "ref:  ${VLLM_FACTORY_GIT_REF}"
    python -m pip uninstall -y vllm-factory >/dev/null 2>&1 || true
    python -m pip install --no-cache-dir "${VLLM_FACTORY_SPEC}"

    echo ""
    echo "=== Installing experiment dependencies ==="
    python -m pip install python-dotenv aiohttp requests

    echo ""
    echo "=== Installing tested vLLM + GLiNER-compatible transformers ==="
    python -m pip install "${VLLM_SPEC}" "${TRANSFORMERS_SPEC}"
fi

echo ""
echo "=== Checking dependency consistency ==="
python -m pip check

echo ""
echo "=== Done ==="
python -m pip show locust | sed -n '1,3p'
if [ "$NO_VLLM" = false ]; then
    python -m pip show vllm-factory | sed -n '1,3p'
    python -m pip show vllm | sed -n '1,3p'
    python -m pip show transformers | sed -n '1,3p'
fi
