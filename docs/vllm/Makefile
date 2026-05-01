.PHONY: update-results v1 v2 v3 v4

REPO_ROOT := $(abspath ../..)
PYTHON ?= python3

update-results: ## Update all experiment docs from results/vllm/*
	cd "$(REPO_ROOT)" && uv run $(PYTHON) scripts/update_vllm_docs_bench.py --doc all

v1: ## Update only v1 experiment doc
	cd "$(REPO_ROOT)" && uv run $(PYTHON) scripts/update_vllm_docs_bench.py --doc v1

v2: ## Update only v2 experiment doc
	cd "$(REPO_ROOT)" && uv run $(PYTHON) scripts/update_vllm_docs_bench.py --doc v2

v3: ## Update only v3 experiment doc
	cd "$(REPO_ROOT)" && uv run $(PYTHON) scripts/update_vllm_docs_bench.py --doc v3

v4: ## Update only v4 experiment doc
	cd "$(REPO_ROOT)" && uv run $(PYTHON) scripts/update_vllm_docs_bench.py --doc v4
