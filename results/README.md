# Benchmark Results

Use this directory for copied benchmark artifacts that should be analyzed by the
vLLM and LitServe experiment notebook.

Expected Locust files per experiment:

```text
<experiment-name>_stats.csv
<experiment-name>_stats_history.csv
<experiment-name>_failures.csv
<experiment-name>_exceptions.csv
```

Expected GPU logs per experiment:

```text
gpu_<experiment-name>_<timestamp>.csv
gpu_procs_<experiment-name>_<timestamp>.csv
server_<experiment-name>_<timestamp>.log
```

Use `litserve-baseline` as the experiment name for the PyTorch/LitServe
baseline GPU accounting.

In `analyze/notebooks/analyze_vllm_experiments.ipynb`, set:

```python
RESULTS_DIR = PROJECT_ROOT / "results"
GPU_LOGS_DIRS = [PROJECT_ROOT / "results"]
```

The notebook currently defaults to the collected data under `analyze/data`.
