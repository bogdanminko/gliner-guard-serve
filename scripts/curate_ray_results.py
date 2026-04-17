#!/usr/bin/env python3
"""Curate raw Ray Serve benchmark outputs into README-ready results layout.

Raw benchmark runs are produced at the repository root under names like:
  results/ray-rest-nobatch-uni-prompts-run2_stats.csv
  results/ray-rest-B4-uni-prompts-run2.html
  results/ray-grpc-B4-bi-prompts-run2_stats.csv

This script copies the chosen raw artifacts into the recursive layout expected
by the upstream benchmark table generator:

  results/ray-serve/<model>/<runtime>.csv
  results/ray-serve/<model>/<runtime>.html

Example:
  python3 scripts/curate_ray_results.py \
    --model gliner-guard-uni \
    --rest-nobatch ray-rest-nobatch-uni-prompts-run2 \
    --rest-dynbatch ray-rest-B4-uni-prompts-run2 \
    --grpc-dynbatch ray-grpc-B4-uni-prompts-run2
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

RUNTIME_MAP = {
    "rest_nobatch": "pytorch-bf16-rest-nobatch",
    "rest_dynbatch": "pytorch-bf16-rest-dynbatch",
    "grpc_dynbatch": "pytorch-bf16-grpc-dynbatch",
}


def _copy_artifact(prefix: str, target_dir: Path, runtime_name: str) -> None:
    stats_src = RESULTS_DIR / f"{prefix}_stats.csv"
    html_src = RESULTS_DIR / f"{prefix}.html"

    if not stats_src.exists():
        raise FileNotFoundError(f"Missing stats CSV: {stats_src}")
    if not html_src.exists():
        raise FileNotFoundError(f"Missing HTML report: {html_src}")

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stats_src, target_dir / f"{runtime_name}.csv")
    shutil.copy2(html_src, target_dir / f"{runtime_name}.html")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy selected Ray Serve raw results into README-ready layout."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Target model directory name, e.g. gliner-guard-uni or gliner-guard-bi",
    )
    parser.add_argument(
        "--rest-nobatch",
        required=True,
        help="Raw result prefix for REST no-batch, without _stats.csv suffix",
    )
    parser.add_argument(
        "--rest-dynbatch",
        required=True,
        help="Raw result prefix for REST dynamic batching, without _stats.csv suffix",
    )
    parser.add_argument(
        "--grpc-dynbatch",
        required=True,
        help="Raw result prefix for gRPC dynamic batching, without _stats.csv suffix",
    )
    args = parser.parse_args()

    target_dir = RESULTS_DIR / "ray-serve" / args.model
    _copy_artifact(args.rest_nobatch, target_dir, RUNTIME_MAP["rest_nobatch"])
    _copy_artifact(args.rest_dynbatch, target_dir, RUNTIME_MAP["rest_dynbatch"])
    _copy_artifact(args.grpc_dynbatch, target_dir, RUNTIME_MAP["grpc_dynbatch"])

    print(f"Curated Ray Serve results written to {target_dir}")


if __name__ == "__main__":
    main()
