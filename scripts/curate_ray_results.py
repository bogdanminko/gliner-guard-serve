#!/usr/bin/env python3
"""Curate raw Ray Serve benchmark outputs into README-ready results layout.

Raw benchmark runs are produced at the repository root under names like:
  results/ray-rest-bf16-nobatch-uni-prompts-run2_stats.csv
  results/ray-rest-fp16-B4-uni-prompts-run2.html
  results/ray-grpc-bf16-B16-bi-prompts-run2_stats.csv

This script copies the chosen raw artifacts into the recursive layout expected
by the upstream benchmark table generator:

  results/ray-serve/<model>/<runtime>.csv
  results/ray-serve/<model>/<runtime>.html

Example:
  python3 scripts/curate_ray_results.py \
    --source-dir artifacts/raw-results \
    --model gliner-guard-uni \
    --dtype bf16 \
    --rest-nobatch ray-rest-bf16-nobatch-uni-prompts-run2 \
    --rest-dynbatch ray-rest-bf16-B4-uni-prompts-run2 \
    --grpc-dynbatch ray-grpc-bf16-B16-uni-prompts-run2
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

DTYPE_RUNTIME_PREFIX = {
    "bf16": "pytorch-bf16",
    "bfloat16": "pytorch-bf16",
    "fp16": "pytorch-fp16",
    "float16": "pytorch-fp16",
}


def _copy_artifact(
    source_dir: Path, prefix: str, target_dir: Path, runtime_name: str
) -> None:
    stats_src = source_dir / f"{prefix}_stats.csv"
    html_src = source_dir / f"{prefix}.html"

    if not stats_src.exists():
        raise FileNotFoundError(f"Missing stats CSV: {stats_src}")
    if not html_src.exists():
        raise FileNotFoundError(f"Missing HTML report: {html_src}")

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stats_src, target_dir / f"{runtime_name}.csv")
    shutil.copy2(html_src, target_dir / f"{runtime_name}.html")


def _runtime_map(dtype: str) -> dict[str, str]:
    runtime_prefix = DTYPE_RUNTIME_PREFIX.get(dtype.strip().lower())
    if runtime_prefix is None:
        supported = ", ".join(sorted(DTYPE_RUNTIME_PREFIX))
        raise ValueError(
            f"Unsupported --dtype={dtype!r}. Supported values: {supported}"
        )
    return {
        "rest_nobatch": f"{runtime_prefix}-rest-nobatch",
        "rest_dynbatch": f"{runtime_prefix}-rest-dynbatch",
        "grpc_dynbatch": f"{runtime_prefix}-grpc-dynbatch",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy selected Ray Serve raw results into README-ready layout."
    )
    parser.add_argument(
        "--source-dir",
        default=str(RESULTS_DIR),
        help="Directory containing raw benchmark artifacts (default: results/)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Target model directory name, e.g. gliner-guard-uni or gliner-guard-bi",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        help="Precision tag for curated runtime names: bf16 or fp16 (aliases: bfloat16, float16)",
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

    runtime_map = _runtime_map(args.dtype)
    source_dir = Path(args.source_dir).expanduser()
    if not source_dir.is_absolute():
        source_dir = ROOT / source_dir
    target_dir = RESULTS_DIR / "ray-serve" / args.model
    _copy_artifact(
        source_dir, args.rest_nobatch, target_dir, runtime_map["rest_nobatch"]
    )
    _copy_artifact(
        source_dir, args.rest_dynbatch, target_dir, runtime_map["rest_dynbatch"]
    )
    _copy_artifact(
        source_dir, args.grpc_dynbatch, target_dir, runtime_map["grpc_dynbatch"]
    )

    print(f"Curated Ray Serve results written to {target_dir}")


if __name__ == "__main__":
    main()
