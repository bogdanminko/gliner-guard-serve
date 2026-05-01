#!/usr/bin/env python3
"""Build comparative runtime charts for gliner2-multi-v1 from README benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


BENCH_START = "<!-- BENCH:START -->"
BENCH_END = "<!-- BENCH:END -->"
TARGET_MODEL = "gliner2-multi-v1"


def parse_number(value: str) -> float:
    cleaned = value.replace("**", "").replace(" ", "").strip()
    if not cleaned:
        return 0.0
    return float(cleaned)


def extract_model_rows(readme_path: Path, model_name: str) -> List[Dict[str, float | str]]:
    text = readme_path.read_text(encoding="utf-8")
    if BENCH_START not in text or BENCH_END not in text:
        raise ValueError("Benchmark markers were not found in README.")

    block = text.split(BENCH_START, maxsplit=1)[1].split(BENCH_END, maxsplit=1)[0]
    rows: List[Dict[str, float | str]] = []
    current_model = ""

    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue
        if set(line.replace("|", "").strip()) == {"-"}:
            continue

        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 8 or parts[0] == "Model":
            continue

        model = parts[0] or current_model
        runtime = parts[2]
        current_model = model

        if model != model_name or not runtime:
            continue

        rows.append(
            {
                "runtime": runtime,
                "rps": parse_number(parts[3]),
                "p50": parse_number(parts[4]),
                "p95": parse_number(parts[5]),
                "p99": parse_number(parts[6]),
                "err_rate": parse_number(parts[7]),
            }
        )

    if not rows:
        raise ValueError(f"No benchmark rows found for model '{model_name}'.")

    return rows


def plot_rps_err(rows: List[Dict[str, float | str]], output_path: Path) -> None:
    runtimes = [str(row["runtime"]) for row in rows]
    rps = [float(row["rps"]) for row in rows]
    err = [float(row["err_rate"]) for row in rows]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()
    bars = ax1.bar(runtimes, rps, color="#4C78A8", label="RPS")
    ax2.plot(runtimes, err, color="#E45756", marker="o", linewidth=2.0, label="Err rate (%)")

    ax1.set_title("gliner2-multi-v1: Throughput and Error Rate by Runtime")
    ax1.set_ylabel("RPS")
    ax2.set_ylabel("Err rate (%)")
    ax1.tick_params(axis="x", rotation=20)

    for bar, value in zip(bars, rps):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    for idx, value in enumerate(err):
        ax2.text(idx, value + 0.2, f"{value:.2f}", color="#E45756", ha="center", fontsize=9)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_latency(rows: List[Dict[str, float | str]], output_path: Path) -> None:
    runtimes = [str(row["runtime"]) for row in rows]
    p50 = [float(row["p50"]) for row in rows]
    p95 = [float(row["p95"]) for row in rows]
    p99 = [float(row["p99"]) for row in rows]

    x = list(range(len(runtimes)))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_50 = ax.bar([i - width for i in x], p50, width=width, label="P50 (ms)", color="#54A24B")
    bars_95 = ax.bar(x, p95, width=width, label="P95 (ms)", color="#F58518")
    bars_99 = ax.bar([i + width for i in x], p99, width=width, label="P99 (ms)", color="#B279A2")

    ax.set_title("gliner2-multi-v1: Latency Percentiles by Runtime")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(runtimes, rotation=20, ha="right")
    ax.legend()

    for bars in (bars_50, bars_95, bars_99):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 20, f"{int(value)}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create runtime comparison plots for gliner2-multi-v1 from README benchmark table.")
    parser.add_argument(
        "--readme",
        type=Path,
        default=repo_root / "README.md",
        help="Path to README.md with benchmark table.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "results/plots",
        help="Directory to write generated plots.",
    )
    args = parser.parse_args()

    rows = extract_model_rows(args.readme, TARGET_MODEL)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rps_err_path = args.out_dir / "gliner2_multi_v1_rps_err.png"
    latency_path = args.out_dir / "gliner2_multi_v1_latency.png"

    plot_rps_err(rows, rps_err_path)
    plot_latency(rows, latency_path)

    print(f"Generated: {rps_err_path}")
    print(f"Generated: {latency_path}")


if __name__ == "__main__":
    main()
