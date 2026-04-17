#!/usr/bin/env python3
"""Parse results/**/*.csv (Locust stats) and print a markdown benchmark table."""

import csv
import glob
import os
import sys

RUNTIME_ORDER = [
    "pytorch",
    "pytorch-fp16",
    "pytorch-bf16-rest-nobatch",
    "pytorch-bf16-rest-dynbatch",
    "pytorch-bf16-grpc-dynbatch",
    "onnx-cuda-fp16",
    "onnx-trt-fp16",
    "onnx-int8-cpu",
]


def runtime_key(r):
    try:
        return RUNTIME_ORDER.index(r)
    except ValueError:
        return len(RUNTIME_ORDER)


def parse_csv(path: str) -> dict | None:
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Name"] == "Aggregated":
                total = int(row["Request Count"])
                failed = int(row["Failure Count"])
                return {
                    "rps": float(row["Requests/s"]),
                    "p50": int(row["50%"]),
                    "p95": int(row["95%"]),
                    "p99": int(row["99%"]),
                    "failure_rate": failed / total if total else 0.0,
                }
    return None


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    files = glob.glob(os.path.join(results_dir, "**", "*.csv"), recursive=True)
    if not files:
        print("No CSV files found in results/", file=sys.stderr)
        sys.exit(1)

    rows = []
    for path in files:
        rel = os.path.relpath(path, results_dir)
        parts = rel.replace("\\", "/").split("/")
        if len(parts) == 3:
            serving, model, runtime_file = parts
            runtime = os.path.splitext(runtime_file)[0]
        else:
            serving, model, runtime = "-", "-", os.path.splitext(os.path.basename(path))[0]
        stats = parse_csv(path)
        if stats is None:
            continue
        rows.append((serving, model, runtime, stats))

    rows.sort(key=lambda r: (r[0], r[1], runtime_key(r[2])))

    col_model = max(len("Model"), max(len(r[1]) for r in rows))
    col_serving = max(len("Serving"), max(len(r[0]) for r in rows))
    col_runtime = max(len("Runtime"), max(len(r[2]) for r in rows))

    def row_line(model, serving, runtime, rps, p50, p95, p99, failures):
        return (
            f"| {model:<{col_model}} | {serving:<{col_serving}} | {runtime:<{col_runtime}}"
            f" | {rps:>7} | {p50:>8} | {p95:>8} | {p99:>8} | {failures:>8} |"
        )

    def blank_line(runtime, rps, p50, p95, p99, failures):
        return (
            f"| {'':<{col_model}} | {'':<{col_serving}} | {runtime:<{col_runtime}}"
            f" | {rps:>7} | {p50:>8} | {p95:>8} | {p99:>8} | {failures:>8} |"
        )

    sep = (
        f"| {'-' * col_model} | {'-' * col_serving} | {'-' * col_runtime}"
        f" | ------: | -------: | -------: | -------: | -------: |"
    )
    section_sep = (
        f"| {'':<{col_model}} | {'':<{col_serving}} | {'':<{col_runtime}}"
        f" |         |          |          |          |          |"
    )

    best_rps = max(r[3]["rps"] for r in rows)
    best_p50 = min(r[3]["p50"] for r in rows)
    best_p95 = min(r[3]["p95"] for r in rows)
    best_p99 = min(r[3]["p99"] for r in rows)

    def fmt(val, best, fmt_str):
        s = fmt_str.format(val)
        return f"**{s}**" if val == best else s

    lines = [
        row_line("Model", "Serving", "Runtime", "RPS", "P50 (ms)", "P95 (ms)", "P99 (ms)", "Err rate (%)"),
        sep,
    ]

    prev_serving = None
    prev_model = None
    for serving, model, runtime, stats in rows:
        rps = fmt(stats["rps"], best_rps, "{:.1f}")
        p50 = fmt(stats["p50"], best_p50, "{}")
        p95 = fmt(stats["p95"], best_p95, "{}")
        p99 = fmt(stats["p99"], best_p99, "{}")
        failures = f"{stats['failure_rate'] * 100:.2f}"

        if prev_serving is not None and serving != prev_serving:
            lines.append(section_sep)

        if model == prev_model and serving == prev_serving:
            lines.append(blank_line(runtime, rps, p50, p95, p99, failures))
        else:
            lines.append(row_line(model, serving, runtime, rps, p50, p95, p99, failures))

        prev_serving = serving
        prev_model = model

    print("\n".join(lines))


if __name__ == "__main__":
    main()
