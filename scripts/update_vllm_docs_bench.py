#!/usr/bin/env python3
"""Update docs/vllm experiment markdown files from results/vllm/*_stats.csv."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


MARKER_START = "<!-- AUTO_VLLM_RESULTS:START -->"
MARKER_END = "<!-- AUTO_VLLM_RESULTS:END -->"


@dataclass(frozen=True)
class DocConfig:
    key: str
    doc_path: str
    results_dir: str
    order: tuple[str, ...]
    aliases: dict[str, str]
    notes: tuple[str, ...] = ()


DOCS: dict[str, DocConfig] = {
    "v1": DocConfig(
        key="v1",
        doc_path="docs/vllm/vllm-experiments-v1.md",
        results_dir="results/vllm/gliner-guard_v1",
        order=(
            "bfloat16-eager",
            "float16-eager",
            "bfloat16-cudagraph",
            "float16-cudagraph",
            "bfloat16-eager-batch16k",
            "bfloat16-eager-mem90",
        ),
        aliases={},
    ),
    "v2": DocConfig(
        key="v2",
        doc_path="docs/vllm/vllm-experiments-v2.md",
        results_dir="results/vllm/gliner-guard_v2",
        order=(
            "sched-safe",
            "sched-balanced",
            "sched-aggressive",
            "sched-short",
            "multi-4x",
        ),
        aliases={},
    ),
    "v3": DocConfig(
        key="v3",
        doc_path="docs/vllm/vllm-experiments-v3.md",
        results_dir="results/vllm/gliner-guard_v3",
        order=(
            "litserve-baseline",
            "single-safe",
            "single-dense",
            "single-short",
            "multi-2x",
            "multi-4x",
        ),
        aliases={"sched-safe": "single-safe"},
        notes=(
            "В каталоге результатов `single-safe` сохранён под именем `sched-safe`.",
            "Дополнительно показан `litserve-baseline` как внешний ориентир для сравнения.",
        ),
    ),
    "v4": DocConfig(
        key="v4",
        doc_path="docs/vllm/vllm_experiments-v4-final.md",
        results_dir="results/vllm/gliner_guard_v4-final",
        order=(
            "baseline-len8192-tokens262k",
            "len4096-tokens131k",
            "len2048-tokens131k",
            "len4096-tokens65k",
            "len2048-tokens65k",
            "len2048-tokens49k",
            "best-no-chunked-prefill-len2048-tokens131k",
            "best-enforce-eager-len2048-tokens131k",
        ),
        aliases={},
    ),
}


def parse_stats_csv(path: Path) -> dict[str, float | int] | None:
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("Name") != "Aggregated":
                continue
            total = int(row["Request Count"])
            failed = int(row["Failure Count"])
            return {
                "rps": float(row["Requests/s"]),
                "p50": int(row["50%"]),
                "p95": int(row["95%"]),
                "p99": int(row["99%"]),
                "err_pct": (failed / total * 100.0) if total else 0.0,
            }
    return None


def collect_rows(repo_root: Path, config: DocConfig) -> list[tuple[str, dict[str, float | int]]]:
    results_dir = repo_root / config.results_dir
    rows: dict[str, dict[str, float | int]] = {}
    for path in sorted(results_dir.glob("*_stats.csv")):
        raw_name = path.name.removesuffix("_stats.csv")
        display_name = config.aliases.get(raw_name, raw_name)
        stats = parse_stats_csv(path)
        if stats is None:
            continue
        rows[display_name] = stats

    ordered: list[tuple[str, dict[str, float | int]]] = []
    seen = set()
    for name in config.order:
        if name in rows:
            ordered.append((name, rows[name]))
            seen.add(name)
    for name in sorted(rows):
        if name not in seen:
            ordered.append((name, rows[name]))
    return ordered


def render_block(config: DocConfig, rows: list[tuple[str, dict[str, float | int]]]) -> str:
    body: list[str] = [MARKER_START, "", f"Источник: `{config.results_dir}`", ""]
    if not rows:
        body.extend(["Результаты пока не найдены.", "", MARKER_END])
        return "\n".join(body)

    best_rps = max(row["rps"] for _, row in rows)
    best_p50 = min(row["p50"] for _, row in rows)
    best_p95 = min(row["p95"] for _, row in rows)
    best_p99 = min(row["p99"] for _, row in rows)
    failures = [name for name, row in rows if row["err_pct"] > 0]

    def fmt_num(value: float | int, best: float | int, pattern: str) -> str:
        rendered = pattern.format(value)
        return f"**{rendered}**" if value == best else rendered

    body.extend(
        [
            "| Experiment | RPS | P50 (ms) | P95 (ms) | P99 (ms) | Err rate (%) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in rows:
        body.append(
            "| {name} | {rps} | {p50} | {p95} | {p99} | {err:.2f} |".format(
                name=name,
                rps=fmt_num(row["rps"], best_rps, "{:.1f}"),
                p50=fmt_num(row["p50"], best_p50, "{}"),
                p95=fmt_num(row["p95"], best_p95, "{}"),
                p99=fmt_num(row["p99"], best_p99, "{}"),
                err=row["err_pct"],
            )
        )

    best_throughput = next(name for name, row in rows if row["rps"] == best_rps)
    best_tail = next(name for name, row in rows if row["p95"] == best_p95)
    body.extend(
        [
            "",
            "Короткий вывод:",
            f"- лучший throughput: `{best_throughput}`",
            f"- лучший p95: `{best_tail}`",
            "- конфиги с failures: "
            + (", ".join(f"`{name}`" for name in failures) if failures else "нет"),
        ]
    )
    if config.notes:
        body.extend(["", "Примечания:"])
        body.extend(f"- {note}" for note in config.notes)
    body.extend(["", MARKER_END])
    return "\n".join(body)


def update_doc(repo_root: Path, config: DocConfig) -> None:
    doc_path = repo_root / config.doc_path
    text = doc_path.read_text()
    block = render_block(config, collect_rows(repo_root, config))
    pattern = re.compile(
        rf"{re.escape(MARKER_START)}.*?{re.escape(MARKER_END)}",
        re.DOTALL,
    )
    if not pattern.search(text):
        raise SystemExit(f"Markers not found in {config.doc_path}")
    doc_path.write_text(pattern.sub(block, text, count=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--doc",
        choices=["all", *DOCS.keys()],
        default="all",
        help="Which doc block to update.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    keys = DOCS.keys() if args.doc == "all" else [args.doc]
    for key in keys:
        update_doc(repo_root, DOCS[key])


if __name__ == "__main__":
    main()
