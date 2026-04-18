# Ray Serve Sweep Snapshot 2026-04-18

This file snapshots the no-Docker Ray Serve batching sweep from Runpod after a
manual budget stop.

- Initial snapshot at `2026-04-18 06:07:50 UTC`
- Initial snapshot at `2026-04-18 09:07:50 Europe/Moscow`
- Budget stop finalized at `2026-04-18 06:19:54 UTC`
- Budget stop finalized at `2026-04-18 09:19:54 Europe/Moscow`
- Source pod summary: `/root/gliner-guard-serve-pr-sweep-20260417/artifacts/raw-results/sweep-summary.csv`
- Completed rows captured: `48`
- Matrix size: `64`
- Last completed run before stop: `ray-grpc-bf16-B8-bi-prompts-run1`
- Runner and Ray Serve processes were stopped immediately after the `48th` row was written
- Remaining matrix not executed: `16/64` runs, which is the entire `bi + fp16` block

The tracked CSV snapshot is stored next to this note:

- [ray-serve-sweep-2026-04-18-partial.csv](./ray-serve-sweep-2026-04-18-partial.csv)

This snapshot is intentionally stored under `docs/`, not `results/`, so it does not
pollute `make bench-readme` output while the sweep is incomplete.
