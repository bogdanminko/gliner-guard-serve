# Ray Serve Sweep Results 2026-04-18

This file captures the completed no-Docker Ray Serve dynamic batching sweep for
GLiNER Guard on Runpod A100.

- Completed at `2026-04-18 10:42:04 UTC`
- Completed at `2026-04-18 13:42:04 Europe/Moscow`
- Matrix size: `64/64`
- Source summary: [ray-serve-sweep-2026-04-18-final.csv](./ray-serve-sweep-2026-04-18-final.csv)
- Precision modes covered: `bf16`, `fp16`
- Models covered: `uni`, `bi`
- Protocols covered: `REST`, `gRPC`
- Sweep configs covered: `B1` .. `B8`
- Selection rule for PR-facing `results/ray-serve/*.csv`: highest `RPS` within
  each `(protocol, model, dtype)` slice
- `REST` no-batch baselines remain tracked separately and were not part of this
  sweep

## Selected winners

- `uni bf16`: REST `B5` (`batch_size=32`, `batch_timeout=0.05`) -> `66.07 RPS`,
  `p50/p95/p99 = 1500/1700/1800 ms`; gRPC `B3`
  (`batch_size=16`, `batch_timeout=0.01`) -> `23.27 RPS`,
  `p50/p95/p99 = 42/46/48 ms`
- `uni fp16`: REST `B5` (`batch_size=32`, `batch_timeout=0.05`) -> `71.07 RPS`,
  `p50/p95/p99 = 1400/1600/1700 ms`; gRPC `B1`
  (`batch_size=8`, `batch_timeout=0.01`) -> `23.50 RPS`,
  `p50/p95/p99 = 41/47/51 ms`
- `bi bf16`: REST `B7` (`batch_size=64`, `batch_timeout=0.05`) -> `67.01 RPS`,
  `p50/p95/p99 = 1400/1900/2200 ms`; gRPC `B1`
  (`batch_size=8`, `batch_timeout=0.01`) -> `17.44 RPS`,
  `p50/p95/p99 = 56/62/67 ms`
- `bi fp16`: REST `B7` (`batch_size=64`, `batch_timeout=0.05`) -> `68.32 RPS`,
  `p50/p95/p99 = 1500/1800/2000 ms`; gRPC `B3`
  (`batch_size=16`, `batch_timeout=0.01`) -> `17.68 RPS`,
  `p50/p95/p99 = 56/60/65 ms`

## Notes

- Best overall REST result in the sweep: `uni fp16 B5` at `71.07 RPS`
- Best overall gRPC result in the sweep: `uni fp16 B1` at `23.50 RPS`
- Final sweep row was `gRPC + bi + fp16 + B8`, which completed successfully and
  confirmed the matrix was fully closed
