# A100-SXM4-80GB benchmark artifacts

This directory contains raw benchmark evidence packs captured from Runpod A100
runs. These artifacts are committed for PR review and auditability; curated
summary tables live in `docs/` and small comparison CSVs live in `results/`.

## Evidence packs

| Directory | Scope | Runs |
|---|---|---:|
| `ray-serve-uni-4worker-2026-04-29/` | Ray Serve UNI REST/gRPC, 4 replicas on one A100, batches 16/32/64 | 18 |

Keep each evidence pack self-contained. Do not rewrite raw CSV, HTML, log, or
plot files after capture unless the checksum manifest is regenerated at the
same time.
