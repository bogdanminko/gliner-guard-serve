"""
Quick benchmark for vLLM GLiNER Guard — sends async requests and reports latency.
No Locust required. Uses the same test data as the full benchmark.

Usage:
    python bench.py                                  # 128 requests, default host
    python bench.py --num-requests 500 --concurrency 32
    python bench.py --host http://gpu-server:8000
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time

import aiohttp

sys.path.insert(0, os.path.dirname(__file__))

MODEL = os.getenv("VLLM_MODEL", "/tmp/gliner-guard-uni-vllm")
HOST = os.getenv("GLINER_HOST", "http://localhost:8000")

PII_LABELS = ["person", "address", "email", "phone"]
SAFETY_LABELS = ["safe", "unsafe"]

SCHEMA = {
    "entities": PII_LABELS,
    "classifications": [
        {"task": "safety", "labels": SAFETY_LABELS}
    ],
}


def load_texts() -> list[str]:
    import csv
    data_dir = os.path.join(os.path.dirname(__file__), "..", "test-script")
    texts = []
    for fname, col in [("prompts.csv", "user_msg"), ("responses.csv", "assistant_msg")]:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row[col])
    if not texts:
        texts = ["John Smith lives at 123 Main St, email john@example.com, phone 555-0123"] * 100
    return texts


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    text: str,
) -> tuple[float, int]:
    payload = {
        "model": MODEL,
        "task": "plugin",
        "data": {
            "text": text,
            "schema": SCHEMA,
            "threshold": 0.4,
        },
    }
    start = time.perf_counter()
    async with session.post(url, json=payload) as resp:
        await resp.read()
        status = resp.status
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, status


async def run(host: str, num_requests: int, concurrency: int, warmup: int):
    url = f"{host}/pooling"
    texts = load_texts()

    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        print(f"Warming up ({warmup} requests)...")
        sem = asyncio.Semaphore(concurrency)

        async def bounded(text: str):
            async with sem:
                return await send_request(session, url, text)

        await asyncio.gather(*[bounded(texts[i % len(texts)]) for i in range(warmup)])

        # Timed run
        print(f"Running {num_requests} requests (concurrency={concurrency})...")
        start = time.perf_counter()
        results = await asyncio.gather(
            *[bounded(texts[i % len(texts)]) for i in range(num_requests)]
        )
        total_elapsed = time.perf_counter() - start

    latencies = sorted([r[0] for r in results])
    errors = sum(1 for r in results if r[1] != 200)
    n = len(latencies)

    print()
    print("=" * 60)
    print("vLLM GLiNER Guard — Quick Benchmark")
    print("=" * 60)
    print(f"  Host:           {host}")
    print(f"  Requests:       {num_requests}")
    print(f"  Concurrency:    {concurrency}")
    print(f"  Elapsed:        {total_elapsed:.2f}s")
    print(f"  RPS:            {num_requests / total_elapsed:.1f}")
    print(f"  P50:            {latencies[int(n * 0.50)]:.1f}ms")
    print(f"  P95:            {latencies[int(n * 0.95)]:.1f}ms")
    print(f"  P99:            {latencies[int(n * 0.99)]:.1f}ms")
    print(f"  Mean:           {statistics.mean(latencies):.1f}ms")
    print(f"  Errors:         {errors}/{num_requests} ({errors/num_requests*100:.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Quick vLLM GLiNER Guard benchmark")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    asyncio.run(run(args.host, args.num_requests, args.concurrency, args.warmup))


if __name__ == "__main__":
    main()
