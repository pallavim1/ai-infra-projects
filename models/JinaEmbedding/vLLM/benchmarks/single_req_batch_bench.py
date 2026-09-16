#!/usr/bin/env python3
"""
Single-HTTP-Request Multi-Prompt Batch Benchmark Script (Client)
Matches Zhemin's "Batch Request Testing: A single HTTP request contains multiple prompts"
  - Client config: Concurrently sends a batch of N random character prompts
    (prompts = [prompt_1, ..., prompt_N]) in a single HTTP inference request
    and measures the response latency in milliseconds (p50, p95, p99, avg) and throughput.
"""

import os
import time
import json
import urllib.request
from datetime import datetime, timezone

URL = os.environ.get("TARGET_URL", "http://jina-embedding-service:8000/prompt_c2")
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789     "


def make_random_chars(length: int, seed: int = 0) -> str:
    return "".join(CHARS[(i * 31 + 17 + seed) % len(CHARS)] for i in range(length))


def run_batch_benchmark():
    print(f"=== Running Multi-Prompt Single-HTTP-Request Batch Benchmark against {URL} ===", flush=True)
    # Warmup XLA shapes for batch sizes 1, 4, 8, 16 across 1KB - 4KB
    for kb in [1, 2, 3, 4]:
        for b in [1, 4, 8, 16]:
            prompts = [make_random_chars(kb * 1024, seed=s) for s in range(b)]
            payload = json.dumps({"text": prompts}).encode("utf-8")
            for _ in range(3):
                req = urllib.request.Request(
                    URL, data=payload, headers={"Content-Type": "application/json"}, method="POST"
                )
                urllib.request.urlopen(req, timeout=15).read()

    batch_results = []
    for kb in [1, 2, 3, 4]:
        for batch_size in [1, 4, 8, 16]:
            # Construct a single HTTP request containing `batch_size` random character strings
            prompts = [make_random_chars(kb * 1024, seed=s) for s in range(batch_size)]
            payload = json.dumps({"text": prompts}).encode("utf-8")
            iters = 40 if batch_size <= 8 else 25
            lats = []
            errs = 0
            t_start = time.perf_counter()
            for _ in range(iters):
                t0 = time.perf_counter()
                try:
                    req = urllib.request.Request(
                        URL, data=payload, headers={"Content-Type": "application/json"}, method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=15) as r:
                        r.read()
                        if r.status != 200:
                            errs += 1
                except Exception:
                    errs += 1
                lats.append((time.perf_counter() - t0) * 1000.0)

            total_s = time.perf_counter() - t_start
            lats.sort()
            p50 = round(lats[int(len(lats) * 0.50)], 2)
            p95 = round(lats[int(len(lats) * 0.95)], 2)
            p99 = round(lats[min(int(len(lats) * 0.99), len(lats) - 1)], 2)
            avg = round(sum(lats) / len(lats), 2)
            tput = round((iters * batch_size) / total_s, 2)
            row = {
                "payload_kb": kb,
                "chars": kb * 1024,
                "batch_size": batch_size,
                "throughput_rps": tput,
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "avg_ms": avg,
                "err_pct": round((errs / iters) * 100.0, 2),
            }
            batch_results.append(row)
            print(
                f"BATCH_SINGLE_REQ | {kb}KB ({kb*1024} chars) | Batch={batch_size:2d} | "
                f"Tput={tput:6.1f}/s | p50={p50:6.1f}ms | p95={p95:6.1f}ms | "
                f"p99={p99:6.1f}ms | avg={avg:6.1f}ms | err={row['err_pct']}%",
                flush=True,
            )

    out_path = f"single_req_batch_results_{TIMESTAMP}.json"
    with open(out_path, "w") as f:
        json.dump({"timestamp": TIMESTAMP, "batch_results": batch_results}, f, indent=2)
    print(f"Saved results to {out_path}", flush=True)


if __name__ == "__main__":
    run_batch_benchmark()
