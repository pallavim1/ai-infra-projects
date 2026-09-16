#!/usr/bin/env python3
"""
Customer-Matching Benchmark Suite for Jina AI Embedding Model + TPU v5e + vLLM
Matches exact test tables in 'ATP AIC2 Benchmarks' (gid=1161755388 & gid=1972899730):
  1. Concurrency Testing (1KB, 2KB, 3KB, 4KB random chars at Concurrency 1, 4, 8, 16)
  2. Multi-Payload Concurrent Sweep (50, 60, 70, 80, 90 RPS across 1KB, 2KB, 5KB, 7KB)
  3. Dedicated RPS Saturation Sweeps:
     - 1KB Dedicated Saturation (100, 120, 140, 160, 180, 190, 200, 220 RPS)
     - 2KB Dedicated Saturation (70, 80, 90, 95, 100, 110 RPS)
     - 3KB Dedicated Saturation (40, 50, 60, 70, 80, 90 RPS)
"""

import asyncio
import aiohttp
import time
import json
import random
import string
import os
import statistics
from datetime import datetime, timezone

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"/workspace/benchmark_runs/customer_match_{TIMESTAMP}"
os.makedirs(OUT_DIR, exist_ok=True)

# Direct vLLM endpoint and Proxy endpoint
TARGET_URL = os.environ.get("TARGET_URL", "http://jina-embedding-service:8000/prompt_c2")
SLA_P99_MS = 50.0

def make_random_payload(num_chars: int) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits + "     ", k=num_chars))

PAYLOAD_POOL = {
    "1KB": [make_random_payload(1024) for _ in range(32)],
    "2KB": [make_random_payload(2048) for _ in range(32)],
    "3KB": [make_random_payload(3072) for _ in range(32)],
    "4KB": [make_random_payload(4096) for _ in range(32)],
    "5KB": [make_random_payload(5120) for _ in range(32)],
    "7KB": [make_random_payload(7168) for _ in range(32)],
}

def calc_percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

async def warmup(session):
    print("Running warmup requests across payload sizes (1KB - 7KB)...")
    for label, pool in PAYLOAD_POOL.items():
        for i in range(3):
            try:
                async with session.post(TARGET_URL, json={"text": pool[i]}) as resp:
                    await resp.read()
            except Exception:
                pass

async def run_closed_loop_concurrency(session, payload_label: str, concurrency: int, duration_sec: float = 15.0):
    """
    Closed-loop concurrency benchmark: N workers continuously send requests back-to-back
    for `duration_sec`, measuring throughput (req/s), p50, p95, p99, avg, and error rate.
    """
    pool = PAYLOAD_POOL[payload_label]
    latencies = []
    errors = 0
    stop_time = time.perf_counter() + duration_sec

    async def worker(worker_id: int):
        nonlocal errors
        idx = worker_id
        while time.perf_counter() < stop_time:
            text = pool[idx % len(pool)]
            idx += 1
            t0 = time.perf_counter()
            try:
                async with session.post(TARGET_URL, json={"text": text}) as resp:
                    await resp.read()
                    dt = (time.perf_counter() - t0) * 1000.0
                    if resp.status == 200:
                        latencies.append(dt)
                    else:
                        errors += 1
            except Exception:
                errors += 1

    t_start = time.perf_counter()
    await asyncio.gather(*(worker(w) for w in range(concurrency)))
    actual_dur = time.perf_counter() - t_start

    sorted_lat = sorted(latencies)
    total_reqs = len(sorted_lat) + errors
    tput = len(sorted_lat) / actual_dur if actual_dur > 0 else 0.0
    p50 = calc_percentile(sorted_lat, 50)
    p90 = calc_percentile(sorted_lat, 90)
    p95 = calc_percentile(sorted_lat, 95)
    p99 = calc_percentile(sorted_lat, 99)
    avg = statistics.mean(sorted_lat) if sorted_lat else 0.0
    err_pct = (errors / total_reqs * 100.0) if total_reqs > 0 else 0.0

    res = {
        "payload": payload_label,
        "concurrency": concurrency,
        "throughput_rps": round(tput, 2),
        "p50_ms": round(p50, 2),
        "p90_ms": round(p90, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "avg_ms": round(avg, 2),
        "total_requests": total_reqs,
        "errors": errors,
        "error_pct": round(err_pct, 2)
    }
    print(f"[Concurrency] {payload_label} | Conc={concurrency:2d} | Tput={tput:6.1f}/s | p50={p50:6.1f}ms | p95={p95:6.1f}ms | p99={p99:6.1f}ms | err={err_pct:.2f}%")
    return res

async def run_open_loop_rps(session, payload_labels, target_rps: int, duration_sec: float = 15.0):
    """
    Open-loop constant arrival rate benchmark at `target_rps` for `duration_sec`.
    Supports single payload label (e.g. ['1KB']) or multi-payload concurrent (['1KB','2KB','5KB','7KB']).
    """
    interval = 1.0 / float(target_rps)
    total_to_send = int(target_rps * duration_sec)
    results_by_label = {lbl: {"latencies": [], "errors": 0} for lbl in payload_labels}
    tasks = []
    sem = asyncio.Semaphore(300)

    async def one_req(lbl: str, text: str):
        async with sem:
            t0 = time.perf_counter()
            try:
                async with session.post(TARGET_URL, json={"text": text}, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    await resp.read()
                    dt = (time.perf_counter() - t0) * 1000.0
                    if resp.status == 200:
                        results_by_label[lbl]["latencies"].append(dt)
                    else:
                        results_by_label[lbl]["errors"] += 1
            except Exception:
                results_by_label[lbl]["errors"] += 1

    t_start = time.perf_counter()
    for i in range(total_to_send):
        lbl = payload_labels[i % len(payload_labels)]
        pool = PAYLOAD_POOL[lbl]
        text = pool[i % len(pool)]
        tasks.append(asyncio.create_task(one_req(lbl, text)))
        target_next = t_start + (i + 1) * interval
        sleep_dur = target_next - time.perf_counter()
        if sleep_dur > 0:
            await asyncio.sleep(sleep_dur)

    await asyncio.gather(*tasks)
    actual_dur = time.perf_counter() - t_start

    all_lat = []
    total_err = 0
    per_label_summary = {}
    for lbl in payload_labels:
        lats = sorted(results_by_label[lbl]["latencies"])
        errs = results_by_label[lbl]["errors"]
        all_lat.extend(lats)
        total_err += errs
        tot_lbl = len(lats) + errs
        per_label_summary[lbl] = {
            "count": tot_lbl,
            "errors": errs,
            "error_pct": round(errs / tot_lbl * 100.0, 2) if tot_lbl > 0 else 0.0,
            "p50_ms": round(calc_percentile(lats, 50), 2),
            "p90_ms": round(calc_percentile(lats, 90), 2),
            "p95_ms": round(calc_percentile(lats, 95), 2),
            "p99_ms": round(calc_percentile(lats, 99), 2),
        }

    all_lat.sort()
    achieved_rps = len(all_lat) / actual_dur if actual_dur > 0 else 0.0
    p50 = calc_percentile(all_lat, 50)
    p90 = calc_percentile(all_lat, 90)
    p99 = calc_percentile(all_lat, 99)
    sla_pass = (p99 <= SLA_P99_MS) and (total_err == 0) and (achieved_rps >= target_rps * 0.92)

    summary = {
        "target_rps": target_rps,
        "achieved_rps": round(achieved_rps, 2),
        "p50_ms": round(p50, 2),
        "p90_ms": round(p90, 2),
        "p99_ms": round(p99, 2),
        "errors": total_err,
        "sla_status": "✅ PASS" if sla_pass else "⚠️ SATURATED",
        "per_payload": per_label_summary
    }
    if len(payload_labels) == 1:
        print(f"[Dedicated {payload_labels[0]}] Target={target_rps:3d} RPS | Achieved={achieved_rps:6.1f} | P50={p50:6.1f}ms | P99={p99:6.1f}ms | Err={total_err} | {summary['sla_status']}")
    else:
        p1 = per_label_summary["1KB"]["p99_ms"]
        p2 = per_label_summary["2KB"]["p99_ms"]
        p5 = per_label_summary["5KB"]["p99_ms"]
        p7 = per_label_summary["7KB"]["p99_ms"]
        print(f"[Multi-Payload] Target={target_rps:3d} RPS | 1KB p99={p1:5.1f}ms | 2KB p99={p2:5.1f}ms | 5KB p99={p5:5.1f}ms | 7KB p99={p7:5.1f}ms | Err={total_err}")
    return summary

async def main():
    connector = aiohttp.TCPConnector(limit=500, limit_per_host=500, keepalive_timeout=60)
    async with aiohttp.ClientSession(connector=connector) as session:
        await warmup(session)

        print("\n=== SUITE 1: Concurrency Request Testing (1KB, 2KB, 3KB, 4KB at Conc 1, 4, 8, 16) ===")
        suite1_results = []
        for payload_lbl in ["1KB", "2KB", "3KB", "4KB"]:
            for conc in [1, 4, 8, 16]:
                res = await run_closed_loop_concurrency(session, payload_lbl, conc, duration_sec=12.0)
                suite1_results.append(res)
                await asyncio.sleep(1.0)

        print("\n=== SUITE 2: Multi-Payload Concurrent Sweep (50-90 RPS across 1KB, 2KB, 5KB, 7KB) ===")
        suite2_results = []
        for rps in [50, 60, 70, 80, 90]:
            res = await run_open_loop_rps(session, ["1KB", "2KB", "5KB", "7KB"], rps, duration_sec=15.0)
            suite2_results.append(res)
            await asyncio.sleep(2.0)

        print("\n=== SUITE 3A: 1KB Dedicated Saturation Sweep ===")
        suite3_1kb = []
        for rps in [100, 120, 140, 160, 180, 190, 200, 220]:
            res = await run_open_loop_rps(session, ["1KB"], rps, duration_sec=15.0)
            suite3_1kb.append(res)
            await asyncio.sleep(2.0)

        print("\n=== SUITE 3B: 2KB Dedicated Saturation Sweep ===")
        suite3_2kb = []
        for rps in [70, 80, 90, 95, 100, 110]:
            res = await run_open_loop_rps(session, ["2KB"], rps, duration_sec=15.0)
            suite3_2kb.append(res)
            await asyncio.sleep(2.0)

        print("\n=== SUITE 3C: 3KB Dedicated Saturation Sweep ===")
        suite3_3kb = []
        for rps in [40, 50, 60, 70, 80, 90]:
            res = await run_open_loop_rps(session, ["3KB"], rps, duration_sec=15.0)
            suite3_3kb.append(res)
            await asyncio.sleep(2.0)

        full_output = {
            "timestamp": TIMESTAMP,
            "model": "jinaai/jina-embeddings-v2-small-en",
            "hardware": "Cloud TPU v5e (ct5lp-hightpu-1t)",
            "vllm_config": "--runner pooling --convert embed --trust-remote-code --max-model-len 8192 --max-num-batched-tokens 8192 --dtype float32",
            "suite1_concurrency": suite1_results,
            "suite2_multi_payload": suite2_results,
            "suite3_1kb_saturation": suite3_1kb,
            "suite3_2kb_saturation": suite3_2kb,
            "suite3_3kb_saturation": suite3_3kb,
        }

        json_path = os.path.join(OUT_DIR, f"customer_matching_results_{TIMESTAMP}.json")
        with open(json_path, "w") as f:
            json.dump(full_output, f, indent=2)
        print(f"\nSaved full JSON results to: {json_path}")

if __name__ == "__main__":
    asyncio.run(main())
