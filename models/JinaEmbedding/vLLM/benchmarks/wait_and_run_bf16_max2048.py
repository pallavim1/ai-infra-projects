#!/usr/bin/env python3
import time
import urllib.request
import json
import subprocess
import sys
from datetime import datetime, timezone

URL = "http://jina-embedding-service:8000/prompt_c2"
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789     "

def make_random_chars(length, seed=0):
    return "".join(CHARS[(i * 31 + 17 + seed) % len(CHARS)] for i in range(length))

print(f"=== [BF16 + max-model-len=2048] Waiting for {URL} to become ready ({TIMESTAMP}) ===", flush=True)
start_wait = time.time()
while True:
    try:
        req = urllib.request.Request(
            URL,
            data=json.dumps({"text": make_random_chars(1024)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                print(f"BF16 Server is READY after {int(time.time() - start_wait)}s!", flush=True)
                break
    except Exception:
        pass
    if int(time.time() - start_wait) % 15 == 0:
        print(f"Still waiting ({int(time.time() - start_wait)}s elapsed)...", flush=True)
    time.sleep(3)

print("\n=== [Step 1] Warming up XLA shapes for 1KB-7KB and Multi-Prompt Batch Sizes (1, 4, 8, 16) ===", flush=True)
for kb in [1, 2, 3, 4, 5, 7]:
    payload = json.dumps({"text": make_random_chars(kb * 1024)}).encode("utf-8")
    for _ in range(3):
        req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=15).read()

for kb in [1, 2, 3, 4]:
    for b in [1, 4, 8, 16]:
        prompts = [make_random_chars(kb * 1024, seed=s) for s in range(b)]
        payload = json.dumps({"text": prompts}).encode("utf-8")
        for _ in range(3):
            req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=15).read()

print("\n=== [Step 2] Running Zhemin's Multi-Prompt Single-Request Batch Test (Batch = 1, 4, 8, 16) ===", flush=True)
batch_results = []
for kb in [1, 2, 3, 4]:
    for batch_size in [1, 4, 8, 16]:
        prompts = [make_random_chars(kb * 1024, seed=s) for s in range(batch_size)]
        payload = json.dumps({"text": prompts}).encode("utf-8")
        iters = 40 if batch_size <= 8 else 25
        lats = []
        errs = 0
        t_start = time.perf_counter()
        for _ in range(iters):
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
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
            "err_pct": round((errs / iters) * 100.0, 2)
        }
        batch_results.append(row)
        print(f"BATCH_SINGLE_REQ | {kb}KB ({kb*1024} chars) | Batch={batch_size:2d} | Tput={tput:6.1f}/s | p50={p50:6.1f}ms | p95={p95:6.1f}ms | p99={p99:6.1f}ms | avg={avg:6.1f}ms | err={row['err_pct']}%", flush=True)

batch_json_path = f"/workspace/benchmark_runs/bf16_single_req_batch_{TIMESTAMP}.json"
with open(batch_json_path, "w") as f:
    json.dump({"timestamp": TIMESTAMP, "dtype": "bfloat16", "max_model_len": 2048, "batch_results": batch_results}, f, indent=2)
print(f"Saved Multi-Prompt Single-Request Batch results to: {batch_json_path}", flush=True)

print("\n=== [Step 3] Launching /workspace/run_customer_matching_k6.py (BF16) ===", flush=True)
ret = subprocess.call(["python3", "-u", "/workspace/run_customer_matching_k6.py"])
sys.exit(ret)
