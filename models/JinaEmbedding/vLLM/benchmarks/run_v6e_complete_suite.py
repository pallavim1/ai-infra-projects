#!/usr/bin/env python3
import subprocess
import urllib.request
import json
import os
import sys
import time
from datetime import datetime, timezone

DTYPE = sys.argv[1] if len(sys.argv) > 1 else "bfloat16"
HTTP_URL = sys.argv[2] if len(sys.argv) > 2 else "http://jina-embeddings-v2-tpu-v6e-bf16-svc:8000/prompt_c2"
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"/workspace/benchmark_runs/v6e_{DTYPE}_{TIMESTAMP}"
os.makedirs(OUT_DIR, exist_ok=True)
K6_SCRIPT = "/workspace/k6_customer_match.js"

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789     "

def make_random_chars(length, seed=0):
    return "".join(CHARS[(i * 31 + 17 + seed) % len(CHARS)] for i in range(length))

print(f"=== [TPU v6e ({DTYPE}) + max-model-len=2048] Waiting for {HTTP_URL} to become ready ({TIMESTAMP}) ===", flush=True)
start_wait = time.time()
while True:
    try:
        req = urllib.request.Request(
            HTTP_URL,
            data=json.dumps({"text": make_random_chars(1024)}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                print(f"TPU v6e ({DTYPE}) Server is READY after {int(time.time() - start_wait)}s!", flush=True)
                break
    except Exception:
        pass
    if int(time.time() - start_wait) % 15 == 0:
        print(f"Still waiting ({int(time.time() - start_wait)}s elapsed)...", flush=True)
    time.sleep(3)

print("\n=== [Step 1] Warming up XLA shapes for 1KB-7KB and Multi-Prompt Batch Sizes (1, 4, 8, 16, 32, 64, 128) ===", flush=True)
for kb in [1, 2, 3, 4, 5, 7]:
    payload = json.dumps({"text": make_random_chars(kb * 1024)}).encode("utf-8")
    for _ in range(4):
        req = urllib.request.Request(HTTP_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=20).read()

for kb in [1, 2, 3, 4]:
    for b in [1, 4, 8, 16]:
        prompts = [make_random_chars(kb * 1024, seed=s) for s in range(b)]
        payload = json.dumps({"text": prompts}).encode("utf-8")
        for _ in range(3):
            req = urllib.request.Request(HTTP_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=20).read()

print(f"\n=== [Step 2] Running Zhemin's Multi-Prompt Single-Request Batch Test (Batch = 1, 4, 8, 16) on TPU v6e ({DTYPE}) ===", flush=True)
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
                req = urllib.request.Request(HTTP_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=20) as r:
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

print(f"\n=== [Step 2B] Running High-Batch Token Sweep (B=1, 8, 16, 32, 64, 128 across 128..2048 tokens) on TPU v6e ({DTYPE}) ===", flush=True)
token_batch_results = []
for tok_len in [128, 256, 512, 1024, 2048]:
    char_len = tok_len * 4
    for batch_size in [1, 8, 16, 32, 64, 128]:
        prompts = [make_random_chars(char_len, seed=s) for s in range(batch_size)]
        payload = json.dumps({"text": prompts}).encode("utf-8")
        # Warmup 2 iters
        for _ in range(2):
            try:
                req = urllib.request.Request(HTTP_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
                urllib.request.urlopen(req, timeout=30).read()
            except Exception:
                pass
        iters = 15 if batch_size <= 32 else 10
        lats = []
        errs = 0
        t_start = time.perf_counter()
        for _ in range(iters):
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(HTTP_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=30) as r:
                    r.read()
                    if r.status != 200:
                        errs += 1
            except Exception:
                errs += 1
            lats.append((time.perf_counter() - t0) * 1000.0)
        total_s = time.perf_counter() - t_start
        lats.sort()
        p50 = round(lats[int(len(lats) * 0.50)], 2)
        p99 = round(lats[min(int(len(lats) * 0.99), len(lats) - 1)], 2)
        emb_per_sec = round((iters * batch_size) / total_s, 2)
        tok_per_sec = round(emb_per_sec * tok_len, 1)
        trow = {
            "tokens": tok_len,
            "batch_size": batch_size,
            "embeddings_per_sec": emb_per_sec,
            "tokens_per_sec": tok_per_sec,
            "p50_ms": p50,
            "p99_ms": p99,
            "err_pct": round((errs / iters) * 100.0, 2)
        }
        token_batch_results.append(trow)
        print(f"TOKEN_BATCH | L={tok_len:4d} | B={batch_size:3d} | Emb/s={emb_per_sec:7.1f} | Tok/s={tok_per_sec:9.1f} | p50={p50:6.1f}ms | p99={p99:6.1f}ms", flush=True)

def run_k6(env_vars: dict, summary_name: str):
    summary_path = os.path.join(OUT_DIR, f"{summary_name}.json")
    cmd = [
        "k6", "run",
        "--quiet",
        "--no-thresholds",
        "--summary-trend-stats=min,avg,med,p(90),p(95),p(99),max",
        f"--summary-export={summary_path}",
    ]
    for k, v in env_vars.items():
        cmd.extend(["-e", f"{k}={v}"])
    cmd.append(K6_SCRIPT)
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            return json.load(f)
    return {}

def extract_metric(data: dict, metric_name: str):
    m = data.get("metrics", {}).get(metric_name, {})
    return {
        "avg": round(m.get("avg", 0.0), 2),
        "p50": round(m.get("med", 0.0), 2),
        "p90": round(m.get("p(90)", 0.0), 2),
        "p95": round(m.get("p(95)", 0.0), 2),
        "p99": round(m.get("p(99)", 0.0), 2),
        "max": round(m.get("max", 0.0), 2),
    }

print(f"\n=== [Step 3] Starting Customer-Matching k6 Online Suite on TPU v6e ({DTYPE}) ===", flush=True)
run_k6({"MODE": "concurrency", "VUS": "4", "PAYLOAD_KB": "1", "DURATION": "4s", "HTTP_URL": HTTP_URL}, "warmup")

print("\n--- SUITE 1: Concurrency Request Testing (k6 constant-vus) ---", flush=True)
suite1 = []
for kb in [1, 2, 3, 4]:
    for vus in [1, 4, 8, 16]:
        data = run_k6({
            "MODE": "concurrency",
            "VUS": str(vus),
            "PAYLOAD_KB": str(kb),
            "DURATION": "10s",
            "HTTP_URL": HTTP_URL,
        }, f"conc_{kb}kb_vus{vus}")
        reqs = data.get("metrics", {}).get("http_reqs", {})
        tput = round(reqs.get("rate", 0.0), 2)
        count = reqs.get("count", 0)
        fails = data.get("metrics", {}).get("http_req_failed", {}).get("passes", 0)
        err_pct = round((fails / count * 100.0) if count > 0 else 0.0, 2)
        lat = extract_metric(data, "http_req_duration")
        row = {
            "payload": f"{kb}KB ({kb*1024} chars)",
            "concurrency": vus,
            "throughput_rps": tput,
            "p50_ms": lat["p50"],
            "p95_ms": lat["p95"],
            "p99_ms": lat["p99"],
            "avg_ms": lat["avg"],
            "err_pct": err_pct,
        }
        suite1.append(row)
        print(f"{kb}KB | Conc={vus:2d} | Tput={tput:6.1f}/s | p50={lat['p50']:6.1f}ms | p95={lat['p95']:6.1f}ms | p99={lat['p99']:6.1f}ms | err={err_pct:.2f}%", flush=True)
        time.sleep(1)

print("\n--- SUITE 2: Multi-Payload Concurrent Sweep (50-160 RPS across 1KB, 2KB, 5KB, 7KB) ---", flush=True)
suite2 = []
for rps in [50, 60, 70, 80, 90, 100, 110, 120, 140, 160]:
    data = run_k6({
        "MODE": "multi_rps",
        "RPS": str(rps),
        "DURATION": "12s",
        "HTTP_URL": HTTP_URL,
    }, f"multi_{rps}rps")
    l1 = extract_metric(data, "lat_1kb")
    l2 = extract_metric(data, "lat_2kb")
    l5 = extract_metric(data, "lat_5kb")
    l7 = extract_metric(data, "lat_7kb")
    e1 = round(data.get("metrics", {}).get("err_1kb", {}).get("value", 0.0) * 100.0, 2)
    e2 = round(data.get("metrics", {}).get("err_2kb", {}).get("value", 0.0) * 100.0, 2)
    e5 = round(data.get("metrics", {}).get("err_5kb", {}).get("value", 0.0) * 100.0, 2)
    e7 = round(data.get("metrics", {}).get("err_7kb", {}).get("value", 0.0) * 100.0, 2)
    row = {
        "rps": rps,
        "1kb_p99_ms": l1["p99"], "1kb_err_pct": e1,
        "2kb_p99_ms": l2["p99"], "2kb_err_pct": e2,
        "5kb_p99_ms": l5["p99"], "5kb_err_pct": e5,
        "7kb_p99_ms": l7["p99"], "7kb_err_pct": e7,
    }
    suite2.append(row)
    print(f"RPS={rps:3d} | 1KB p99={l1['p99']:5.1f}ms (err {e1}%) | 2KB p99={l2['p99']:5.1f}ms (err {e2}%) | 5KB p99={l5['p99']:5.1f}ms (err {e5}%) | 7KB p99={l7['p99']:5.1f}ms (err {e7}%)", flush=True)
    time.sleep(1)

def run_dedicated_sweep(kb: int, rps_list: list):
    print(f"\n--- SUITE 3 ({kb}KB): Dedicated RPS Saturation Sweep ---", flush=True)
    rows = []
    for rps in rps_list:
        data = run_k6({
            "MODE": "dedicated_rps",
            "PAYLOAD_KB": str(kb),
            "RPS": str(rps),
            "DURATION": "12s",
            "HTTP_URL": HTTP_URL,
        }, f"ded_{kb}kb_{rps}rps")
        reqs = data.get("metrics", {}).get("http_reqs", {})
        achieved = round(reqs.get("rate", 0.0), 2)
        count = reqs.get("count", 0)
        fails = data.get("metrics", {}).get("http_req_failed", {}).get("passes", 0)
        err_pct = round((fails / count * 100.0) if count > 0 else 0.0, 2)
        lat = extract_metric(data, "http_req_duration")
        sla = "✅ PASS" if (lat["p99"] <= 50.0 and err_pct == 0.0 and achieved >= rps * 0.90) else "⚠️ SATURATED"
        row = {
            "target_rps": rps,
            "achieved_rps": achieved,
            "p50_ms": lat["p50"],
            "p90_ms": lat["p90"],
            "p99_ms": lat["p99"],
            "err_pct": err_pct,
            "sla": sla,
        }
        rows.append(row)
        print(f"{kb}KB | Target={rps:3d} | Achieved={achieved:6.1f} | P50={lat['p50']:6.1f}ms | P99={lat['p99']:6.1f}ms | Err={err_pct:.2f}% | {sla}", flush=True)
        time.sleep(1)
    return rows

suite3_1kb = run_dedicated_sweep(1, [100, 120, 140, 160, 180, 190, 200, 220, 240, 260, 280, 300, 320, 350, 380, 400])
suite3_2kb = run_dedicated_sweep(2, [70, 80, 90, 95, 100, 110, 120, 140, 160, 180, 200, 220])
suite3_3kb = run_dedicated_sweep(3, [40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160])

final_json = {
    "timestamp": TIMESTAMP,
    "accelerator": "TPU v6e (ct6e-standard-1t, 1 chip)",
    "dtype": DTYPE,
    "max_model_len": 2048,
    "batch_single_req_results": batch_results,
    "token_batch_results": token_batch_results,
    "suite1_concurrency": suite1,
    "suite2_multi_payload": suite2,
    "suite3_1kb_saturation": suite3_1kb,
    "suite3_2kb_saturation": suite3_2kb,
    "suite3_3kb_saturation": suite3_3kb,
}
out_file = os.path.join(OUT_DIR, f"v6e_{DTYPE}_complete_results_{TIMESTAMP}.json")
with open(out_file, "w") as f:
    json.dump(final_json, f, indent=2)
print(f"\nCOMPLETED TPU v6e ({DTYPE})! Results saved to: {out_file}", flush=True)
