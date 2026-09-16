#!/usr/bin/env python3
import subprocess
import json
import os
import time
from datetime import datetime, timezone

TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"/workspace/benchmark_runs/customer_k6_{TIMESTAMP}"
os.makedirs(OUT_DIR, exist_ok=True)
K6_SCRIPT = "/workspace/k6_customer_match.js"
HTTP_URL = "http://jina-embedding-service:8000/prompt_c2"

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

def main():
    print(f"=== Starting Customer-Matching k6 + vLLM Benchmark Suite ({TIMESTAMP}) ===")

    # Warmup
    run_k6({"MODE": "concurrency", "VUS": "4", "PAYLOAD_KB": "1", "DURATION": "4s", "HTTP_URL": HTTP_URL}, "warmup")

    # 1. Concurrency Request Testing (1KB, 2KB, 3KB, 4KB at Concurrency 1, 4, 8, 16)
    print("\n--- SUITE 1: Concurrency Request Testing (k6 constant-vus) ---")
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
            print(f"{kb}KB | Conc={vus:2d} | Tput={tput:6.1f}/s | p50={lat['p50']:6.1f}ms | p95={lat['p95']:6.1f}ms | p99={lat['p99']:6.1f}ms | err={err_pct:.2f}%")
            time.sleep(1)

    # 2. Multi-Payload Concurrent Sweep (50, 60, 70, 80, 90 RPS across 1KB, 2KB, 5KB, 7KB)
    print("\n--- SUITE 2: Multi-Payload Concurrent Sweep (50-90 RPS across 1KB, 2KB, 5KB, 7KB) ---")
    suite2 = []
    for rps in [50, 60, 70, 80, 90]:
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
        print(f"RPS={rps:3d} | 1KB p99={l1['p99']:5.1f}ms (err {e1}%) | 2KB p99={l2['p99']:5.1f}ms (err {e2}%) | 5KB p99={l5['p99']:5.1f}ms (err {e5}%) | 7KB p99={l7['p99']:5.1f}ms (err {e7}%)")
        time.sleep(1)

    # 3. Dedicated RPS Saturation Sweeps (1KB, 2KB, 3KB)
    def run_dedicated_sweep(kb: int, rps_list: list):
        print(f"\n--- SUITE 3 ({kb}KB): Dedicated RPS Saturation Sweep ---")
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
            print(f"{kb}KB | Target={rps:3d} | Achieved={achieved:6.1f} | P50={lat['p50']:6.1f}ms | P99={lat['p99']:6.1f}ms | Err={err_pct:.2f}% | {sla}")
            time.sleep(1)
        return rows

    suite3_1kb = run_dedicated_sweep(1, [100, 120, 140, 160, 180, 190, 200, 220])
    suite3_2kb = run_dedicated_sweep(2, [70, 80, 90, 95, 100, 110])
    suite3_3kb = run_dedicated_sweep(3, [40, 50, 60, 70, 80, 90])

    final_json = {
        "timestamp": TIMESTAMP,
        "suite1_concurrency": suite1,
        "suite2_multi_payload": suite2,
        "suite3_1kb_saturation": suite3_1kb,
        "suite3_2kb_saturation": suite3_2kb,
        "suite3_3kb_saturation": suite3_3kb,
    }
    out_file = os.path.join(OUT_DIR, f"customer_k6_results_{TIMESTAMP}.json")
    with open(out_file, "w") as f:
        json.dump(final_json, f, indent=2)
    print(f"\nCOMPLETED! Results saved to: {out_file}")

if __name__ == "__main__":
    main()
