#!/usr/bin/env python3
"""
CPU Node to TPU v5e Service Saturation Benchmark Runner (FP16 Precision)
Runs from: cpu-benchmark-runner (n2-standard-8 in cpu-benchmark-pool)
Targets:   http://jina-embedding-service:8000/prompt_c2 (TPU v5e in pm-panw-jina-tpu-pool)
Zone:      europe-west4-b
Precision: FP16 (vLLM --dtype float16)
Target SLA: Strict P99 Round-Trip Latency < 50 ms
"""

import subprocess, json, time, os, sys, argparse
from datetime import datetime

RESULTS_DIR = '/workspace/cpu_to_tpu_results_fp16'
os.makedirs(RESULTS_DIR, exist_ok=True)
TPU_SERVICE_URL = 'http://jina-embedding-service:8000'

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)

def run_single(js_script, suite, rps, duration, prefix, extra=''):
    raw = f'{RESULTS_DIR}/{prefix}_{rps}rps.ndjson'
    summary = f'{RESULTS_DIR}/{prefix}_{rps}rps_summary'
    
    cmd_k6 = (
        f'cd /workspace && rm -f {raw} && '
        f'k6 run --no-thresholds --out json={raw} '
        f'-e SUITE={suite} -e ENDPOINT=prompt_c2 -e HTTP_URL={TPU_SERVICE_URL} '
        f'-e STAGE_DURATION={duration} -e PAYLOAD_RPS={rps} {extra} {js_script}'
    )
    log(f'[K6 RUN] {cmd_k6}')
    subprocess.run(cmd_k6, shell=True, check=False)
    
    cmd_ana = f'python3 /workspace/analyze_k6_results.py {raw} --out {summary}'
    log(f'[ANALYZE] {cmd_ana}')
    subprocess.run(cmd_ana, shell=True, check=False)
    
    try:
        with open(f'{summary}.json', 'r') as f:
            data = json.load(f)
            
        print(f'\n┌' + '─'*98 + '┐', flush=True)
        print(f'│ Results for {rps:3d} RPS (Duration: {duration}) - FP16 Mode (CPU -> TPU v5e)' + ' '*30 + '│', flush=True)
        print(f'├' + '─'*22 + '┬' + '─'*14 + '┬' + '─'*10 + '┬' + '─'*10 + '┬' + '─'*10 + '┬' + '─'*10 + '┬' + '─'*16 + '┤', flush=True)
        print(f'│ {"Scenario":20s} │ {"Achieved RPS":12s} │ {"P50 (ms)":8s} │ {"P90 (ms)":8s} │ {"P95 (ms)":8s} │ {"P99 (ms)":8s} │ {"SLA (<50ms)":14s} │', flush=True)
        print(f'├' + '─'*22 + '┼' + '─'*14 + '┼' + '─'*10 + '┼' + '─'*10 + '┼' + '─'*10 + '┼' + '─'*10 + '┼' + '─'*16 + '┤', flush=True)
        for sc, m in sorted(data.items()):
            if isinstance(m, dict) and 'metrics' in m:
                met = m['metrics'].get('prompt_c2_latency_ms', {})
                p50, p90, p95, p99, ach = met.get('p50',0), met.get('p90',0), met.get('p95',0), met.get('p99',0), met.get('achieved_rps',0)
                status = '✅ PASS' if p99 < 50.0 else '⚠️ SATURATED'
                print(f'│ {sc:20s} │ {ach:12.2f} │ {p50:8.1f} │ {p90:8.1f} │ {p95:8.1f} │ {p99:8.1f} │ {status:14s} │', flush=True)
        print(f'└' + '─'*22 + '┴' + '─'*14 + '┴' + '─'*10 + '┴' + '─'*10 + '┴' + '─'*10 + '┴' + '─'*10 + '┴' + '─'*16 + '┘\n', flush=True)
    except Exception as e:
        log(f'[WARN] Could not parse summary: {e}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', default='60s')
    parser.add_argument('--phase', default='all', choices=['all', 'multi', '2k', '1k'])
    args = parser.parse_args()
    
    log('==========================================================================')
    log(f'  CPU NODE -> TPU v5e Saturation Benchmark Suite (FP16 Precision)')
    log(f'  Client Runner:  cpu-benchmark-runner (Node: cpu-benchmark-pool)')
    log(f'  TPU Endpoint:   {TPU_SERVICE_URL}')
    log(f'  Stage Duration: {args.duration}')
    log(f'  Results Dir:    {RESULTS_DIR}')
    log('==========================================================================')
    
    if args.phase in ['all', 'multi']:
        log('\n>>> Phase 1: Multi-Payload Sweep (50 -> 90 RPS) <<<')
        for rps in [50, 60, 70, 80, 90]:
            run_single('k6_high_rps_saturation_test.js', 'payload_size', rps, args.duration, 'multi_fp16')
        
    if args.phase in ['all', '2k']:
        log('\n>>> Phase 2: 2 KB Dedicated Saturation Sweep (90, 95, 100, 110 RPS) <<<')
        for rps in [90, 95, 100, 110]:
            run_single('k6_2kb_saturation_test.js', 'payload_size', rps, args.duration, '2kb_fp16', '-e PAYLOAD_VUS=200')
        
    if args.phase in ['all', '1k']:
        log('\n>>> Phase 3: 1 KB Dedicated Saturation Sweep (100 -> 220 RPS) <<<')
        for rps in [100, 120, 140, 160, 180, 190, 200, 220]:
            run_single('k6_1kb_saturation_test.js', 'payload_size', rps, args.duration, '1kb_fp16', '-e PAYLOAD_VUS=250')
        
    log('==========================================================================')
    log(f'✅ All FP16 CPU -> TPU Sweeps Complete! Results in {RESULTS_DIR}')
    log('==========================================================================')

if __name__ == '__main__':
    main()
