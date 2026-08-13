# Jina Embeddings v2 on Google Cloud TPU v5e (vLLM)
## Production Benchmarks, Economic Analysis & Deployment Guide

This directory contains the deployment configurations, load testing harnesses, live benchmark results, and Performance-per-Dollar economic models for serving `jinaai/jina-embeddings-v2-small-en` on **Google Cloud TPU v5e** using **vLLM**.

---

## 1. Directory Overview

```
models/JinaEmbedding/vLLM/
├── deploy/
│   ├── jina_v5e_deployment.yaml    # Kubernetes Deployment & Service with vLLM + high-performance proxy
│   └── cluster_setup.sh            # GKE cluster & TPU/CPU nodepool creation script
├── benchmarks/
│   ├── k6_ray_serve_test.js        # Original PANW k6 test suite
│   ├── k6_high_rps_saturation_test.js # Multi-payload saturation load script (1K, 2K, 5K, 7K)
│   ├── k6_1kb_saturation_test.js   # 1 KB dedicated saturation script (100–220 RPS)
│   ├── k6_2kb_saturation_test.js   # 2 KB dedicated saturation script (90–110 RPS)
│   ├── run_cpu_to_tpu_saturation_fp16.py # Automated test runner for FP16 precision
│   ├── analyze_k6_results.py       # Latency percentile & telemetry parser
│   └── generate_consolidated_excel.py # Multi-tab Excel workbook builder
├── reports/
│   ├── cpu_to_tpu_saturation_fp16_report.md  # FP16 live benchmark report across GKE network
│   ├── cpu_to_tpu_saturation_report.md       # FP32 live benchmark report across GKE network
│   ├── tpu_v5e_vs_l4_perf_per_dollar_analysis.md      # FP32 Performance/$ and TCO business case
│   ├── tpu_v5e_vs_l4_perf_per_dollar_analysis_fp16.md # FP16 Performance/$ and TCO business case
│   └── jina_embeddings_v2_tpu_v5e_benchmarks.xlsx     # Consolidated multi-tab Excel workbook
└── results/
    ├── fp32/                       # Raw summary JSONs and Excel sheets for FP32 live runs
    └── fp16/                       # Raw summary JSONs and Excel sheets for FP16 live runs
```

---

## 2. Key Benchmark Findings vs. NVIDIA L4 ($P_{99} < 50\text{ ms}$ SLA)

All tests were conducted from a dedicated **CPU Node Pool (`cpu-benchmark-pool`, `n2-standard-8`)** targeting the TPU v5e service over the real **GKE internal cluster network** with sustained 60-second stages:

| Payload Size | NVIDIA L4 Max RPS | TPU v5e Max RPS (FP32) | TPU v5e Max RPS (FP16) | TPU Throughput Advantage | TPU Net Cost Savings (3-Yr CUD) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1 KB** (`1024 B`) | **40 RPS** | **180 RPS** ($P_{99}=38.9\text{ms}$) | **160 RPS** ($P_{99}=28.9\text{ms}$) | **4.0x – 4.5x Higher** | **57.8% – 62.5% Cheaper** |
| **2 KB** (`2048 B`) | **40 RPS** | **90 RPS** ($P_{99}=28.4\text{ms}$) | **80 RPS** ($P_{99}=24.8\text{ms}$) | **2.0x – 2.25x Higher** | **15.6% – 25.0% Cheaper** |
| **5 KB** (`5120 B`) | **20 RPS** | **90 RPS** ($P_{99}=44.9\text{ms}$) | **80 RPS** ($P_{99}=39.5\text{ms}$) | **4.0x – 4.5x Higher** | **57.8% – 62.5% Cheaper** |
| **7 KB** (`7168 B`) | **10 RPS** | **80 RPS** ($P_{99}=41.5\text{ms}$) | **80 RPS** ($P_{99}=48.0\text{ms}$) | **8.0x Higher** | **78.9% Cheaper** |
| **Blended Avg** | **33.0 RPS** | **125.0 RPS** | **112.0 RPS** | **3.4x – 3.8x Higher** | **50.3% – 55.5% Cheaper** |

---

## 3. Quick Start & Reproduction

### Step 1: Deploy Jina Embeddings on TPU v5e
```bash
kubectl apply -f deploy/jina_v5e_deployment.yaml
kubectl wait --for=condition=ready pod -l app=jina-embeddings-v2-tpu --timeout=300s
```

### Step 2: Test Endpoint Health
```bash
curl -X POST http://jina-embedding-service:8000/prompt_c2 \
  -H "Content-Type: application/json" \
  -d '{"text": "Palo Alto Networks ATP Jina Embeddings Verification Test"}'
```

### Step 3: Run Full Benchmark Suite from CPU Nodepool
```bash
python3 benchmarks/run_cpu_to_tpu_saturation_fp16.py --duration 60s --phase all
```

---

## 4. Reports & Deliverables
* **Excel Workbook**: [`reports/jina_embeddings_v2_tpu_v5e_benchmarks.xlsx`](reports/jina_embeddings_v2_tpu_v5e_benchmarks.xlsx)
* **FP16 Benchmark Report**: [`reports/cpu_to_tpu_saturation_fp16_report.md`](reports/cpu_to_tpu_saturation_fp16_report.md)
* **FP32 Benchmark Report**: [`reports/cpu_to_tpu_saturation_report.md`](reports/cpu_to_tpu_saturation_report.md)
* **FP32 Economic Model**: [`reports/tpu_v5e_vs_l4_perf_per_dollar_analysis.md`](reports/tpu_v5e_vs_l4_perf_per_dollar_analysis.md)
* **FP16 Economic Model**: [`reports/tpu_v5e_vs_l4_perf_per_dollar_analysis_fp16.md`](reports/tpu_v5e_vs_l4_perf_per_dollar_analysis_fp16.md)
