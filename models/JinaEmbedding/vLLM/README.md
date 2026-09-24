# Jina Embeddings v2 on Google Cloud TPU v6e & TPU v5e (vLLM)
## Production Benchmarks, Economic Analysis & Deployment Guide

This directory contains the deployment configurations, load testing harnesses, live benchmark results (`Online k6` + `Batch`), and Performance-per-Dollar / TCO economic models for serving `jinaai/jina-embeddings-v2-small-en` on **Google Cloud TPU v6e (`ct6e-standard-1t`)** and **Google Cloud TPU v5e (`ct5lp-hightpu-1t`)** using **vLLM (`0.26.0`)** vs **NVIDIA L4 (`g2-standard-4`, Triton + TensorRT)**.

---

## 1. Latest Reports & Consolidated Excel Workbooks

- **TPU v6e (`FP32`) vs NVIDIA L4 & TPU v5e (`Performance & TCO`)**: [`benchmarks/reports/06A_tpu_v6e_fp32_vs_l4_and_v5e_performance_comparison.md`](benchmarks/reports/06A_tpu_v6e_fp32_vs_l4_and_v5e_performance_comparison.md)
- **TPU v6e (`BF16`) vs NVIDIA L4 & TPU v5e (`Performance & TCO`)**: [`benchmarks/reports/06B_tpu_v6e_bf16_vs_l4_and_v5e_performance_comparison.md`](benchmarks/reports/06B_tpu_v6e_bf16_vs_l4_and_v5e_performance_comparison.md)
- **TPU v5e (`FP32`) vs NVIDIA L4 (`Performance & TCO`)**: [`benchmarks/reports/05A_tpu_v5e_fp32_vs_l4_performance_comparison.md`](benchmarks/reports/05A_tpu_v5e_fp32_vs_l4_performance_comparison.md)
- **TPU v5e (`BF16`) vs NVIDIA L4 (`Performance & TCO`)**: [`benchmarks/reports/05B_tpu_v5e_bf16_vs_l4_performance_comparison.md`](benchmarks/reports/05B_tpu_v5e_bf16_vs_l4_performance_comparison.md)
- **Consolidated 8-Tab Excel Workbook (`TPU v6e + TPU v5e + NVIDIA L4`)**: [`benchmarks/reports/ATP_AIC2_Benchmarks_TPU_v6e_v5e_FP32_and_BF16_vs_L4.xlsx`](benchmarks/reports/ATP_AIC2_Benchmarks_TPU_v6e_v5e_FP32_and_BF16_vs_L4.xlsx)
- **Raw TPU v6e JSON Results**:
  - [`benchmarks/results/v6e_float32_complete_results_20260924_184725.json`](benchmarks/results/v6e_float32_complete_results_20260924_184725.json)
  - [`benchmarks/results/v6e_bfloat16_complete_results_20260924_175613.json`](benchmarks/results/v6e_bfloat16_complete_results_20260924_175613.json)

---

## 2. Key Benchmark Findings: TPU v6e vs TPU v5e vs NVIDIA L4 ($P_{99} \le 50\text{ ms}$ SLA)

All tests were conducted from a dedicated **CPU Node Pool (`cpu-benchmark-pool`, `n2-standard-8`)** on `pm-panw-jina-cluster` targeting the TPU v6e (`pm-panw-jina-v6e-pool`, `ct6e-standard-1t`) and TPU v5e (`pm-panw-jina-tpu-pool`, `ct5lp-hightpu-1t`) services over the real **GKE internal cluster network**:

| Payload Size | NVIDIA L4 Max RPS | TPU v5e (`FP32` / `BF16`) | **TPU v6e (`FP32` / `BF16`)** | **TPU v6e vs L4 Advantage** | **TPU v6e Cost Reduction vs L4 (1-Yr / 3-Yr CUD)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1 KB** (`1024 B`) | **70 RPS** | 140 / 160 RPS | **160–180 (`FP32`) / 160–200 (`BF16`) RPS** | **2.29x – 2.86x Higher** | **15.0% (`1-Yr`) – 38.9% (`3-Yr`) Cheaper** |
| **2 KB** (`2048 B`) | **40 RPS** | 70 / 90 RPS | **100 (`FP32`) / 110 (`BF16`) RPS** | **2.50x – 2.75x Higher** | **11.7% (`1-Yr`) – 36.6% (`3-Yr`) Cheaper** |
| **3 KB** (`3072 B`) | **30 RPS** | 70 / 90 RPS | **110 (`FP32`) / 100 (`BF16`) RPS** | **3.33x – 3.67x Higher** | **27.1% (`1-Yr`) – 52.5% (`3-Yr`) Cheaper** |
| **5 KB** (`5120 B`) | **20 RPS** | 70 / 90 RPS | **100 (`FP32`) / 110 (`BF16`) RPS** | **5.00x – 5.50x Higher** | **55.9% (`1-Yr`) – 68.3% (`3-Yr`) Cheaper** *(29.9% On-Demand)* |
| **7 KB** (`7168 B`) | **10 RPS** | 70 / 90 RPS | **100 (`FP32`) / 110 (`BF16`) RPS** | **10.0x – 11.0x Higher** | **77.9% (`1-Yr`) – 84.2% (`3-Yr`) Cheaper** *(64.9% On-Demand)* |

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
