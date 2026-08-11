# Jina Embeddings v2 on Google Cloud TPU v5e (GKE)

This repository provides an end-to-end guide, reproducible Kubernetes manifests, environment patches, and load-testing scripts for deploying and benchmarking **Jina Embeddings v2 Small** (`jinaai/jina-embeddings-v2-small-en`) on **Google Cloud TPU v5e** using **vLLM-TPU** on **Google Kubernetes Engine (GKE)**.

---

## 1. Problem Statement & Customer SLOs

Palo Alto Networks (PANW) runs real-time embedding inference for threat detection and MCP workloads:
- **Baseline Hardware**: `g2-standard-4` VM with 1x NVIDIA L4 GPU running ONNX Runtime (~40 ms P99 latency).
- **Target SLO**: **Round-trip P99 Latency < 50 ms** across byte sizes `[1024, 2048, 5120, 7168]` at **20–40 RPS**.
- **Traffic Patterns**:
  - `POST /prompt_c2`: Text payloads spanning 1 KB to 7 KB (1024, 2048, 5120, 7168 bytes).
  - `POST /mcp_c2`: Text payloads spanning 100 B to 400 B.

---

## 2. Performance & Benchmark Results

Benchmarked with PANW-provided k6 load-testing suite against a **single TPU v5e chip** (`ct5lp-hightpu-1t`, topology `1x1`):

### 20 RPS Fixed Payload Size Suite (`SUITE=payload_size @ 20 RPS`)

| Scenario | Payload Size | Target RPS | Achieved RPS | Min (ms) | Avg (ms) | P50 (ms) | P90 (ms) | P95 (ms) | **P99 (ms)** | Max (ms) | Errors | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `prompt_c2_1024b` | **1 KB (1024 B)** | 20.0 | 20.04 | 11.0 | 12.2 | 11.9 | 13.1 | 13.7 | **14.7 ms** | 15.4 | 0 | **PASSED (<50ms)** |
| `prompt_c2_2048b` | **2 KB (2048 B)** | 20.0 | 20.03 | 16.0 | 17.6 | 17.4 | 18.7 | 19.2 | **20.1 ms** | 21.0 | 0 | **PASSED (<50ms)** |
| `prompt_c2_5120b` | **5 KB (5120 B)** | 20.0 | 20.03 | 19.8 | 21.4 | 21.3 | 22.0 | 22.3 | **23.1 ms** | 24.1 | 0 | **PASSED (<50ms)** |
| `prompt_c2_7168b` | **7 KB (7168 B)** | 20.0 | 20.03 | 22.3 | 23.7 | 23.6 | 24.5 | 24.8 | **25.3 ms** | 31.7 | 0 | **PASSED (<50ms)** |

### 40 RPS Stress Test Suite (`SUITE=payload_size @ 40 RPS`)

| Scenario | Payload Size | Target RPS | Achieved RPS | Min (ms) | Avg (ms) | P50 (ms) | P90 (ms) | P95 (ms) | **P99 (ms)** | Max (ms) | Errors | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `prompt_c2_1024b` | **1 KB (1024 B)** | 40.0 | 40.06 | 11.1 | 12.1 | 12.0 | 12.7 | 13.1 | **15.2 ms** | 17.4 | 0 | **PASSED (<50ms)** |
| `prompt_c2_2048b` | **2 KB (2048 B)** | 40.0 | 40.05 | 16.2 | 17.2 | 16.9 | 18.1 | 19.1 | **20.7 ms** | 22.0 | 0 | **PASSED (<50ms)** |
| `prompt_c2_5120b` | **5 KB (5120 B)** | 40.0 | 40.05 | 19.6 | 21.7 | 21.5 | 23.5 | 24.2 | **26.3 ms** | 28.6 | 0 | **PASSED (<50ms)** |
| `prompt_c2_7168b` | **7 KB (7168 B)** | 40.0 | 40.06 | 22.7 | 28.4 | 27.1 | 34.3 | 36.2 | **42.5 ms** | 61.0 | 0 | **PASSED (<50ms)** |

### Hardware Comparison Summary

| Metric | PANW Baseline (NVIDIA L4 GPU) | Single Google Cloud TPU v5e |
| :--- | :--- | :--- |
| **P99 Latency (20 RPS)** | ~40 ms | **14.7 – 25.3 ms** (~1.6x–2.7x faster) |
| **P99 Latency (40 RPS)** | Saturated / >50 ms | **15.2 – 42.5 ms** (All payloads < 50 ms) |
| **Error Rate** | 0% | **0.00%** across all stages |
| **Serving Stack** | ONNX Runtime | `vllm-tpu` + JAX / Flax NNX |

---

## 3. Architecture & Architecture Details

```
+-------------------------------------------------------------------------------+
| GKE Node: ct5lp-hightpu-1t (Single v5e Host, 1x1 Topology)                    |
|                                                                               |
|  +-------------------------------------------------------------------------+  |
|  | Pod: jina-embeddings-v2-tpu                                             |  |
|  |                                                                         |  |
|  |  +---------------------------+       +-------------------------------+  |  |
|  |  | PANW Async Adapter Proxy  | ----> | vLLM TPU Engine (Port 8001)   |  |  |
|  |  | (aiohttp on Port 8000)    |       | jinaai/jina-embeddings-v2     |  |  |
|  |  | - POST /prompt_c2         |       | - runner: pooling             |  |  |
|  |  | - POST /mcp_c2            |       | - convert: embed              |  |  |
|  |  | - GET /health             |       | - max-model-len: 2048         |  |  |
|  |  +---------------------------+       +-------------------------------+  |  |
|  |                                                        |                |  |
|  |                                                        v                |  |
|  |                                               +-------------------+     |  |
|  |                                               | TPU v5e (1 chip)  |     |  |
|  |                                               | HBM: 16 GB        |     |  |
|  |                                               +-------------------+     |  |
|  +-------------------------------------------------------------------------+  |
+-------------------------------------------------------------------------------+
```

### Key Components:
1. **Jina Model Architecture**: `JinaBertForMaskedLM` / `JinaBertModel` encoder-only pooling network. Zero KV cache overhead.
2. **vLLM-TPU Serving Flags**:
   - `--runner pooling`: Uses encoder pooling pipeline instead of generative decoding.
   - `--convert embed`: Directs vLLM to treat masked-LM architecture as embedding model.
   - `--max-model-len 2048`: Allocates execution buckets up to 2048 tokens.
   - `--dtype float32`: Single precision embedding representations.
3. **Async Adapter Proxy**: Built with Python `aiohttp` to accept PANW JSON payload (`{"text": "..."}`), route to vLLM `/v1/embeddings` with `truncate_prompt_tokens: 2048`, and return standardized embeddings with keepalive HTTP pooling.
4. **Environment Patches (`scripts/setup_jina_env.py`)**:
   - Patches `transformers.onnx` and `transformers.pytorch_utils` for dynamic registration of custom `JinaBertForMaskedLM` modules.
   - Installs sitecustomize hooks so sub-processes inherit registrations.

---

## 4. Repository Structure

```
.
├── README.md                           # Main documentation & runbook
├── gke/
│   ├── cluster_setup.sh                # Automated VPC, GKE cluster & TPU pool provisioner
│   └── jina_v5e_deployment.yaml        # GKE Deployment, Service, ConfigMap & Secrets manifest
├── benchmarks/
│   ├── k6_ray_serve_test.js            # PANW k6 test suite (fixed payload sizes, RPS sweeps)
│   ├── analyze_k6_results.py           # Per-scenario results parser & Excel generator
│   └── run_benchmark.sh                # End-to-end benchmark execution script
└── results/
    ├── benchmark_report.md             # Markdown performance audit report
    ├── raw_scenario_summary.xlsx       # Excel report for 20 RPS benchmark run
    ├── raw_scenario_summary.json       # JSON summary for 20 RPS benchmark run
    ├── raw_40rps_scenario_summary.xlsx # Excel report for 40 RPS stress run
    └── raw_40rps_scenario_summary.json # JSON summary for 40 RPS stress run
```

---

## 5. Step-by-Step Reproduction Guide

### Step 1: Provision GKE Cluster with TPU v5e
```bash
cd gke/
chmod +x cluster_setup.sh
./cluster_setup.sh
```

### Step 2: Verify Pod Readiness
```bash
kubectl get pods -l app=jina-embeddings-v2 -w
```
Once the pod shows `1/1 Running`, verify the health check:
```bash
kubectl exec -it deployment/jina-embeddings-v2-tpu -- curl http://127.0.0.1:8000/health
```

### Step 3: Run the Benchmark Suite
```bash
cd ../benchmarks/
chmod +x run_benchmark.sh
./run_benchmark.sh
```

### Step 4: Inspect Reports & Excel Exports
All generated `.xlsx`, `.json`, and `.txt` reports are saved to `results/`.
