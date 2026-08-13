# Palo Alto Networks (PANW) - TPU Inference Benchmarks & Architectures

This repository contains production benchmarks, deployment manifests, performance-per-dollar economic models, and reproduction runbooks for evaluating Google Cloud TPUs (v5e / v6e) against GPU baselines for Palo Alto Networks (PANW) workloads.

---

## Directory Structure

```
panw-tpu-inference/
└── models/
    └── JinaEmbedding/
        └── vLLM/
            ├── deploy/       # Kubernetes manifests, GKE cluster setup, and serving configs
            ├── benchmarks/   # k6 load scripts, Python test harnesses, and telemetry analyzers
            ├── reports/      # Executive reports, Performance/$ models, and Excel workbooks
            └── results/      # Raw and summarized benchmark telemetry (FP32 & FP16)
                ├── fp32/
                └── fp16/
```

---

## Models Evaluated

### 1. `jina-embeddings-v2-small-en` (vLLM on Cloud TPU v5e)
* **Model**: `jinaai/jina-embeddings-v2-small-en` (512-dim embedding representation)
* **Serving Framework**: vLLM (v0.26.0) on Cloud TPU v5e (`ct5lp-hightpu-1t`)
* **Precision Modes**: FP32 & FP16 (`bfloat16`)
* **Strict SLA Target**: $P_{99} \text{ Round-Trip Latency} < 50\text{ ms}$
* **Key Finding**: TPU v5e delivers **2.0x to 8.0x higher throughput** and **50% to 78.9% cost reduction** compared to NVIDIA L4 GPUs.

👉 See full documentation, setup guides, and benchmark reports in [`models/JinaEmbedding/vLLM/`](models/JinaEmbedding/vLLM/README.md).
