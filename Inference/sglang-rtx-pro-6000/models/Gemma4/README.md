# Gemma 4 26B Benchmarks on G4 (NVIDIA RTX PRO 6000 Blackwell)

This directory contains deployment manifests, benchmark execution scripts, and complete performance results for **Gemma 4 26B** (`google/gemma-4-26B-A4B-it`) served via **vLLM v0.26.0** on GCE **G4 instances** (`g4-standard-48`, 1x NVIDIA RTX PRO 6000 Blackwell GPU, 96 GB VRAM).

## Files Included

- [`vllm-gemma4-26b.yaml`](./vllm-gemma4-26b.yaml): Kubernetes Deployment spec with full optimization parameters (FP8 KV Cache, QWIX FP8 Quantization, async scheduling).
- [`run_benchmarks.sh`](./run_benchmarks.sh): Standalone automation script to execute the 20-run benchmark matrix (4 ISL/OSL workloads × 5 concurrency levels).
- [`benchmark_report.md`](./benchmark_report.md): Full performance report with summary tables, latency percentiles, Mermaid scaling charts, and saturation analysis.

## Quick Start

### 1. Launch Server via Docker
```bash
export HF_TOKEN="your_hf_token_here"

docker run -d --name vllm-gemma4-opt26b \
  --gpus all \
  --ipc=host \
  --net=host \
  -e HF_TOKEN="$HF_TOKEN" \
  -v /tmp/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  google/gemma-4-26B-A4B-it \
  --host 0.0.0.0 \
  --port 8000 \
  --seed 42 \
  --max-model-len 16384 \
  --max-num-seqs 1024 \
  --tensor-parallel-size 1 \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 1, "video": 0, "audio": 0}' \
  --block-size 256 \
  --additional-config '{"quantization": { "qwix": { "rules": [{ "module_path": ".*", "weight_qtype": "float8_e4m3fn", "act_qtype": "float8_e4m3fn"}]}}}' \
  --trust-remote-code \
  --enforce-eager
```

### 2. Run Benchmark Matrix
```bash
./run_benchmarks.sh
```
