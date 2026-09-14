# Gemma 4 26B Benchmarks on G4 (NVIDIA RTX PRO 6000 Blackwell)

This directory contains deployment manifests, benchmark execution scripts, and complete performance results for **Gemma 4 26B** (`google/gemma-4-26B-A4B-it`) served via **vLLM v0.26.0** on GCE **G4 instances** (`g4-standard-48`, 1x NVIDIA RTX PRO 6000 Blackwell GPU, 96 GB VRAM).

## Files Included

- **[Google Slides Presentation Deck](https://docs.google.com/presentation/d/1VmMbPQSLC-9ZBAqypqno8g6GYZ43arnNkPHWXOdfLzk/edit)**: 7-slide executive deck covering Gemma 4 26B MoE 10K/500 serving benchmarks & `vllm bench serve` vs. `sglang.bench_serving` methodology analysis ([PDF](./results/Gemma4_26B_G4_10K_500_Benchmark_Deck.pdf) | [PPTX](./results/Gemma4_26B_G4_10K_500_Benchmark_Deck.pptx)).
- [`results/sglang_benchmark_10k_300_optimized_report.md`](./results/sglang_benchmark_10k_300_optimized_report.md): **10K Input / 300 Output Optimized Concurrency Sweep Report (C=1 to 512)** meeting customer's **$\le 4.7\text{s}$ P99 Latency SLA up to C=8 (`3.45s` P99, `519.83` output tok/s, `+77.4%` gain)**.
- [`results/sglang_benchmark_10k_300_report.md`](./results/sglang_benchmark_10k_300_report.md): **10K Input / 300 Output Baseline Concurrency Sweep Report (C=1 to 512)** using `sglang.bench_serving` on GKE `g4-standard-48`.
- [`vllm_gemma4_26b_g4_1GPU_optimized_10k_300.yaml`](./vllm_gemma4_26b_g4_1GPU_optimized_10k_300.yaml): Optimized GKE StatefulSet + Service manifest for `google/gemma-4-26B-A4B` (`--quantization=fp8`, `--enable-chunked-prefill`, `--max-num-batched-tokens=4096`, `--limit-mm-per-prompt={"image":0,"video":0,"audio":0}`, `--enable-prefix-caching`).
- [`sglang-gemma4-10k-300-benchmark-sweep-optimized.yaml`](./sglang-gemma4-10k-300-benchmark-sweep-optimized.yaml): Automated GKE benchmark runner pod executing `sglang.bench_serving` against the optimized 10K/300 vLLM server.
- [`sglang-gemma4-10k-300-benchmark-sweep.yaml`](./sglang-gemma4-10k-300-benchmark-sweep.yaml): Automated GKE benchmark runner pod executing `sglang.bench_serving` against the baseline 10K/300 vLLM server.
- [`results/vllm_vs_sglang_client_comparison.md`](./results/vllm_vs_sglang_client_comparison.md): **Side-by-side comparison of `vllm bench serve` vs. `sglang.bench_serving`** on the same vLLM G4 server (`1x g4-standard-48`).
- [`results/vllm_benchmark_10k_500_report.md`](./results/vllm_benchmark_10k_500_report.md): **10K Input / 500 Output Native vLLM Concurrency Sweep Report (C=1 to 512)** using `vllm bench serve` on GKE `g4-standard-48` (`pm-g4-sglang-cluster`).
- [`results/sglang_benchmark_10k_500_report.md`](./results/sglang_benchmark_10k_500_report.md): **10K Input / 500 Output Concurrency Sweep Report (C=1 to 512)** using `sglang.bench_serving` on GKE `g4-standard-48` (`pm-g4-sglang-cluster`).
- [`vllm_gemma4_26b_g4_1GPU.yaml`](./vllm_gemma4_26b_g4_1GPU.yaml): GKE StatefulSet + Service manifest for serving `google/gemma-4-26B-A4B` with vLLM on `g4-standard-48` (TP=1, FP8).
- [`sglang-gemma4-10k-500-benchmark-sweep.yaml`](./sglang-gemma4-10k-500-benchmark-sweep.yaml): Automated GKE benchmark runner pod executing `sglang.bench_serving` across concurrencies 1, 8, 16, 32, 64, 128, 256, 512.
- [`vllm-gemma4-10k-500-benchmark-sweep.yaml`](./vllm-gemma4-10k-500-benchmark-sweep.yaml): Automated GKE benchmark runner pod executing native `vllm bench serve` across concurrencies 1, 8, 16, 32, 64, 128, 256, 512.
- [`run_vllm_benchmark_10k_500.sh`](./run_vllm_benchmark_10k_500.sh): Standalone shell script to run native `vllm bench serve` against a vLLM server (or inside Docker) for the 10K Input / 500 Output concurrency sweep (C=1 to 512).
- [`results/gemma4_26b_g4_benchmark_sweep_report.md`](./results/gemma4_26b_g4_benchmark_sweep_report.md): **New Structured Extended Benchmark Report** (Peak throughput summary, sweet-spot SLA analysis, and full concurrency sweep breakdown).
- [`vllm-gemma4-26b.yaml`](./vllm-gemma4-26b.yaml): Kubernetes Deployment spec with full optimization parameters (FP8 KV Cache, QWIX FP8 Quantization, async scheduling).
- [`run_benchmarks.sh`](./run_benchmarks.sh): Standalone automation script to execute the 20-run benchmark matrix (4 ISL/OSL workloads × 5 concurrency levels).
- [`benchmark_report_concurrency_matrix.md`](./benchmark_report_concurrency_matrix.md): Max-concurrency benchmark report (`--request-rate inf --max-concurrency $C`, Aug 8, 2026).
- [`benchmark_report.md`](./benchmark_report.md): Fixed-rate benchmark report (`--request-rate $C`, Aug 7, 2026).
- **Raw Benchmark Results**:
  - [`results/sglang_benchmark_10k_500_report.md`](./results/sglang_benchmark_10k_500_report.md): Report for 10K Input / 500 Output sweep.
  - [`results/gemma4_26b_manual_results_20260808_071935.txt`](./results/gemma4_26b_manual_results_20260808_071935.txt): Full 20-run raw console output for max-concurrency mode.
  - [`results/gemma4_26b_manual_results.txt`](./results/gemma4_26b_manual_results.txt): Full 20-run raw console output for fixed-rate mode.

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
