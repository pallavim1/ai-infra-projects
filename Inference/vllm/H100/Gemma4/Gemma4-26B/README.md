# Gemma 4 26B MoE (`google/gemma-4-26B-A4B`) on 2x NVIDIA H100 GPUs (`TP=2`)

This directory contains the Kubernetes configurations, automation scripts, and verified benchmark results for serving **`google/gemma-4-26B-A4B`** (26B total parameters, 4B active parameters per token) on **2x NVIDIA H100-SXM5-80GB GPUs (`TP=2`)** from a single H100 node on GKE (`pallaviam-gke-h100-tcpx-cluster`, `us-east5-a`).

---

## Directory Structure

| File / Directory | Description |
| :--- | :--- |
| [`vllm_gemma4_26b_h100_2GPU.yaml`](./vllm_gemma4_26b_h100_2GPU.yaml) | StatefulSet (`vllm-gemma4-1node-2gpu`) and Service (`vllm-gemma4-h100-service`) deploying `google/gemma-4-26B-A4B` on **2x H100 GPUs (`TP=2`)** with FP8 weights + KV cache, Triton Attention (`TRITON_ATTN`), and host TCPXO plugin isolation for NVLink P2P (`P2P/CUMEM`). |
| [`sglang-gemma4-h100-2gpu-10k-500-benchmark-sweep.yaml`](./sglang-gemma4-h100-2gpu-10k-500-benchmark-sweep.yaml) | Kubernetes Benchmark Runner Pod executing `sglang.bench_serving --backend vllm` across concurrency **1, 8, 16, 32, 64, 128, 256, 512** for **ISL=10,240 (10K) / OSL=500**. |
| [`run_gemma4_26b_h100_2gpu_benchmark.sh`](./run_gemma4_26b_h100_2gpu_benchmark.sh) | Automation script to deploy both YAMLs and stream logs on `pallaviam-gke-h100-tcpx-cluster`. |
| [`results/sglang_benchmark_h100_2gpu_10k_500_report.md`](./results/sglang_benchmark_h100_2gpu_10k_500_report.md) | Full verified benchmark report for **10K Input / 500 Output** on **2x H100 GPUs (`TP=2`)**, including side-by-side comparison against **1x G4 GPU (`TP=1`)**. |
| `results/result_c{1,8,16,32,64,128,256,512}.json` | Raw JSON benchmark outputs directly from `sglang.bench_serving`. |

---

## Summary of 10K / 500 Results on 2x H100 GPUs (`TP=2`)

| Concurrency | Output Tok/s | Input Tok/s | Total Tok/s | Req/s | Median TTFT (ms) | P99 TTFT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Median E2E (s) | P99 E2E (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **18.88** | 386.65 | 405.53 | 0.04 | **267.47** | 531.46 | **52.39** | 53.19 | **26.41** | 27.07 |
| **8** | **135.90** | 2,782.91 | 2,918.81 | 0.27 | **961.33** | 2,703.29 | **55.56** | 57.90 | **28.89** | 30.32 |
| **16** | **261.46** | 5,354.13 | 5,615.58 | 0.52 | **779.69** | 3,346.99 | **59.47** | 60.97 | **30.34** | 30.96 |
| **32** | **505.75** | 10,356.80 | 10,862.56 | 1.01 | **1,223.96** | 6,182.36 | **56.60** | 66.50 | **31.58** | 34.16 |
| **64** | **921.04** | 18,860.99 | 19,782.03 | 1.84 | **1,298.26** | 12,646.08 | **56.80** | 77.64 | **34.28** | 39.83 |
| **128** | **1,533.44** | 31,401.72 | 32,935.16 | 3.07 | **1,759.33** | 20,239.88 | **64.78** | 101.39 | **41.71** | 51.86 |
| **256** 🏆 | **2,223.66** | 45,536.10 | 47,759.76 | 4.45 | **3,848.03** | 43,243.34 | **78.50** | 152.64 | **56.80** | 78.12 |
| **512** | **1,765.25** | 36,148.79 | 37,914.04 | 3.53 | **5,629.38** | 118,099.70 | **262.00** | 274.67 | **136.02** | 253.17 |
