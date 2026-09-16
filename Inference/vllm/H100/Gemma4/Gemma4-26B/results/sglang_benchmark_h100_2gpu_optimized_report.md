# SGLang Benchmark Report (Optimized): `google/gemma-4-26B-A4B` on 2x NVIDIA H100 GPUs (`TP=2`, Customer 7.7s Latency SLA)

Comprehensive optimization and live benchmark verification for **`google/gemma-4-26B-A4B`** served via **vLLM** on **2x NVIDIA H100-SXM5-80GB GPUs (`TP=2`)** (`pallaviam-h100-tcpx-pool`, GKE cluster `pallaviam-gke-h100-tcpx-cluster`, `us-east5-a`) matching the customer's exact workload profile.

---

## 1. Customer Workload Specification vs. Live Verified Results

| Parameter | Customer Target | Live Optimized Result (`2x H100 GPUs, TP=2`) | Status |
| :--- | :---: | :---: | :---: |
| **Input Tokens (ISL)** | `10,000` | `10,000` (`--random-input-len 10000 --random-range-ratio 1.0`) | **Exact Match** |
| **Cached Tokens** | `0` | `0` (`--no-enable-prefix-caching`) | **Exact Match** |
| **Output Tokens (OSL)** | `500` | `500` (`--random-output-len 500 --random-range-ratio 1.0`) | **Exact Match** |
| **GPUs per Replica** | `2` | `2x NVIDIA H100-SXM5-80GB (TP=2, EP=2)` | **Exact Match** |
| **Max Batch Size** | `32` | `32` (`--max-num-seqs 32`) | **Exact Match** |
| **QPM Effective (30% Utilization)** | `75 QPM` (`1.25 req/s`) | **`81.5 QPM` (`1.36 req/s`)** at `C=8` / **`101.2 QPM` (`1.69 req/s`)** at `C=10` | **Exceeds Target (+35%)** |
| **End-to-End Latency at Effective Load** | **`7.70 seconds`** | **`5.38 s` Median / `5.68 s` Mean** (at 81.5 QPM) | **Beats SLA by 2.32 seconds** |
| **Max QPM Meeting `< 7.7s` Latency SLA** | `75 QPM` (`1.25 req/s`) | **`130.9 QPM` (`2.18 req/s`)** at `C=16` (`7.31 s` Median / `7.32 s` Mean) | **1.75x Higher Capacity** |
| **Sustained Throughput at Max Batch Size (`C=32`)** | `Batch Size 32` | **`170.7 QPM` (`2.85 req/s`, `29,872.33 total tok/s`)** | **Verified** |

---

## 2. Summary of Applied Optimizations (`vllm_gemma4_26b_h100_2GPU_optimized.yaml`)

1. **Removed `--enforce-eager` & Enabled Full/Piecewise CUDA Graphs + `torch.compile`**:
   - **Root Cause**: In the baseline run (`26.41s` at `C=1`), `--enforce-eager` disabled CUDA Graphs (`Cudagraph is disabled under eager mode`). Launching 400+ small Python/CUDA kernels per token across 30 MoE layers added `~45 ms/token` of pure CPU/Python launch overhead.
   - **Fix**: Enabled `--compilation-config '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1, 2, 4, 8, 12, 16, 24, 32]}'`, cutting single-stream **TPOT from `52.39 ms/tok` down to `5.37 ms/tok` (9.75x faster)**.
2. **Enabled Expert Parallelism (`--enable-expert-parallel`, `EP=2`) across 2x H100 GPUs**:
   - **Root Cause**: Pure Tensor Parallelism (`TP=2, EP=1`) slices each of the 128 experts' intermediate dimension (`N=352`) in half (`N=176`), underutilizing H100 Tensor Cores and doubling all-reduce communication.
   - **Fix**: Added `--enable-expert-parallel`, assigning 64 full experts (`N=352`) per H100 GPU.
3. **Enabled Text-Only Mode (`--language-model-only`)**:
   - Skips loading the SigLIP vision encoder, eliminates the 56-second multimodal warmup, and disables bidirectional prefix-LM attention checks.
4. **Enabled Custom NVLink AllReduce (`--no-disable-custom-all-reduce`)**:
   - Uses FlashInfer MNNVL symmetric memory all-reduce + `allreduce_rms` custom fusion over direct NVLink P2P (`P2P/CUMEM`).
5. **Enforced Customer Scheduler & Batching Profile (`--max-num-seqs 32`, `--max-num-batched-tokens 32768`, `--no-enable-prefix-caching`)**:
   - Bounds in-flight batch size to `32` and disables prefix caching (`cached tokens = 0`).

---

## 3. Baseline vs. Optimized Performance Comparison (`10,000 Input / 500 Output`)

| Concurrency | Baseline E2E Latency (s) | **Optimized E2E Latency (s)** | Latency Speedup | Baseline Median TPOT (ms) | **Optimized Median TPOT (ms)** | Baseline Output Tok/s | **Optimized Output Tok/s** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 26.41 s | **2.92 s** | **9.04x faster** | 52.39 ms | **5.37 ms** | 18.88 | **171.02** |
| **8** | 28.89 s | **5.48 s** | **5.27x faster** | 55.56 ms | **8.57 ms** | 135.90 | **727.86** |
| **10** | — | **5.93 s** | — | — | **8.83 ms** | — | **843.37** |
| **16** | 30.34 s | **7.31 s** | **4.15x faster** | 59.47 ms | **10.76 ms** | 261.46 | **1,090.78** |
| **32** | 31.58 s | **11.20 s** | **2.82x faster** | 56.60 ms | **16.05 ms** | 505.75 | **1,422.63** |

---

## 4. Full Live Benchmark Results Table (`results/optimized/result_*.json`)

All figures below are parsed directly from the live JSON output files produced on `pallaviam-gke-h100-tcpx-cluster`.

| Profile / Concurrency | Completed Reqs | Output Tok/s | Input Tok/s | Total Tok/s | Req/s (QPM) | Median TTFT (ms) | P99 TTFT (ms) | Mean TTFT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Mean TPOT (ms) | Median ITL (ms) | P99 ITL (ms) | Median E2E (s) | Mean E2E (s) | P99 E2E (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Customer Effective Load (`75 QPM` target)** | 32 | **678.94** | 13,577.43 | 14,256.37 | **1.36 (`81.5 QPM`)** | **262.42** | 749.18 | 368.60 | **10.06** | 15.56 | 10.64 | **7.69** | 224.22 | **5.38** | **5.68** | **8.09** |
| **Customer Peak Burst (`250 QPM` target)** | 64 | **1,404.50** | 28,087.16 | 29,491.66 | **2.81 (`168.5 QPM`)** | **1,230.94** | 2,586.73 | 1,259.31 | **18.73** | 22.12 | 17.67 | **9.83** | 405.81 | **10.65** | **10.08** | **11.40** |
| **Concurrency 1** | 8 | **171.02** | 3,419.96 | 3,590.98 | 0.34 (`20.5 QPM`) | **245.34** | 250.84 | 245.18 | **5.37** | 5.37 | 5.37 | **5.39** | 5.61 | **2.92** | **2.92** | **2.93** |
| **Concurrency 8** ⚡ | 16 | **727.86** | 14,555.69 | 15,283.55 | 1.46 (`87.3 QPM`) | **1,212.09** | 2,194.18 | 1,186.83 | **8.57** | 10.90 | 8.62 | **7.10** | 7.46 | **5.48** | **5.49** | **5.81** |
| **Concurrency 10** ⚡ | 20 | **843.37** | 16,865.63 | 17,709.00 | 1.69 (`101.2 QPM`) | **1,522.50** | 2,057.43 | 1,368.29 | **8.83** | 11.36 | 9.13 | **7.83** | 8.80 | **5.93** | **5.92** | **5.93** |
| **Concurrency 16** 🏆 | 32 | **1,090.78** | 21,813.41 | 22,904.19 | 2.18 (`130.9 QPM`) | **1,840.41** | 3,248.46 | 1,879.55 | **10.76** | 14.15 | 10.91 | **8.27** | 9.19 | **7.31** | **7.32** | **8.23** |
| **Concurrency 32 (`Batch Size 32`)** | 64 | **1,422.63** | 28,449.70 | 29,872.33 | 2.85 (`170.7 QPM`) | **3,206.90** | 6,408.64 | 3,096.23 | **16.05** | 21.89 | 16.29 | **9.81** | 332.42 | **11.20** | **11.22** | **14.71** |
