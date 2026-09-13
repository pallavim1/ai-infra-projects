# Gemma 4 26B vLLM Benchmark Report (10K Input / 500 Output)

**Configuration**
- **Model**: `google/gemma-4-26B-A4B`
- **Input Prompt Length**: 10,240 tokens (10K Context)
- **Max Output Tokens**: 500 tokens (`--ignore-eos`, `--random-range-ratio 0.0`)
- **Framework**: vLLM Serving (`vllm/vllm-openai:gemma4`) + Native vLLM Benchmark CLI (`vllm bench serve`)
- **Hardware**: 1x NVIDIA RTX PRO 6000 Blackwell Server Edition (GKE `g4-standard-48`, TP=1)

---

## 1. Concurrency Sweep Summary Table

| Concurrency | Prompts | Output Tok/s | Total Tok/s | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Mean Latency (s) | Median Latency (s) | P99 Latency (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 8 | 109.45 | 2350.85 | 368.72 | 368.51 | 372.90 | 8.42 | 8.41 | 8.48 | 4.57 | 4.57 | 4.60 |
| **8** | 16 | 395.76 | 8500.21 | 1560.09 | 1729.81 | 2674.27 | 17.10 | 16.76 | 19.52 | 10.09 | 10.09 | 10.71 |
| **16** | 32 | 555.73 | 11935.96 | 2417.51 | 2096.29 | 5299.56 | 23.93 | 25.00 | 28.18 | 14.36 | 14.31 | 17.90 |
| **32** | 64 | 701.89 | 15075.21 | 3931.14 | 2651.97 | 10464.16 | 37.62 | 40.19 | 44.48 | 22.70 | 22.62 | 30.65 |
| **64** | 128 | 884.24 | 18991.66 | 6668.30 | 2675.55 | 21010.99 | 58.72 | 68.08 | 70.77 | 35.97 | 35.66 | 55.11 |
| **128** | 256 | 1000.53 | 21489.31 | 11924.67 | 2176.53 | 42330.31 | 103.13 | 120.23 | 124.14 | 63.39 | 62.69 | 102.66 |
| **256** | 512 | 1020.99 | 21928.85 | 48185.38 | 37146.25 | 107200.99 | 131.36 | 144.94 | 154.20 | 113.74 | 109.40 | 179.74 |
| **512** | 1024 | 1038.83 | 22311.93 | 139140.04 | 166089.84 | 235171.29 | 138.75 | 145.36 | 154.01 | 208.37 | 237.53 | 311.29 |

---

## 2. Key Performance Observations & Comparison with SGLang Client

1. **Exact Length Enforcement (`--random-range-ratio 0.0`)**:
   - Using `vllm bench serve --dataset-name random --random-input-len 10240 --random-output-len 500 --random-range-ratio 0.0 --ignore-eos` guarantees that every single prompt has **exactly 10,239 input tokens** and generates **exactly 500 output tokens** (`4,000` generated tokens for `C=1`, `512,000` generated tokens for `C=512`).
   - By contrast, SGLang's `--random-range-ratio 1` samples uniformly over `[0.5 * L, 1.0 * L]` in some client versions unless patched, whereas vLLM's `--random-range-ratio 0.0` strictly disables variance (`[L * (1 - 0), L * (1 + 0)]`).

2. **Scaling Efficiency (Concurrency 1 to 128)**:
   - **Single-Stream (C=1)**: Achieves **109.45 output tokens/sec** (`2,350.85 total tok/s`) with a **368.51 ms median TTFT** and **8.41 ms median TPOT** (~119 tokens/sec streaming speed).
   - **Sweet-Spot Concurrency (C=32 to C=64)**:
     - At **C=32**, output throughput reaches **701.89 tok/s** (**15,075.21 total tok/s**) with **2.65 s median TTFT** and **40.19 ms median TPOT** (~25 tok/s per user).
     - At **C=64**, output throughput climbs to **884.24 tok/s** (**18,991.66 total tok/s**) while keeping median TTFT flat at **2.68 s**.
   - **High-Throughput Regime (C=128 to C=512)**:
     - At **C=128**, output token throughput crosses **1,000.53 tok/s** (**21,489.31 total tok/s**) with a **2.18 s median TTFT**.
     - Peak throughput is achieved at **C=512** with **1,038.83 output tok/s** and **22,311.93 total tok/s**.
