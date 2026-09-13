# SGLang Benchmark Report: `google/gemma-4-26B-A4B` (10K Input / 500 Output - 1x GPU TP=1)

Performance benchmark across concurrency levels **1, 8, 16, 32** for **google/gemma-4-26B-A4B** served on **1x NVIDIA RTX PRO 6000 Blackwell GPU (TP=1, FP8)**.

- **Input Length**: 10,240 tokens (10K)
- **Max Output Length**: 500 tokens
- **Backend**: vLLM OpenAI API Server (TP=1)
- **Client**: `sglang.bench_serving` on `default-pool`

---

## 📊 Concurrency Sweep Summary (Median & P99 Metrics)

| Concurrency | Completed Req | Output Tok/s | Input Tok/s | Total Tok/s | Req/s | Median TTFT (ms) | P99 TTFT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Median ITL (ms) | Median Latency (s) | P99 Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 8 | **117.93** | 2315.77 | 2433.70 | 0.52 | **178.53** | 249.14 | **7.78** | 8.08 | **7.84** | **1.35** | 3.90 |
| **8** | 16 | **319.98** | 5436.58 | 5756.56 | 1.21 | **275.97** | 1083.44 | **19.50** | 39.05 | **16.91** | **5.79** | 9.74 |
| **16** | 32 | **437.92** | 8268.06 | 8705.98 | 1.74 | **349.16** | 2152.10 | **28.99** | 59.74 | **22.33** | **8.10** | 14.46 |
| **32** | 64 | **565.34** | 11424.56 | 11989.90 | 2.19 | **593.24** | 4589.30 | **45.18** | 90.79 | **30.73** | **12.90** | 24.40 |

---
