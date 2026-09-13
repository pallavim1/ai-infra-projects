# SGLang Benchmark Report: `google/gemma-4-26B-A4B` (10K Input / 500 Output - 1x GPU TP=1)

Performance benchmark at concurrency **32** for **google/gemma-4-26B-A4B** served on **1x NVIDIA RTX PRO 6000 Blackwell GPU (TP=1, FP8)**.

- **Input Length**: 10,240 tokens (10K)
- **Max Output Length**: 500 tokens
- **Backend**: vLLM OpenAI API Server (TP=1)
- **Client**: `sglang.bench_serving` on `default-pool`

---

## 📊 Concurrency 32 Benchmark Results (Median & P99 Metrics)

| Concurrency | Completed Req | Output Tok/s | Input Tok/s | Total Tok/s | Req/s | Median TTFT (ms) | P99 TTFT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Median ITL (ms) | Median Latency (s) | P99 Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **32** | 64 | **564.88** | 11415.36 | 11980.25 | 2.19 | **728.72** | 4574.72 | **45.41** | 92.83 | **30.68** | **12.91** | 24.40 |

---
