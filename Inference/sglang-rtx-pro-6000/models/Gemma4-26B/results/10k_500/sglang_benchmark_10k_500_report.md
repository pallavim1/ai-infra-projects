# SGLang Benchmark Report: `google/gemma-4-26B-A4B` (10K Input / 500 Output)

Comprehensive performance sweep across concurrency levels **1 to 32** for **`google/gemma-4-26B-A4B`** served via **vLLM** on a single **NVIDIA RTX PRO 6000 Blackwell GPU** (`g4-standard-48`, TP=1, FP8 quantization).

- **Input Prompt Length**: 10,240 tokens (10K Context)
- **Max Output Tokens**: 500 tokens
- **Backend**: vLLM OpenAI API Server
- **Client**: `sglang.bench_serving` executed from `default-pool`

---

## 1. Concurrency Sweep Summary Table (Median & P99 Metrics)

| Concurrency | Completed Req | Output Tok/s | Input Tok/s | Total Tok/s | Req/s | Median TTFT (ms) | P99 TTFT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Median ITL (ms) | Median Latency (s) | P99 Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 8 | **117.93** | 2,315.77 | 2,433.70 | 0.52 | **178.53** | 249.14 | **7.78** | 8.08 | **7.84** | **1.35** | 3.90 |
| **8** | 16 | **319.98** | 5,436.58 | 5,756.56 | 1.21 | **275.97** | 1,083.44 | **19.50** | 39.05 | **16.91** | **5.79** | 9.74 |
| **16** ⚡ | 32 | **437.92** | 8,268.06 | 8,705.98 | 1.74 | **349.16** | 2,152.10 | **28.99** | 59.74 | **22.33** | **8.10** | 14.46 |
| **32** 🏆 | 64 | **565.34** | 11,424.56 | **11,989.90** | **2.19** | **593.24** | 4,589.30 | **45.18** | 90.79 | **30.73** | **12.90** | 24.40 |

---

## 2. Detailed Performance Comparison: Mean vs. Median

| Concurrency | Output Tok/s | Total Tok/s | Req/s | Mean TTFT (ms) | Median TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | Mean Latency (s) | Median Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 117.93 | 2,433.70 | 0.52 | 156.01 | **178.53** | 7.72 | **7.78** | 7.82 | **7.84** | 1.92 | **1.35** |
| **8** | 319.98 | 5,756.56 | 1.21 | 453.65 | **275.97** | 20.23 | **19.50** | 18.56 | **16.91** | 5.33 | **5.79** |
| **16** | 437.92 | 8,705.98 | 1.74 | 798.56 | **349.16** | 29.55 | **28.99** | 26.85 | **22.33** | 7.52 | **8.10** |
| **32** | 565.34 | 11,989.90 | 2.19 | 1,421.02 | **593.24** | 44.95 | **45.18** | 42.41 | **30.73** | 12.33 | **12.90** |

---

## 3. Key Observations & Sizing Guidance

1. **Sub-Second TTFT across All Concurrency Levels (1–32)**:
   - Even with a large 10K prompt context (`10,240` tokens), the Blackwell GPU sustains **Median TTFT of 178 ms (C=1)**, **276 ms (C=8)**, **349 ms (C=16)**, and **593 ms (C=32)**.
2. **Interactive Generation Speeds (TPOT)**:
   - At Concurrency 1, token generation speed is **7.78 ms TPOT** (~128.5 tok/s per user stream), providing instant, responsive text generation.
   - At Concurrency 16, token generation speed remains **28.99 ms TPOT** (~34.5 tok/s per user stream).
3. **Throughput Scaling**:
   - Total throughput scales from **2,433.70 tok/s at C=1** to **11,989.90 tok/s at C=32** (a **4.93x increase**), with output throughput reaching **565.34 tok/s**.
4. **Recommended Production Operating Range**:
   - **Concurrency 16 to 32** provides the optimal balance of **high throughput (8.7K–12.0K total tok/s)**, fast turnaround (**8.1–12.9 s total latency**), and responsive Time to First Token (**<600 ms TTFT**).
