# SGLang Benchmark Report: `google/gemma-4-26B-A4B` (10K Input / 300 Output - Baseline)

Comprehensive performance sweep across concurrency levels **1 to 512** (`1, 2, 4, 6, 8, 16, 32, 64, 128, 256, 512`) for **`google/gemma-4-26B-A4B`** served via **vLLM** on a single **NVIDIA RTX PRO 6000 Blackwell GPU** (`g4-standard-48`, TP=1, FP8 quantization).

- **Input Prompt Length**: 10,240 tokens (10K Context)
- **Max Output Tokens**: 300 tokens
- **Backend**: vLLM OpenAI API Server (`vllm_gemma4_26b_g4_1GPU.yaml`: `--max-num-batched-tokens=16384`, `--no-enable-prefix-caching`)
- **Client**: `sglang.bench_serving` executed from `benchmark-client-pool` (`sglang-gemma4-10k-300-benchmark-sweep.yaml`)
- **Customer SLA Target**: **$\le 4.7\text{ seconds}$ P99 End-to-End Latency**

---

## 1. Concurrency Sweep Summary Table (Median & P99 Metrics)

| Concurrency | Completed Req | Output Tok/s | Input Tok/s | Total Tok/s | Req/s | Median TTFT (ms) | P99 TTFT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Median ITL (ms) | Median Latency (s) | P99 Latency (s) | Meets $\le 4.7\text{s}$ P99 SLA? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 8 | **116.43** | 3,036.84 | 3,153.27 | 0.68 | **177.49** | 246.56 | **7.84** | 8.06 | **7.82** | **1.37** | 2.24 | ✅ Yes |
| **2** | 8 | **169.96** | 4,432.78 | 4,602.73 | 1.00 | **184.52** | 272.88 | **10.34** | 13.74 | **10.11** | **1.94** | 3.08 | ✅ Yes |
| **4** | 8 | **218.37** | 5,695.43 | 5,913.80 | 1.28 | **214.43** | 589.42 | **14.43** | 19.63 | **13.10** | **2.66** | 3.71 | ✅ Yes |
| **6** | 12 | **249.67** | 5,972.31 | 6,221.98 | 1.38 | **252.67** | 945.33 | **18.79** | 21.61 | **16.28** | **3.34** | 6.26 | ❌ No (Prefill Stall) |
| **8** | 16 | **293.08** | 8,299.29 | 8,592.37 | 1.85 | **275.72** | 1,074.88 | **20.35** | 38.77 | **16.67** | **3.61** | 6.66 | ❌ No |
| **16** | 32 | **376.39** | 12,121.28 | 12,497.67 | 2.55 | **522.00** | 2,157.21 | **33.21** | 79.49 | **22.65** | **5.15** | 10.49 | ❌ No |
| **32** | 64 | **470.79** | 15,451.53 | 15,922.32 | 2.96 | **517.76** | 4,597.85 | **51.02** | 136.42 | **31.02** | **8.59** | 18.54 | ❌ No |
| **64** | 128 | **589.87** | 20,092.40 | 20,682.27 | 3.74 | **882.70** | 10,348.08 | **79.41** | 450.58 | **37.77** | **14.19** | 30.06 | ❌ No |
| **128** ⚡ | 256 | **695.48** | 23,922.07 | 24,617.55 | 4.56 | **1489.88** | 21,042.68 | **139.96** | 507.70 | **47.97** | **23.85** | 51.53 | ❌ No |
| **256** | 512 | **748.21** | 26,228.98 | 26,977.19 | 5.09 | **13109.12** | 43,047.06 | **218.34** | 509.72 | **93.55** | **42.65** | 92.10 | ❌ No |
| **512** 🏆 | 1024 | **789.01** | 27,119.23 | 27,908.24 | 5.25 | **59507.61** | 89,119.19 | **229.95** | 508.45 | **159.29** | **82.46** | 148.97 | ❌ No |

---

## 2. Detailed Performance Comparison: Mean vs. Median

| Concurrency | Output Tok/s | Total Tok/s | Req/s | Mean TTFT (ms) | Median TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | Mean Latency (s) | Median Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 116.43 | 3,153.27 | 0.68 | 154.25 | **177.49** | 7.76 | **7.84** | 7.73 | **7.82** | 1.46 | **1.37** |
| **2** | 169.96 | 4,602.73 | 1.00 | 163.67 | **184.52** | 10.97 | **10.34** | 10.76 | **10.11** | 1.99 | **1.94** |
| **4** | 218.37 | 5,913.80 | 1.28 | 296.42 | **214.43** | 14.89 | **14.43** | 14.31 | **13.10** | 2.72 | **2.66** |
| **6** | 249.67 | 6,221.98 | 1.38 | 410.75 | **252.67** | 17.51 | **18.79** | 17.38 | **16.28** | 3.54 | **3.34** |
| **8** | 293.08 | 8,592.37 | 1.85 | 484.87 | **275.72** | 21.17 | **20.35** | 19.69 | **16.67** | 3.58 | **3.61** |
| **16** | 376.39 | 12,497.67 | 2.55 | 824.96 | **522.00** | 36.78 | **33.21** | 31.19 | **22.65** | 5.39 | **5.15** |
| **32** | 470.79 | 15,922.32 | 2.96 | 1,414.60 | **517.76** | 53.96 | **51.02** | 50.85 | **31.02** | 9.45 | **8.59** |
| **64** | 589.87 | 20,682.27 | 3.74 | 2,790.14 | **882.70** | 94.54 | **79.41** | 80.62 | **37.77** | 15.42 | **14.19** |
| **128** | 695.48 | 24,617.55 | 4.56 | 5,854.32 | **1489.88** | 159.03 | **139.96** | 132.48 | **47.97** | 25.92 | **23.85** |
| **256** | 748.21 | 26,977.19 | 5.09 | 17,507.73 | **13109.12** | 219.51 | **218.34** | 195.92 | **93.55** | 46.08 | **42.65** |
| **512** | 789.01 | 27,908.24 | 5.25 | 52,369.39 | **59507.61** | 224.29 | **229.95** | 213.58 | **159.29** | 84.18 | **82.46** |

---

## 3. Baseline P99 SLA Analysis ($\le 4.7\text{s}$ P99 Latency)

- In the un-optimized baseline configuration (`--max-num-batched-tokens=16384`, `--no-enable-prefix-caching`):
  - **Concurrencies 1, 2, and 4** satisfy the **$\le 4.7\text{s}$ P99 Latency SLA**, achieving up to **218.37 output tok/s** (`5,913.80 total tok/s`) at `C=4` (`P99 Latency = 3.71s`).
  - At **Concurrency 6 and 8**, **Median Latency is only `3.34s` and `3.61s`** (well below `4.7s`), **but P99 Latency jumps to `6.26s` and `6.66s`**.
  - **Root Cause**: Because `--max-num-batched-tokens=16384` processes each incoming 10K prompt prefill in one monolithic GPU pass without chunking, active decode steps experience a **~490 ms stall (`Max ITL = 490.69 ms`)** per prefill pass, pushing the tail P99 latency above `4.7s`.
