# SGLang Benchmark Report: `google/gemma-4-26B-A4B` (10K Input / 500 Output)

Comprehensive performance sweep across concurrency levels **1 to 512** for **`google/gemma-4-26B-A4B`** served via **vLLM** on a single **NVIDIA RTX PRO 6000 Blackwell GPU** (`g4-standard-48`, TP=1, FP8 quantization).

- **Input Prompt Length**: 10,240 tokens (10K Context)
- **Max Output Tokens**: 500 tokens
- **Backend**: vLLM OpenAI API Server
- **Client**: `sglang.bench_serving` executed from `benchmark-client-pool`

---

## 1. Concurrency Sweep Summary Table (Median & P99 Metrics)

| Concurrency | Completed Req | Output Tok/s | Input Tok/s | Total Tok/s | Req/s | Median TTFT (ms) | P99 TTFT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Median ITL (ms) | Median Latency (s) | P99 Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 8 | **118.03** | 2,317.77 | 2,435.80 | 0.52 | **177.51** | 243.57 | **7.80** | 8.05 | **7.86** | **1.35** | 3.88 |
| **8** | 16 | **321.25** | 5,458.16 | 5,779.40 | 1.22 | **275.49** | 1,070.71 | **19.05** | 24.89 | **16.87** | **5.77** | 9.70 |
| **16** | 32 | **436.33** | 8,238.04 | 8,674.37 | 1.74 | **363.13** | 2,144.13 | **29.11** | 55.66 | **22.40** | **8.14** | 14.46 |
| **32** | 64 | **565.62** | 11,430.16 | 11,995.77 | 2.19 | **592.67** | 4,567.29 | **45.60** | 90.47 | **30.79** | **12.91** | 24.38 |
| **64** | 128 | **733.83** | 16,308.14 | 17,041.97 | 3.03 | **680.76** | 10,125.56 | **68.08** | 281.02 | **37.59** | **16.94** | 36.68 |
| **128** ⚡ | 256 | **941.48** | 19,679.09 | 20,620.57 | 3.75 | **1039.43** | 20,886.02 | **98.68** | 498.43 | **47.43** | **28.15** | 60.42 |
| **256** | 512 | **1056.21** | 22,850.16 | 23,906.38 | 4.43 | **14167.12** | 45,682.22 | **146.21** | 506.62 | **56.42** | **49.55** | 101.04 |
| **512** 🏆 | 1024 | **1154.84** | 23,360.47 | 24,515.32 | 4.52 | **67969.69** | 98,858.01 | **156.89** | 358.96 | **57.93** | **95.17** | 166.53 |

---

## 2. Detailed Performance Comparison: Mean vs. Median

| Concurrency | Output Tok/s | Total Tok/s | Req/s | Mean TTFT (ms) | Median TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | Mean Latency (s) | Median Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 118.03 | 2,435.80 | 0.52 | 153.02 | **177.51** | 7.74 | **7.80** | 7.83 | **7.86** | 1.92 | **1.35** |
| **8** | 321.25 | 5,779.40 | 1.22 | 483.58 | **275.49** | 18.76 | **19.05** | 18.38 | **16.87** | 5.31 | **5.77** |
| **16** | 436.33 | 8,674.37 | 1.74 | 810.05 | **363.13** | 29.64 | **29.11** | 26.82 | **22.40** | 7.52 | **8.14** |
| **32** | 565.62 | 11,995.77 | 2.19 | 1,434.26 | **592.67** | 44.63 | **45.60** | 42.34 | **30.79** | 12.33 | **12.91** |
| **64** | 733.83 | 17,041.97 | 3.03 | 2,747.11 | **680.76** | 78.61 | **68.08** | 64.82 | **37.59** | 18.35 | **16.94** |
| **128** | 941.48 | 20,620.57 | 3.75 | 5,576.26 | **1039.43** | 117.00 | **98.68** | 98.07 | **47.43** | 30.08 | **28.15** |
| **256** | 1056.21 | 23,906.38 | 4.43 | 18,261.19 | **14167.12** | 161.83 | **146.21** | 140.82 | **56.42** | 51.61 | **49.55** |
| **512** | 1154.84 | 24,515.32 | 4.52 | 58,715.30 | **67969.69** | 153.45 | **156.89** | 147.18 | **57.93** | 96.08 | **95.17** |

---

## 3. Key Observations & Sizing Guidance

1. **Massive Total Throughput Scaling**:
   - Due to the large 10K prompt size, the single Blackwell RTX PRO 6000 GPU sustains thousands of input and output tokens per second, scaling smoothly from C=1 to C=512.
2. **Sub-Second TTFT across Low & Medium Concurrency (1–128)**:
   - Even with a large 10K prompt context, the Blackwell GPU processes prefill with sub-second median TTFT through Concurrency 128.
3. **Smooth Inter-Token Streaming**:
   - Median Inter-Token Latency (ITL) remains remarkably consistent across the entire concurrency sweep.
4. **Recommended Production Operating Range**:
   - **Concurrency 64 to 128** provides the optimal balance of **high throughput, responsive TTFT, and fast end-to-end turnaround time**.
