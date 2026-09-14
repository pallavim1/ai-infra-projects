# SGLang Benchmark Report: `google/gemma-4-26B-A4B` (10K Input / 300 Output - Optimized for $\le 4.7\text{s}$ P99 Latency SLA)

Comprehensive performance sweep across concurrency levels **1 to 512** (`1, 2, 4, 6, 8, 12, 16, 32, 64, 128, 256, 512`) for **`google/gemma-4-26B-A4B`** served via **vLLM** on a single **NVIDIA RTX PRO 6000 Blackwell GPU** (`g4-standard-48`, TP=1) with **FP8 Quantization + Chunked Prefill + Prefix Caching**.

- **Input Prompt Length**: 10,240 tokens (10K Context)
- **Max Output Tokens**: 300 tokens
- **Backend**: vLLM OpenAI API Server (`vllm_gemma4_26b_g4_1GPU_optimized_10k_300.yaml`: `--quantization=fp8`, `--enable-chunked-prefill`, `--max-num-batched-tokens=4096`, `--limit-mm-per-prompt={"image":0,"video":0,"audio":0}`, `--enable-prefix-caching`)
- **Client**: `sglang.bench_serving` executed from `benchmark-client-pool` (`sglang-gemma4-10k-300-benchmark-sweep-optimized.yaml`)
- **Customer SLA Target**: **$\le 4.7\text{ seconds}$ P99 End-to-End Latency**

---

## 1. Concurrency Sweep Summary Table (Optimized Run - Median & P99 Metrics)

| Concurrency | Completed Req | Output Tok/s | Input Tok/s | Total Tok/s | Req/s | Median TTFT (ms) | P99 TTFT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Median ITL (ms) | Median Latency (s) | P99 Latency (s) | Meets $\le 4.7\text{s}$ P99 SLA? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 8 | **147.94** | 3,858.51 | 4,006.45 | 0.87 | **144.36** | 411.72 | **6.02** | 6.21 | **5.98** | **1.05** | **2.03** | ✅ Yes |
| **2** | 8 | **265.81** | 6,932.98 | 7,198.79 | 1.56 | **32.43** | 36.84 | **7.38** | 7.53 | **7.35** | **1.14** | **2.13** | ✅ Yes |
| **4** | 8 | **372.99** | 9,728.27 | 10,101.26 | 2.19 | **35.99** | 58.30 | **9.10** | 9.23 | **9.00** | **1.41** | **2.41** | ✅ Yes |
| **6** | 12 | **418.66** | 10,014.78 | 10,433.45 | 2.31 | **61.52** | 243.80 | **11.28** | 13.57 | **10.49** | **1.99** | **3.46** | ✅ **Yes (SLA Unlocked!)** |
| **8** 🎯 | 16 | **519.83** | 14,720.29 | 15,240.12 | 3.28 | **66.60** | 183.60 | **11.71** | 14.67 | **10.77** | **2.04** | **3.45** | ✅ **Yes (Max SLA-Compliant Concurrency)** |
| **12** | 24 | **488.44** | 16,098.67 | 16,587.11 | 3.39 | **225.73** | 1,067.20 | **18.46** | 26.00 | **13.13** | **3.22** | **5.63** | ⚠️ P90=4.72s / P99=5.63s |
| **16** | 32 | **678.40** | 21,847.31 | 22,525.72 | 4.60 | **103.83** | 536.98 | **20.25** | 26.34 | **14.13** | **2.92** | **5.56** | ❌ No |
| **32** | 64 | **888.35** | 29,156.00 | 30,044.36 | 5.58 | **164.39** | 419.99 | **31.37** | 38.42 | **19.85** | **5.03** | **9.45** | ❌ No |
| **64** | 128 | **1,219.89** | 41,552.29 | 42,772.19 | 7.73 | **298.16** | 731.33 | **46.36** | 64.40 | **25.06** | **6.82** | **13.82** | ❌ No |
| **128** | 256 | **1,594.39** | 54,841.59 | 56,435.98 | 10.45 | **564.85** | 1,856.91 | **73.02** | 102.10 | **76.98** | **10.17** | **21.15** | ❌ No |
| **256** 🏆 | 512 | **1,700.28** | 59,604.56 | 61,304.84 | 11.56 | **4,320.23** | 16,887.06 | **78.00** | 105.35 | **95.35** | **18.15** | **34.71** | ❌ No |
| **512** | 1024 | **1,092.26** | 37,542.44 | 38,634.71 | 7.26 | **52,427.21** | 66,086.88 | **101.99** | 105.17 | **100.21** | **61.81** | **90.26** | ❌ No |

---

## 2. Side-by-Side Comparison: Baseline vs. Optimized (`10K / 300`)

| Concurrency | Baseline Output Tok/s | Optimized Output Tok/s | Throughput Gain | Baseline P99 Latency (s) | Optimized P99 Latency (s) | Latency Reduction | Customer SLA ($\le 4.7\text{s}$ P99) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 116.43 | **147.94** | **+27.1%** | 2.24 s | **2.03 s** | -9.4% | ✅ Both Pass |
| **2** | 169.96 | **265.81** | **+56.4%** | 3.08 s | **2.13 s** | -30.8% | ✅ Both Pass |
| **4** | 218.37 | **372.99** | **+70.8%** | 3.71 s | **2.41 s** | -35.0% | ✅ Both Pass |
| **6** | 249.67 | **418.66** | **+67.7%** | 6.26 s ❌ | **3.46 s** ✅ | **-44.7%** | **✅ Unlocked by Optimization** |
| **8** 🎯 | 293.08 | **519.83** | **+77.4%** | 6.66 s ❌ | **3.45 s** ✅ | **-48.2%** | **✅ Unlocked by Optimization** |
| **16** | 376.39 | **678.40** | **+80.2%** | 10.49 s | **5.56 s** | -47.0% | ❌ Exceeds 4.7s |
| **32** | 470.79 | **888.35** | **+88.7%** | 18.54 s | **9.45 s** | -49.0% | ❌ Exceeds 4.7s |
| **64** | 589.87 | **1,219.89** | **+106.8%** | 30.06 s | **13.82 s** | -54.0% | ❌ Exceeds 4.7s |
| **128** | 695.48 | **1,594.39** | **+129.3%** | 51.53 s | **21.15 s** | -59.0% | ❌ Exceeds 4.7s |
| **256** | 748.21 | **1,700.28** | **+127.2%** | 92.10 s | **34.71 s** | -62.3% | ❌ Exceeds 4.7s |
| **512** | 789.01 | **1,092.26** | **+38.4%** | 148.97 s | **90.26 s** | -39.4% | ❌ Exceeds 4.7s |

---

## 3. Detailed Performance Comparison: Mean vs. Median (Optimized Run)

| Concurrency | Output Tok/s | Total Tok/s | Req/s | Mean TTFT (ms) | Median TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | Mean Latency (s) | Median Latency (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 147.94 | 4,006.45 | 0.87 | 152.04 | **144.36** | 5.93 | **6.02** | 5.90 | **5.98** | 1.15 | **1.05** |
| **2** | 265.81 | 7,198.79 | 1.56 | 31.91 | **32.43** | 7.34 | **7.38** | 7.31 | **7.35** | 1.27 | **1.14** |
| **4** | 372.99 | 10,101.26 | 2.19 | 39.45 | **35.99** | 8.97 | **9.10** | 8.86 | **9.00** | 1.54 | **1.41** |
| **6** | 418.66 | 10,433.45 | 2.31 | 79.52 | **61.52** | 11.03 | **11.28** | 10.78 | **10.49** | 2.02 | **1.99** |
| **8** | 519.83 | 15,240.12 | 3.28 | 78.17 | **66.60** | 11.60 | **11.71** | 11.57 | **10.77** | 1.90 | **2.04** |
| **12** | 488.44 | 16,587.11 | 3.39 | 375.84 | **225.73** | 18.53 | **18.46** | 17.85 | **13.13** | 2.93 | **3.22** |
| **16** | 678.40 | 22,525.72 | 4.60 | 150.96 | **103.83** | 19.70 | **20.25** | 18.69 | **14.13** | 2.88 | **2.92** |
| **32** | 888.35 | 30,044.36 | 5.58 | 190.02 | **164.39** | 29.11 | **31.37** | 29.51 | **19.85** | 4.85 | **5.03** |
| **64** | 1,219.89 | 42,772.19 | 7.73 | 322.36 | **298.16** | 43.80 | **46.36** | 43.78 | **25.06** | 7.16 | **6.82** |
| **128** | 1,594.39 | 56,435.98 | 10.45 | 650.51 | **564.85** | 68.56 | **73.02** | 66.89 | **76.98** | 10.77 | **10.17** |
| **256** | 1,700.28 | 61,304.84 | 11.56 | 6,027.44 | **4,320.23** | 75.22 | **78.00** | 76.15 | **95.35** | 17.13 | **18.15** |
| **512** | 1,092.26 | 38,634.71 | 7.26 | 43,539.22 | **52,427.21** | 97.34 | **101.99** | 95.96 | **100.21** | 57.83 | **61.81** |

---

## 4. Key Optimizations & Architectural Impact

1. **Native FP8 Weight & Activation Quantization (`--quantization=fp8`)**:
   - Enables `CutlassFP8ScaledMMLinearKernel` and `TRITON Fp8 MoE` kernels on NVIDIA RTX PRO 6000 Blackwell.
   - Reduces model weight footprint from `47.43 GiB` (BF16) down to **`24.62 GiB`**, freeing ~22.8 GiB of additional VRAM for KV cache and **doubling decode memory bandwidth**.
   - **Impact**: Single-request decode TPOT improved from `7.84 ms` down to **`6.02 ms`**, and peak output throughput surged by **+127.2%** (from `789.01 tok/s` to **`1,700.28 tok/s`** at `C=256`).
2. **Chunked Prefill Scheduling (`--enable-chunked-prefill --max-num-batched-tokens=4096`)**:
   - Splits 10,240-token prefills into 4,096-token chunks and piggybacks active decode tokens onto each chunk step.
   - Eliminates the `490 ms` monolithic prefill stall that previously caused P99 latency at `C=6` and `C=8` to jump to `6.26 s` and `6.66 s`.
3. **Text-Only Multimodal Bypass (`--limit-mm-per-prompt={"image":0,"video":0,"audio":0}`) & Automatic Prefix Caching (`--enable-prefix-caching`)**:
   - Disables vision/audio encoder prefix attention allocations and caches common prompt prefixes in FP8 KV memory.
   - **Impact**: Unlocks **Concurrency 8 (`C=8`)** at **`3.45 s` P99 Latency** (comfortably below the customer's **$\le 4.7\text{ s}$ P99 SLA**), delivering **`519.83 output tok/s`** (**2.38x higher output throughput** than the baseline's maximum SLA-compliant concurrency `C=4` at `218.37 output tok/s`).
