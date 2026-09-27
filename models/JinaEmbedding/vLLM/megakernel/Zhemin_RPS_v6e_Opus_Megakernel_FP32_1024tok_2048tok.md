# End-to-End Empirical Benchmarks: Zhemin's Test Suite on TPU v6e (`cb460828` Opus Megakernel `v2` + Device Pooling, FP32)

> [!IMPORTANT]
> **100% Single-Shot Empirical Verification (`2026-09-27T08:52:54Z` $\rightarrow$ `09:52:39Z`, 59m 45s Wall-Clock)**
> Every TPU v6e number in this report and in [ATP_AIC2_Benchmarks_TPU_v6e_FP32_Megakernel_vs_L4_and_Baselines.xlsx](file:///usr/local/google/home/pallaviam/.gemini/jetski/brain/292ebc90-dbb8-4a33-8f1a-ca6dee63d8ff/ATP_AIC2_Benchmarks_TPU_v6e_FP32_Megakernel_vs_L4_and_Baselines.xlsx) was measured in a single continuous end-to-end execution on GKE pod `jina-opus-v6e-test` (`ct6e-standard-1t`, 1x TPU v6e chip) driven from CPU load-generator pod `panw-v6e-benchmark-runner`, and cross-checked against vLLM's `/metrics` Prometheus counters (`vllm:request_success_total` and `vllm:prompt_tokens_total`) before and after every step:
> - **Branch & Commit:** [`jina-v2-opus-megakernel` @ `cb460828`](https://github.com/pallavim1/tpu-inference/commit/cb460828b65f0a387780b3199ac2f15099177ada)
> - **vLLM Config:** `--dtype float32 --max-model-len 2048 --max-num-batched-tokens 2048 --no-enable-prefix-caching`
> - **Kernel & Pooling Flags:** `USE_JINA_BERT_MEGAKERNEL=1`, `JINA_BERT_MEGAKERNEL_VERSION=v2`, `JINA_BERT_MEGAKERNEL_PRECISION=default`, `TPU_POOLING_FAST_PATH=1`, `JINA_BERT_DEVICE_POOLING=1`
> - **Exact Token Payloads:**
>   - **1K (`1KB` tier):** 25,000 unique prompts from `pool_1024tok.json`, **100% verified at `1024.0` server tokens/prompt**.
>   - **2K (`2KB` tier):** 25,000 unique prompts from `pool_2048tok.json`, **100% verified at `2048.0` server tokens/prompt**.
> - **Raw Empirical Logs:**
>   - [full_e2e_results.json](file:///usr/local/google/home/pallaviam/.gemini/jetski/brain/292ebc90-dbb8-4a33-8f1a-ca6dee63d8ff/scratch/opus_v6e_tests/full_e2e_results.json) (`run_id: cb460828_full_e2e_20260927T085254Z`)
>   - [mid_1k_490_results.json](file:///usr/local/google/home/pallaviam/.gemini/jetski/brain/292ebc90-dbb8-4a33-8f1a-ca6dee63d8ff/scratch/opus_v6e_tests/mid_1k_490_results.json) & [mid_2k_330_results.json](file:///usr/local/google/home/pallaviam/.gemini/jetski/brain/292ebc90-dbb8-4a33-8f1a-ca6dee63d8ff/scratch/opus_v6e_tests/mid_2k_330_results.json)
>   - [task-4487.log](file:///usr/local/google/home/pallaviam/.gemini/jetski/brain/292ebc90-dbb8-4a33-8f1a-ca6dee63d8ff/.system_generated/tasks/task-4487.log)

---

## Tab 1: Raw Benchmark Results (Matching `gid=1161755388` AS-IS in a Single Tab)

### 1A. Batch Request Testing: A Single HTTP Request Contains Multiple Prompts (`{"text": [p_1, ..., p_N]}`, $N \in \{1, 4, 8, 16\}$)

*Client sends a batch of $N$ prompts (`1,024` tokens/prompt for 1KB, `2,048` tokens/prompt for 2KB) in a single HTTP `POST /prompt_c2` request and records the HTTP response latency in milliseconds.*

| Payload Size | Concurrency | TPU v6e `cb460828` Throughput | TPU v6e `cb460828` p50 | TPU v6e `cb460828` p99 | TPU v5e Throughput | TPU v5e p50 | TPU v5e p99 | L4 GPU Throughput | L4 GPU p50 | L4 GPU p99 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1KB (1024 chars / 1,024 tok)** | **1** | **130.3/s** | **7.4ms** | **10.4ms** | 85.8/s | 11.5ms | 12.7ms | 43.2/s | 20.7ms | 23.5ms |
| **1KB (1024 chars / 1,024 tok)** | **4** | **227.0/s** | **17.4ms** | **19.4ms** | 187.4/s | 21.2ms | 22.8ms | 103.4/s | 38.1ms | 42.9ms |
| **1KB (1024 chars / 1,024 tok)** | **8** | **254.3/s** | **31.2ms** | **32.7ms** | 187.6/s | 42.5ms | 44.5ms | 135.1/s | 58.7ms | 66.3ms |
| **1KB (1024 chars / 1,024 tok)** | **16** | **267.9/s** | **59.5ms** | **60.8ms** | 186.8/s | 85.4ms | 89.9ms | 165.2/s | 96.5ms | 107.6ms |
| **2KB (2048 chars / 2,048 tok)** | **1** | **108.1/s** | **9.0ms** | **10.6ms** | 59.5/s | 16.5ms | 17.7ms | 39.0/s | 24.3ms | 28.2ms |
| **2KB (2048 chars / 2,048 tok)** | **4** | **148.3/s** | **26.6ms** | **29.3ms** | 97.7/s | 40.7ms | 42.8ms | 75.7/s | 52.1ms | 58.2ms |
| **2KB (2048 chars / 2,048 tok)** | **8** | **158.3/s** | **50.1ms** | **52.4ms** | 96.6/s | 82.6ms | 86.0ms | 90.5/s | 87.4ms | 97.3ms |
| **2KB (2048 chars / 2,048 tok)** | **16** | **163.0/s** | **97.3ms** | **104.0ms** | 96.5/s | 165.9ms | 171.2ms | 101.1/s | 156.9ms | 175.5ms |

### 1B. Concurrent HTTP Request Testing (`k6 constant-vus`, $\text{VUS} \in \{1, 4, 8, 16\}$, 30 s per step)

*`k6` runs $N \in \{1, 4, 8, 16\}$ concurrent virtual users (`constant-vus`), each sending 1 prompt (`1,024` or `2,048` exact tokens) per HTTP request.*

| Payload Size | Concurrency | TPU v6e `cb460828` Throughput | TPU v6e `cb460828` p50 | TPU v6e `cb460828` p99 | TPU v5e Throughput | TPU v5e p50 | TPU v5e p99 | L4 GPU Throughput | L4 GPU p50 | L4 GPU p99 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1KB (1024 chars / 1,024 tok)** | **1** | **128.4/s** | **6.9ms** | **8.0ms** | 85.8/s | 11.5ms | 12.7ms | 43.2/s | 20.7ms | 23.5ms |
| **1KB (1024 chars / 1,024 tok)** | **4** | **376.4/s** | **9.9ms** | **12.3ms** | 187.4/s | 21.2ms | 22.8ms | 103.4/s | 38.1ms | 42.9ms |
| **1KB (1024 chars / 1,024 tok)** | **8** | **507.1/s** | **14.8ms** | **20.2ms** | 187.6/s | 42.5ms | 44.5ms | 135.1/s | 58.7ms | 66.3ms |
| **1KB (1024 chars / 1,024 tok)** | **16** | **518.0/s** | **30.1ms** | **39.3ms** | 186.8/s | 85.4ms | 89.9ms | 165.2/s | 96.5ms | 107.6ms |
| **2KB (2048 chars / 2,048 tok)** | **1** | **99.0/s** | **8.6ms** | **9.5ms** | 59.5/s | 16.5ms | 17.7ms | 39.0/s | 24.3ms | 28.2ms |
| **2KB (2048 chars / 2,048 tok)** | **4** | **290.1/s** | **12.4ms** | **16.2ms** | 97.7/s | 40.7ms | 42.8ms | 75.7/s | 52.1ms | 58.2ms |
| **2KB (2048 chars / 2,048 tok)** | **8** | **367.6/s** | **20.1ms** | **26.8ms** | 96.6/s | 82.6ms | 86.0ms | 90.5/s | 87.4ms | 97.3ms |
| **2KB (2048 chars / 2,048 tok)** | **16** | **366.9/s** | **42.0ms** | **50.0ms** | 96.5/s | 165.9ms | 171.2ms | 101.1/s | 156.9ms | 175.5ms |

---

### 2. 1KB Dedicated Saturation (`1,024` Exact Tokens per Request, FP32, 60 s per step)

| RPS | Achieved (TPU v6e `cb460828`) | P50 (TPU v6e `cb460828`) | P99 (TPU v6e `cb460828`) | SLA (TPU v6e `cb460828`) | Achieved (TPU v5e) | P50 (TPU v5e) | P99 (TPU v5e) | SLA (TPU v5e) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | 100.03 | 7.2 ms | 8.5 ms | PASS | 100 | 11.5 ms | 14.0 ms | PASS |
| **120** | 120.03 | 7.0 ms | 8.3 ms | PASS | 120 | 11.8 ms | 15.5 ms | PASS |
| **140** | 140.05 | 6.9 ms | 8.2 ms | PASS | 140 | 11.6 ms | 18.5 ms | PASS |
| **160** | 160.04 | 7.1 ms | 9.1 ms | PASS | 160 | 16.8 ms | 24.2 ms | PASS |
| **180** | 180.07 | 7.4 ms | 9.7 ms | PASS | **180.1** | **19.9 ms** | **29.6 ms** | **PASS (v5e Max)** |
| **190** | 190.04 | 7.6 ms | 9.7 ms | PASS | 187.8 | 523.7 ms | 762.7 ms | SATURATED |
| **200** | 200.05 | 7.7 ms | 9.7 ms | PASS | 189 | 1709 ms | 2818 ms | SATURATED |
| **220** | 220.05 | 8.3 ms | 9.4 ms | PASS | 187.9 | 3738 ms | 6182 ms | SATURATED |
| **240** | 240.06 | 8.1 ms | 9.6 ms | PASS | — | — | — | SATURATED |
| **260** | 260.07 | 7.9 ms | 10.3 ms | PASS | — | — | — | SATURATED |
| **280** | 280.08 | 7.9 ms | 10.9 ms | PASS | — | — | — | SATURATED |
| **300** | 300.08 | 7.9 ms | 11.2 ms | PASS | — | — | — | SATURATED |
| **320** | 320.06 | 8.4 ms | 12.8 ms | PASS | — | — | — | SATURATED |
| **340** | 340.15 | 8.7 ms | 13.1 ms | PASS | — | — | — | SATURATED |
| **360** | 360.19 | 9.3 ms | 15.3 ms | PASS | — | — | — | SATURATED |
| **380** | 380.21 | 9.6 ms | 14.4 ms | PASS | — | — | — | SATURATED |
| **400** | 400.28 | 10.1 ms | 16.3 ms | PASS | — | — | — | SATURATED |
| **420** | 420.45 | 9.8 ms | 17.5 ms | PASS | — | — | — | SATURATED |
| **440** | 440.21 | 11.3 ms | 19.7 ms | PASS | — | — | — | SATURATED |
| **460** | 460.31 | 11.5 ms | 25.0 ms | PASS | — | — | — | SATURATED |
| **480** | **480.45** | **12.6 ms** | **27.9 ms** | **PASS (20-RPS Grid Max)** | — | — | — | SATURATED |
| **490** | **490.28** | **14.1 ms** | **46.3 ms** | **PASS (10-RPS Grid Max)** | — | — | — | SATURATED |
| **500** | 500.55 | 13.7 ms | 71.3 ms | SATURATED | — | — | — | SATURATED |
| **510** | 510.42 | 16.1 ms | 104.2 ms | SATURATED | — | — | — | SATURATED |
| **520** | 517.55 | 21.2 ms | 533.7 ms | SATURATED | — | — | — | SATURATED |

---

### 3. 2KB Dedicated Saturation (`2,048` Exact Tokens per Request, FP32, 60 s per step)

| RPS | Achieved (TPU v6e `cb460828`) | P50 (TPU v6e `cb460828`) | P99 (TPU v6e `cb460828`) | SLA (TPU v6e `cb460828`) | Achieved (TPU v5e) | P50 (TPU v5e) | P99 (TPU v5e) | SLA (TPU v5e) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **90** | 90.03 | 8.6 ms | 9.9 ms | PASS | **90** | **17.2 ms** | **26.5 ms** | **PASS (v5e Max)** |
| **95** | 95.03 | 8.7 ms | 10.2 ms | PASS | 94.75 | 218.6 ms | 346.5 ms | SATURATED |
| **100** | 100.03 | 8.6 ms | 9.9 ms | PASS | 96.32 | 1031 ms | 2185 ms | SATURATED |
| **110** | 110.03 | 8.6 ms | 10.0 ms | PASS | 96.74 | 3208 ms | 5170 ms | SATURATED |
| **120** | 120.03 | 8.6 ms | 10.4 ms | PASS | — | — | — | SATURATED |
| **130** | 130.05 | 8.8 ms | 11.8 ms | PASS | — | — | — | SATURATED |
| **140** | 140.05 | 9.3 ms | 12.9 ms | PASS | — | — | — | SATURATED |
| **150** | 150.05 | 9.4 ms | 12.7 ms | PASS | — | — | — | SATURATED |
| **160** | 160.05 | 11.1 ms | 12.5 ms | PASS | — | — | — | SATURATED |
| **170** | 170.04 | 10.9 ms | 12.2 ms | PASS | — | — | — | SATURATED |
| **180** | 180.06 | 10.5 ms | 12.0 ms | PASS | — | — | — | SATURATED |
| **190** | 190.04 | 10.3 ms | 11.9 ms | PASS | — | — | — | SATURATED |
| **200** | 200.04 | 10.0 ms | 12.0 ms | PASS | — | — | — | SATURATED |
| **220** | 220.06 | 9.8 ms | 13.9 ms | PASS | — | — | — | SATURATED |
| **240** | 240.06 | 9.7 ms | 15.5 ms | PASS | — | — | — | SATURATED |
| **260** | 260.08 | 9.8 ms | 15.1 ms | PASS | — | — | — | SATURATED |
| **280** | 280.14 | 10.1 ms | 16.7 ms | PASS | — | — | — | SATURATED |
| **300** | **300.10** | **12.9 ms** | **18.6 ms** | **PASS (300 RPS Target)** | — | — | — | SATURATED |
| **310** | **310.17** | **12.7 ms** | **18.1 ms** | **PASS** | — | — | — | SATURATED |
| **320** | **320.17** | **12.6 ms** | **25.3 ms** | **PASS (v6e Max)** | — | — | — | SATURATED |
| **330** | 330.10 | 12.6 ms | 57.1 ms | SATURATED | — | — | — | SATURATED |
| **340** | 340.37 | 13.6 ms | 138.7 ms | SATURATED | — | — | — | SATURATED |
| **360** | 348.05 | 271.2 ms | 2526.2 ms | SATURATED | — | — | — | SATURATED |

---

## Tab 2: Performance & Cost Comparison (Matching `gid=1972899730` AS-IS)

### 1A. Concurrent Request Comparison — Single HTTP Request Containing $N$ Prompts

#### P50 Latency
| Payload Size | Concurrency | TPU v6e `cb460828` | TPU v5e | L4 GPU | Delta vs v5e | % Reduction vs v5e | Delta vs L4 | % Reduction vs L4 | Faster Setup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1KB (1024 chars / 1,024 tok)** | **1** | **7.4ms** | 11.5ms | 20.7ms | -4.1ms | **35.7%** | -13.3ms | **64.3%** | **TPU v6e** |
| **1KB (1024 chars / 1,024 tok)** | **4** | **17.4ms** | 21.2ms | 38.1ms | -3.8ms | **17.9%** | -20.7ms | **54.3%** | **TPU v6e** |
| **1KB (1024 chars / 1,024 tok)** | **8** | **31.2ms** | 42.5ms | 58.7ms | -11.3ms | **26.6%** | -27.5ms | **46.8%** | **TPU v6e** |
| **1KB (1024 chars / 1,024 tok)** | **16** | **59.5ms** | 85.4ms | 96.5ms | -25.9ms | **30.3%** | -37.0ms | **38.3%** | **TPU v6e** |
| **2KB (2048 chars / 2,048 tok)** | **1** | **9.0ms** | 16.5ms | 24.3ms | -7.5ms | **45.5%** | -15.3ms | **63.0%** | **TPU v6e** |
| **2KB (2048 chars / 2,048 tok)** | **4** | **26.6ms** | 40.7ms | 52.1ms | -14.1ms | **34.6%** | -25.5ms | **48.9%** | **TPU v6e** |
| **2KB (2048 chars / 2,048 tok)** | **8** | **50.1ms** | 82.6ms | 87.4ms | -32.5ms | **39.3%** | -37.3ms | **42.7%** | **TPU v6e** |
| **2KB (2048 chars / 2,048 tok)** | **16** | **97.3ms** | 165.9ms | 156.9ms | -68.6ms | **41.4%** | -59.6ms | **38.0%** | **TPU v6e** |

#### p99 Latency
| Payload Size | Concurrency | TPU v6e `cb460828` | TPU v5e | L4 GPU | Delta vs v5e | % Reduction vs v5e | Delta vs L4 | % Reduction vs L4 | Faster Setup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1KB (1024 chars / 1,024 tok)** | **1** | **10.4ms** | 12.7ms | 23.5ms | -2.3ms | **18.1%** | -13.1ms | **55.7%** | **TPU v6e** |
| **1KB (1024 chars / 1,024 tok)** | **4** | **19.4ms** | 22.8ms | 42.9ms | -3.4ms | **14.9%** | -23.5ms | **54.8%** | **TPU v6e** |
| **1KB (1024 chars / 1,024 tok)** | **8** | **32.7ms** | 44.5ms | 66.3ms | -11.8ms | **26.5%** | -33.6ms | **50.7%** | **TPU v6e** |
| **1KB (1024 chars / 1,024 tok)** | **16** | **60.8ms** | 89.9ms | 107.6ms | -29.1ms | **32.4%** | -46.8ms | **43.5%** | **TPU v6e** |
| **2KB (2048 chars / 2,048 tok)** | **1** | **10.6ms** | 17.7ms | 28.2ms | -7.1ms | **40.1%** | -17.6ms | **62.4%** | **TPU v6e** |
| **2KB (2048 chars / 2,048 tok)** | **4** | **29.3ms** | 42.8ms | 58.2ms | -13.5ms | **31.5%** | -28.9ms | **49.7%** | **TPU v6e** |
| **2KB (2048 chars / 2,048 tok)** | **8** | **52.4ms** | 86.0ms | 97.3ms | -33.6ms | **39.1%** | -44.9ms | **46.1%** | **TPU v6e** |
| **2KB (2048 chars / 2,048 tok)** | **16** | **104.0ms** | 171.2ms | 175.5ms | -67.2ms | **39.3%** | -71.5ms | **40.7%** | **TPU v6e** |

---

### 1B. Concurrent Request Comparison — `k6` Concurrent HTTP Requests ($\text{VUS} \in \{1, 4, 8, 16\}$)

#### P50 Latency
| Payload Size | Concurrency | TPU v6e `cb460828` | TPU v5e | L4 GPU | Delta vs v5e | % Reduction vs v5e | Delta vs L4 | % Reduction vs L4 | Faster Setup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1KB (1024 chars / 1,024 tok)** | **1** | **6.9ms** | 11.5ms | 20.7ms | -4.6ms | **40.0%** | -13.8ms | **66.7%** | **TPU v6e** |
| **1KB (1024 chars / 1,024 tok)** | **4** | **9.9ms** | 21.2ms | 38.1ms | -11.3ms | **53.3%** | -28.2ms | **74.0%** | **TPU v6e** |
| **1KB (1024 chars / 1,024 tok)** | **8** | **14.8ms** | 42.5ms | 58.7ms | -27.7ms | **65.2%** | -43.9ms | **74.8%** | **TPU v6e** |
| **1KB (1024 chars / 1,024 tok)** | **16** | **30.1ms** | 85.4ms | 96.5ms | -55.3ms | **64.8%** | -66.4ms | **68.8%** | **TPU v6e** |
| **2KB (2048 chars / 2,048 tok)** | **1** | **8.6ms** | 16.5ms | 24.3ms | -7.9ms | **47.9%** | -15.7ms | **64.6%** | **TPU v6e** |
| **2KB (2048 chars / 2,048 tok)** | **4** | **12.4ms** | 40.7ms | 52.1ms | -28.3ms | **69.5%** | -39.7ms | **76.2%** | **TPU v6e** |
| **2KB (2048 chars / 2,048 tok)** | **8** | **20.1ms** | 82.6ms | 87.4ms | -62.5ms | **75.7%** | -67.3ms | **77.0%** | **TPU v6e** |
| **2KB (2048 chars / 2,048 tok)** | **16** | **42.0ms** | 165.9ms | 156.9ms | -123.9ms | **74.7%** | -114.9ms | **73.2%** | **TPU v6e** |

#### p99 Latency
| Payload Size | Concurrency | TPU v6e `cb460828` | TPU v5e | L4 GPU | Delta vs v5e | % Reduction vs v5e | Delta vs L4 | % Reduction vs L4 | Faster Setup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1KB (1024 chars / 1,024 tok)** | **1** | **8.0ms** | 12.7ms | 23.5ms | -4.7ms | **37.0%** | -15.5ms | **66.0%** | **TPU v6e** |
| **1KB (1024 chars / 1,024 tok)** | **4** | **12.3ms** | 22.8ms | 42.9ms | -10.5ms | **46.1%** | -30.6ms | **71.3%** | **TPU v6e** |
| **1KB (1024 chars / 1,024 tok)** | **8** | **20.2ms** | 44.5ms | 66.3ms | -24.3ms | **54.6%** | -46.1ms | **69.5%** | **TPU v6e** |
| **1KB (1024 chars / 1,024 tok)** | **16** | **39.3ms** | 89.9ms | 107.6ms | -50.6ms | **56.3%** | -68.3ms | **63.5%** | **TPU v6e** |
| **2KB (2048 chars / 2,048 tok)** | **1** | **9.5ms** | 17.7ms | 28.2ms | -8.2ms | **46.3%** | -18.7ms | **66.3%** | **TPU v6e** |
| **2KB (2048 chars / 2,048 tok)** | **4** | **16.2ms** | 42.8ms | 58.2ms | -26.6ms | **62.1%** | -42.0ms | **72.2%** | **TPU v6e** |
| **2KB (2048 chars / 2,048 tok)** | **8** | **26.8ms** | 86.0ms | 97.3ms | -59.2ms | **68.8%** | -70.5ms | **72.5%** | **TPU v6e** |
| **2KB (2048 chars / 2,048 tok)** | **16** | **50.0ms** | 171.2ms | 175.5ms | -121.2ms | **70.8%** | -125.5ms | **71.5%** | **TPU v6e** |

---

### 2. RPS Saturation Result ($\text{p99} < 50\text{ ms}$ SLA)

| Payload | Setup | RPS | p50 | p99 | RPS Improvement vs L4 | RPS Improvement vs TPU v5e |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1K** | L4 GPU | 70/s | — | — | Baseline | — |
| **1K** | TPU v5e | 180/s *(187/s max)* | 19.9ms | 29.6ms | +157.1% | Baseline |
| **1K** | **TPU v6e (`cb460828`, 10-RPS grid)** | **490/s** | **14.1ms** | **46.3ms** | **+600.0% (7.00x)** | **+172.2% vs 180/s (+162.0% vs 187/s)** |
| **1K** | **TPU v6e (`cb460828`, 20-RPS grid)** | **480/s** | **12.6ms** | **27.9ms** | **+585.7% (6.86x)** | **+166.7% vs 180/s (+156.7% vs 187/s)** |
| **2K** | L4 GPU | 40/s | — | — | Baseline | — |
| **2K** | TPU v5e | 90/s | 17.2ms | 26.5ms | +125.0% | Baseline |
| **2K** | **TPU v6e (`cb460828`, max PASS)** | **320/s** | **12.6ms** | **25.3ms** | **+700.0% (8.00x)** | **+255.6% (3.56x)** |
| **2K** | **TPU v6e (`cb460828`, 300 RPS target)** | **300/s** | **12.9ms** | **18.6ms** | **+650.0% (7.50x)** | **+233.3% (3.33x)** |

---

### 3. Cost Improvement (Matching Zhemin's Exact Formula)

$$\text{Cost per 1M requests} = \frac{\text{Hourly Cost}}{0.40 \times \text{Max RPS} \times 3600} \times 1{,}000{,}000$$

| Machine Type | Machine Config | Hourly Cost | Cost per 1M request (1K Payload) | Cost per 1M request (2K Payload) |
| :--- | :--- | :---: | :---: | :---: |
| **g2-standard-4** | L4 GPUs: 1 \| vCPUs: 4 \| Memory: 16GiB | **0.7** | **6.944444444** | **12.15277778** |
| **ct5lp-hightpu-1t** *(Zhemin sheet \$1.20, 187 / 90 RPS)* | vCPUs: 24 \| Memory: 48 GB \| TPU HBM: 16 GB | **1.2** | **4.456327986** | **9.259259259** |
| **ct5lp-hightpu-1t** *(OD \$1.22, 187 / 90 RPS)* | vCPUs: 24 \| Memory: 48 GB \| TPU HBM: 16 GB | **1.22** | **4.530600119** | **9.413580247** |
| **ct5lp-hightpu-1t** *(OD \$1.22, SLA-passing 180 / 90 RPS)* | vCPUs: 24 \| Memory: 48 GB \| TPU HBM: 16 GB | **1.22** | **4.706790123** | **9.413580247** |
| **ct6e-standard-1t** *(`cb460828`, **490 / 320 RPS** max PASS)* | TPU v6e: 1 chip \| vCPUs: 44 \| Memory: 176 GB \| TPU HBM: 32 GB | **2.7** | **3.826530612** | **5.859375** |
| **ct6e-standard-1t** *(`cb460828`, **480 / 320 RPS** 20-RPS grid)* | TPU v6e: 1 chip \| vCPUs: 44 \| Memory: 176 GB \| TPU HBM: 32 GB | **2.7** | **3.90625** | **5.859375** |
| **Cost Improvement** *(v5e \$1.20 vs L4 \$0.70 — Zhemin Baseline)* | | | **35.83% reduction** | **23.81% reduction** |
| **Cost Improvement** *(**v6e \$2.70 vs L4 \$0.70** @ 490 / 320 RPS)* | | | **44.90% reduction** | **51.79% reduction** |
| **Cost Improvement** *(**v6e \$2.70 vs L4 \$0.70** @ 480 / 320 RPS)* | | | **43.75% reduction** | **51.79% reduction** |
| **Cost Improvement** *(**v6e \$2.70 vs v5e \$1.20** @ 187/90 vs 490/320 RPS)* | | | **14.13% reduction** | **36.72% reduction** |
| **Cost Improvement** *(**v6e \$2.70 vs v5e \$1.22** @ 187/90 vs 490/320 RPS)* | | | **15.54% reduction** | **37.76% reduction** |
| **Cost Improvement** *(**v6e \$2.70 vs v5e \$1.22** @ 180/90 vs 490/320 RPS)* | | | **18.70% reduction** | **37.76% reduction** |
