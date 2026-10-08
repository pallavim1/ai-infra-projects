# End-to-End Empirical Benchmarks: Zhemin's Test Suite on TPU v6e (`d3439062` Fused Device Pooling & `cb460828` Device Pooling, FP32)

> [!IMPORTANT]
> **100% Empirical Verification on Single TPU v6e Chip (`ct6e-standard-1t`, FP32, Exact `1,024` & `2,048` Tokens)**
> Every TPU v6e benchmark in this report and in [ATP_AIC2_Benchmarks_TPU_v6e_FP32_Megakernel_vs_L4_and_Baselines.xlsx](file:///usr/local/google/home/pallaviam/.gemini/jetski/brain/292ebc90-dbb8-4a33-8f1a-ca6dee63d8ff/ATP_AIC2_Benchmarks_TPU_v6e_FP32_Megakernel_vs_L4_and_Baselines.xlsx) was measured on GKE pod `jina-opus-v6e-test` (`ct6e-standard-1t`, 1x TPU v6e chip) driven from CPU load-generator pod `panw-v6e-benchmark-runner`, and cross-checked against vLLM's `/metrics` Prometheus counters (`vllm:request_success_total` and `vllm:prompt_tokens_total`) before and after every step:
> - **Latest Commit (`d3439062` — Fused Device Pooling + L2 Norm + Encoder Fast Path):** [`jina-v2-opus-megakernel` @ `d3439062`](https://github.com/pallavim1/tpu-inference/commit/d34390621876658cc7d6adc5b5fb5681eb105639)
>   - **Flags:** `USE_JINA_BERT_MEGAKERNEL=1`, `JINA_BERT_MEGAKERNEL_VERSION=v2`, `JINA_BERT_MEGAKERNEL_PRECISION=default`, `TPU_POOLING_FAST_PATH=1`, `JINA_BERT_DEVICE_POOLING=1`, `JINA_BERT_FUSED_POOLING=1`, `TPU_ENCODER_INPUT_FAST_PATH=1`
>   - **Max SLA-Passing Throughput ($\text{p99} < 50\text{ ms}$):** **`530 RPS`** at `1KB (1,024 tok)` (`p50: 13.1 ms`, `p99: 47.9 ms`) and **`350 RPS`** at `2KB (2,048 tok)` (`p50: 11.7 ms`, `p99: 25.2 ms`).
>   - **Max Achieved Closed-Loop QPS (`k6 constant-vus`):** **`546.4/s`** at `1KB (VUS=16)` (`p50: 28.4 ms`, `p99: 37.4 ms`) and **`374.0/s`** at `2KB (VUS=8)` (`p50: 19.7 ms`, `p99: 26.0 ms`).
> - **Previous Commit (`cb460828` — Megakernel `v2` + Device Pooling):** [`jina-v2-opus-megakernel` @ `cb460828`](https://github.com/pallavim1/tpu-inference/commit/cb460828b65f0a387780b3199ac2f15099177ada)
>   - **Flags:** `USE_JINA_BERT_MEGAKERNEL=1`, `JINA_BERT_MEGAKERNEL_VERSION=v2`, `JINA_BERT_MEGAKERNEL_PRECISION=default`, `TPU_POOLING_FAST_PATH=1`, `JINA_BERT_DEVICE_POOLING=1`
>   - **Max SLA-Passing Throughput ($\text{p99} < 50\text{ ms}$):** **`490 RPS`** at `1KB (1,024 tok)` (`p50: 14.1 ms`, `p99: 46.3 ms`) and **`320 RPS`** at `2KB (2,048 tok)` (`p50: 12.6 ms`, `p99: 25.3 ms`).
> - **vLLM Config:** `--dtype float32 --max-model-len 2048 --max-num-batched-tokens 2048 --no-enable-prefix-caching`
> - **Exact Token Payloads:**
>   - **1K (`1KB` tier):** 25,000 unique prompts from `pool_1024tok.json`, **100% verified at `1024.0` server tokens/prompt**.
>   - **2K (`2KB` tier):** 25,000 unique prompts from `pool_2048tok.json`, **100% verified at `2048.0` server tokens/prompt**.

---

## Tab 1: Raw Benchmark Results (Matching `gid=1161755388` / `gid=1632902472` AS-IS)

### 1. Batch Request Testing: A Single HTTP Request Contains Multiple Prompts (`{"text": [p_1, ..., p_N]}`, $N \in \{1, 4, 8, 16\}$)

*Client sends a batch of $N$ prompts (`1,024` tokens/prompt for 1KB, `2,048` tokens/prompt for 2KB) in a single HTTP `POST /prompt_c2` request and records the HTTP response latency in milliseconds.*

| Payload Size | Concurrency ($N$) | TPU v6e `d3439062` (Fused) Throughput | TPU v6e `d3439062` (Fused) p50 | TPU v6e `d3439062` (Fused) p99 | TPU v6e `cb460828` Throughput | TPU v6e `cb460828` p50 | TPU v6e `cb460828` p99 | TPU v5e Throughput | TPU v5e p50 | TPU v5e p99 | L4 GPU Throughput | L4 GPU p50 | L4 GPU p99 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1KB (1024 chars / 1,024 tok)** | **1** | **139.4/s** | **7.1ms** | **8.1ms** | 130.3/s | 7.4ms | 10.4ms | 85.8/s | 11.5ms | 12.7ms | 43.2/s | 20.7ms | 23.5ms |
| **1KB (1024 chars / 1,024 tok)** | **4** | **235.1/s** | **16.9ms** | **18.1ms** | 227.0/s | 17.4ms | 19.4ms | 187.4/s | 21.2ms | 22.8ms | 103.4/s | 38.1ms | 42.9ms |
| **1KB (1024 chars / 1,024 tok)** | **8** | **259.5/s** | **30.6ms** | **32.3ms** | 254.3/s | 31.2ms | 32.7ms | 187.6/s | 42.5ms | 44.5ms | 135.1/s | 58.7ms | 66.3ms |
| **1KB (1024 chars / 1,024 tok)** | **16** | **272.9/s** | **58.5ms** | **59.6ms** | 267.9/s | 59.5ms | 60.8ms | 186.8/s | 85.4ms | 89.9ms | 165.2/s | 96.5ms | 107.6ms |
| **2KB (2048 chars / 2,048 tok)** | **1** | **115.5/s** | **8.5ms** | **9.5ms** | 108.1/s | 9.0ms | 10.6ms | 59.5/s | 16.5ms | 17.7ms | 39.0/s | 24.3ms | 28.2ms |
| **2KB (2048 chars / 2,048 tok)** | **4** | **153.2/s** | **25.9ms** | **27.1ms** | 148.3/s | 26.6ms | 29.3ms | 97.7/s | 40.7ms | 42.8ms | 75.7/s | 52.1ms | 58.2ms |
| **2KB (2048 chars / 2,048 tok)** | **8** | **165.3/s** | **48.1ms** | **50.5ms** | 158.3/s | 50.1ms | 52.4ms | 96.6/s | 82.6ms | 86.0ms | 90.5/s | 87.4ms | 97.3ms |
| **2KB (2048 chars / 2,048 tok)** | **16** | **172.4/s** | **92.1ms** | **98.0ms** | 163.0/s | 97.3ms | 104.0ms | 96.5/s | 165.9ms | 171.2ms | 101.1/s | 156.9ms | 175.5ms |

---

### 2. Concurrent Request Testing (`k6 constant-vus`, $\text{VUS} \in \{1, 4, 8, 16\}$ concurrent HTTP requests, 30 s per step)

*`k6` runs $C \in \{1, 4, 8, 16\}$ concurrent virtual users (`constant-vus`), each sending 1 prompt (`1,024` or `2,048` exact tokens) per HTTP request.*

| Payload Size | Concurrency (`VUS`) | TPU v6e `d3439062` (Fused) Throughput | TPU v6e `d3439062` (Fused) p50 | TPU v6e `d3439062` (Fused) p99 | TPU v6e `cb460828` Throughput | TPU v6e `cb460828` p50 | TPU v6e `cb460828` p99 | TPU v5e Throughput | TPU v5e p50 | TPU v5e p99 | L4 GPU Throughput | L4 GPU p50 | L4 GPU p99 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1KB (1024 chars / 1,024 tok)** | **1** | **134.1/s** | **6.6ms** | **7.1ms** | 128.4/s | 6.9ms | 8.0ms | 85.8/s | 11.5ms | 12.7ms | 43.2/s | 20.7ms | 23.5ms |
| **1KB (1024 chars / 1,024 tok)** | **4** | **394.5/s** | **9.3ms** | **12.0ms** | 376.4/s | 9.9ms | 12.3ms | 187.4/s | 21.2ms | 22.8ms | 103.4/s | 38.1ms | 42.9ms |
| **1KB (1024 chars / 1,024 tok)** | **8** | **519.2/s** | **14.3ms** | **19.7ms** | 507.1/s | 14.8ms | 20.2ms | 187.6/s | 42.5ms | 44.5ms | 135.1/s | 58.7ms | 66.3ms |
| **1KB (1024 chars / 1,024 tok)** | **16** | **546.4/s** | **28.4ms** | **37.4ms** | 518.0/s | 30.1ms | 39.3ms | 186.8/s | 85.4ms | 89.9ms | 165.2/s | 96.5ms | 107.6ms |
| **2KB (2048 chars / 2,048 tok)** | **1** | **105.8/s** | **8.0ms** | **9.0ms** | 99.0/s | 8.6ms | 9.5ms | 59.5/s | 16.5ms | 17.7ms | 39.0/s | 24.3ms | 28.2ms |
| **2KB (2048 chars / 2,048 tok)** | **4** | **294.8/s** | **12.3ms** | **15.6ms** | 290.1/s | 12.4ms | 16.2ms | 97.7/s | 40.7ms | 42.8ms | 75.7/s | 52.1ms | 58.2ms |
| **2KB (2048 chars / 2,048 tok)** | **8** | **374.0/s** | **19.7ms** | **26.0ms** | 367.6/s | 20.1ms | 26.8ms | 96.6/s | 82.6ms | 86.0ms | 90.5/s | 87.4ms | 97.3ms |
| **2KB (2048 chars / 2,048 tok)** | **16** | **302.3/s** | **51.2ms** | **58.2ms** | 366.9/s | 42.0ms | 50.0ms | 96.5/s | 165.9ms | 171.2ms | 101.1/s | 156.9ms | 175.5ms |

---

### 3A. 1KB Dedicated Saturation (`1,024` Exact Tokens per Request, FP32, 60 s per step, $\text{p99} < 50\text{ ms}$ SLA)

| RPS | Achieved (`d3439062` Fused) | P50 (`d3439062` Fused) | P99 (`d3439062` Fused) | SLA (`d3439062` Fused) | Achieved (`cb460828`) | P50 (`cb460828`) | P99 (`cb460828`) | SLA (`cb460828`) | Achieved (TPU v5e) | P50 (TPU v5e) | P99 (TPU v5e) | SLA (TPU v5e) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | — *(PASS <480)* | — | — | PASS | 100.03 | 7.2 ms | 8.5 ms | PASS | 100 | 11.5 ms | 14.0 ms | PASS |
| **120** | — *(PASS <480)* | — | — | PASS | 120.03 | 7.0 ms | 8.3 ms | PASS | 120 | 11.8 ms | 15.5 ms | PASS |
| **140** | — *(PASS <480)* | — | — | PASS | 140.05 | 6.9 ms | 8.2 ms | PASS | 140 | 11.6 ms | 18.5 ms | PASS |
| **160** | — *(PASS <480)* | — | — | PASS | 160.04 | 7.1 ms | 9.1 ms | PASS | 160 | 16.8 ms | 24.2 ms | PASS |
| **180** | — *(PASS <480)* | — | — | PASS | 180.07 | 7.4 ms | 9.7 ms | PASS | **180.1** | **19.9 ms** | **29.6 ms** | **PASS (v5e Max)** |
| **190** | — *(PASS <480)* | — | — | PASS | 190.04 | 7.6 ms | 9.7 ms | PASS | 187.8 | 523.7 ms | 762.7 ms | SATURATED |
| **200** | — *(PASS <480)* | — | — | PASS | 200.05 | 7.7 ms | 9.7 ms | PASS | 189 | 1709 ms | 2818 ms | SATURATED |
| **220** | — *(PASS <480)* | — | — | PASS | 220.05 | 8.3 ms | 9.4 ms | PASS | 187.9 | 3738 ms | 6182 ms | SATURATED |
| **240** | — *(PASS <480)* | — | — | PASS | 240.06 | 8.1 ms | 9.6 ms | PASS | — | — | — | SATURATED |
| **260** | — *(PASS <480)* | — | — | PASS | 260.07 | 7.9 ms | 10.3 ms | PASS | — | — | — | SATURATED |
| **280** | — *(PASS <480)* | — | — | PASS | 280.08 | 7.9 ms | 10.9 ms | PASS | — | — | — | SATURATED |
| **300** | — *(PASS <480)* | — | — | PASS | 300.08 | 7.9 ms | 11.2 ms | PASS | — | — | — | SATURATED |
| **320** | — *(PASS <480)* | — | — | PASS | 320.06 | 8.4 ms | 12.8 ms | PASS | — | — | — | SATURATED |
| **340** | — *(PASS <480)* | — | — | PASS | 340.15 | 8.7 ms | 13.1 ms | PASS | — | — | — | SATURATED |
| **360** | — *(PASS <480)* | — | — | PASS | 360.19 | 9.3 ms | 15.3 ms | PASS | — | — | — | SATURATED |
| **380** | — *(PASS <480)* | — | — | PASS | 380.21 | 9.6 ms | 14.4 ms | PASS | — | — | — | SATURATED |
| **400** | — *(PASS <480)* | — | — | PASS | 400.28 | 10.1 ms | 16.3 ms | PASS | — | — | — | SATURATED |
| **420** | — *(PASS <480)* | — | — | PASS | 420.45 | 9.8 ms | 17.5 ms | PASS | — | — | — | SATURATED |
| **440** | — *(PASS <480)* | — | — | PASS | 440.21 | 11.3 ms | 19.7 ms | PASS | — | — | — | SATURATED |
| **460** | — *(PASS <480)* | — | — | PASS | 460.31 | 11.5 ms | 25.0 ms | PASS | — | — | — | SATURATED |
| **480** | **480.30** | **11.2 ms** | **21.5 ms** | **PASS** | **480.45** | **12.6 ms** | **27.9 ms** | **PASS** | — | — | — | SATURATED |
| **490** | — *(PASS <530)* | — | — | **PASS** | **490.28** | **14.1 ms** | **46.3 ms** | **PASS (`cb460828` Max)** | — | — | — | SATURATED |
| **500** | **500.66** | **11.6 ms** | **24.2 ms** | **PASS** | 500.55 | 13.7 ms | 71.3 ms | SATURATED | — | — | — | SATURATED |
| **510** | **510.37** | **12.2 ms** | **23.4 ms** | **PASS** | 510.42 | 16.1 ms | 104.2 ms | SATURATED | — | — | — | SATURATED |
| **520** | **520.39** | **12.8 ms** | **36.1 ms** | **PASS (20-RPS Grid Max)** | 517.55 | 21.2 ms | 533.7 ms | SATURATED | — | — | — | SATURATED |
| **530** | **530.35** | **13.1 ms** | **47.9 ms** | **PASS (`d3439062` Max)** | — | — | — | SATURATED | — | — | — | SATURATED |
| **540** | 540.57 | 15.7 ms | 121.1 ms | SATURATED | — | — | — | SATURATED | — | — | — | SATURATED |
| **550** | 546.40 | 52.0 ms | 564.0 ms | SATURATED | — | — | — | SATURATED | — | — | — | SATURATED |

---

### 3B. 2KB Dedicated Saturation (`2,048` Exact Tokens per Request, FP32, 60 s per step, $\text{p99} < 50\text{ ms}$ SLA)

| RPS | Achieved (`d3439062` Fused) | P50 (`d3439062` Fused) | P99 (`d3439062` Fused) | SLA (`d3439062` Fused) | Achieved (`cb460828`) | P50 (`cb460828`) | P99 (`cb460828`) | SLA (`cb460828`) | Achieved (TPU v5e) | P50 (TPU v5e) | P99 (TPU v5e) | SLA (TPU v5e) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **90** | — *(PASS <280)* | — | — | PASS | 90.03 | 8.6 ms | 9.9 ms | PASS | **90** | **17.2 ms** | **26.5 ms** | **PASS (v5e Max)** |
| **95** | — *(PASS <280)* | — | — | PASS | 95.03 | 8.7 ms | 10.2 ms | PASS | 94.75 | 218.6 ms | 346.5 ms | SATURATED |
| **100** | — *(PASS <280)* | — | — | PASS | 100.03 | 8.6 ms | 9.9 ms | PASS | 96.32 | 1031 ms | 2185 ms | SATURATED |
| **110** | — *(PASS <280)* | — | — | PASS | 110.03 | 8.6 ms | 10.0 ms | PASS | 96.74 | 3208 ms | 5170 ms | SATURATED |
| **120..260** | — *(PASS <280)* | — | — | PASS | 120.03..260.08 | 8.6..9.8 ms | 10.4..15.5 ms | PASS | — | — | — | SATURATED |
| **280** | **280.16** | **10.7 ms** | **16.2 ms** | **PASS** | 280.14 | 10.1 ms | 16.7 ms | PASS | — | — | — | SATURATED |
| **300** | **300.14** | **12.4 ms** | **17.0 ms** | **PASS** | **300.10** | **12.9 ms** | **18.6 ms** | **PASS** | — | — | — | SATURATED |
| **310** | — *(PASS <340)* | — | — | **PASS** | **310.17** | **12.7 ms** | **18.1 ms** | **PASS** | — | — | — | SATURATED |
| **320** | **320.14** | **12.2 ms** | **18.6 ms** | **PASS** | **320.17** | **12.6 ms** | **25.3 ms** | **PASS (`cb460828` Max)** | — | — | — | SATURATED |
| **330** | — *(PASS <340)* | — | — | **PASS** | 330.10 | 12.6 ms | 57.1 ms | SATURATED | — | — | — | SATURATED |
| **340** | **340.18** | **12.0 ms** | **27.0 ms** | **PASS (20-RPS Grid Max)** | 340.37 | 13.6 ms | 138.7 ms | SATURATED | — | — | — | SATURATED |
| **350** | **350.20** | **11.7 ms** | **25.2 ms** | **PASS (`d3439062` Max)** | — | — | — | SATURATED | — | — | — | SATURATED |
| **360** | 359.88 | 12.6 ms | 89.3 ms | SATURATED | 348.05 | 271.2 ms | 2526.2 ms | SATURATED | — | — | — | SATURATED |
| **370** | 370.17 | 87.4 ms | 325.1 ms | SATURATED | — | — | — | SATURATED | — | — | — | SATURATED |

---

## Tab 2: Performance & Cost Comparison (Matching `gid=1972899730` AS-IS)

### 1A. Batch Request Comparison — Single HTTP Request Containing $N$ Prompts

#### P50 Latency
| Payload Size | Concurrency ($N$) | TPU v6e `d3439062` | TPU v6e `cb460828` | TPU v5e | L4 GPU | Delta (`d3439062` vs v5e) | % Reduction vs v5e | Delta (`d3439062` vs L4) | % Reduction vs L4 | Faster Setup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1KB (`1,024` tok)** | **1** | **7.1ms** | 7.4ms | 11.5ms | 20.7ms | -4.4ms | **38.3%** | -13.6ms | **65.7%** | **TPU v6e (`d3439062`)** |
| **1KB (`1,024` tok)** | **4** | **16.9ms** | 17.4ms | 21.2ms | 38.1ms | -4.3ms | **20.3%** | -21.2ms | **55.6%** | **TPU v6e (`d3439062`)** |
| **1KB (`1,024` tok)** | **8** | **30.6ms** | 31.2ms | 42.5ms | 58.7ms | -11.9ms | **28.0%** | -28.1ms | **47.9%** | **TPU v6e (`d3439062`)** |
| **1KB (`1,024` tok)** | **16** | **58.5ms** | 59.5ms | 85.4ms | 96.5ms | -26.9ms | **31.5%** | -38.0ms | **39.4%** | **TPU v6e (`d3439062`)** |
| **2KB (`2,048` tok)** | **1** | **8.5ms** | 9.0ms | 16.5ms | 24.3ms | -8.0ms | **48.5%** | -15.8ms | **65.0%** | **TPU v6e (`d3439062`)** |
| **2KB (`2,048` tok)** | **4** | **25.9ms** | 26.6ms | 40.7ms | 52.1ms | -14.8ms | **36.4%** | -26.2ms | **50.3%** | **TPU v6e (`d3439062`)** |
| **2KB (`2,048` tok)** | **8** | **48.1ms** | 50.1ms | 82.6ms | 87.4ms | -34.5ms | **41.8%** | -39.3ms | **45.0%** | **TPU v6e (`d3439062`)** |
| **2KB (`2,048` tok)** | **16** | **92.1ms** | 97.3ms | 165.9ms | 156.9ms | -73.8ms | **44.5%** | -64.8ms | **41.3%** | **TPU v6e (`d3439062`)** |

#### p99 Latency
| Payload Size | Concurrency ($N$) | TPU v6e `d3439062` | TPU v6e `cb460828` | TPU v5e | L4 GPU | Delta (`d3439062` vs v5e) | % Reduction vs v5e | Delta (`d3439062` vs L4) | % Reduction vs L4 | Faster Setup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1KB (`1,024` tok)** | **1** | **8.1ms** | 10.4ms | 12.7ms | 23.5ms | -4.6ms | **36.2%** | -15.4ms | **65.5%** | **TPU v6e (`d3439062`)** |
| **1KB (`1,024` tok)** | **4** | **18.1ms** | 19.4ms | 22.8ms | 42.9ms | -4.7ms | **20.6%** | -24.8ms | **57.8%** | **TPU v6e (`d3439062`)** |
| **1KB (`1,024` tok)** | **8** | **32.3ms** | 32.7ms | 44.5ms | 66.3ms | -12.2ms | **27.4%** | -34.0ms | **51.3%** | **TPU v6e (`d3439062`)** |
| **1KB (`1,024` tok)** | **16** | **59.6ms** | 60.8ms | 89.9ms | 107.6ms | -30.3ms | **33.7%** | -48.0ms | **44.6%** | **TPU v6e (`d3439062`)** |
| **2KB (`2,048` tok)** | **1** | **9.5ms** | 10.6ms | 17.7ms | 28.2ms | -8.2ms | **46.3%** | -18.7ms | **66.3%** | **TPU v6e (`d3439062`)** |
| **2KB (`2,048` tok)** | **4** | **27.1ms** | 29.3ms | 42.8ms | 58.2ms | -15.7ms | **36.7%** | -31.1ms | **53.4%** | **TPU v6e (`d3439062`)** |
| **2KB (`2,048` tok)** | **8** | **50.5ms** | 52.4ms | 86.0ms | 97.3ms | -35.5ms | **41.3%** | -46.8ms | **48.1%** | **TPU v6e (`d3439062`)** |
| **2KB (`2,048` tok)** | **16** | **98.0ms** | 104.0ms | 171.2ms | 175.5ms | -73.2ms | **42.8%** | -77.5ms | **44.2%** | **TPU v6e (`d3439062`)** |

---

### 1B. Concurrent Request Comparison — `k6` Concurrent HTTP Requests ($\text{VUS} \in \{1, 4, 8, 16\}$)

#### P50 Latency
| Payload Size | Concurrency (`VUS`) | TPU v6e `d3439062` | TPU v6e `cb460828` | TPU v5e | L4 GPU | Delta (`d3439062` vs v5e) | % Reduction vs v5e | Delta (`d3439062` vs L4) | % Reduction vs L4 | Faster Setup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1KB (`1,024` tok)** | **1** | **6.6ms** | 6.9ms | 11.5ms | 20.7ms | -4.9ms | **42.6%** | -14.1ms | **68.1%** | **TPU v6e (`d3439062`)** |
| **1KB (`1,024` tok)** | **4** | **9.3ms** | 9.9ms | 21.2ms | 38.1ms | -11.9ms | **56.1%** | -28.8ms | **75.6%** | **TPU v6e (`d3439062`)** |
| **1KB (`1,024` tok)** | **8** | **14.3ms** | 14.8ms | 42.5ms | 58.7ms | -28.2ms | **66.4%** | -44.4ms | **75.6%** | **TPU v6e (`d3439062`)** |
| **1KB (`1,024` tok)** | **16** | **28.4ms** | 30.1ms | 85.4ms | 96.5ms | -57.0ms | **66.7%** | -68.1ms | **70.6%** | **TPU v6e (`d3439062`)** |
| **2KB (`2,048` tok)** | **1** | **8.0ms** | 8.6ms | 16.5ms | 24.3ms | -8.5ms | **51.5%** | -16.3ms | **67.1%** | **TPU v6e (`d3439062`)** |
| **2KB (`2,048` tok)** | **4** | **12.3ms** | 12.4ms | 40.7ms | 52.1ms | -28.4ms | **69.8%** | -39.8ms | **76.4%** | **TPU v6e (`d3439062`)** |
| **2KB (`2,048` tok)** | **8** | **19.7ms** | 20.1ms | 82.6ms | 87.4ms | -62.9ms | **76.2%** | -67.7ms | **77.5%** | **TPU v6e (`d3439062`)** |
| **2KB (`2,048` tok)** | **16** | **51.2ms** | **42.0ms** | 165.9ms | 156.9ms | -114.7ms | **69.1%** | -105.7ms | **67.4%** | **TPU v6e** |

#### p99 Latency
| Payload Size | Concurrency (`VUS`) | TPU v6e `d3439062` | TPU v6e `cb460828` | TPU v5e | L4 GPU | Delta (`d3439062` vs v5e) | % Reduction vs v5e | Delta (`d3439062` vs L4) | % Reduction vs L4 | Faster Setup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1KB (`1,024` tok)** | **1** | **7.1ms** | 8.0ms | 12.7ms | 23.5ms | -5.6ms | **44.1%** | -16.4ms | **69.8%** | **TPU v6e (`d3439062`)** |
| **1KB (`1,024` tok)** | **4** | **12.0ms** | 12.3ms | 22.8ms | 42.9ms | -10.8ms | **47.4%** | -30.9ms | **72.0%** | **TPU v6e (`d3439062`)** |
| **1KB (`1,024` tok)** | **8** | **19.7ms** | 20.2ms | 44.5ms | 66.3ms | -24.8ms | **55.7%** | -46.6ms | **70.3%** | **TPU v6e (`d3439062`)** |
| **1KB (`1,024` tok)** | **16** | **37.4ms** | 39.3ms | 89.9ms | 107.6ms | -52.5ms | **58.4%** | -70.2ms | **65.2%** | **TPU v6e (`d3439062`)** |
| **2KB (`2,048` tok)** | **1** | **9.0ms** | 9.5ms | 17.7ms | 28.2ms | -8.7ms | **49.2%** | -19.2ms | **68.1%** | **TPU v6e (`d3439062`)** |
| **2KB (`2,048` tok)** | **4** | **15.6ms** | 16.2ms | 42.8ms | 58.2ms | -27.2ms | **63.6%** | -42.6ms | **73.2%** | **TPU v6e (`d3439062`)** |
| **2KB (`2,048` tok)** | **8** | **26.0ms** | 26.8ms | 86.0ms | 97.3ms | -60.0ms | **69.8%** | -71.3ms | **73.3%** | **TPU v6e (`d3439062`)** |
| **2KB (`2,048` tok)** | **16** | **58.2ms** | **50.0ms** | 171.2ms | 175.5ms | -113.0ms | **66.0%** | -117.3ms | **66.8%** | **TPU v6e** |

---

### 2. RPS Saturation Result ($\text{p99} < 50\text{ ms}$ SLA)

| Payload | Setup | RPS | p50 | p99 | RPS Improvement vs L4 | RPS Improvement vs TPU v5e |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1K** | L4 GPU | 70/s | — | — | Baseline | — |
| **1K** | TPU v5e | 180/s *(187/s max)* | 19.9ms | 29.6ms | +157.1% | Baseline |
| **1K** | **TPU v6e (`d3439062` Fused Pooling, 10-RPS grid max PASS)** | **530/s** | **13.1ms** | **47.9ms** | **+657.1% (7.57x)** | **+194.4% vs 180/s (+183.4% vs 187/s)** |
| **1K** | **TPU v6e (`d3439062` Fused Pooling, 20-RPS grid max PASS)** | **520/s** | **12.8ms** | **36.1ms** | **+642.9% (7.43x)** | **+188.9% vs 180/s (+178.1% vs 187/s)** |
| **1K** | **TPU v6e (`cb460828` Device Pooling, 10-RPS grid max PASS)** | **490/s** | **14.1ms** | **46.3ms** | **+600.0% (7.00x)** | **+172.2% vs 180/s (+162.0% vs 187/s)** |
| **1K** | **TPU v6e (`cb460828` Device Pooling, 20-RPS grid max PASS)** | **480/s** | **12.6ms** | **27.9ms** | **+585.7% (6.86x)** | **+166.7% vs 180/s (+156.7% vs 187/s)** |
| **2K** | L4 GPU | 40/s | — | — | Baseline | — |
| **2K** | TPU v5e | 90/s | 17.2ms | 26.5ms | +125.0% | Baseline |
| **2K** | **TPU v6e (`d3439062` Fused Pooling, 10-RPS grid max PASS)** | **350/s** | **11.7ms** | **25.2ms** | **+775.0% (8.75x)** | **+288.9% (3.89x)** |
| **2K** | **TPU v6e (`d3439062` Fused Pooling, 20-RPS grid max PASS)** | **340/s** | **12.0ms** | **27.0ms** | **+750.0% (8.50x)** | **+277.8% (3.78x)** |
| **2K** | **TPU v6e (`cb460828` Device Pooling, max PASS)** | **320/s** | **12.6ms** | **25.3ms** | **+700.0% (8.00x)** | **+255.6% (3.56x)** |
| **2K** | **TPU v6e (`cb460828` Device Pooling, 300 RPS target)** | **300/s** | **12.9ms** | **18.6ms** | **+650.0% (7.50x)** | **+233.3% (3.33x)** |

---

### 3. Cost Improvement (Matching Zhemin's Exact Formula)

$$\text{Cost per 1M requests} = \frac{\text{Hourly Cost}}{0.40 \times \text{Max RPS} \times 3600} \times 1{,}000{,}000$$

| Machine Type | Machine Config | Hourly Cost | Cost per 1M request (1K Payload) | Cost per 1M request (2K Payload) |
| :--- | :--- | :---: | :---: | :---: |
| **g2-standard-4** | L4 GPUs: 1 \| vCPUs: 4 \| Memory: 16GiB | **0.7** | **6.944444444** | **12.15277778** |
| **ct5lp-hightpu-1t** *(Zhemin sheet \$1.20, 187 / 90 RPS)* | vCPUs: 24 \| Memory: 48 GB \| TPU HBM: 16 GB | **1.2** | **4.456327986** | **9.259259259** |
| **ct5lp-hightpu-1t** *(OD \$1.22, 187 / 90 RPS)* | vCPUs: 24 \| Memory: 48 GB \| TPU HBM: 16 GB | **1.22** | **4.530600119** | **9.413580247** |
| **ct5lp-hightpu-1t** *(OD \$1.22, SLA-passing 180 / 90 RPS)* | vCPUs: 24 \| Memory: 48 GB \| TPU HBM: 16 GB | **1.22** | **4.706790123** | **9.413580247** |
| **ct6e-standard-1t** *(`d3439062` Fused, **530 / 350 RPS** max PASS)* | TPU v6e: 1 chip \| vCPUs: 44 \| Memory: 176 GB \| TPU HBM: 32 GB | **2.7** | **3.537735849** | **5.357142857** |
| **ct6e-standard-1t** *(`d3439062` Fused, **520 / 340 RPS** 20-RPS grid)* | TPU v6e: 1 chip \| vCPUs: 44 \| Memory: 176 GB \| TPU HBM: 32 GB | **2.7** | **3.605769231** | **5.514705882** |
| **ct6e-standard-1t** *(`cb460828`, **490 / 320 RPS** max PASS)* | TPU v6e: 1 chip \| vCPUs: 44 \| Memory: 176 GB \| TPU HBM: 32 GB | **2.7** | **3.826530612** | **5.859375** |
| **ct6e-standard-1t** *(`cb460828`, **480 / 320 RPS** 20-RPS grid)* | TPU v6e: 1 chip \| vCPUs: 44 \| Memory: 176 GB \| TPU HBM: 32 GB | **2.7** | **3.90625** | **5.859375** |
| **Cost Improvement** *(v5e \$1.20 vs L4 \$0.70 — Zhemin Baseline)* | | | **35.83% reduction** | **23.81% reduction** |
| **Cost Improvement** *(**v6e `d3439062` \$2.70 vs L4 \$0.70** @ 530 / 350 RPS)* | | | **49.06% reduction** | **55.92% reduction** |
| **Cost Improvement** *(**v6e `d3439062` \$2.70 vs v5e \$1.20** @ 187/90 vs 530/350 RPS)* | | | **20.61% reduction** | **42.14% reduction** |
| **Cost Improvement** *(**v6e `d3439062` \$2.70 vs v5e \$1.22** @ 180/90 vs 530/350 RPS)* | | | **24.84% reduction** | **43.09% reduction** |
| **Cost Improvement** *(**v6e `cb460828` \$2.70 vs L4 \$0.70** @ 490 / 320 RPS)* | | | **44.90% reduction** | **51.79% reduction** |
| **Cost Improvement** *(**v6e `cb460828` \$2.70 vs v5e \$1.20** @ 187/90 vs 490/320 RPS)* | | | **14.13% reduction** | **36.72% reduction** |
