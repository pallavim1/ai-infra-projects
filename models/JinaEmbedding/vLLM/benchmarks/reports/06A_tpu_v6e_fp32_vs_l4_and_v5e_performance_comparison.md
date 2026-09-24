# Table 2A: Cloud TPU v6e (`FP32`) vs NVIDIA L4 GPU & TPU v5e (`Performance & TCO Comparison`)

* **Model:** `jinaai/jina-embeddings-v2-small-en` (`--max-model-len 2048`, restricted at `2,048` tokens via `BertTokenizerFast`, `--max-num-seqs 40 --max-num-batched-tokens 8192`)
* **TPU v6e Configuration:** `ct6e-standard-1t` (1x Cloud TPU v6e Trillium chip, 44 vCPUs, 176 GB RAM, 32 GB HBM, `vLLM 0.26.0 --dtype float32`, GKE cluster `pm-panw-jina-cluster`, node pool `pm-panw-jina-v6e-pool` in `europe-west4-a`)
* **TPU v5e Configuration:** `ct5lp-hightpu-1t` (1x Cloud TPU v5e chip, 24 vCPUs, 45.6 GB RAM, 16 GB HBM, `vLLM 0.26.0 --dtype float32`, `pm-panw-jina-cluster`)
* **GPU Baseline Configuration:** `g2-standard-4` (1x NVIDIA L4 GPU, 4 vCPUs, 16 GiB RAM, 24 GB VRAM, Triton Inference Server + TensorRT)

---

## 1. Online Concurrent Request Comparison (`k6 constant-vus`)

### 1A. `P50` Latency & Throughput (`TPU v6e FP32` vs `NVIDIA L4` & `TPU v5e FP32`)

| Payload Size | Concurrency | **TPU v6e (`FP32`) Tput** | **TPU v6e (`p50`)** | TPU v5e (`FP32`) (`p50`) | L4 GPU (`p50`) | **v6e vs L4 Delta (`% Reduction`)** | **v6e vs v5e (`FP32`) Delta (`% Reduction`)** | Faster Setup |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: |
| **1KB** | **1** | **100.0 req/s** | **9.8ms** | 12.1ms | 20.7ms | **v6e is 10.9ms faster (`52.7%` lower)** | **v6e is 2.3ms faster (`19.0%` lower)** | **TPU v6e** |
| **1KB** | **4** | **219.4 req/s** | **17.8ms** | 22.8ms | 38.1ms | **v6e is 20.3ms faster (`53.3%` lower)** | **v6e is 5.0ms faster (`21.9%` lower)** | **TPU v6e** |
| **1KB** | **8** | **181.2 req/s** | **41.3ms** | 51.5ms | 58.7ms | **v6e is 17.4ms faster (`29.6%` lower)** | **v6e is 10.2ms faster (`19.8%` lower)** | **TPU v6e** |
| **1KB** | **16** | **91.3 req/s** | **174.3ms** | 183.1ms | 96.5ms | GPU is 77.8ms faster (`44.6%` lower) | **v6e is 8.8ms faster (`4.8%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **2KB** | **1** | **71.5 req/s** | **13.8ms** | 16.2ms | 24.3ms | **v6e is 10.5ms faster (`43.2%` lower)** | **v6e is 2.4ms faster (`14.8%` lower)** | **TPU v6e** |
| **2KB** | **4** | **47.4 req/s** | **97.4ms** | 46.5ms | 52.1ms | GPU is 45.3ms faster (`46.5%` lower) | v5e is 50.9ms faster | **TPU v5e / L4** |
| **2KB** | **8** | **46.3 req/s** | **172.8ms** | 183.1ms | 87.4ms | GPU is 85.4ms faster (`49.4%` lower) | **v6e is 10.3ms faster (`5.6%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **2KB** | **16** | **57.5 req/s** | **260.7ms** | 275.4ms | 156.9ms | GPU is 103.8ms faster (`39.8%` lower) | **v6e is 14.7ms faster (`5.3%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **3KB** | **1** | **65.5 req/s** | **15.0ms** | 17.6ms | 26.6ms | **v6e is 11.6ms faster (`43.6%` lower)** | **v6e is 2.6ms faster (`14.8%` lower)** | **TPU v6e** |
| **3KB** | **4** | **97.9 req/s** | **40.4ms** | 45.4ms | 57.0ms | **v6e is 16.6ms faster (`29.1%` lower)** | **v6e is 5.0ms faster (`11.0%` lower)** | **TPU v6e** |
| **3KB** | **8** | **46.1 req/s** | **172.9ms** | 183.2ms | 90.8ms | GPU is 82.1ms faster (`47.5%` lower) | **v6e is 10.3ms faster (`5.6%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **3KB** | **16** | **45.9 req/s** | **347.9ms** | 367.3ms | 167.8ms | GPU is 180.1ms faster (`51.8%` lower) | **v6e is 19.4ms faster (`5.3%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **4KB** | **1** | **66.2 req/s** | **14.8ms** | 17.5ms | 28.6ms | **v6e is 13.8ms faster (`48.3%` lower)** | **v6e is 2.7ms faster (`15.4%` lower)** | **TPU v6e** |
| **4KB** | **4** | **97.6 req/s** | **40.6ms** | 45.5ms | 60.6ms | **v6e is 20.0ms faster (`33.0%` lower)** | **v6e is 4.9ms faster (`10.8%` lower)** | **TPU v6e** |
| **4KB** | **8** | **46.1 req/s** | **172.8ms** | 183.8ms | 100.2ms | GPU is 72.6ms faster (`42.0%` lower) | **v6e is 11.0ms faster (`6.0%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **4KB** | **16** | **46.1 req/s** | **346.5ms** | 368.1ms | 180.5ms | GPU is 166.0ms faster (`47.9%` lower) | **v6e is 21.6ms faster (`5.9%` lower)** | **L4 GPU** *(v6e beats v5e)* |

---

### 1B. `P99` Latency (`TPU v6e FP32` vs `NVIDIA L4` & `TPU v5e FP32`)

| Payload Size | Concurrency | **TPU v6e (`p99`)** | TPU v5e (`FP32`) (`p99`) | L4 GPU (`p99`) | **v6e vs L4 Delta (`% Reduction`)** | **v6e vs v5e (`FP32`) Delta (`% Reduction`)** | Faster Setup |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- | :---: |
| **1KB** | **1** | **11.5ms** | 13.9ms | 23.5ms | **v6e is 12.0ms faster (`51.1%` lower)** | **v6e is 2.4ms faster (`17.3%` lower)** | **TPU v6e** |
| **1KB** | **4** | **32.8ms** | 43.1ms | 42.9ms | **v6e is 10.1ms faster (`23.5%` lower)** | **v6e is 10.3ms faster (`23.9%` lower)** | **TPU v6e** |
| **1KB** | **8** | **110.7ms** | 126.9ms | 66.3ms | GPU is 44.4ms faster (`40.1%` lower) | **v6e is 16.2ms faster (`12.8%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **1KB** | **16** | **179.6ms** | 189.6ms | 107.6ms | GPU is 72.0ms faster (`40.1%` lower) | **v6e is 10.0ms faster (`5.3%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **2KB** | **1** | **15.0ms** | 18.0ms | 28.2ms | **v6e is 13.2ms faster (`46.8%` lower)** | **v6e is 3.0ms faster (`16.7%` lower)** | **TPU v6e** |
| **2KB** | **4** | **107.0ms** | 49.3ms | 58.2ms | GPU is 48.8ms faster (`45.6%` lower) | v5e is 57.7ms faster | **TPU v5e / L4** |
| **2KB** | **8** | **176.7ms** | 188.7ms | 97.3ms | GPU is 79.4ms faster (`44.9%` lower) | **v6e is 12.0ms faster (`6.4%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **2KB** | **16** | **348.0ms** | 369.5ms | 175.5ms | GPU is 172.5ms faster (`49.6%` lower) | **v6e is 21.5ms faster (`5.8%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **3KB** | **1** | **16.8ms** | 19.8ms | 31.3ms | **v6e is 14.5ms faster (`46.3%` lower)** | **v6e is 3.0ms faster (`15.2%` lower)** | **TPU v6e** |
| **3KB** | **4** | **43.3ms** | 48.7ms | 63.7ms | **v6e is 20.4ms faster (`32.0%` lower)** | **v6e is 5.4ms faster (`11.1%` lower)** | **TPU v6e** |
| **3KB** | **8** | **177.0ms** | 189.1ms | 100.2ms | GPU is 76.8ms faster (`43.4%` lower) | **v6e is 12.1ms faster (`6.4%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **3KB** | **16** | **352.2ms** | 373.2ms | 189.0ms | GPU is 163.2ms faster (`46.3%` lower) | **v6e is 21.0ms faster (`5.6%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **4KB** | **1** | **16.1ms** | 19.3ms | 34.2ms | **v6e is 18.1ms faster (`52.9%` lower)** | **v6e is 3.2ms faster (`16.6%` lower)** | **TPU v6e** |
| **4KB** | **4** | **43.1ms** | 48.9ms | 68.5ms | **v6e is 25.4ms faster (`37.1%` lower)** | **v6e is 5.8ms faster (`11.9%` lower)** | **TPU v6e** |
| **4KB** | **8** | **176.8ms** | 189.4ms | 113.9ms | GPU is 62.9ms faster (`35.6%` lower) | **v6e is 12.6ms faster (`6.7%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **4KB** | **16** | **350.3ms** | 374.1ms | 204.0ms | GPU is 146.3ms faster (`41.8%` lower) | **v6e is 23.8ms faster (`6.4%` lower)** | **L4 GPU** *(v6e beats v5e)* |

---

## 2. RPS Saturation & Multi-Payload SLA Results (`P99 < 50 ms` SLA)

### 2A. Per-Payload SLA Saturation Ceiling (`P99 <= 50 ms` in `FP32`)

| Payload | Setup | Max SLA RPS (`P99 <= 50 ms`) | `p50` @ Max SLA | `p99` @ Max SLA | **v6e vs L4 Improvement (`(v6e - L4) / L4`)** | **v6e vs v5e (`FP32`) Improvement** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1K** | L4 + Triton | 70 | 20.0ms | 37.0ms | — | — |
| **1K** | TPU v5e + vLLM (`FP32`) | 140 | 18.7ms | 27.6ms | `+100.0%` (`2.00x`) | — |
| **1K** | **TPU v6e + vLLM (`FP32`)** | **160** *(**180** @ **39.4ms $P_{50}$, 50.1ms $P_{99}$**)* | **10.3ms** *(39.4ms @ 180)* | **13.7ms** *(50.1ms @ 180)* | **`+128.6%` (`2.29x` @ 160 RPS)**<br>**`+157.1%` (`2.57x` @ 180 RPS)** | **`+14.3%` (`160` vs `140` RPS)**<br>**`+28.6%` (`180` vs `140` RPS)** |
| **2K** | L4 + Triton | 40 | 22.0ms | 28.0ms | — | — |
| **2K** | TPU v5e + vLLM (`FP32`) | 70 | 17.6ms | 21.8ms | `+75.0%` (`1.75x`) | — |
| **2K** | **TPU v6e + vLLM (`FP32`)** | **100** | **14.4ms** | **18.9ms** | **`+150.0%` (`2.50x` total RPS)** | **`+42.9%` (`100` vs `70` RPS)** |
| **3K** | L4 + Triton | 30 | 25.0ms | 35.0ms | — | — |
| **3K** | TPU v5e + vLLM (`FP32`) | 70 | 18.2ms | 22.1ms | `+133.3%` (`2.33x`) | — |
| **3K** | **TPU v6e + vLLM (`FP32`)** | **110** | **15.2ms** | **20.7ms** | **`+266.7%` (`3.67x` total RPS)** | **`+57.1%` (`110` vs `70` RPS)** |
| **5K** | L4 + Triton | 20 | 31.2ms | 49.1ms | — | — |
| **5K** | TPU v5e + vLLM (`FP32`) | 70 | 18.0ms | 22.4ms | `+250.0%` (`3.50x`) | — |
| **5K** | **TPU v6e + vLLM (`FP32`)** | **100** *(110 @ 18.6ms)* | **15.8ms** | **21.6ms** | **`+400.0%` (`5.00x` total RPS)** | **`+42.9%` (`100` vs `70` RPS)** |
| **7K** | L4 + Triton | 10 | 36.6ms | 46.4ms | — | — |
| **7K** | TPU v5e + vLLM (`FP32`) | 70 | 18.5ms | 23.9ms | `+600.0%` (`7.00x`) | — |
| **7K** | **TPU v6e + vLLM (`FP32`)** | **100** | **16.1ms** | **19.7ms** | **`+900.0%` (`10.00x` total RPS)** | **`+42.9%` (`100` vs `70` RPS)** |

### 2B. Suite 2 Multi-Payload Concurrent Sweep (`1KB, 2KB, 5KB, 7KB` Simultaneously on `TPU v6e FP32`)

| Target RPS | `1KB p99` | `2KB p99` | `5KB p99` | `7KB p99` | Error Rate | SLA Status (`All 4 Payloads <= 50ms`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **50 RPS** | 12.0ms | 15.5ms | 17.2ms | 17.6ms | `0.0%` | **PASS** *(L4 max ceiling)* |
| **60 RPS** | 12.0ms | 15.4ms | 16.4ms | 17.4ms | `0.0%` | **PASS** |
| **70 RPS** | 11.7ms | 15.8ms | 19.5ms | 21.2ms | `0.0%` | **PASS** *(TPU v5e FP32 ceiling = 70 RPS)* |
| **80 RPS** | 12.3ms | 17.1ms | 19.7ms | 20.7ms | `0.0%` | **PASS** |
| **90 RPS** | 15.7ms | 17.6ms | 20.1ms | 20.4ms | `0.0%` | **PASS** |
| **100 RPS** | **12.7ms** | **19.2ms** | **21.6ms** | **19.7ms** | **`0.0%`** | **PASS (`TPU v6e FP32 Multi-Payload Ceiling`)** |
| **110 RPS** | 192.6ms | 1907.0ms | 18.6ms | 105.3ms | `0.0%` | Saturated |

---

## 3. Batch Benchmark Results (`Multi-Prompt Single-Request` & `High-Batch Token Sweep`)

### 3A. Zhemin's Multi-Prompt Single-Request Batch Test (`Batch = 1, 4, 8, 16` in `FP32`)

| Payload Size | Batch Size | **TPU v6e (`FP32`) Tput** | **TPU v6e `p50`** | **TPU v6e `p99`** | TPU v5e (`FP32`) Tput | TPU v5e `p50` | **v6e vs v5e (`FP32`) Speedup** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1KB (`1024 chars`)** | **1** | **88.0 emb/s** | **11.1ms** | **16.5ms** | 69.8 emb/s | 14.1ms | **`+26.1%` higher tput** |
| **1KB (`1024 chars`)** | **4** | **119.1 emb/s** | **32.4ms** | **38.8ms** | 104.2 emb/s | 38.0ms | **`+14.3%` higher tput** |
| **1KB (`1024 chars`)** | **8** | **99.7 emb/s** | **60.7ms** | **116.3ms** | 84.5 emb/s | 94.7ms | **`+18.0%` higher tput** |
| **1KB (`1024 chars`)** | **16** | **97.2 emb/s** | **149.0ms** | **211.1ms** | 86.1 emb/s | 185.8ms | **`+12.9%` higher tput** |
| **2KB (`2048 chars`)** | **1** | **66.2 emb/s** | **15.0ms** | **16.2ms** | 53.5 emb/s | 18.6ms | **`+23.7%` higher tput** |
| **2KB (`2048 chars`)** | **4** | **40.0 emb/s** | **102.0ms** | **111.9ms** | 36.8 emb/s | 108.7ms | **`+8.7%` higher tput** |
| **2KB (`2048 chars`)** | **8** | **40.5 emb/s** | **197.8ms** | **199.3ms** | 37.9 emb/s | 211.1ms | **`+6.9%` higher tput** |
| **2KB (`2048 chars`)** | **16** | **49.4 emb/s** | **313.1ms** | **389.5ms** | 45.2 emb/s | 354.0ms | **`+9.3%` higher tput** |
| **3KB (`3072 chars`)** | **1** | **62.2 emb/s** | **15.9ms** | **17.2ms** | 51.0 emb/s | 19.5ms | **`+22.0%` higher tput** |
| **3KB (`3072 chars`)** | **16** | **40.1 emb/s** | **401.6ms** | **411.2ms** | 37.5 emb/s | 426.7ms | **`+6.9%` higher tput** |
| **4KB (`4096 chars`)** | **1** | **60.9 emb/s** | **16.3ms** | **18.0ms** | 49.8 emb/s | 20.0ms | **`+22.3%` higher tput** |
| **4KB (`4096 chars`)** | **16** | **39.6 emb/s** | **404.3ms** | **413.8ms** | 37.4 emb/s | 427.8ms | **`+5.9%` higher tput** |

### 3B. High-Batch Token Sweep (`B = 1..128` across `128..2048` Tokens on `TPU v6e FP32`)

| Token Length (`L`) | Peak Batch (`B`) | **Embeddings / sec (`Emb/s`)** | **Tokens / sec (`Tok/s`)** | `p50` Latency | `p99` Latency |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **128 tokens** | **B = 8** | **206.3 emb/s** | **26,406.4 tok/s** | 40.8ms | 44.5ms |
| **256 tokens** | **B = 8 / 128** | **103.4 emb/s** *(102.9 @ B=128)* | **26,473.0 tok/s** | 59.7ms | 114.5ms |
| **512 tokens** | **B = 1 / 32** | **64.8 emb/s** *(50.4 @ B=32)* | **33,182.7 tok/s** | 15.2ms | 16.4ms |
| **1024 tokens** | **B = 1 / 16** | **62.2 emb/s** *(40.4 @ B=16)* | **63,723.5 tok/s** | 15.8ms | 17.4ms |
| **2048 tokens** | **B = 1 / 8** | **59.6 emb/s** *(41.5 @ B=8)* | **122,081.3 tok/s** | 16.7ms | 18.0ms |

---

## 4. TCO & Cost per 1M Requests Analysis (`TPU v6e FP32` vs `L4` & `TPU v5e FP32`)

| Machine Type & Pricing Tier | Machine Config | Hourly Cost | Cost / 1M Req (`1K` Payload) | Cost / 1M Req (`2K` Payload) | Cost / 1M Req (`3K` Payload) | Cost / 1M Req (`5K` Payload) | Cost / 1M Req (`7K` Payload) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`g2-standard-4` (L4 On-Demand)** | 1x L4 GPU (24GB)<br>4 vCPUs, 16 GiB | **\$0.70** | **\$6.94** *(70 RPS)* | **\$12.15** *(40 RPS)* | **\$16.20** *(30 RPS)* | **\$24.31** *(20 RPS)* | **\$48.61** *(10 RPS)* |
| **`ct5lp-hightpu-1t` (`v5e FP32` On-Demand)** | 1x TPU v5e (16GB)<br>24 vCPUs, 45.6 GB | **\$1.20** | **\$5.95** *(140 RPS)* | **\$11.90** *(70 RPS)* | **\$11.90** *(70 RPS)* | **\$11.90** *(70 RPS)* | **\$11.90** *(70 RPS)* |
| **`ct6e-standard-1t` (`v6e FP32` On-Demand)** | 1x TPU v6e (32GB)<br>44 vCPUs, 176 GB | **\$2.70** | **\$11.72** *(160 RPS)*<br>**\$10.42** *(180 RPS)* | **\$18.75** *(100 RPS)* | **\$17.05** *(110 RPS)* | **\$18.75** *(100 RPS)* | **\$18.75** *(100 RPS)* |
| **`ct6e-standard-1t` (`v6e FP32` 1-Yr CUD)** | 1x TPU v6e (32GB)<br>37% CUD Discount | **\$1.70** | **\$7.38** *(160 RPS)*<br>**\$6.56** *(180 RPS)* | **\$11.81** *(100 RPS)* | **\$10.73** *(110 RPS)* | **\$11.81** *(100 RPS)* | **\$11.81** *(100 RPS)* |
| **`ct6e-standard-1t` (`v6e FP32` 3-Yr CUD)** | 1x TPU v6e (32GB)<br>55% CUD Discount | **\$1.22** | **\$5.30** *(160 RPS)*<br>**\$4.71** *(180 RPS)* | **\$8.47** *(100 RPS)* | **\$7.70** *(110 RPS)* | **\$8.47** *(100 RPS)* | **\$8.47** *(100 RPS)* |
| **v6e On-Demand vs L4 Cost Reduction** | | | *+50.1% (@180 RPS)* | *+54.3% (@100 RPS)* | *+5.2% (@110 RPS)* | **22.87% reduction** | **61.43% reduction** |
| **v6e 1-Yr CUD vs L4 Cost Reduction** | | | **5.48% reduction** *(@180 RPS)* | **2.80% reduction** | **33.77% reduction** | **51.42% reduction** | **75.70% reduction** |
| **v6e 3-Yr CUD vs L4 Cost Reduction** | | | **32.13% reduction** *(@180 RPS)* | **30.29% reduction** | **52.47% reduction** | **65.16% reduction** | **82.58% reduction** |

---

## 5. TPU v6e vs TPU v5e vs L4 Regional Availability

| Region Name | Region ID | TPU v6e (`ct6e`) Zones | TPU v5e (`ct5lp`) Zones | L4 GPU (`g2`) Zones | Overlap / Multi-Accelerator Region? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Netherlands** | `europe-west4` | `europe-west4-a` | `europe-west4-b` | `a, b, c` | **Yes (All 3: v6e, v5e, L4)** |
| **South Carolina** | `us-east1` | `us-east1-d` | — | `b, c, d` | **Yes (v6e + L4)** |
| **Columbus, Ohio** | `us-east5` | `us-east5-a, b, c` | `us-east5-a` | `a, b, c` | **Yes (All 3: v6e, v5e, L4)** |
| **Iowa** | `us-central1` | `us-central1-b` | `a, b, c, f` | `a, b, c, f` | **Yes (All 3: v6e, v5e, L4)** |
| **Oregon** | `us-west1` | — | `us-west1-c` | `a, b, c` | Yes (v5e + L4) |
| **Las Vegas** | `us-west4` | `us-west4-a` | `us-west4-a` | `a, b, c` | **Yes (All 3: v6e, v5e, L4)** |
