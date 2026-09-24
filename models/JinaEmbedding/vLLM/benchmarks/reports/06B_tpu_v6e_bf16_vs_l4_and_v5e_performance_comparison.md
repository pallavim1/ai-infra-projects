# Table 2B: Cloud TPU v6e (`BF16`) vs NVIDIA L4 GPU & TPU v5e (`Performance & TCO Comparison`)

* **Model:** `jinaai/jina-embeddings-v2-small-en` (`--max-model-len 2048`, restricted at `2,048` tokens via `BertTokenizerFast`, `--max-num-seqs 40 --max-num-batched-tokens 8192`)
* **TPU v6e Configuration:** `ct6e-standard-1t` (1x Cloud TPU v6e Trillium chip, 44 vCPUs, 176 GB RAM, 32 GB HBM, `vLLM 0.26.0 --dtype bfloat16`, GKE cluster `pm-panw-jina-cluster`, node pool `pm-panw-jina-v6e-pool` in `europe-west4-a`)
* **TPU v5e Configuration:** `ct5lp-hightpu-1t` (1x Cloud TPU v5e chip, 24 vCPUs, 45.6 GB RAM, 16 GB HBM, `vLLM 0.26.0 --dtype bfloat16`, `pm-panw-jina-cluster`)
* **GPU Baseline Configuration:** `g2-standard-4` (1x NVIDIA L4 GPU, 4 vCPUs, 16 GiB RAM, 24 GB VRAM, Triton Inference Server + TensorRT)

---

## 1. Online Concurrent Request Comparison (`k6 constant-vus`)

### 1A. `P50` Latency & Throughput (`TPU v6e BF16` vs `NVIDIA L4` & `TPU v5e BF16`)

| Payload Size | Concurrency | **TPU v6e (`BF16`) Tput** | **TPU v6e (`p50`)** | TPU v5e (`p50`) | L4 GPU (`p50`) | **v6e vs L4 Delta (`% Reduction`)** | **v6e vs v5e Delta (`% Reduction`)** | Faster Setup |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :---: |
| **1KB** | **1** | **96.8 req/s** | **10.1ms** | 11.8ms | 20.7ms | **v6e is 10.6ms faster (`51.2%` lower)** | **v6e is 1.7ms faster (`14.4%` lower)** | **TPU v6e** |
| **1KB** | **4** | **208.9 req/s** | **18.7ms** | 21.7ms | 38.1ms | **v6e is 19.4ms faster (`50.9%` lower)** | **v6e is 3.0ms faster (`13.8%` lower)** | **TPU v6e** |
| **1KB** | **8** | **144.5 req/s** | **43.1ms** | 47.1ms | 58.7ms | **v6e is 15.6ms faster (`26.6%` lower)** | **v6e is 4.0ms faster (`8.5%` lower)** | **TPU v6e** |
| **1KB** | **16** | **119.2 req/s** | **133.5ms** | 140.3ms | 96.5ms | GPU is 37.0ms faster (`27.7%` lower) | **v6e is 6.8ms faster (`4.8%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **2KB** | **1** | **69.7 req/s** | **14.1ms** | 16.5ms | 24.3ms | **v6e is 10.2ms faster (`42.0%` lower)** | **v6e is 2.4ms faster (`14.5%` lower)** | **TPU v6e** |
| **2KB** | **4** | **59.4 req/s** | **76.0ms** | 47.3ms | 52.1ms | GPU is 23.9ms faster (`31.4%` lower) | v5e is 28.7ms faster | **TPU v5e / L4** |
| **2KB** | **8** | **61.3 req/s** | **129.8ms** | 130.9ms | 87.4ms | GPU is 42.4ms faster (`32.7%` lower) | **v6e is 1.1ms faster (`0.8%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **2KB** | **16** | **76.0 req/s** | **197.2ms** | 210.4ms | 156.9ms | GPU is 40.3ms faster (`20.4%` lower) | **v6e is 13.2ms faster (`6.3%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **3KB** | **1** | **65.2 req/s** | **15.1ms** | 17.2ms | 26.6ms | **v6e is 11.5ms faster (`43.2%` lower)** | **v6e is 2.1ms faster (`12.2%` lower)** | **TPU v6e** |
| **3KB** | **4** | **94.9 req/s** | **41.9ms** | 46.1ms | 57.0ms | **v6e is 15.1ms faster (`26.5%` lower)** | **v6e is 4.2ms faster (`9.1%` lower)** | **TPU v6e** |
| **3KB** | **8** | **60.8 req/s** | **130.9ms** | 138.9ms | 90.8ms | GPU is 40.1ms faster (`30.6%` lower) | **v6e is 8.0ms faster (`5.8%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **3KB** | **16** | **61.0 req/s** | **261.3ms** | 279.7ms | 167.8ms | GPU is 93.5ms faster (`35.8%` lower) | **v6e is 18.4ms faster (`6.6%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **4KB** | **1** | **62.9 req/s** | **15.7ms** | 18.2ms | 28.6ms | **v6e is 12.9ms faster (`45.1%` lower)** | **v6e is 2.5ms faster (`13.7%` lower)** | **TPU v6e** |
| **4KB** | **4** | **94.7 req/s** | **41.9ms** | 46.8ms | 60.6ms | **v6e is 18.7ms faster (`30.9%` lower)** | **v6e is 4.9ms faster (`10.5%` lower)** | **TPU v6e** |
| **4KB** | **8** | **61.0 req/s** | **130.4ms** | 139.9ms | 100.2ms | GPU is 30.2ms faster (`23.2%` lower) | **v6e is 9.5ms faster (`6.8%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **4KB** | **16** | **60.9 req/s** | **261.8ms** | 281.5ms | 180.5ms | GPU is 81.3ms faster (`31.1%` lower) | **v6e is 19.7ms faster (`7.0%` lower)** | **L4 GPU** *(v6e beats v5e)* |

---

### 1B. `P99` Latency (`TPU v6e BF16` vs `NVIDIA L4` & `TPU v5e BF16`)

| Payload Size | Concurrency | **TPU v6e (`p99`)** | TPU v5e (`p99`) | L4 GPU (`p99`) | **v6e vs L4 Delta (`% Reduction`)** | **v6e vs v5e Delta (`% Reduction`)** | Faster Setup |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- | :---: |
| **1KB** | **1** | **12.2ms** | 13.4ms | 23.5ms | **v6e is 11.3ms faster (`48.1%` lower)** | **v6e is 1.2ms faster (`9.0%` lower)** | **TPU v6e** |
| **1KB** | **4** | **34.6ms** | 41.3ms | 42.9ms | **v6e is 8.3ms faster (`19.3%` lower)** | **v6e is 6.7ms faster (`16.2%` lower)** | **TPU v6e** |
| **1KB** | **8** | **90.2ms** | 97.0ms | 66.3ms | GPU is 23.9ms faster (`26.5%` lower) | **v6e is 6.8ms faster (`7.0%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **1KB** | **16** | **140.1ms** | 146.1ms | 107.6ms | GPU is 32.5ms faster (`23.2%` lower) | **v6e is 6.0ms faster (`4.1%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **2KB** | **1** | **15.2ms** | 18.1ms | 28.2ms | **v6e is 13.0ms faster (`46.1%` lower)** | **v6e is 2.9ms faster (`16.0%` lower)** | **TPU v6e** |
| **2KB** | **4** | **85.9ms** | 93.1ms | 58.2ms | GPU is 27.7ms faster (`32.2%` lower) | **v6e is 7.2ms faster (`7.7%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **2KB** | **8** | **133.7ms** | 142.9ms | 97.3ms | GPU is 36.4ms faster (`27.2%` lower) | **v6e is 9.2ms faster (`6.4%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **2KB** | **16** | **264.1ms** | 281.4ms | 175.5ms | GPU is 88.6ms faster (`33.5%` lower) | **v6e is 17.3ms faster (`6.1%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **3KB** | **1** | **16.7ms** | 19.6ms | 31.3ms | **v6e is 14.6ms faster (`46.6%` lower)** | **v6e is 2.9ms faster (`14.8%` lower)** | **TPU v6e** |
| **3KB** | **4** | **44.4ms** | 49.4ms | 63.7ms | **v6e is 19.3ms faster (`30.3%` lower)** | **v6e is 5.0ms faster (`10.1%` lower)** | **TPU v6e** |
| **3KB** | **8** | **134.1ms** | 143.3ms | 100.2ms | GPU is 33.9ms faster (`25.3%` lower) | **v6e is 9.2ms faster (`6.4%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **3KB** | **16** | **265.9ms** | 284.7ms | 189.0ms | GPU is 76.9ms faster (`28.9%` lower) | **v6e is 18.8ms faster (`6.6%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **4KB** | **1** | **16.8ms** | 19.2ms | 34.2ms | **v6e is 17.4ms faster (`50.9%` lower)** | **v6e is 2.4ms faster (`12.5%` lower)** | **TPU v6e** |
| **4KB** | **4** | **44.5ms** | 49.7ms | 68.5ms | **v6e is 24.0ms faster (`35.0%` lower)** | **v6e is 5.2ms faster (`10.5%` lower)** | **TPU v6e** |
| **4KB** | **8** | **133.8ms** | 143.9ms | 113.9ms | GPU is 19.9ms faster (`14.9%` lower) | **v6e is 10.1ms faster (`7.0%` lower)** | **L4 GPU** *(v6e beats v5e)* |
| **4KB** | **16** | **265.7ms** | 286.5ms | 204.0ms | GPU is 61.7ms faster (`23.2%` lower) | **v6e is 20.8ms faster (`7.3%` lower)** | **L4 GPU** *(v6e beats v5e)* |

---

## 2. RPS Saturation & Multi-Payload SLA Results (`P99 < 50 ms` SLA)

### 2A. Per-Payload SLA Saturation Ceiling (`P99 <= 50 ms`)

| Payload | Setup | Max SLA RPS (`P99 <= 50 ms`) | `p50` @ Max SLA | `p99` @ Max SLA | **v6e vs L4 Improvement (`(v6e - L4) / L4`)** | **v6e vs v5e Improvement (`(v6e - v5e) / v5e`)** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1K** | L4 + Triton | 70 | 20.0ms | 37.0ms | — | — |
| **1K** | TPU v5e + vLLM (`BF16`) | 160 *(180 @ 48.5ms $P_{50}$)* | 19.1ms | 27.3ms | `+128.6%` (`2.29x`) | — |
| **1K** | **TPU v6e + vLLM (`BF16`)** | **160** *(**200** @ **43.7ms $P_{50}$, 55.3ms $P_{99}$**)* | **10.2ms** *(43.7ms @ 200)* | **13.6ms** *(55.3ms @ 200)* | **`+128.6%` (`2.29x` @ 160 RPS)**<br>**`+185.7%` (`2.86x` @ 200 RPS)** | **50% lower $P_{99}$ at 160 RPS (`13.6ms` vs `27.3ms`)**<br>**`+11.1%` higher peak (`200` vs `180` RPS)** |
| **2K** | L4 + Triton | 40 | 22.0ms | 28.0ms | — | — |
| **2K** | TPU v5e + vLLM (`BF16`) | 90 | 18.1ms | 23.2ms | `+125.0%` (`2.25x`) | — |
| **2K** | **TPU v6e + vLLM (`BF16`)** | **100** *( Dedicated )* / **110** *( Multi )* | **15.1ms** | **17.7ms** *(22.1ms @ 110)* | **`+150.0%` (`2.50x` @ 100 RPS)**<br>**`+175.0%` (`2.75x` @ 110 RPS)** | **`+11.1%` (`100` vs `90` RPS)**<br>**`+22.2%` (`110` vs `90` RPS)** |
| **3K** | L4 + Triton | 30 | 25.0ms | 35.0ms | — | — |
| **3K** | TPU v5e + vLLM (`BF16`) | 90 | 19.0ms | 22.6ms | `+200.0%` (`3.00x`) | — |
| **3K** | **TPU v6e + vLLM (`BF16`)** | **100** *( Dedicated )* | **15.7ms** | **19.5ms** | **`+233.3%` (`3.33x` total RPS)** | **`+11.1%` (`100` vs `90` RPS, 13.7% lower $P_{99}$)** |
| **5K** | L4 + Triton | 20 | 31.2ms | 49.1ms | — | — |
| **5K** | TPU v5e + vLLM (`BF16`) | 90 | 17.8ms | 21.2ms | `+350.0%` (`4.50x`) | — |
| **5K** | **TPU v6e + vLLM (`BF16`)** | **110** *( Multi-Payload )* | **15.0ms** | **17.1ms** | **`+450.0%` (`5.50x` total RPS)** | **`+22.2%` (`110` vs `90` RPS, 19.3% lower $P_{99}$)** |
| **7K** | L4 + Triton | 10 | 36.6ms | 46.4ms | — | — |
| **7K** | TPU v5e + vLLM (`BF16`) | 90 | 18.4ms | 24.0ms | `+800.0%` (`9.00x`) | — |
| **7K** | **TPU v6e + vLLM (`BF16`)** | **110** *( Multi-Payload )* | **16.2ms** | **19.6ms** | **`+1000.0%` (`11.00x` total RPS)** | **`+22.2%` (`110` vs `90` RPS, 18.3% lower $P_{99}$)** |

### 2B. Suite 2 Multi-Payload Concurrent Sweep (`1KB, 2KB, 5KB, 7KB` Simultaneously on `TPU v6e BF16`)

| Target RPS | `1KB p99` | `2KB p99` | `5KB p99` | `7KB p99` | Error Rate | SLA Status (`All 4 Payloads <= 50ms`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **50 RPS** | 12.4ms | 15.8ms | 16.3ms | 16.9ms | `0.0%` | **PASS** *(L4 max ceiling)* |
| **60 RPS** | 12.4ms | 15.5ms | 17.2ms | 16.9ms | `0.0%` | **PASS** |
| **70 RPS** | 13.0ms | 17.1ms | 18.8ms | 21.3ms | `0.0%` | **PASS** |
| **80 RPS** | 15.4ms | 18.4ms | 19.6ms | 20.1ms | `0.0%` | **PASS** *(TPU v5e BF16 ceiling = 80–90 RPS)* |
| **90 RPS** | 12.4ms | 17.1ms | 17.7ms | 18.6ms | `0.0%` | **PASS** |
| **100 RPS** | 13.2ms | 17.3ms | 25.9ms | 21.2ms | `0.0%` | **PASS** |
| **110 RPS** | **13.2ms** | **22.1ms** | **17.1ms** | **19.6ms** | **`0.0%`** | **PASS (`TPU v6e BF16 Multi-Payload Ceiling`)** |
| **120 RPS** | 4422.9ms | 3464.2ms | 3083.3ms | 4428.9ms | `0.0%` | Saturated |

---

## 3. Batch Benchmark Results (`Multi-Prompt Single-Request` & `High-Batch Token Sweep`)

### 3A. Zhemin's Multi-Prompt Single-Request Batch Test (`Batch = 1, 4, 8, 16`)

| Payload Size | Batch Size | **TPU v6e (`BF16`) Tput** | **TPU v6e `p50`** | **TPU v6e `p99`** | TPU v5e (`BF16`) Tput | TPU v5e `p50` | **v6e vs v5e Speedup** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1KB (`1024 chars`)** | **1** | **86.4 emb/s** | **11.3ms** | **16.2ms** | 71.9 emb/s | 13.8ms | **`+20.2%` higher tput** |
| **1KB (`1024 chars`)** | **4** | **117.9 emb/s** | **33.0ms** | **40.5ms** | 109.1 emb/s | 36.1ms | **`+8.1%` higher tput** |
| **1KB (`1024 chars`)** | **8** | **97.2 emb/s** | **91.2ms** | **95.1ms** | 89.4 emb/s | 97.4ms | **`+8.7%` higher tput** |
| **1KB (`1024 chars`)** | **16** | **115.0 emb/s** | **131.2ms** | **167.7ms** | 108.6 emb/s | 141.5ms | **`+5.9%` higher tput** |
| **2KB (`2048 chars`)** | **1** | **64.0 emb/s** | **15.5ms** | **16.6ms** | 54.6 emb/s | 18.2ms | **`+17.2%` higher tput** |
| **2KB (`2048 chars`)** | **4** | **50.1 emb/s** | **80.6ms** | **93.5ms** | 46.8 emb/s | 85.4ms | **`+7.1%` higher tput** |
| **2KB (`2048 chars`)** | **8** | **53.1 emb/s** | **154.9ms** | **156.6ms** | 49.3 emb/s | 162.1ms | **`+7.7%` higher tput** |
| **2KB (`2048 chars`)** | **16** | **63.7 emb/s** | **250.7ms** | **260.6ms** | 59.1 emb/s | 270.6ms | **`+7.8%` higher tput** |
| **3KB (`3072 chars`)** | **1** | **61.2 emb/s** | **16.2ms** | **17.6ms** | 52.1 emb/s | 19.1ms | **`+17.5%` higher tput** |
| **3KB (`3072 chars`)** | **16** | **50.6 emb/s** | **316.1ms** | **324.0ms** | 46.9 emb/s | 341.0ms | **`+7.9%` higher tput** |
| **4KB (`4096 chars`)** | **1** | **59.2 emb/s** | **16.8ms** | **18.0ms** | 50.4 emb/s | 19.8ms | **`+17.5%` higher tput** |
| **4KB (`4096 chars`)** | **16** | **50.5 emb/s** | **320.4ms** | **328.9ms** | 46.7 emb/s | 342.5ms | **`+8.1%` higher tput** |

### 3B. High-Batch Token Sweep (`B = 1..128` across `128..2048` Tokens on `TPU v6e BF16`)

| Token Length (`L`) | Peak Batch (`B`) | **Embeddings / sec (`Emb/s`)** | **Tokens / sec (`Tok/s`)** | `p50` Latency | `p99` Latency |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **128 tokens** | **B = 32** | **214.2 emb/s** | **27,418.9 tok/s** | 151.7ms | 155.2ms |
| **256 tokens** | **B = 128** | **124.8 emb/s** | **31,959.0 tok/s** | 1037.1ms | 1042.2ms |
| **512 tokens** | **B = 1 / 32** | **64.6 emb/s** | **33,090.6 tok/s** | 15.4ms | 16.7ms |
| **1024 tokens** | **B = 1 / 128** | **59.7 emb/s** *(50.8 @ B=128)* | **61,102.1 tok/s** *(52,060 @ B=128)* | 16.6ms | 17.6ms |
| **2048 tokens** | **B = 1 / 128** | **57.4 emb/s** *(48.8 @ B=128)* | **117,534.7 tok/s** *(99,861 @ B=128)* | 17.3ms | 18.5ms |

---

## 4. TCO & Cost per 1M Requests Analysis (`TPU v6e BF16` vs `L4` & `TPU v5e BF16`)

> [!NOTE]
> Following Zhemin's customer TCO formula (`Cost per 1M Requests = Hourly_Cost * 1,000,000 / (Max_SLA_RPS * 0.40_Utilization * 3,600)`), we provide TCO for **Cloud TPU v6e (`ct6e-standard-1t`)** across **On-Demand (`\$2.70/hr`)**, **1-Year CUD (`\$1.70/hr`, 37% discount)**, and **3-Year CUD (`\$1.22/hr`, 55% discount)** alongside **NVIDIA L4 (`g2-standard-4`, `\$0.70/hr`)** and **TPU v5e (`ct5lp-hightpu-1t`, `\$1.20/hr`)**.

| Machine Type & Pricing Tier | Machine Config | Hourly Cost | Cost / 1M Req (`1K` Payload) | Cost / 1M Req (`2K` Payload) | Cost / 1M Req (`3K` Payload) | Cost / 1M Req (`5K` Payload) | Cost / 1M Req (`7K` Payload) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`g2-standard-4` (L4 On-Demand)** | 1x L4 GPU (24GB)<br>4 vCPUs, 16 GiB | **\$0.70** | **\$6.94** *(70 RPS)* | **\$12.15** *(40 RPS)* | **\$16.20** *(30 RPS)* | **\$24.31** *(20 RPS)* | **\$48.61** *(10 RPS)* |
| **`ct5lp-hightpu-1t` (`v5e BF16` On-Demand)** | 1x TPU v5e (16GB)<br>24 vCPUs, 45.6 GB | **\$1.20** | **\$5.21** *(160 RPS)*<br>**\$4.63** *(180 RPS)* | **\$9.26** *(90 RPS)* | **\$9.26** *(90 RPS)* | **\$9.26** *(90 RPS)* | **\$9.26** *(90 RPS)* |
| **`ct6e-standard-1t` (`v6e BF16` On-Demand)** | 1x TPU v6e (32GB)<br>44 vCPUs, 176 GB | **\$2.70** | **\$11.72** *(160 RPS)*<br>**\$9.38** *(200 RPS)* | **\$17.05** *(110 RPS)* | **\$18.75** *(100 RPS)* | **\$17.05** *(110 RPS)* | **\$17.05** *(110 RPS)* |
| **`ct6e-standard-1t` (`v6e BF16` 1-Yr CUD)** | 1x TPU v6e (32GB)<br>37% CUD Discount | **\$1.70** | **\$7.38** *(160 RPS)*<br>**\$5.90** *(200 RPS)* | **\$10.73** *(110 RPS)* | **\$11.81** *(100 RPS)* | **\$10.73** *(110 RPS)* | **\$10.73** *(110 RPS)* |
| **`ct6e-standard-1t` (`v6e BF16` 3-Yr CUD)** | 1x TPU v6e (32GB)<br>55% CUD Discount | **\$1.22** | **\$5.30** *(160 RPS)*<br>**\$4.24** *(200 RPS)* | **\$7.70** *(110 RPS)* | **\$8.47** *(100 RPS)* | **\$7.70** *(110 RPS)* | **\$7.70** *(110 RPS)* |
| **v6e On-Demand vs L4 Cost Reduction** | | | *+35.1% (@200 RPS)* | *+40.3% (@110 RPS)* | *+15.7% (@100 RPS)* | **29.86% reduction** | **64.93% reduction** |
| **v6e 1-Yr CUD vs L4 Cost Reduction** | | | **15.00% reduction** *(@200 RPS)* | **11.69% reduction** | **27.10% reduction** | **55.86% reduction** | **77.93% reduction** |
| **v6e 3-Yr CUD vs L4 Cost Reduction** | | | **38.90% reduction** *(@200 RPS)* | **36.63% reduction** | **47.72% reduction** | **68.33% reduction** | **84.16% reduction** |

### Key TCO Takeaways (`TPU v6e BF16`):
1. **Long-Context (`5K–7K` Payloads)**: Even at **On-Demand pricing (`\$2.70/hr`)**, **TPU v6e (`ct6e-standard-1t`) reduces cost per 1M requests by `29.9%` on `5K` (`\$17.05` vs `\$24.31`) and `64.9%` on `7K` (`\$17.05` vs `\$48.61`)** compared to NVIDIA L4 (`g2-standard-4`) because 1 TPU v6e chip sustains **`110 RPS`** across `5K–7K` (`11.0x` the throughput of 1 L4 GPU at `10 RPS`).
2. **With 1-Year CUD (`\$1.70/hr`) or 3-Year CUD (`\$1.22/hr`)**: **TPU v6e (`BF16`) beats NVIDIA L4 on EVERY payload size (`1K` through `7K`)**, reducing cost per 1M requests by **`11.7%–77.9%` (1-Yr CUD)** and **`36.6%–84.2%` (3-Yr CUD)** — while also beating TPU v5e on `2K–7K` at 3-Yr CUD (`\$7.70` vs `\$9.26` per 1M requests) and delivering **`2x` the HBM (`32 GB` vs `16 GB`)** and **14–50% lower `p50`/`p99` latency**.
