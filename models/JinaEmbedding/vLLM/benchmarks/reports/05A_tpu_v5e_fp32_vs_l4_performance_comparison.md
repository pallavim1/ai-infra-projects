# Table 1: TPU V5e (`FP32`) vs L4 (`Performance Comparison`)

* **Source TPU Run:** `customer_k6_results_20260916_075137.json` (`2026-09-16 07:51:37 UTC`, `--dtype float32 --max-model-len 2048`, `truncate_prompt_tokens: 2048`)
* **TPU Configuration:** `ct5lp-hightpu-1t` (1x Cloud TPU v5e chip, `--tpu-topology=1x1`, 24 vCPUs, 48 GB RAM, 16 GB HBM)
* **GPU Configuration:** `g2-standard-4` (1x NVIDIA L4 GPU, 4 vCPUs, 16 GB RAM, Triton Inference Server + TensorRT)

---

## 1. Concurrent Request Comparison (`100% From 20260916_075137 FP32 Test Run`)

### `P50`
| Payload Size | Concurrency | TPU + vLLM (`p50`) | L4 GPU (`p50`) | Absolute Delta | % Reduction | Faster Setup |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| **1KB** | **1** | 11.5ms | 20.7ms | TPU is 9.2ms faster | 44.4% lower latency | **TPU** |
| **1KB** | **4** | 21.3ms | 38.1ms | TPU is 16.8ms faster | 44.1% lower latency | **TPU** |
| **1KB** | **8** | 44.6ms | 58.7ms | TPU is 14.1ms faster | 24.0% lower latency | **TPU** |
| **1KB** | **16** | 153.2ms | 96.5ms | GPU is 56.7ms faster | 37.0% lower latency | **GPU** |
| **2KB** | **1** | 16.0ms | 24.3ms | TPU is 8.3ms faster | 34.2% lower latency | **TPU** |
| **2KB** | **4** | 89.1ms | 52.1ms | GPU is 37.0ms faster | 41.5% lower latency | **GPU** |
| **2KB** | **8** | 153.2ms | 87.4ms | GPU is 65.8ms faster | 43.0% lower latency | **GPU** |
| **2KB** | **16** | 560.6ms | 156.9ms | GPU is 403.7ms faster | 72.0% lower latency | **GPU** |
| **3KB** | **1** | 17.1ms | 26.6ms | TPU is 9.5ms faster | 35.7% lower latency | **TPU** |
| **3KB** | **4** | 44.5ms | 57.0ms | TPU is 12.5ms faster | 21.9% lower latency | **TPU** |
| **3KB** | **8** | 356.5ms | 90.8ms | GPU is 265.7ms faster | 74.5% lower latency | **GPU** |
| **3KB** | **16** | 563.2ms | 167.8ms | GPU is 395.4ms faster | 70.2% lower latency | **GPU** |
| **4KB** | **1** | 17.4ms | 28.6ms | TPU is 11.2ms faster | 39.2% lower latency | **TPU** |
| **4KB** | **4** | 45.1ms | 60.6ms | TPU is 15.5ms faster | 25.6% lower latency | **TPU** |
| **4KB** | **8** | 152.9ms | 100.2ms | GPU is 52.7ms faster | 34.5% lower latency | **GPU** |
| **4KB** | **16** | 562.4ms | 180.5ms | GPU is 381.9ms faster | 67.9% lower latency | **GPU** |

---

### `p99`
| Payload Size | Concurrency | TPU + vLLM (`p99`) | L4 GPU (`p99`) | Absolute Delta | % Reduction | Faster Setup |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| **1KB** | **1** | 12.7ms | 23.5ms | TPU is 10.8ms faster | 46.0% lower latency | **TPU** |
| **1KB** | **4** | 40.4ms | 42.9ms | TPU is 2.5ms faster | 5.8% lower latency | **TPU** |
| **1KB** | **8** | 101.5ms | 66.3ms | GPU is 35.2ms faster | 34.7% lower latency | **GPU** |
| **1KB** | **16** | 155.8ms | 107.6ms | GPU is 48.2ms faster | 30.9% lower latency | **GPU** |
| **2KB** | **1** | 17.2ms | 28.2ms | TPU is 11.0ms faster | 39.0% lower latency | **TPU** |
| **2KB** | **4** | 99.9ms | 58.2ms | GPU is 41.7ms faster | 41.7% lower latency | **GPU** |
| **2KB** | **8** | 339.1ms | 97.3ms | GPU is 241.8ms faster | 71.3% lower latency | **GPU** |
| **2KB** | **16** | 597.2ms | 175.5ms | GPU is 421.7ms faster | 70.6% lower latency | **GPU** |
| **3KB** | **1** | 19.2ms | 31.3ms | TPU is 12.1ms faster | 38.7% lower latency | **TPU** |
| **3KB** | **4** | 46.4ms | 63.7ms | TPU is 17.3ms faster | 27.2% lower latency | **TPU** |
| **3KB** | **8** | 358.4ms | 100.2ms | GPU is 258.2ms faster | 72.0% lower latency | **GPU** |
| **3KB** | **16** | 625.9ms | 189.0ms | GPU is 436.9ms faster | 69.8% lower latency | **GPU** |
| **4KB** | **1** | 18.9ms | 34.2ms | TPU is 15.3ms faster | 44.7% lower latency | **TPU** |
| **4KB** | **4** | 47.6ms | 68.5ms | TPU is 20.9ms faster | 30.5% lower latency | **TPU** |
| **4KB** | **8** | 155.8ms | 113.9ms | GPU is 41.9ms faster | 26.9% lower latency | **GPU** |
| **4KB** | **16** | 610.0ms | 204.0ms | GPU is 406.0ms faster | 66.6% lower latency | **GPU** |

**Conclusion:**
* **Low-to-Medium Concurrency (`Concurrency 1–4` across `1KB`, `3KB`, `4KB`, and `Concurrency 1–8` on `1KB` `P50`)**: TPU v5e (`FP32`) achieves **22% to 44% lower `P50` latency** than L4 GPU and keeps `3KB` (`46.4 ms`) and `4KB` (`47.6 ms`) `P99` latency under the `< 50 ms` SLA at Concurrency 4.
* **High Concurrency (`Concurrency 8–16`)**: In `FP32` mode, TPU hits its matrix compute ceiling around Concurrency 4–8 (`172.4 req/s` on `1KB`, `89.0 req/s` on `3KB`), while L4 GPU exhibits lower per-batch latency at Concurrency 8–16.

---

## 2. RPS Saturation Result (`P99 < 50 ms` SLA — `20260916_075137` Run)

| Payload | Setup | RPS | `p50` | `p99` | RPS Improvement (`(TPU - L4) / L4`) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1K** | L4 + Triton | 70 | 20.0ms | 37.0ms | — |
| **1K** | **TPU V5e + vLLM (`FP32`)** | **140** *(180 @ 26.5ms $P_{50}$ / 57.4ms $P_{99}$)* | **12.2ms** *(26.5ms @ 180)* | **18.1ms** *(57.4ms @ 180)* | **`1.00` (`+100.0%` / `2.00x` @ 140 RPS)**<br>**`1.57` (`+157.1%` / `2.57x` @ 180 RPS)** |
| **2K** | L4 + Triton | 40 | 22.0ms | 28.0ms | — |
| **2K** | **TPU V5e + vLLM (`FP32`)** | **100** | **18.9ms** | **25.6ms** | **`1.50` (`+150.0%` / `2.50x` total RPS)** |
| **3K** | L4 + Triton | 30 | 25.0ms | 35.0ms | — |
| **3K** | **TPU V5e + vLLM (`FP32`)** | **90** | **18.0ms** | **20.3ms** | **`2.00` (`+200.0%` / `3.00x` total RPS)** |
| **5K** | L4 + Triton | 20 | 31.2ms | 49.1ms | — |
| **5K** | **TPU V5e + vLLM (`FP32`)** | **90** | **17.5ms** | **20.4ms** | **`3.50` (`+350.0%` / `4.50x` total RPS)** |
| **7K** | L4 + Triton | 10 | 36.6ms | 46.4ms | — |
| **7K** | **TPU V5e + vLLM (`FP32`)** | **90** | **18.0ms** | **20.0ms** | **`8.00` (`+800.0%` / `9.00x` total RPS)** |

---

## 3. Cost Improvement (`TPU V5e FP32` vs `L4`)

| Machine Type | Machine Config | Hourly Cost | Cost per 1M request (`1K` Payload) | Cost per 1M request (`2K` Payload) | Cost per 1M request (`3K` Payload) | Cost per 1M request (`5K` Payload) | Cost per 1M request (`7K` Payload) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`g2-standard-4`** | L4 GPUs: 1<br>vCPUs: 4<br>Memory: 16GiB | \$0.70 | \$6.94 | \$12.15 | \$16.20 | \$24.31 | \$48.61 |
| **`ct5lp-hightpu-1t` (`FP32`)** | vCPUs: 24<br>Memory: 45.6 GB<br>TPU Memory: 15.75 GB | \$1.20 | **\$5.95** *(140 RPS)*<br>**\$4.63** *(180 RPS)* | **\$8.33** *(100 RPS)* | **\$9.26** *(90 RPS)* | **\$9.26** *(90 RPS)* | **\$9.26** *(90 RPS)* |
| **Cost Improvement** | | | **14.29% reduction** *(140 RPS)*<br>**33.33% reduction** *(180 RPS)* | **31.43% reduction** | **42.86% reduction** | **61.90% reduction** | **80.95% reduction** |
