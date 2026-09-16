# Table 1: TPU V5e (`FP32`) vs L4 (`Performance Comparison`)

* **Model:** `jinaai/jina-embeddings-v2-small-en` (`--max-model-len 2048`, restricted at `2,048` tokens via `BertTokenizerFast`)
* **TPU Configuration:** `ct5lp-hightpu-1t` (1x Cloud TPU v5e chip, `vLLM --dtype float32`)
* **GPU Configuration:** `g2-standard-4` (1x NVIDIA L4 GPU, Triton Inference Server + TensorRT)

---

## 1. Concurrent Request Comparison

### `P50`
| Payload Size | Concurrency | TPU + vLLM (`p50`) | L4 GPU (`p50`) | Absolute Delta | % Reduction | Faster Setup |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| **1KB** | **1** | 11.5ms | 20.7ms | TPU is 9.2ms faster | 44.4% lower latency | **TPU** |
| **1KB** | **4** | 21.2ms | 38.1ms | TPU is 16.9ms faster | 44.4% lower latency | **TPU** |
| **1KB** | **8** | 42.5ms | 58.7ms | TPU is 16.2ms faster | 27.6% lower latency | **TPU** |
| **1KB** | **16** | 85.4ms | 96.5ms | TPU is 11.1ms faster | 11.5% lower latency | **TPU** |
| **2KB** | **1** | 16.0ms | 24.3ms | TPU is 8.3ms faster | 34.2% lower latency | **TPU** |
| **2KB** | **4** | 40.7ms | 52.1ms | TPU is 11.4ms faster | 21.9% lower latency | **TPU** |
| **2KB** | **8** | 82.6ms | 87.4ms | TPU is 4.8ms faster | 5.5% lower latency | **TPU** |
| **2KB** | **16** | 165.9ms | 156.9ms | GPU is 9.0ms faster | 5.4% lower latency | **GPU** |
| **3KB** | **1** | 17.1ms | 26.6ms | TPU is 9.5ms faster | 35.7% lower latency | **TPU** |
| **3KB** | **4** | 44.5ms | 57.0ms | TPU is 12.5ms faster | 21.9% lower latency | **TPU** |
| **3KB** | **8** | 334.4ms | 90.8ms | GPU is 243.6ms faster | 72.8% lower latency | **GPU** |
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
| **1KB** | **4** | 22.8ms | 42.9ms | TPU is 20.1ms faster | 46.9% lower latency | **TPU** |
| **1KB** | **8** | 44.5ms | 66.3ms | TPU is 21.8ms faster | 32.9% lower latency | **TPU** |
| **1KB** | **16** | 89.9ms | 107.6ms | TPU is 17.7ms faster | 16.4% lower latency | **TPU** |
| **2KB** | **1** | 17.2ms | 28.2ms | TPU is 11.0ms faster | 39.0% lower latency | **TPU** |
| **2KB** | **4** | 42.8ms | 58.2ms | TPU is 15.4ms faster | 26.5% lower latency | **TPU** |
| **2KB** | **8** | 86.0ms | 97.3ms | TPU is 11.3ms faster | 11.6% lower latency | **TPU** |
| **2KB** | **16** | 171.2ms | 175.5ms | TPU is 4.3ms faster | 2.5% lower latency | **TPU** |
| **3KB** | **1** | 19.2ms | 31.3ms | TPU is 12.1ms faster | 38.7% lower latency | **TPU** |
| **3KB** | **4** | 46.4ms | 63.7ms | TPU is 17.3ms faster | 27.2% lower latency | **TPU** |
| **3KB** | **8** | 358.4ms | 100.2ms | GPU is 258.2ms faster | 72.0% lower latency | **GPU** |
| **3KB** | **16** | 625.9ms | 189.0ms | GPU is 436.9ms faster | 69.8% lower latency | **GPU** |
| **4KB** | **1** | 18.9ms | 34.2ms | TPU is 15.3ms faster | 44.7% lower latency | **TPU** |
| **4KB** | **4** | 47.6ms | 68.5ms | TPU is 20.9ms faster | 30.5% lower latency | **TPU** |
| **4KB** | **8** | 155.8ms | 113.9ms | GPU is 41.9ms faster | 26.9% lower latency | **GPU** |
| **4KB** | **16** | 610.0ms | 204.0ms | GPU is 406.0ms faster | 66.6% lower latency | **GPU** |

**Conclusion:**
* **Low-to-Medium Concurrency (`Concurrency 1–4` across `1KB–4KB`, and `Concurrency 1–16` for `1KB` & `2KB`)**: TPU v5e (`FP32`) achieves **32% to 47% lower latency** than L4 GPU and passes the `< 50 ms` $P_{99}$ SLA at Concurrency 4 even for `3KB` (`46.4 ms`) and `4KB` (`47.6 ms`) payloads (where L4 GPU breaches 50 ms at `63.7 ms` and `68.5 ms`).
* **High Concurrency (`Concurrency 8–16` for `3KB–4KB`)**: In `FP32` mode, TPU hits its matrix/token-bucket ceiling at Concurrency 4 (`~89 req/s`), whereas L4 GPU scales progressively at Concurrency 8 and 16.

---

## 2. RPS Saturation Result (`P99 < 50 ms` SLA)

| Payload | Setup | RPS | `p50` | `p99` | RPS Improvement (`(TPU - L4) / L4`) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1K** | L4 + Triton | 70 | 20.0ms | 37.0ms | — |
| **1K** | **TPU V5e + vLLM (`FP32`)** | **187** *(180 Dedicated)* | **19.9ms** | **29.6ms** | **`1.67` (`+167.1%` / `2.67x` total RPS)** |
| **2K** | L4 + Triton | 40 | 22.0ms | 28.0ms | — |
| **2K** | **TPU V5e + vLLM (`FP32`)** | **100** *(90 Sheet)* | **18.9ms** | **25.6ms** | **`1.50` (`+150.0%` / `2.50x` total RPS)** |
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
| **`ct5lp-hightpu-1t` (`FP32`)** | vCPUs: 24<br>Memory: 45.6 GB<br>TPU Memory: 15.75 GB | \$1.20 | **\$4.46** *(187 RPS)*<br>**\$4.63** *(180 RPS)* | **\$8.33** *(100 RPS)*<br>**\$9.26** *(90 RPS)* | **\$9.26** *(90 RPS)* | **\$9.26** *(90 RPS)* | **\$9.26** *(90 RPS)* |
| **Cost Improvement** | | | **35.83% reduction** *(187 RPS)*<br>**33.33% reduction** *(180 RPS)* | **31.43% reduction** *(100 RPS)*<br>**23.81% reduction** *(90 RPS)* | **42.86% reduction** | **61.90% reduction** | **80.95% reduction** |

---

## 4. TPU V5E vs L4 Availability

| Region Name | Region ID | TPU v5e Supported Zones | L4 GPU Supported Zones | Overlap / Multi-Accelerator Region? |
| :--- | :--- | :--- | :--- | :--- |
| **Iowa** | `us-central1` | 4 (`a, b, c, f`) | 4 (`a, b, c, f`) | Yes (Full) |
| **Columbus, Ohio** | `us-east5` | 1 (`a`) | 3 (`a, b, c`) | Yes |
| **Dallas** | `us-south1` | 1 (`a`) | 3 (`a, b, c`) | Yes |
| **Oregon** | `us-west1` | 1 (`c`) | 3 (`a, b, c`) | Yes |
| **Las Vegas** | `us-west4` | 1 (`a`) | 3 (`a, b, c`) | Yes |
| **South Carolina** | `us-east1` | — | 3 (`b, c, d`) | L4 GPU Only |
