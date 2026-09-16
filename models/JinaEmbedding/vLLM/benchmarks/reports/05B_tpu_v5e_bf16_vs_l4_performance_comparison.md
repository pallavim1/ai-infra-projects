# Table 2: TPU V5e (`BF16`) vs L4 (`Performance Comparison`)

* **Model:** `jinaai/jina-embeddings-v2-small-en` (`--max-model-len 2048`, restricted at `2,048` tokens via `BertTokenizerFast`)
* **TPU Configuration:** `ct5lp-hightpu-1t` (1x Cloud TPU v5e chip, `vLLM --dtype bfloat16`)
* **GPU Configuration:** `g2-standard-4` (1x NVIDIA L4 GPU, Triton Inference Server + TensorRT)

---

## 1. Concurrent Request Comparison

### `P50`
| Payload Size | Concurrency | TPU + vLLM (`p50`) | L4 GPU (`p50`) | Absolute Delta | % Reduction | Faster Setup |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| **1KB** | **1** | 11.8ms | 20.7ms | TPU is 8.9ms faster | 43.0% lower latency | **TPU** |
| **1KB** | **4** | 21.7ms | 38.1ms | TPU is 16.4ms faster | 43.0% lower latency | **TPU** |
| **1KB** | **8** | 47.1ms | 58.7ms | TPU is 11.6ms faster | 19.8% lower latency | **TPU** |
| **1KB** | **16** | 140.3ms | 96.5ms | GPU is 43.8ms faster | 31.2% lower latency | **GPU** |
| **2KB** | **1** | 16.5ms | 24.3ms | TPU is 7.8ms faster | 32.1% lower latency | **TPU** |
| **2KB** | **4** | 47.3ms | 52.1ms | TPU is 4.8ms faster | 9.2% lower latency | **TPU** |
| **2KB** | **8** | 130.9ms | 87.4ms | GPU is 43.5ms faster | 33.2% lower latency | **GPU** |
| **2KB** | **16** | 210.4ms | 156.9ms | GPU is 53.5ms faster | 25.4% lower latency | **GPU** |
| **3KB** | **1** | 17.2ms | 26.6ms | TPU is 9.4ms faster | 35.3% lower latency | **TPU** |
| **3KB** | **4** | 46.1ms | 57.0ms | TPU is 10.9ms faster | 19.1% lower latency | **TPU** |
| **3KB** | **8** | 138.9ms | 90.8ms | GPU is 48.1ms faster | 34.6% lower latency | **GPU** |
| **3KB** | **16** | 279.7ms | 167.8ms | GPU is 111.9ms faster | 40.0% lower latency | **GPU** |
| **4KB** | **1** | 18.2ms | 28.6ms | TPU is 10.4ms faster | 36.4% lower latency | **TPU** |
| **4KB** | **4** | 46.8ms | 60.6ms | TPU is 13.8ms faster | 22.8% lower latency | **TPU** |
| **4KB** | **8** | 139.9ms | 100.2ms | GPU is 39.7ms faster | 28.4% lower latency | **GPU** |
| **4KB** | **16** | 281.5ms | 180.5ms | GPU is 101.0ms faster | 35.9% lower latency | **GPU** |

---

### `p99`
| Payload Size | Concurrency | TPU + vLLM (`p99`) | L4 GPU (`p99`) | Absolute Delta | % Reduction | Faster Setup |
| :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| **1KB** | **1** | 13.4ms | 23.5ms | TPU is 10.1ms faster | 43.0% lower latency | **TPU** |
| **1KB** | **4** | 41.3ms | 42.9ms | TPU is 1.6ms faster | 3.7% lower latency | **TPU** |
| **1KB** | **8** | 97.0ms | 66.3ms | GPU is 30.7ms faster | 31.6% lower latency | **GPU** |
| **1KB** | **16** | 146.1ms | 107.6ms | GPU is 38.5ms faster | 26.4% lower latency | **GPU** |
| **2KB** | **1** | 18.1ms | 28.2ms | TPU is 10.1ms faster | 35.8% lower latency | **TPU** |
| **2KB** | **4** | 93.1ms | 58.2ms | GPU is 34.9ms faster | 37.5% lower latency | **GPU** |
| **2KB** | **8** | 142.9ms | 97.3ms | GPU is 45.6ms faster | 31.9% lower latency | **GPU** |
| **2KB** | **16** | 281.4ms | 175.5ms | GPU is 105.9ms faster | 37.6% lower latency | **GPU** |
| **3KB** | **1** | 19.6ms | 31.3ms | TPU is 11.7ms faster | 37.4% lower latency | **TPU** |
| **3KB** | **4** | 49.4ms | 63.7ms | TPU is 14.3ms faster | 22.4% lower latency | **TPU** |
| **3KB** | **8** | 143.3ms | 100.2ms | GPU is 43.1ms faster | 30.1% lower latency | **GPU** |
| **3KB** | **16** | 284.7ms | 189.0ms | GPU is 95.7ms faster | 33.6% lower latency | **GPU** |
| **4KB** | **1** | 19.2ms | 34.2ms | TPU is 15.0ms faster | 43.9% lower latency | **TPU** |
| **4KB** | **4** | 49.7ms | 68.5ms | TPU is 18.8ms faster | 27.4% lower latency | **TPU** |
| **4KB** | **8** | 143.9ms | 113.9ms | GPU is 30.0ms faster | 20.8% lower latency | **GPU** |
| **4KB** | **16** | 286.5ms | 204.0ms | GPU is 82.5ms faster | 28.8% lower latency | **GPU** |

**Conclusion:**
* **Low-to-Medium Concurrency (`Concurrency 1–4` across `1KB–4KB`)**: TPU v5e (`BF16`) delivers **19% to 43% lower `P50` latency** than L4 GPU (`11.8ms` vs `20.7ms` at `1KB`; `46.1ms` vs `57.0ms` at `3KB` Concurrency 4) and keeps `3KB` (`49.4 ms`) and `4KB` (`49.7 ms`) `P99` latency under the `< 50 ms` SLA at Concurrency 4.
* **High Concurrency (`Concurrency 8–16`)**: At Concurrency 8 and 16, TPU v5e (`BF16`) sustains high batch throughput (`112.3–163.7 req/s` on `1KB`, `57–71 req/s` on `2KB–4KB`), while L4 GPU exhibits lower per-batch latency (`96.5–180.5 ms` vs `140.3–281.5 ms`).

---

## 2. RPS Saturation Result (`P99 < 50 ms` SLA)

| Payload | Setup | RPS | `p50` | `p99` | RPS Improvement (`(TPU - L4) / L4`) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1K** | L4 + Triton | 70 | 20.0ms | 37.0ms | — |
| **1K** | **TPU V5e + vLLM (`BF16`)** | **160** *(180 @ 48.5ms $P_{50}$)* | **19.1ms** | **27.3ms** | **`1.29` (`+128.6%` / `2.29x` total RPS)** *(1.57 @ 180 RPS)* |
| **2K** | L4 + Triton | 40 | 22.0ms | 28.0ms | — |
| **2K** | **TPU V5e + vLLM (`BF16`)** | **90** | **18.1ms** | **23.2ms** | **`1.25` (`+125.0%` / `2.25x` total RPS)** |
| **3K** | L4 + Triton | 30 | 25.0ms | 35.0ms | — |
| **3K** | **TPU V5e + vLLM (`BF16`)** | **90** | **19.0ms** | **22.6ms** | **`2.00` (`+200.0%` / `3.00x` total RPS)** |
| **5K** | L4 + Triton | 20 | 31.2ms | 49.1ms | — |
| **5K** | **TPU V5e + vLLM (`BF16`)** | **90** | **17.8ms** | **21.2ms** | **`3.50` (`+350.0%` / `4.50x` total RPS)** |
| **7K** | L4 + Triton | 10 | 36.6ms | 46.4ms | — |
| **7K** | **TPU V5e + vLLM (`BF16`)** | **90** | **18.4ms** | **24.0ms** | **`8.00` (`+800.0%` / `9.00x` total RPS)** |

---

## 3. Cost Improvement (`TPU V5e BF16` vs `L4`)

| Machine Type | Machine Config | Hourly Cost | Cost per 1M request (`1K` Payload) | Cost per 1M request (`2K` Payload) | Cost per 1M request (`3K` Payload) | Cost per 1M request (`5K` Payload) | Cost per 1M request (`7K` Payload) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`g2-standard-4`** | L4 GPUs: 1<br>vCPUs: 4<br>Memory: 16GiB | \$0.70 | \$6.94 | \$12.15 | \$16.20 | \$24.31 | \$48.61 |
| **`ct5lp-hightpu-1t` (`BF16`)** | vCPUs: 24<br>Memory: 45.6 GB<br>TPU Memory: 15.75 GB | \$1.20 | **\$5.21** *(160 RPS)*<br>**\$4.63** *(180 RPS)* | **\$9.26** *(90 RPS)* | **\$9.26** *(90 RPS)* | **\$9.26** *(90 RPS)* | **\$9.26** *(90 RPS)* |
| **Cost Improvement** | | | **25.00% reduction** *(160 RPS)*<br>**33.33% reduction** *(180 RPS)* | **23.81% reduction** | **42.86% reduction** | **61.90% reduction** | **80.95% reduction** |

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
