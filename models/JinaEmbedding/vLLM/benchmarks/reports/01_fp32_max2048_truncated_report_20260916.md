# Report 1: Jina AI Embedding Model + Cloud TPU v5e + vLLM (`--max-model-len 2048`, Restricted at `2048`, `FP32`)

* **Run Timestamp (UTC):** `2026-09-16 07:51:37 UTC` (`20260916_075137`)
* **Model:** `jinaai/jina-embeddings-v2-small-en` (512-dim BERT/ALiBi Embeddings)
* **Accelerator Hardware:** 1x Cloud TPU v5e chip (`ct5lp-hightpu-1t`, GKE cluster `pm-panw-jina-cluster` in `europe-west4-b`)
* **Serving Configuration:**
  * Engine: `vLLM v0.26.0` (`--runner pooling --convert embed --trust-remote-code`)
  * Precision: **`--dtype float32` (`FP32`)**
  * Max Model Length: **`--max-model-len 2048`**
  * Server-Side Fast Tokenizer Truncation: **`truncate_prompt_tokens: 2048`** (`BertTokenizerFast` in Rust)

---

## 🔗 Quick Links to All 4 Clean Reports

| # | Report Name | Local Artifact (Click to Open) | GitHub (`ai-infra-projects`) | GitHub (`tpu-inference`) |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **`--max-model-len 2048`, Restricted at `2048`, `FP32`** | [`01_fp32_max2048_truncated_report_20260916.md`](file:///usr/local/google/home/pallaviam/.gemini/jetski/brain/292ebc90-dbb8-4a33-8f1a-ca6dee63d8ff/01_fp32_max2048_truncated_report_20260916.md) | [View on GitHub](https://github.com/pallavim1/ai-infra-projects/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/01_fp32_max2048_truncated_report_20260916.md) | [View on GitHub](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/01_fp32_max2048_truncated_report_20260916.md) |
| **2** | **`--max-model-len 2048`, Restricted at `2048`, `BF16`** | [`02_bf16_max2048_truncated_report_20260916.md`](file:///usr/local/google/home/pallaviam/.gemini/jetski/brain/292ebc90-dbb8-4a33-8f1a-ca6dee63d8ff/02_bf16_max2048_truncated_report_20260916.md) | [View on GitHub](https://github.com/pallavim1/ai-infra-projects/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/02_bf16_max2048_truncated_report_20260916.md) | [View on GitHub](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/02_bf16_max2048_truncated_report_20260916.md) |
| **3** | **Comparison of `FP32` vs `BF16` Runs (TPU v5e)** | [`03_fp32_vs_bf16_comparison_report_20260916.md`](file:///usr/local/google/home/pallaviam/.gemini/jetski/brain/292ebc90-dbb8-4a33-8f1a-ca6dee63d8ff/03_fp32_vs_bf16_comparison_report_20260916.md) | [View on GitHub](https://github.com/pallavim1/ai-infra-projects/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/03_fp32_vs_bf16_comparison_report_20260916.md) | [View on GitHub](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/03_fp32_vs_bf16_comparison_report_20260916.md) |
| **4** | **3-Way Comparison: TPU `FP32` vs TPU `BF16` vs Zhemin's Triton (`L4 GPU`)** | [`04_tpu_fp32_bf16_vs_zhemin_triton_l4_report_20260916.md`](file:///usr/local/google/home/pallaviam/.gemini/jetski/brain/292ebc90-dbb8-4a33-8f1a-ca6dee63d8ff/04_tpu_fp32_bf16_vs_zhemin_triton_l4_report_20260916.md) | [View on GitHub](https://github.com/pallavim1/ai-infra-projects/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/04_tpu_fp32_bf16_vs_zhemin_triton_l4_report_20260916.md) | [View on GitHub](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/04_tpu_fp32_bf16_vs_zhemin_triton_l4_report_20260916.md) |

---

## 📊 Key Highlights Across All 4 Reports

### 1. `FP32` vs `BF16` on Cloud TPU v5e (Single-Request Multi-Prompt Batch Test, `Batch = 1..16`)
*At `Batch = 8` and `Batch = 16`, `BF16` delivers **`1.83x–2.23x` higher throughput** and **`45%–59%` lower $P_{50}$ latency** because TPU v5e MXUs execute `bfloat16` at 2x the compute rate of `float32`:*

| Payload Size | Batch (`N`) | **TPU `FP32` Throughput** | **TPU `BF16` Throughput** | **TPU `FP32` $P_{50}$ / $P_{99}$** | **TPU `BF16` $P_{50}$ / $P_{99}$** | **`BF16` Gain Over `FP32`** |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1 KB** | **16** | 50.4 prompts/s | **112.3 prompts/s** | 331.4 ms / 342.9 ms | **142.8 ms / 147.8 ms** | **`2.23x` throughput, `-56.9%` $P_{50}$** |
| **2 KB** | **8** | 28.6 prompts/s | **59.1 prompts/s** | 319.1 ms / 331.0 ms | **130.9 ms / 167.8 ms** | **`2.07x` throughput, `-59.0%` $P_{50}$** |
| **3 KB** *(truncated @ 2048)* | **16** | 25.5 prompts/s | **47.6 prompts/s** | 632.9 ms / 700.2 ms | **339.1 ms / 349.8 ms** | **`1.87x` throughput, `-46.4%` $P_{50}$** |
| **4 KB** *(truncated @ 2048)* | **16** | 24.9 prompts/s | **46.7 prompts/s** | 648.2 ms / 704.2 ms | **353.7 ms / 357.4 ms** | **`1.88x` throughput, `-45.4%` $P_{50}$** |

### 2. 3-Way Production SLA Comparison (`P99 < 50 ms`): TPU `FP32` vs TPU `BF16` vs Zhemin's Triton (`L4 GPU`)

| Payload Size | **Zhemin Triton L4 (`FP16`)**<br>Max Sustained RPS (`P99 < 50 ms`) | **Cloud TPU v5e (`FP32`)**<br>Max Sustained RPS (`P99 < 50 ms`) | **Cloud TPU v5e (`BF16`)**<br>Max Sustained RPS (`P99 < 50 ms`) | **Cloud TPU v5e Advantage Over Triton L4** |
| :--- | :---: | :---: | :---: | :--- |
| **`1 KB` (`1024 B`)** | **`70 RPS`** *(Dedicated)* / **`40 RPS`** *(Ray Serve)* | **`180 RPS`** ($P_{99}=29.6\text{ ms}$) | **`160–180 RPS`** ($P_{99}=27.3\text{ ms}$ @ 160) | 🏆 **`2.57x` higher sustained RPS** |
| **`2 KB` (`2048 B`)** | **`40 RPS`** ($P_{99}=46.5\text{ ms}$) | **`100 RPS`** ($P_{99}=25.6\text{ ms}$) | **`90 RPS`** ($P_{99}=23.2\text{ ms}$) | 🏆 **`2.25x–2.50x` higher sustained RPS** |
| **`3 KB` (`3072 B`, truncated @ 2048)** | **`~30 RPS`** | **`90 RPS`** ($P_{99}=20.3\text{ ms}$) | **`90 RPS`** ($P_{99}=22.6\text{ ms}$) | 🏆 **`3.00x` higher sustained RPS (`0%` errors)** |
| **`5 KB` (`5120 B`, truncated @ 2048)** | **`20 RPS`** ($P_{99}=49.1\text{ ms}$; fails @ 30 RPS) | **`90 RPS`** ($P_{99}=20.4\text{ ms}$) | **`90 RPS`** ($P_{99}=21.2\text{ ms}$) | 🏆 **`4.50x` higher sustained RPS (`0%` errors)** |
| **`7 KB` (`7168 B`, truncated @ 2048)** | **`10 RPS`** ($P_{99}=46.4\text{ ms}$; fails @ 20 RPS) | **`90 RPS`** ($P_{99}=20.0\text{ ms}$) | **`90 RPS`** ($P_{99}=24.0\text{ ms}$) | 🏆 **`9.00x` higher sustained RPS (`0%` errors)** |

---

## 1. Executive Summary (`FP32`, Restricted at `2,048` Tokens)

* **Zero Errors Across All Payload Sizes (`1 KB` to `7 KB`)**: Restricting input sequences to `2,048` tokens via `vLLM`'s built-in `BertTokenizerFast` (`truncate_prompt_tokens: 2048`) eliminates all HTTP 400 errors on `3 KB`, `4 KB`, `5 KB`, and `7 KB` random-character payloads (`0.00%` error rate across every test).
* **Multi-Payload Concurrent Sweep (`50–90 RPS` across `1 KB, 2 KB, 5 KB, 7 KB`)**: Passes the strict **$< 50\text{ ms}$ $P_{99}$ SLA** at every rate from `50 RPS` through `90 RPS` ($P_{99} = 13.2\text{ ms} \text{ to } 24.8\text{ ms}$).
* **Dedicated Saturation Capacity (`P99 < 50 ms` SLA)**:
  * **`1 KB` (736 tokens)**: Up to **`180 RPS`** ($P_{50} = 19.9\text{ ms}, P_{99} = 29.6\text{ ms}$ with 8K token bucket; `140–180 RPS` range).
  * **`2 KB` (1,468 tokens)**: Up to **`100 RPS`** ($P_{50} = 18.9\text{ ms}, P_{99} = 25.6\text{ ms}$).
  * **`3 KB` (truncated to 2,048 tokens)**: Up to **`90 RPS`** ($P_{50} = 18.0\text{ ms}, P_{99} = 20.3\text{ ms}$).

---

## 2. Suite 1A: Single-HTTP-Request Multi-Prompt Batch Test (`Python Batch Client`)
*Client sends 1 HTTP `POST` request containing an array of $N \in \{1, 4, 8, 16\}$ random character prompts (`{"text": [prompt_1, ..., prompt_N]}`).*

| Payload Size | Token Count (Post-Tokenizer) | Batch Size (`N` prompts/req) | Throughput (`prompts/s`) | $P_{50}$ (`ms`) | $P_{95}$ (`ms`) | $P_{99}$ (`ms`) | Avg (`ms`) | Error Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB (1024 chars)** | 736 tokens | **1** | **78.8/s** | **12.7 ms** | 13.1 ms | **13.2 ms** | 12.7 ms | `0.00%` |
| **1 KB (1024 chars)** | 736 tokens | **4** | **104.9/s** | **41.1 ms** | 42.4 ms | **42.8 ms** | 38.1 ms | `0.00%` |
| **1 KB (1024 chars)** | 736 tokens | **8** | **75.6/s** | **107.8 ms** | 110.0 ms | **110.0 ms** | 105.7 ms | `0.00%` |
| **1 KB (1024 chars)** | 736 tokens | **16** | **50.4/s** | **331.4 ms** | 342.9 ms | **342.9 ms** | 317.2 ms | `0.00%` |
| **2 KB (2048 chars)** | 1,468 tokens | **1** | **57.9/s** | **17.2 ms** | 17.9 ms | **18.0 ms** | 17.3 ms | `0.00%` |
| **2 KB (2048 chars)** | 1,468 tokens | **4** | **43.9/s** | **93.0 ms** | 102.4 ms | **102.8 ms** | 91.2 ms | `0.00%` |
| **2 KB (2048 chars)** | 1,468 tokens | **8** | **28.6/s** | **319.1 ms** | 330.7 ms | **331.0 ms** | 280.0 ms | `0.00%` |
| **2 KB (2048 chars)** | 1,468 tokens | **16** | **37.8/s** | **419.1 ms** | 430.3 ms | **430.3 ms** | 422.7 ms | `0.00%` |
| **3 KB (3072 chars)** | 2,048 tokens *(truncated)* | **1** | **55.2/s** | **18.1 ms** | 18.7 ms | **18.8 ms** | 18.1 ms | `0.00%` |
| **3 KB (3072 chars)** | 2,048 tokens *(truncated)* | **4** | **40.2/s** | **104.5 ms** | 106.1 ms | **106.4 ms** | 99.5 ms | `0.00%` |
| **3 KB (3072 chars)** | 2,048 tokens *(truncated)* | **8** | **24.7/s** | **334.4 ms** | 388.9 ms | **390.6 ms** | 323.5 ms | `0.00%` |
| **3 KB (3072 chars)** | 2,048 tokens *(truncated)* | **16** | **25.5/s** | **632.9 ms** | 700.2 ms | **700.2 ms** | 628.5 ms | `0.00%` |
| **4 KB (4096 chars)** | 2,048 tokens *(truncated)* | **1** | **53.7/s** | **18.5 ms** | 19.3 ms | **20.1 ms** | 18.6 ms | `0.00%` |
| **4 KB (4096 chars)** | 2,048 tokens *(truncated)* | **4** | **41.0/s** | **96.5 ms** | 107.0 ms | **108.4 ms** | 97.5 ms | `0.00%` |
| **4 KB (4096 chars)** | 2,048 tokens *(truncated)* | **8** | **25.3/s** | **327.1 ms** | 340.5 ms | **390.8 ms** | 316.6 ms | `0.00%` |
| **4 KB (4096 chars)** | 2,048 tokens *(truncated)* | **16** | **24.9/s** | **648.2 ms** | 704.2 ms | **704.2 ms** | 643.7 ms | `0.00%` |

---

## 3. Suite 1B: `k6` Concurrent Virtual Users (`1 Prompt per HTTP Request` — Server Continuous Batching)

| Payload Size | Concurrency (`VUs`) | Throughput (`req/s`) | $P_{50}$ (`ms`) | $P_{95}$ (`ms`) | $P_{99}$ (`ms`) | Error Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB (1024 chars)** | **1** | **85.4/s** | **11.5 ms** | 12.1 ms | **12.7 ms** | `0.00%` |
| **1 KB (1024 chars)** | **4** | **153.8/s** *(187.4/s\*)* | **21.3 ms** | 38.8 ms | **40.4 ms** *(22.8 ms\*)* | `0.00%` |
| **1 KB (1024 chars)** | **8** | **172.4/s** *(187.6/s\*)* | **44.6 ms** | 46.0 ms | **101.5 ms** *(44.5 ms\*)* | `0.00%` |
| **1 KB (1024 chars)** | **16** | **103.9/s** *(186.8/s\*)* | **153.2 ms** *(85.4 ms\*)* | 154.6 ms | **155.8 ms** *(89.9 ms\*)* | `0.00%` |
| **2 KB (2048 chars)** | **1** | **61.7/s** | **16.0 ms** | 16.6 ms | **17.2 ms** | `0.00%` |
| **2 KB (2048 chars)** | **4** | **51.2/s** *(97.7/s\*)* | **89.1 ms** *(40.7 ms\*)* | 98.9 ms | **99.9 ms** *(42.8 ms\*)* | `0.00%` |
| **2 KB (2048 chars)** | **8** | **50.6/s** *(96.6/s\*)* | **153.2 ms** *(82.6 ms\*)* | 155.0 ms | **339.1 ms** *(86.0 ms\*)* | `0.00%` |
| **2 KB (2048 chars)** | **16** | **28.5/s** *(96.5/s\*)* | **560.6 ms** *(165.9 ms\*)* | 562.2 ms | **597.2 ms** *(171.2 ms\*)* | `0.00%` |
| **3 KB (3072 chars)** | **1** | **57.4/s** | **17.1 ms** | 18.0 ms | **19.2 ms** | `0.00%` |
| **3 KB (3072 chars)** | **4** | **89.0/s** | **44.5 ms** | 45.4 ms | **46.4 ms** | `0.00%` |
| **3 KB (3072 chars)** | **8** | **22.3/s** | **356.5 ms** | 357.6 ms | **358.4 ms** | `0.00%` |
| **3 KB (3072 chars)** | **16** | **28.6/s** | **563.2 ms** | 564.6 ms | **625.9 ms** | `0.00%` |
| **4 KB (4096 chars)** | **1** | **56.7/s** | **17.4 ms** | 18.1 ms | **18.9 ms** | `0.00%` |
| **4 KB (4096 chars)** | **4** | **87.7/s** | **45.1 ms** | 46.4 ms | **47.6 ms** | `0.00%` |
| **4 KB (4096 chars)** | **8** | **52.1/s** | **152.9 ms** | 154.8 ms | **155.8 ms** | `0.00%` |
| **4 KB (4096 chars)** | **16** | **28.5/s** | **562.4 ms** | 564.1 ms | **610.0 ms** | `0.00%` |

*\*Values in parentheses are with `--max-num-batched-tokens 8192 --max-num-seqs=40`.*

---

## 4. Suite 2: `k6` Multi-Payload Concurrent Sweep (`50–90 RPS` Across `1 KB, 2 KB, 5 KB, 7 KB`)

| Target RPS | **1 KB $P_{99}$** | **2 KB $P_{99}$** | **5 KB $P_{99}$** *(Truncated to 2,048)* | **7 KB $P_{99}$** *(Truncated to 2,048)* | Error Rate | SLA Status (`< 50 ms`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **50 RPS** | **13.2 ms** | **17.6 ms** | **17.9 ms** | **19.1 ms** | `0.00%` | ✅ **ALL PASS** |
| **60 RPS** | **13.7 ms** | **18.7 ms** | **20.2 ms** | **24.8 ms** | `0.00%` | ✅ **ALL PASS** |
| **70 RPS** | **13.8 ms** | **20.8 ms** | **23.0 ms** | **23.0 ms** | `0.00%` | ✅ **ALL PASS** |
| **80 RPS** | **13.8 ms** | **19.3 ms** | **21.2 ms** | **20.2 ms** | `0.00%` | ✅ **ALL PASS** |
| **90 RPS** | **15.5 ms** | **19.5 ms** | **20.4 ms** | **20.0 ms** | `0.00%` | ✅ **ALL PASS** |

---

## 5. Suite 3: `k6` Dedicated RPS Saturation Sweeps (`1 KB`, `2 KB`, `3 KB`)

### 5A. `1 KB` Dedicated Saturation Sweep (`FP32`)
| Target RPS | Achieved RPS | $P_{50}$ (`ms`) | $P_{95}$ (`ms`) | $P_{99}$ (`ms`) | Error Rate | SLA Status (`P99 < 50 ms`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100 RPS** | 100.0 | **13.4 ms** | 14.9 ms | **15.8 ms** | `0.00%` | ✅ **PASS** |
| **120 RPS** | 119.9 | **12.8 ms** | 13.8 ms | **14.3 ms** | `0.00%` | ✅ **PASS** |
| **140 RPS** | 139.9 | **12.2 ms** | 14.5 ms | **18.1 ms** | `0.00%` | ✅ **PASS** |
| **160 RPS** | 159.6 | **18.8 ms** *(16.8 ms\*)* | 45.7 ms | **54.6 ms** *(24.2 ms\*)* | `0.00%` | ✅ **PASS (`24.2 ms` w/ 8K bucket)** |
| **180 RPS** | 179.5 | **26.5 ms** *(19.9 ms\*)* | 47.4 ms | **57.4 ms** *(29.6 ms\*)* | `0.00%` | ✅ **PASS (`29.6 ms` w/ 8K bucket)** |
| **190 RPS** | 123.9 | 51.8 ms | 4017.8 ms | 4216.0 ms | `0.00%` | ⚠️ **SATURATED** |

### 5B. `2 KB` Dedicated Saturation Sweep (`FP32`)
| Target RPS | Achieved RPS | $P_{50}$ (`ms`) | $P_{95}$ (`ms`) | $P_{99}$ (`ms`) | Error Rate | SLA Status (`P99 < 50 ms`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **70 RPS** | 70.0 | **17.6 ms** | 18.8 ms | **20.6 ms** | `0.00%` | ✅ **PASS** |
| **80 RPS** | 80.0 | **17.9 ms** | 19.0 ms | **20.0 ms** | `0.00%` | ✅ **PASS** |
| **90 RPS** | 90.0 | **17.0 ms** | 19.3 ms | **20.8 ms** | `0.00%` | ✅ **PASS** |
| **95 RPS** | 95.0 | **16.4 ms** | 18.7 ms | **20.3 ms** | `0.00%` | ✅ **PASS** |
| **100 RPS** | 99.9 | **18.9 ms** | 21.7 ms | **25.6 ms** | `0.00%` | ✅ **PASS (Peak `100 RPS`)** |
| **110 RPS** | 39.2 | 6026.6 ms | 8456.6 ms | 8636.2 ms | `0.00%` | ⚠️ **SATURATED** |

### 5C. `3 KB` Dedicated Saturation Sweep (`FP32`, Restricted at `2,048` Tokens)
| Target RPS | Achieved RPS | $P_{50}$ (`ms`) | $P_{95}$ (`ms`) | $P_{99}$ (`ms`) | Error Rate | SLA Status (`P99 < 50 ms`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **40 RPS** | 40.0 | **16.9 ms** | 17.8 ms | **18.5 ms** | `0.00%` | ✅ **PASS** |
| **50 RPS** | 50.0 | **16.9 ms** | 17.8 ms | **18.3 ms** | `0.00%` | ✅ **PASS** |
| **60 RPS** | 60.0 | **17.4 ms** | 18.5 ms | **20.2 ms** | `0.00%` | ✅ **PASS** |
| **70 RPS** | 70.0 | **20.7 ms** | 21.8 ms | **22.4 ms** | `0.00%` | ✅ **PASS** |
| **80 RPS** | 80.0 | **18.7 ms** | 19.8 ms | **20.5 ms** | `0.00%` | ✅ **PASS** |
| **90 RPS** | 90.0 | **18.0 ms** | 19.4 ms | **20.3 ms** | `0.00%` | ✅ **PASS (90 RPS Sustained)** |
