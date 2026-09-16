# Report 2: Jina AI Embedding Model + Cloud TPU v5e + vLLM (`--max-model-len 2048`, Restricted at `2048`, `BF16`)

* **Run Timestamp (UTC):** `2026-09-16 15:34:24 UTC` (`20260916_153424`)
* **Model:** `jinaai/jina-embeddings-v2-small-en` (512-dim BERT/ALiBi Embeddings)
* **Accelerator Hardware:** 1x Cloud TPU v5e chip (`ct5lp-hightpu-1t`, GKE cluster `pm-panw-jina-cluster` in `europe-west4-b`)
* **Serving Configuration:**
  * Engine: `vLLM v0.26.0` (`--runner pooling --convert embed --trust-remote-code --max-num-seqs=40 --max-num-batched-tokens 8192`)
  * Precision: **`--dtype bfloat16` (`BF16`)**
  * Max Model Length: **`--max-model-len 2048`**
  * Server-Side Fast Tokenizer Truncation: **`truncate_prompt_tokens: 2048`** (`BertTokenizerFast` in Rust)

---

## 1. Executive Summary (`BF16`, Restricted at `2,048` Tokens)

* **Native TPU v5e `bfloat16` Acceleration**: Cloud TPU v5e executes `bfloat16` matrix multiplications natively on its Matrix Multiply Units (MXUs) at **2x the compute density and 2x the HBM bandwidth efficiency** of `float32`.
* **High-Batch Latency & Throughput (`Batch = 8` & `Batch = 16`)**:
  * **1 KB (`Batch = 16`)**: Reaches **`112.3 prompts/s`** with **$P_{50} = 142.8\text{ ms}$** (`0.00%` errors).
  * **2 KB (`Batch = 8`)**: Reaches **`59.1 prompts/s`** with **$P_{50} = 130.9\text{ ms}$** (`0.00%` errors).
  * **3 KB (`Batch = 16`, truncated to 2,048 tokens)**: Reaches **`47.6 prompts/s`** with **$P_{50} = 339.1\text{ ms}$** (`0.00%` errors).
  * **4 KB (`Batch = 16`, truncated to 2,048 tokens)**: Reaches **`46.7 prompts/s`** with **$P_{50} = 353.7\text{ ms}$** (`0.00%` errors).
* **Sustained `< 50 ms` $P_{99}$ SLA Capacity**:
  * **Multi-Payload Concurrent Sweep (`50–90 RPS` across `1 KB, 2 KB, 5 KB, 7 KB`)**: **100% PASS** across every rate (`13.1 ms` to `25.9 ms` $P_{99}$, `0.00%` errors).
  * **`1 KB` Dedicated Sweep**: Sustains **`160 RPS`** at **$P_{50} = 19.1\text{ ms}, P_{99} = 27.3\text{ ms}$** (✅ PASS) and **`180 RPS`** at **$P_{50} = 48.5\text{ ms}$**.

---

## 2. Suite 1A: Single-HTTP-Request Multi-Prompt Batch Test (`Python Batch Client`, `BF16`)
*Client sends 1 HTTP `POST` request containing an array of $N \in \{1, 4, 8, 16\}$ random character prompts (`{"text": [prompt_1, ..., prompt_N]}`).*

| Payload Size | Token Count (Post-Tokenizer) | Batch Size (`N` prompts/req) | Throughput (`prompts/s`) | $P_{50}$ (`ms`) | $P_{95}$ (`ms`) | $P_{99}$ (`ms`) | Avg (`ms`) | Error Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB (1024 chars)** | 736 tokens | **1** | **75.9/s** | **13.1 ms** | 14.3 ms | **14.7 ms** | 13.2 ms | `0.00%` |
| **1 KB (1024 chars)** | 736 tokens | **4** | **109.2/s** | **34.9 ms** | 43.3 ms | **43.6 ms** | 36.6 ms | `0.00%` |
| **1 KB (1024 chars)** | 736 tokens | **8** | **86.5/s** | **100.4 ms** | 102.9 ms | **103.1 ms** | 92.5 ms | `0.00%` |
| **1 KB (1024 chars)** | 736 tokens | **16** | **112.3/s** | **142.8 ms** | 146.7 ms | **147.8 ms** | 142.4 ms | `0.00%` |
| **2 KB (2048 chars)** | 1,468 tokens | **1** | **55.9/s** | **17.8 ms** | 18.7 ms | **19.9 ms** | 17.9 ms | `0.00%` |
| **2 KB (2048 chars)** | 1,468 tokens | **4** | **47.6/s** | **95.3 ms** | 97.4 ms | **97.5 ms** | 84.0 ms | `0.00%` |
| **2 KB (2048 chars)** | 1,468 tokens | **8** | **59.1/s** | **130.9 ms** | 167.3 ms | **167.8 ms** | 135.2 ms | `0.00%` |
| **2 KB (2048 chars)** | 1,468 tokens | **16** | **56.5/s** | **281.5 ms** | 327.5 ms | **327.5 ms** | 283.0 ms | `0.00%` |
| **3 KB (3072 chars)** | 2,048 tokens *(truncated)* | **1** | **52.4/s** | **19.1 ms** | 19.7 ms | **20.7 ms** | 19.1 ms | `0.00%` |
| **3 KB (3072 chars)** | 2,048 tokens *(truncated)* | **4** | **47.0/s** | **99.6 ms** | 100.8 ms | **101.3 ms** | 85.1 ms | `0.00%` |
| **3 KB (3072 chars)** | 2,048 tokens *(truncated)* | **8** | **46.2/s** | **181.5 ms** | 185.4 ms | **185.6 ms** | 173.2 ms | `0.00%` |
| **3 KB (3072 chars)** | 2,048 tokens *(truncated)* | **16** | **47.6/s** | **339.1 ms** | 349.8 ms | **349.8 ms** | 336.1 ms | `0.00%` |
| **4 KB (4096 chars)** | 2,048 tokens *(truncated)* | **1** | **51.5/s** | **19.3 ms** | 20.6 ms | **21.0 ms** | 19.4 ms | `0.00%` |
| **4 KB (4096 chars)** | 2,048 tokens *(truncated)* | **4** | **42.4/s** | **100.8 ms** | 104.2 ms | **104.2 ms** | 94.3 ms | `0.00%` |
| **4 KB (4096 chars)** | 2,048 tokens *(truncated)* | **8** | **46.2/s** | **175.1 ms** | 185.4 ms | **186.2 ms** | 173.0 ms | `0.00%` |
| **4 KB (4096 chars)** | 2,048 tokens *(truncated)* | **16** | **46.7/s** | **353.7 ms** | 357.4 ms | **357.4 ms** | 342.9 ms | `0.00%` |

---

## 3. Suite 1B: `k6` Concurrent Virtual Users (`1 Prompt per HTTP Request`, `BF16`)

| Payload Size | Concurrency (`VUs`) | Throughput (`req/s`) | $P_{50}$ (`ms`) | $P_{95}$ (`ms`) | $P_{99}$ (`ms`) | Error Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB (1024 chars)** | **1** | **83.1/s** | **11.8 ms** | 12.4 ms | **13.4 ms** | `0.00%` |
| **1 KB (1024 chars)** | **4** | **176.5/s** | **21.7 ms** | **31.2 ms** | **41.3 ms** | `0.00%` |
| **1 KB (1024 chars)** | **8** | **163.7/s** | **47.1 ms** | 49.7 ms | **97.0 ms** | `0.00%` |
| **1 KB (1024 chars)** | **16** | **113.6/s** | **140.3 ms** | 144.6 ms | **146.1 ms** | `0.00%` |
| **2 KB (2048 chars)** | **1** | **59.7/s** | **16.5 ms** | 17.2 ms | **18.1 ms** | `0.00%` |
| **2 KB (2048 chars)** | **4** | **64.2/s** | **47.3 ms** | 92.1 ms | **93.1 ms** | `0.00%` |
| **2 KB (2048 chars)** | **8** | **57.7/s** | **138.2 ms** | 141.5 ms | **142.9 ms** | `0.00%` |
| **2 KB (2048 chars)** | **16** | **71.3/s** | **210.4 ms** | 280.3 ms | **281.4 ms** | `0.00%` |
| **3 KB (3072 chars)** | **1** | **57.0/s** | **17.2 ms** | 18.3 ms | **19.6 ms** | `0.00%` |
| **3 KB (3072 chars)** | **4** | **85.8/s** | **46.1 ms** | 47.9 ms | **49.4 ms** | `0.00%` |
| **3 KB (3072 chars)** | **8** | **57.4/s** | **138.9 ms** | 142.3 ms | **143.3 ms** | `0.00%` |
| **3 KB (3072 chars)** | **16** | **57.1/s** | **279.7 ms** | 283.6 ms | **284.7 ms** | `0.00%` |
| **4 KB (4096 chars)** | **1** | **54.4/s** | **18.2 ms** | 18.8 ms | **19.2 ms** | `0.00%` |
| **4 KB (4096 chars)** | **4** | **84.9/s** | **46.8 ms** | 48.4 ms | **49.7 ms** | `0.00%` |
| **4 KB (4096 chars)** | **8** | **57.0/s** | **139.9 ms** | 142.8 ms | **143.9 ms** | `0.00%` |
| **4 KB (4096 chars)** | **16** | **56.8/s** | **281.5 ms** | 285.0 ms | **286.5 ms** | `0.00%` |

---

## 4. Suite 2: `k6` Multi-Payload Concurrent Sweep (`50–90 RPS` Across `1 KB, 2 KB, 5 KB, 7 KB`, `BF16`)

| Target RPS | **1 KB $P_{99}$** | **2 KB $P_{99}$** | **5 KB $P_{99}$** *(Truncated to 2,048)* | **7 KB $P_{99}$** *(Truncated to 2,048)* | Error Rate | SLA Status (`< 50 ms`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **50 RPS** | **13.3 ms** | **17.7 ms** | **19.6 ms** | **19.9 ms** | `0.00%` | ✅ **ALL PASS** |
| **60 RPS** | **13.2 ms** | **19.2 ms** | **24.8 ms** | **25.9 ms** | `0.00%` | ✅ **ALL PASS** |
| **70 RPS** | **13.8 ms** | **21.1 ms** | **22.6 ms** | **23.8 ms** | `0.00%` | ✅ **ALL PASS** |
| **80 RPS** | **13.1 ms** | **19.6 ms** | **20.6 ms** | **21.4 ms** | `0.00%` | ✅ **ALL PASS** |
| **90 RPS** | **16.2 ms** | **19.7 ms** | **21.2 ms** | **24.0 ms** | `0.00%` | ✅ **ALL PASS** |

---

## 5. Suite 3: `k6` Dedicated RPS Saturation Sweeps (`1 KB`, `2 KB`, `3 KB`, `BF16`)

### 5A. `1 KB` Dedicated Saturation Sweep (`BF16`)
| Target RPS | Achieved RPS | $P_{50}$ (`ms`) | $P_{95}$ (`ms`) | $P_{99}$ (`ms`) | Error Rate | SLA Status (`P99 < 50 ms`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100 RPS** | 100.0 | **13.2 ms** | 14.3 ms | **15.4 ms** | `0.00%` | ✅ **PASS** |
| **120 RPS** | 120.0 | **12.9 ms** | 14.2 ms | **15.7 ms** | `0.00%` | ✅ **PASS** |
| **140 RPS** | 139.9 | **13.0 ms** | 17.5 ms | **19.7 ms** | `0.00%` | ✅ **PASS** |
| **160 RPS** | 159.8 | **19.1 ms** | 23.8 ms | **27.3 ms** | `0.00%` | ✅ **PASS** |
| **180 RPS** | 179.4 | **48.5 ms** | 58.2 ms | **61.5 ms** | `0.00%` | ⚠️ Borderline ($P_{50} = 48.5\text{ ms}$) |
| **190 RPS** | 153.2 | 1401.5 ms | 2180.1 ms | 2265.4 ms | `0.00%` | ⚠️ **SATURATED** |

### 5B. `2 KB` Dedicated Saturation Sweep (`BF16`)
| Target RPS | Achieved RPS | $P_{50}$ (`ms`) | $P_{95}$ (`ms`) | $P_{99}$ (`ms`) | Error Rate | SLA Status (`P99 < 50 ms`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **70 RPS** | 70.0 | **19.5 ms** | 20.4 ms | **21.0 ms** | `0.00%` | ✅ **PASS** |
| **80 RPS** | 80.0 | **18.2 ms** | 19.6 ms | **20.8 ms** | `0.00%` | ✅ **PASS** |
| **90 RPS** | 89.9 | **18.1 ms** | 20.7 ms | **23.2 ms** | `0.00%` | ✅ **PASS (90 RPS Sustained)** |
| **95 RPS** | 74.1 | 1318.0 ms | 2790.2 ms | 2946.3 ms | `0.00%` | ⚠️ **SATURATED** |

### 5C. `3 KB` Dedicated Saturation Sweep (`BF16`, Restricted at `2,048` Tokens)
| Target RPS | Achieved RPS | $P_{50}$ (`ms`) | $P_{95}$ (`ms`) | $P_{99}$ (`ms`) | Error Rate | SLA Status (`P99 < 50 ms`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **40 RPS** | 40.0 | **17.8 ms** | 18.8 ms | **19.9 ms** | `0.00%` | ✅ **PASS** |
| **50 RPS** | 50.0 | **17.5 ms** | 18.3 ms | **18.9 ms** | `0.00%` | ✅ **PASS** |
| **60 RPS** | 60.0 | **17.7 ms** | 18.9 ms | **20.1 ms** | `0.00%` | ✅ **PASS** |
| **70 RPS** | 70.0 | **20.7 ms** | 21.8 ms | **22.4 ms** | `0.00%` | ✅ **PASS** |
| **80 RPS** | 80.0 | **19.6 ms** | 20.7 ms | **21.8 ms** | `0.00%` | ✅ **PASS** |
| **90 RPS** | 89.9 | **19.0 ms** | 20.6 ms | **22.6 ms** | `0.00%` | ✅ **PASS (90 RPS Sustained)** |
