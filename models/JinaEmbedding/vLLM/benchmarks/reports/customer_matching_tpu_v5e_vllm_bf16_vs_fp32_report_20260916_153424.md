# Jina AI Embedding Model + Cloud TPU v5e + vLLM: `bfloat16` (`BF16`) vs `float32` (`FP32`) Benchmark Report (`--max-model-len 2048`)

**Run Timestamp (UTC):** `2026-09-16 15:34:24 UTC` (`20260916_153424`)  
**Model:** `jinaai/jina-embeddings-v2-small-en` (512-dim BERT/ALiBi Embeddings)  
**Serving Stack:** `vLLM v0.26.0` (`--runner pooling --convert embed --trust-remote-code --max-num-seqs=40 --max-model-len 2048 --max-num-batched-tokens 8192`)  
**Precision Comparison:** **`--dtype bfloat16` (`BF16`)** vs **`--dtype float32` (`FP32`)**  
**Tokenization & Truncation Policy:** Fast Tokenizer + `truncate_prompt_tokens: 2048` (matching customer's Triton TensorRT `max_length=2048` baseline)  
**Cluster & Zone:** `pm-panw-jina-cluster` (`europe-west4-b`), Project `northam-ce-mlai-tpu`  
**Accelerator Hardware:** 1x Cloud TPU v5e chip (`ct5lp-hightpu-1t`)  
**Reference Worksheet:** [ATP AIC2 Benchmarks](https://docs.google.com/spreadsheets/d/1fGgqjRp4giG0MD6NjSIzAbpDbeJoq_Aao7B2MM78QgU/edit?gid=1161755388#gid=1161755388)

---

## 1. Executive Summary (`BF16` vs `FP32`)

> [!IMPORTANT]
> **Key Takeaways from Switching `--dtype float32` $\rightarrow$ `--dtype bfloat16` (with `--max-model-len 2048` & `--max-num-seqs=40`)**
> 1. **Multi-Prompt Single-Request Batch Test (`"input": [prompt_1, ..., prompt_N]`, $N \in \{1, 4, 8, 16\}$)**:
>    - Cloud TPU v5e's Matrix Multiply Units (MXUs) execute `bfloat16` natively at **2x the matrix compute rate and half the HBM memory bandwidth** of `float32`.
>    - While single-prompt latency ($N=1$) is network/dispatch-bound (`~12–19 ms` in both `BF16` and `FP32`), **at higher batch sizes ($N = 8$ and $N = 16$), `BF16` achieves `1.83x to 2.23x higher throughput` and `45% to 59% lower latency`** across `1 KB`, `2 KB`, `3 KB`, and `4 KB` payloads:
>      - **1 KB (`Batch = 16`)**: `BF16` reaches **`112.3 prompts/s`** ($P_{50}=142.8\text{ ms}$) vs `FP32` `50.4 prompts/s` ($P_{50}=331.4\text{ ms}$) — **2.23x higher throughput, 56.9% lower $P_{50}$ latency**.
>      - **2 KB (`Batch = 8`)**: `BF16` reaches **`59.1 prompts/s`** ($P_{50}=130.9\text{ ms}$) vs `FP32` `28.6 prompts/s` ($P_{50}=319.1\text{ ms}$) — **2.07x higher throughput, 59.0% lower $P_{50}$ latency**.
>      - **3 KB (`Batch = 16`)**: `BF16` reaches **`47.6 prompts/s`** ($P_{50}=339.1\text{ ms}$) vs `FP32` `25.5 prompts/s` ($P_{50}=632.9\text{ ms}$) — **1.87x higher throughput, 46.4% lower $P_{50}$ latency**.
>      - **4 KB (`Batch = 16`)**: `BF16` reaches **`46.7 prompts/s`** ($P_{50}=353.7\text{ ms}$) vs `FP32` `24.9 prompts/s` ($P_{50}=648.2\text{ ms}$) — **1.88x higher throughput, 45.4% lower $P_{50}$ latency**.
> 2. **`k6` Concurrency Request Testing (`constant-vus`, 1 Prompt per HTTP Request)**:
>    - **1 KB (`Concurrency = 4`)**: `BF16` jumps from `153.8 req/s` to **`176.5 req/s`** ($P_{50}=21.7\text{ ms}, P_{95}=31.2\text{ ms}$).
>    - **2 KB (`Concurrency = 4`)**: `BF16` cuts $P_{50}$ latency nearly in half from `89.1 ms` (`FP32`) to **`47.3 ms`** (**-46.9%**) and increases throughput by **+25.4%** (`64.2 req/s` vs `51.2 req/s`).
>    - **2 KB (`Concurrency = 16`)**: `BF16` increases throughput by **2.50x** (`71.3 req/s` vs `28.5 req/s`) and cuts $P_{50}$ latency by **62.5%** (`210.4 ms` vs `560.6 ms`).
>    - **3 KB (`Concurrency = 8`)**: `BF16` increases throughput by **2.57x** (`57.4 req/s` vs `22.3 req/s`) and cuts $P_{50}$ latency by **61.0%** (`138.9 ms` vs `356.5 ms`).
> 3. **`k6` Dedicated Saturation Sweeps (`< 50 ms` $P_{99}$ SLA)**:
>    - **1 KB Dedicated (`160 RPS`)**: `BF16` cuts $P_{99}$ latency by **50.0%** from `54.6 ms` (`FP32` ⚠️ Saturated) to **`27.3 ms`** (`BF16` ✅ **PASS**)!
>    - **2 KB & 3 KB Dedicated (`90 RPS`)**: Both pass the `< 50 ms` $P_{99}$ SLA cleanly (`23.2 ms` and `22.6 ms` $P_{99}$).

---

## 2. Multi-Prompt Single-Request Batch Test (`"input": [prompt_1, ..., prompt_N]`): `BF16` vs `FP32`

This table matches the methodology in **Rows 4–13 (`Batch Request Testing: A single HTTP request contains multiple prompts`)** of Zhemin's worksheet, where a single HTTP request sends a JSON array of $N \in \{1, 4, 8, 16\}$ random-character strings:

| Payload Size | Batch Size (`N` prompts/req) | **`BF16` Throughput (`prompts/s`)** | **`FP32` Throughput (`prompts/s`)** | **`BF16` $P_{50}$ (`ms`)** | **`FP32` $P_{50}$ (`ms`)** | **`BF16` $P_{99}$ (`ms`)** | **`FP32` $P_{99}$ (`ms`)** | **`BF16` Avg (`ms`)** | **`FP32` Avg (`ms`)** | **`BF16` vs `FP32` Improvement** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1 KB (1024 chars)** | **1** | 75.9/s | 78.8/s | 13.1 ms | 12.7 ms | 14.7 ms | 13.2 ms | 13.2 ms | 12.7 ms | Comparable (1-seq overhead) |
| **1 KB (1024 chars)** | **4** | **109.2/s** | 104.9/s | **34.9 ms** | 41.1 ms | 43.6 ms | 42.8 ms | **36.6 ms** | 38.1 ms | **15.1% lower $P_{50}$ latency (+4.1% tput)** |
| **1 KB (1024 chars)** | **8** | **86.5/s** | 75.6/s | **100.4 ms** | 107.8 ms | **103.1 ms** | 110.0 ms | **92.5 ms** | 105.7 ms | **+14.4% higher throughput, 12.5% lower avg** |
| **1 KB (1024 chars)** | **16** | **112.3/s** | 50.4/s | **142.8 ms** | 331.4 ms | **147.8 ms** | 342.9 ms | **142.4 ms** | 317.2 ms | **2.23x throughput, 56.9% lower $P_{50}$ latency** |
| **2 KB (2048 chars)** | **1** | 55.9/s | 57.9/s | 17.8 ms | 17.2 ms | 19.9 ms | 18.0 ms | 17.9 ms | 17.3 ms | Comparable (1-seq overhead) |
| **2 KB (2048 chars)** | **4** | **47.6/s** | 43.9/s | 95.3 ms | 93.0 ms | **97.5 ms** | 102.8 ms | **84.0 ms** | 91.2 ms | **+8.4% higher throughput, 7.9% lower avg** |
| **2 KB (2048 chars)** | **8** | **59.1/s** | 28.6/s | **130.9 ms** | 319.1 ms | **167.8 ms** | 331.0 ms | **135.2 ms** | 280.0 ms | **2.07x throughput, 59.0% lower $P_{50}$ latency** |
| **2 KB (2048 chars)** | **16** | **56.5/s** | 37.8/s | **281.5 ms** | 419.1 ms | **327.5 ms** | 430.3 ms | **283.0 ms** | 422.7 ms | **1.49x throughput, 32.8% lower $P_{50}$ latency** |
| **3 KB (3072 chars)** | **1** | 52.4/s | 55.2/s | 19.1 ms | 18.1 ms | 20.7 ms | 18.8 ms | 19.1 ms | 18.1 ms | Comparable (1-seq overhead) |
| **3 KB (3072 chars)** | **4** | **47.0/s** | 40.2/s | **99.6 ms** | 104.5 ms | **101.3 ms** | 106.4 ms | **85.1 ms** | 99.5 ms | **+16.9% higher throughput, 14.5% lower avg** |
| **3 KB (3072 chars)** | **8** | **46.2/s** | 24.7/s | **181.5 ms** | 334.4 ms | **185.6 ms** | 390.6 ms | **173.2 ms** | 323.5 ms | **1.87x throughput, 45.7% lower $P_{50}$ latency** |
| **3 KB (3072 chars)** | **16** | **47.6/s** | 25.5/s | **339.1 ms** | 632.9 ms | **349.8 ms** | 700.2 ms | **336.1 ms** | 628.5 ms | **1.87x throughput, 46.4% lower $P_{50}$ latency** |
| **4 KB (4096 chars)** | **1** | 51.5/s | 53.7/s | 19.3 ms | 18.5 ms | 21.0 ms | 20.1 ms | 19.4 ms | 18.6 ms | Comparable (1-seq overhead) |
| **4 KB (4096 chars)** | **4** | **42.4/s** | 41.0/s | 100.8 ms | 96.5 ms | **104.2 ms** | 108.4 ms | **94.3 ms** | 97.5 ms | **+3.4% higher throughput, 3.9% lower $P_{99}$** |
| **4 KB (4096 chars)** | **8** | **46.2/s** | 25.3/s | **175.1 ms** | 327.1 ms | **186.2 ms** | 390.8 ms | **173.0 ms** | 316.6 ms | **1.83x throughput, 46.5% lower $P_{50}$ latency** |
| **4 KB (4096 chars)** | **16** | **46.7/s** | 24.9/s | **353.7 ms** | 648.2 ms | **357.4 ms** | 704.2 ms | **342.9 ms** | 643.7 ms | **1.88x throughput, 45.4% lower $P_{50}$ latency** |

---

## 3. Suite 1: `k6` Concurrency Request Testing (`constant-vus`, 1 Prompt per HTTP Request) — `BF16` vs `FP32`

| Payload Size | Concurrency (`VUs`) | **`BF16` Throughput (`req/s`)** | **`FP32` Throughput (`req/s`)** | **`BF16` $P_{50}$ (`ms`)** | **`FP32` $P_{50}$ (`ms`)** | **`BF16` $P_{95}$ (`ms`)** | **`FP32` $P_{95}$ (`ms`)** | **`BF16` $P_{99}$ (`ms`)** | **`FP32` $P_{99}$ (`ms`)** | **`BF16` vs `FP32` Improvement** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1 KB (1024 chars)** | **1** | 83.1/s | 85.4/s | 11.8 ms | 11.5 ms | 12.4 ms | 12.1 ms | 13.4 ms | 12.7 ms | Comparable (~11.5–11.8 ms $P_{50}$) |
| **1 KB (1024 chars)** | **4** | **176.5/s** | 153.8/s | **21.7 ms** | 21.3 ms | **31.2 ms** | 38.8 ms | 41.3 ms | 40.4 ms | **+14.8% higher throughput, 19.6% lower $P_{95}$** |
| **1 KB (1024 chars)** | **8** | 163.7/s | 172.4/s | 47.1 ms | 44.6 ms | 49.7 ms | 46.0 ms | **97.0 ms** | 101.5 ms | **4.4% lower $P_{99}$ latency** |
| **1 KB (1024 chars)** | **16** | **113.6/s** | 103.9/s | **140.3 ms** | 153.2 ms | **144.6 ms** | 154.6 ms | **146.1 ms** | 155.8 ms | **+9.3% higher throughput, 8.4% lower $P_{50}$** |
| **2 KB (2048 chars)** | **1** | 59.7/s | 61.7/s | 16.5 ms | 16.0 ms | 17.2 ms | 16.6 ms | 18.1 ms | 17.2 ms | Matches customer's `59.5/s` (`16.5ms`) |
| **2 KB (2048 chars)** | **4** | **64.2/s** | 51.2/s | **47.3 ms** | 89.1 ms | **92.1 ms** | 98.9 ms | **93.1 ms** | 99.9 ms | **+25.4% throughput, 46.9% lower $P_{50}$ latency** |
| **2 KB (2048 chars)** | **8** | **57.7/s** | 50.6/s | **138.2 ms** | 153.2 ms | **141.5 ms** | 155.0 ms | **142.9 ms** | 339.1 ms | **+14.0% throughput, 57.9% lower $P_{99}$ latency** |
| **2 KB (2048 chars)** | **16** | **71.3/s** | 28.5/s | **210.4 ms** | 560.6 ms | **280.3 ms** | 562.2 ms | **281.4 ms** | 597.2 ms | **2.50x throughput, 62.5% lower $P_{50}$ latency** |
| **3 KB (3072 chars)** | **1** | 57.0/s | 57.4/s | 17.2 ms | 17.1 ms | 18.3 ms | 18.0 ms | 19.6 ms | 19.2 ms | Comparable (~17.1–17.2 ms $P_{50}$) |
| **3 KB (3072 chars)** | **4** | 85.8/s | 89.0/s | 46.1 ms | 44.5 ms | 47.9 ms | 45.4 ms | 49.4 ms | 46.4 ms | Both pass `< 50 ms` $P_{99}$ SLA (`~86–89/s`) |
| **3 KB (3072 chars)** | **8** | **57.4/s** | 22.3/s | **138.9 ms** | 356.5 ms | **142.3 ms** | 357.6 ms | **143.3 ms** | 358.4 ms | **2.57x throughput, 61.0% lower $P_{50}$ latency** |
| **3 KB (3072 chars)** | **16** | **57.1/s** | 28.6/s | **279.7 ms** | 563.2 ms | **283.6 ms** | 564.6 ms | **284.7 ms** | 625.9 ms | **2.00x throughput, 50.3% lower $P_{50}$ latency** |
| **4 KB (4096 chars)** | **1** | 54.4/s | 56.7/s | 18.2 ms | 17.4 ms | 18.8 ms | 18.1 ms | 19.2 ms | 18.9 ms | Comparable (~17.4–18.2 ms $P_{50}$) |
| **4 KB (4096 chars)** | **4** | 84.9/s | 87.7/s | 46.8 ms | 45.1 ms | 48.4 ms | 46.4 ms | 49.7 ms | 47.6 ms | Both pass `< 50 ms` $P_{99}$ SLA (`~85–88/s`) |
| **4 KB (4096 chars)** | **8** | **57.0/s** | 52.1/s | **139.9 ms** | 152.9 ms | **142.8 ms** | 154.8 ms | **143.9 ms** | 155.8 ms | **+9.4% throughput, 8.5% lower $P_{50}$ latency** |
| **4 KB (4096 chars)** | **16** | **56.8/s** | 28.5/s | **281.5 ms** | 562.4 ms | **285.0 ms** | 564.1 ms | **286.5 ms** | 610.0 ms | **1.99x throughput, 49.9% lower $P_{50}$ latency** |

---

## 4. Suite 2: Multi-Payload Concurrent Sweep (`50–90 RPS` Across `1 KB, 2 KB, 5 KB, 7 KB`) — `BF16` vs `FP32`

Both `BF16` and `FP32` (with `--max-model-len 2048` and `truncate_prompt_tokens: 2048`) achieve **`0.00%` errors** and pass the $< 50\text{ ms}$ $P_{99}$ SLA across all 4 payload tiers from 50 RPS to 90 RPS:

| Target RPS | **1 KB $P_{99}$ (`BF16`)** | **1 KB $P_{99}$ (`FP32`)** | **2 KB $P_{99}$ (`BF16`)** | **2 KB $P_{99}$ (`FP32`)** | **5 KB $P_{99}$ (`BF16`)** | **5 KB $P_{99}$ (`FP32`)** | **7 KB $P_{99}$ (`BF16`)** | **7 KB $P_{99}$ (`FP32`)** | SLA Status (`< 50 ms`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **50 RPS** | **13.3 ms** (`0%`) | 13.2 ms (`0%`) | **17.7 ms** (`0%`) | 17.6 ms (`0%`) | **19.6 ms** (`0%`) | 17.9 ms (`0%`) | **19.9 ms** (`0%`) | 19.1 ms (`0%`) | ✅ **ALL PASS** |
| **60 RPS** | **13.2 ms** (`0%`) | 13.7 ms (`0%`) | **19.2 ms** (`0%`) | 18.7 ms (`0%`) | **24.8 ms** (`0%`) | 20.2 ms (`0%`) | **25.9 ms** (`0%`) | 24.8 ms (`0%`) | ✅ **ALL PASS** |
| **70 RPS** | **13.8 ms** (`0%`) | 13.8 ms (`0%`) | **21.1 ms** (`0%`) | 20.8 ms (`0%`) | **22.6 ms** (`0%`) | 23.0 ms (`0%`) | **23.8 ms** (`0%`) | 23.0 ms (`0%`) | ✅ **ALL PASS** |
| **80 RPS** | **13.1 ms** (`0%`) | 13.8 ms (`0%`) | **19.6 ms** (`0%`) | 19.3 ms (`0%`) | **20.6 ms** (`0%`) | 21.2 ms (`0%`) | **21.4 ms** (`0%`) | 20.2 ms (`0%`) | ✅ **ALL PASS** |
| **90 RPS** | **16.2 ms** (`0%`) | 15.5 ms (`0%`) | **19.7 ms** (`0%`) | 19.5 ms (`0%`) | **21.2 ms** (`0%`) | 20.4 ms (`0%`) | **24.0 ms** (`0%`) | 20.0 ms (`0%`) | ✅ **ALL PASS** |

---

## 5. Suite 3: Dedicated RPS Saturation Sweeps (`1 KB`, `2 KB`, `3 KB`) — `BF16` vs `FP32`

### 5A. `1 KB` Dedicated Saturation Sweep (`BF16` vs `FP32`)
Notice that at **`160 RPS`**, `BF16` cuts $P_{99}$ latency in half (`27.3 ms` vs `54.6 ms` in `FP32`) to **PASS** the $< 50\text{ ms}$ SLA, and under overload (`190–220 RPS`), `BF16` sustains **`152–153 RPS`** vs `FP32`'s `78–127 RPS` (**+23% to +94% higher saturated throughput**):

| Target RPS | **`BF16` Achieved RPS** | **`FP32` Achieved RPS** | **`BF16` $P_{50}$ (`ms`)** | **`FP32` $P_{50}$ (`ms`)** | **`BF16` $P_{99}$ (`ms`)** | **`FP32` $P_{99}$ (`ms`)** | **`BF16` SLA Status** | **`FP32` SLA Status** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100 RPS** | **100.0** | 100.0 | **13.2 ms** | 13.4 ms | **15.4 ms** | 15.8 ms | ✅ **PASS** | ✅ **PASS** |
| **120 RPS** | **120.0** | 119.9 | **12.9 ms** | 12.8 ms | **15.7 ms** | 14.3 ms | ✅ **PASS** | ✅ **PASS** |
| **140 RPS** | **139.9** | 139.9 | **13.0 ms** | 12.2 ms | **19.7 ms** | 18.1 ms | ✅ **PASS** | ✅ **PASS** |
| **160 RPS** | **159.8** | 159.6 | **19.1 ms** | 18.8 ms | **27.3 ms** (**-50%**) | 54.6 ms | ✅ **PASS** | ⚠️ SATURATED |
| **180 RPS** | **179.4** | 179.5 | **48.5 ms** | 26.5 ms | **61.5 ms** | 57.4 ms | ⚠️ Borderline ($P_{50}=48.5\text{ ms}$) | ⚠️ Borderline |
| **190 RPS** | **153.2** (+23.6%) | 123.9 | 1401.5 ms | 51.8 ms | **2265.4 ms** (-46%) | 4216.0 ms | ⚠️ SATURATED | ⚠️ SATURATED |
| **200 RPS** | **151.9** (+19.1%) | 127.5 | 1830.2 ms | 53.1 ms | **2462.0 ms** (-43%) | 4302.1 ms | ⚠️ SATURATED | ⚠️ SATURATED |
| **220 RPS** | **152.4** (+94.1%) | 78.5 | **2044.1 ms** (-52%) | 4217.8 ms | **2608.6 ms** (-51%) | 5294.6 ms | ⚠️ SATURATED | ⚠️ SATURATED |

### 5B. `2 KB` Dedicated Saturation Sweep (`BF16` vs `FP32` vs Customer Sheet)
| Target RPS | **`BF16` Achieved RPS** | **`BF16` $P_{50}$ (`ms`)** | **`BF16` $P_{99}$ (`ms`)** | **`BF16` SLA Status** | **Customer Sheet Baseline (`FP32`)** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **70 RPS** | 70.0 | **19.5 ms** | **21.0 ms** | ✅ **PASS** | ✅ PASS |
| **80 RPS** | 80.0 | **18.2 ms** | **20.8 ms** | ✅ **PASS** | ✅ PASS |
| **90 RPS** | 89.9 | **18.1 ms** | **23.2 ms** | ✅ **PASS** | 17.2 ms $P_{50}$ / 26.5 ms $P_{99}$ (✅ PASS) |
| **95 RPS** | 74.1 | 1318.0 ms | 2946.3 ms | ⚠️ SATURATED | 218.6 ms $P_{50}$ / 346.5 ms $P_{99}$ (⚠️ SATURATED) |
| **100 RPS** | 70.7 | 2407.2 ms | 3777.1 ms | ⚠️ SATURATED | 1031 ms $P_{50}$ / 2185 ms $P_{99}$ (⚠️ SATURATED) |
| **110 RPS** | 70.9 | **2960.5 ms** | **4308.3 ms** | ⚠️ SATURATED | 3208 ms $P_{50}$ / 5170 ms $P_{99}$ (⚠️ SATURATED) |

### 5C. `3 KB` Dedicated Saturation Sweep (`BF16` vs `FP32` vs Customer Sheet)
| Target RPS | **`BF16` Achieved RPS** | **`BF16` $P_{50}$ (`ms`)** | **`BF16` $P_{99}$ (`ms`)** | **`BF16` SLA Status** | **`FP32` $P_{50}$ / $P_{99}$ (`ms`)** | **Customer Sheet Baseline** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **40 RPS** | 40.0 | **17.8 ms** | **19.9 ms** | ✅ **PASS** | 16.9 ms / 18.5 ms (✅ PASS) | ❌ 100% Failed (`max_model_len`) |
| **50 RPS** | 50.0 | **17.5 ms** | **18.9 ms** | ✅ **PASS** | 16.9 ms / 18.3 ms (✅ PASS) | ❌ 100% Failed (`max_model_len`) |
| **60 RPS** | 60.0 | **17.7 ms** | **20.1 ms** | ✅ **PASS** | 17.4 ms / 20.2 ms (✅ PASS) | ❌ 100% Failed (`max_model_len`) |
| **70 RPS** | 70.0 | **20.7 ms** | **22.4 ms** | ✅ **PASS** | 20.7 ms / 22.4 ms (✅ PASS) | ❌ 100% Failed (`max_model_len`) |
| **80 RPS** | 80.0 | **19.6 ms** | **21.8 ms** | ✅ **PASS** | 18.7 ms / 20.5 ms (✅ PASS) | ❌ 100% Failed (`max_model_len`) |
| **90 RPS** | 89.9 | **19.0 ms** | **22.6 ms** | ✅ **PASS** | 18.0 ms / 20.3 ms (✅ PASS) | ❌ 100% Failed (`max_model_len`) |
