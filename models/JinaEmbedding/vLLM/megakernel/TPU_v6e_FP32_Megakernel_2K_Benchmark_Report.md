# Jina Embedding (`jina-embeddings-v2-small-en`) — TPU v6e (`FP32` Megakernel) Benchmark Results (Zhemin AS-IS Format)

## Test Configuration (Matching Zhemin's Baseline AS-IS)
* **Model**: `jinaai/jina-embeddings-v2-small-en` (`FP32` / `float32`)
* **`vLLM` Server Config**:
  ```bash
  vllm serve jinaai/jina-embeddings-v2-small-en \
    --runner pooling --convert embed --trust-remote-code \
    --max-model-len 2048 --dtype float32 \
    --max-num-seqs 40 --max-num-batched-tokens 8192
  ```
* **Tokenization & Payload Tiers (`1KB` & `2KB` Random Characters ONLY — No `3K`, `5K`, `7K`)**:
  * `BertTokenizerFast` with `1KB (1,024 random chars ≈ 1,009 tokens)` and `2KB (2,048 random chars ≈ 2,016 tokens)`
  * **SLA Threshold**: `P99 < 50 ms` (`✅ PASS` when `P99 < 50 ms`, `⚠️ SATURATED` when `P99 >= 50 ms`)

---

## 1. Batch Request Testing: A single HTTP request contains multiple prompts

**Client Config**: Concurrently sends a batch of random characters (`1KB (1024 chars)` $\approx$ `1K tokens`, `2KB (2048 chars)` $\approx$ `2K tokens`) in a single `vLLM` inference request and returns the response latency in milliseconds.

| Payload Size | Concurrency | **TPU v6e + vLLM Megakernel (`FP32`)**<br>Throughput | **TPU v6e + vLLM Megakernel (`FP32`)**<br>`p50` | **TPU v6e + vLLM Megakernel (`FP32`)**<br>`p99` | **Zhemin TPU v5e + vLLM (`FP32`)**<br>Throughput / `p50` / `p99` | **Zhemin L4 GPU + Triton (`FP16`)**<br>Throughput / `p50` / `p99` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1KB (1024 chars)** | **1** | **`116.3/s`** | **`8.4ms`** | **`10.0ms`** | `85.8/s` \| `11.5ms` \| `12.7ms` | `43.2/s` \| `20.7ms` \| `23.5ms` |
| **1KB (1024 chars)** | **4** | **`213.8/s`** | **`19.1ms`** | **`25.3ms`** | `187.4/s` \| `21.2ms` \| `22.8ms` | `103.4/s` \| `38.1ms` \| `42.9ms` |
| **1KB (1024 chars)** | **8** | **`269.9/s`** | **`27.3ms`** | **`49.6ms`** | `187.6/s` \| `42.5ms` \| `44.5ms` | `135.1/s` \| `58.7ms` \| `66.3ms` |
| **1KB (1024 chars)** | **16** | **`281.7/s`** | **`57.0ms`** | **`78.4ms`** | `186.8/s` \| `85.4ms` \| `89.9ms` | `165.2/s` \| `96.5ms` \| `107.6ms` |
| **2KB (2048 chars)** | **1** | **`92.7/s`** | **`10.6ms`** | **`12.0ms`** | `59.5/s` \| `16.5ms` \| `17.7ms` | `39.0/s` \| `24.3ms` \| `28.2ms` |
| **2KB (2048 chars)** | **4** | **`125.6/s`** | **`32.1ms`** | **`39.0ms`** | `97.7/s` \| `40.7ms` \| `42.8ms` | `75.7/s` \| `52.1ms` \| `58.2ms` |
| **2KB (2048 chars)** | **8** | **`203.3/s`** | **`38.1ms`** | **`60.0ms`** | `96.6/s` \| `82.6ms` \| `86.0ms` | `90.5/s` \| `87.4ms` \| `97.3ms` |
| **2KB (2048 chars)** | **16** | **`202.4/s`** | **`77.4ms`** | **`97.2ms`** | `96.5/s` \| `165.9ms` \| `171.2ms` | `101.1/s` \| `156.9ms` \| `175.5ms` |

---

## 2. 1KB Dedicated Saturation (`1,024 chars ≈ 1K tokens`, `FP32`)

| RPS | **TPU v6e Megakernel (`FP32`)**<br>Achieved | **TPU v6e Megakernel (`FP32`)**<br>P50 | **TPU v6e Megakernel (`FP32`)**<br>P99 | **TPU v6e Megakernel (`FP32`)**<br>SLA (`P99 < 50 ms`) | **Zhemin TPU v5e (`FP32`)**<br>Achieved / P50 / P99 / SLA |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | **99.99** | **8.6 ms** | **10.1 ms** | **✅ PASS** | `100` \| `11.5 ms` \| `14.0 ms` \| `✅ PASS` |
| **120** | **119.97** | **8.7 ms** | **12.4 ms** | **✅ PASS** | `120` \| `11.8 ms` \| `15.5 ms` \| `✅ PASS` |
| **140** | **139.94** | **11.0 ms** | **12.6 ms** | **✅ PASS** | `140` \| `11.6 ms` \| `18.5 ms` \| `✅ PASS` |
| **160** | **159.91** | **10.6 ms** | **12.3 ms** | **✅ PASS** | `160` \| `16.8 ms` \| `24.2 ms` \| `✅ PASS` |
| **180** | **179.89** | **10.4 ms** | **12.7 ms** | **✅ PASS** | `180.1` \| `19.9 ms` \| `29.6 ms` \| `✅ PASS` |
| **190** | **189.86** | **10.5 ms** | **15.4 ms** | **✅ PASS** | `187.8` \| `523.7 ms` \| `762.7 ms` \| `⚠️ SATURATED` |
| **200** | **199.75** | **17.1 ms** | **43.7 ms** | **✅ PASS** | `189` \| `1709 ms` \| `2818 ms` \| `⚠️ SATURATED` |
| **220** | **219.64** | **40.0 ms** | **79.1 ms** | **⚠️ SATURATED** | `187.9` \| `3738 ms` \| `6182 ms` \| `⚠️ SATURATED` |

---

## 3. 2KB Dedicated Saturation (`2,048 chars ≈ 2K tokens`, `FP32`)

| RPS | **TPU v6e Megakernel (`FP32`)**<br>Achieved | **TPU v6e Megakernel (`FP32`)**<br>P50 | **TPU v6e Megakernel (`FP32`)**<br>P99 | **TPU v6e Megakernel (`FP32`)**<br>SLA (`P99 < 50 ms`) | **Zhemin TPU v5e (`FP32`)**<br>Achieved / P50 / P99 / SLA |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **90** | **89.99** | **10.7 ms** | **12.9 ms** | **✅ PASS** | `90` \| `17.2 ms` \| `26.5 ms` \| `✅ PASS` |
| **95** | **94.99** | **10.7 ms** | **12.7 ms** | **✅ PASS** | `94.75` \| `218.6 ms` \| `346.5 ms` \| `⚠️ SATURATED` |
| **100** | **99.98** | **11.7 ms** | **15.4 ms** | **✅ PASS** | `96.32` \| `1031 ms` \| `2185 ms` \| `⚠️ SATURATED` |
| **110** | **109.95** | **13.7 ms** | **15.1 ms** | **✅ PASS** | `96.74` \| `3208 ms` \| `5170 ms` \| `⚠️ SATURATED` |
| **120** | **119.94** | **12.9 ms** | **14.3 ms** | **✅ PASS** | — *(Saturated at 95 RPS)* |
| **130** | **129.92** | **12.4 ms** | **13.8 ms** | **✅ PASS** | — *(Saturated at 95 RPS)* |
| **140** | **139.92** | **12.2 ms** | **14.2 ms** | **✅ PASS** | — *(Saturated at 95 RPS)* |
| **150** | **149.91** | **12.2 ms** | **15.4 ms** | **✅ PASS** | — *(Saturated at 95 RPS)* |
| **160** | **159.90** | **49.2 ms** | **111.2 ms** | **⚠️ SATURATED** | — *(Saturated at 95 RPS)* |
