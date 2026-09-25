# Cloud TPU v6e (`FP32` 4-Layer Fused Megakernel) Benchmark & Cost Report — Strictly `max_model_len = 2048` (`jina-v2-embeddings-clean`)

> [!IMPORTANT]
> **Tested Engine Branch & Strict `max_model_len = 2048` Configuration**
> - **Engine Branch Tested**: [`pallavim1/tpu-inference@jina-v2-embeddings-clean`](https://github.com/pallavim1/tpu-inference/tree/jina-v2-embeddings-clean) (`commit a8733e93`)
> - **Reports & Docs Branch**: [`pallavim1/tpu-inference@panw-tpu-inference`](https://github.com/pallavim1/tpu-inference/tree/panw-tpu-inference/models/JinaEmbedding/vLLM/megakernel)
> - **Model & Precision**: `jinaai/jina-embeddings-v2-small-en` (`JinaBertForMaskedLM`, `dtype = float32`)
> - **Strict `2048` Token Limit**:
>   1. `JinaBertModel` (`tpu_inference/models/jax/jina_bert.py`) and `jina_v6e_4layer_megakernel` (`tpu_inference/kernels/jina_v6e_megakernel.py`) strictly enforce **`MAX_MODEL_LEN = 2048`** (`T <= 2048`; `4096` and `8192` token buckets are never compiled or executed).
>   2. `vllm serve` runs with **`--max-model-len 2048 --max-num-batched-tokens 2048 --dtype float32`**.
>   3. Adapter proxy (`megakernel_proxy.py`) enforces **`"truncate_prompt_tokens": 2048`** on every request.
> - **Payloads Tested**: Strictly **`1KB` (`1,024` random chars $\approx$ `1,009` tokens)** and **`2KB` (`2,048` random chars $\approx$ `2,016` tokens)**. (`3K`, `5K`, and `7K` tiers are excluded.)

---

## Part 1: Benchmark Results Matching Zhemin's `gid=1161755388` Tab AS-IS

### 1.1 Batch Request Testing: A Single HTTP Request Contains Multiple Prompts
**Client Config**: Concurrently sends a batch of random characters (`1KB = 1,024 chars` or `2KB = 2,048 chars`) in a single vLLM inference request and returns the response latency in milliseconds.

| Payload Size | Concurrency | TPU v6e Megakernel Throughput (`max_len=2048`) | TPU v6e Megakernel `p50` | TPU v6e Megakernel `p99` | TPU v5e Baseline Throughput (Zhemin) | TPU v5e Baseline `p50` | TPU v5e Baseline `p99` | NVIDIA L4 Throughput (Zhemin) | NVIDIA L4 `p50` | NVIDIA L4 `p99` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1KB (1024 chars)`** | **1** | **`109.5/s`** | **`9.0ms`** | **`11.4ms`** | `85.8/s` | `11.5ms` | `12.7ms` | `43.2/s` | `20.7ms` | `23.5ms` |
| **`1KB (1024 chars)`** | **4** | **`196.3/s`** | **`20.5ms`** | **`23.9ms`** | `187.4/s` | `21.2ms` | `22.8ms` | `103.4/s` | `38.1ms` | `42.9ms` |
| **`1KB (1024 chars)`** | **8** | **`252.4/s`** | **`31.4ms`** | **`37.5ms`** | `187.6/s` | `42.5ms` | `44.5ms` | `135.1/s` | `58.7ms` | `66.3ms` |
| **`1KB (1024 chars)`** | **16** | **`280.2/s`** | **`56.9ms`** | **`65.1ms`** | `186.8/s` | `85.4ms` | `89.9ms` | `165.2/s` | `96.5ms` | `107.6ms` |
| **`2KB (2048 chars)`** | **1** | **`89.0/s`** | **`11.3ms`** | **`13.0ms`** | `59.5/s` | `16.5ms` | `17.7ms` | `39.0/s` | `24.3ms` | `28.2ms` |
| **`2KB (2048 chars)`** | **4** | **`146.1/s`** | **`27.4ms`** | **`32.2ms`** | `97.7/s` | `40.7ms` | `42.8ms` | `75.7/s` | `52.1ms` | `58.2ms` |
| **`2KB (2048 chars)`** | **8** | **`163.5/s`** | **`50.1ms`** | **`54.4ms`** | `96.6/s` | `82.6ms` | `86.0ms` | `90.5/s` | `87.4ms` | `97.3ms` |
| **`2KB (2048 chars)`** | **16** | **`175.0/s`** | **`91.5ms`** | **`101.4ms`** | `96.5/s` | `165.9ms` | `171.2ms` | `101.1/s` | `156.9ms` | `175.5ms` |

---

### 1.2 `1KB Dedicated Saturation` (`1,024 chars ≈ 1,009 tokens`, `< 50 ms` `P99` SLA)

> [!TIP]
> By restricting `JinaBert` and the 4-layer Megakernel on `jina-v2-embeddings-clean` strictly to **`max_model_len = 2048` (`max_num_batched_tokens = 2048`)**, two `1KB` (`1,009`-token) requests pack into a single `T = 2048` forward step (`2,018 <= 2048` tokens) without ever triggering `T = 4096` or `T = 8192` padded attention grids. As a result, **`1KB` saturation reaches `270 RPS` (`P50 = 19.0 ms`, `P99 = 28.3 ms`, `✅ PASS`)** — **`+50.0%` higher RPS than TPU v5e (`180 RPS`)** and **`3.86x` higher RPS than NVIDIA L4 (`70 RPS`)**.

| RPS | TPU v6e Megakernel Achieved | TPU v6e Megakernel `P50` | TPU v6e Megakernel `P99` | TPU v6e Megakernel SLA (`<50ms`) | TPU v5e Achieved (Zhemin) | TPU v5e `P50` (Zhemin) | TPU v5e `P99` (Zhemin) | TPU v5e SLA (Zhemin) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | `99.97` | `11.5 ms` | `13.6 ms` | `✅ PASS` | `100` | `11.5 ms` | `14.0 ms` | `✅ PASS` |
| **120** | `119.96` | `12.2 ms` | `15.0 ms` | `✅ PASS` | `120` | `11.8 ms` | `15.5 ms` | `✅ PASS` |
| **140** | `139.92` | `13.1 ms` | `15.1 ms` | `✅ PASS` | `140` | `11.6 ms` | `18.5 ms` | `✅ PASS` |
| **160** | `159.91` | `12.5 ms` | `16.3 ms` | `✅ PASS` | `160` | `16.8 ms` | `24.2 ms` | `✅ PASS` |
| **180** | `179.89` | `12.7 ms` | `25.5 ms` | `✅ PASS` | `180.1` | `19.9 ms` | `29.6 ms` | `✅ PASS` |
| **190** | `189.87` | `12.3 ms` | `22.7 ms` | `✅ PASS` | `187.8` | `523.7 ms` | `762.7 ms` | `⚠️ SATURATED` |
| **200** | `199.84` | `17.9 ms` | `26.6 ms` | `✅ PASS` | `189` | `1709 ms` | `2818 ms` | `⚠️ SATURATED` |
| **220** | `219.76` | `18.5 ms` | `27.2 ms` | `✅ PASS` | `187.9` | `3738 ms` | `6182 ms` | `⚠️ SATURATED` |
| **240** | `239.71` | `19.7 ms` | `28.4 ms` | `✅ PASS` | — | — | — | `⚠️ SATURATED` |
| **260** | `259.69` | `20.7 ms` | `36.2 ms` | `✅ PASS` | — | — | — | `⚠️ SATURATED` |
| **270** | **`269.73`** | **`19.0 ms`** | **`28.3 ms`** | **`✅ PASS`** | — | — | — | `⚠️ SATURATED` |
| **280** | `275.68` | `47.6 ms` | `190.1 ms` | `⚠️ SATURATED` | — | — | — | `⚠️ SATURATED` |
| **290** | `271.16` | `507.9 ms` | `836.9 ms` | `⚠️ SATURATED` | — | — | — | `⚠️ SATURATED` |

---

### 1.3 `2KB Dedicated Saturation` (`2,048 chars ≈ 2,016 tokens`, `< 50 ms` `P99` SLA)

| RPS | TPU v6e Megakernel Achieved | TPU v6e Megakernel `P50` | TPU v6e Megakernel `P99` | TPU v6e Megakernel SLA (`<50ms`) | TPU v5e Achieved (Zhemin) | TPU v5e `P50` (Zhemin) | TPU v5e `P99` (Zhemin) | TPU v5e SLA (Zhemin) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **90** | `89.96` | `17.0 ms` | `18.9 ms` | `✅ PASS` | `90` | `17.2 ms` | `26.5 ms` | `✅ PASS` |
| **95** | `94.97` | `16.6 ms` | `18.3 ms` | `✅ PASS` | `94.75` | `218.6 ms` | `346.5 ms` | `⚠️ SATURATED` |
| **100** | `99.96` | `16.3 ms` | `18.0 ms` | `✅ PASS` | `96.32` | `1031 ms` | `2185 ms` | `⚠️ SATURATED` |
| **110** | `109.94` | `15.4 ms` | `16.9 ms` | `✅ PASS` | `96.74` | `3208 ms` | `5170 ms` | `⚠️ SATURATED` |
| **120** | `119.94` | `14.5 ms` | `16.4 ms` | `✅ PASS` | — | — | — | `⚠️ SATURATED` |
| **130** | `129.92` | `14.1 ms` | `17.9 ms` | `✅ PASS` | — | — | — | `⚠️ SATURATED` |
| **140** | `139.92` | `14.1 ms` | `30.3 ms` | `✅ PASS` | — | — | — | `⚠️ SATURATED` |
| **150** | **`149.90`** | **`13.9 ms`** | **`31.1 ms`** | **`✅ PASS`** | — | — | — | `⚠️ SATURATED` |
| **155** | `151.34` | `122.9 ms` | `292.1 ms` | `⚠️ SATURATED` | — | — | — | `⚠️ SATURATED` |
| **160** | `149.57` | `456.9 ms` | `836.6 ms` | `⚠️ SATURATED` | — | — | — | `⚠️ SATURATED` |

---

## Part 2: Platform Comparison & Cost Improvement Matching Zhemin's `gid=1972899730` Tab (`1KB` & `2KB`)

### 2.1 Concurrent Request Latency Comparison (`L4` vs `TPU v5e` vs `TPU v6e Megakernel`)

| Payload Category | L4 `P50` | TPU v5e `P50` | TPU v6e Megakernel `P50` | v6e vs L4 `P50` | v6e vs v5e `P50` | L4 `P99` | TPU v5e `P99` | TPU v6e Megakernel `P99` | v6e vs L4 `P99` | v6e vs v5e `P99` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1KB (1024 chars)` @ C=1** | `20.7 ms` | `11.5 ms` | **`9.0 ms`** | **`-56.5%`** | **`-21.7%`** | `23.5 ms` | `12.7 ms` | **`11.4 ms`** | **`-51.5%`** | **`-10.2%`** |
| **`2KB (2048 chars)` @ C=1** | `24.3 ms` | `16.5 ms` | **`11.3 ms`** | **`-53.5%`** | **`-31.5%`** | `28.2 ms` | `17.7 ms` | **`13.0 ms`** | **`-53.9%`** | **`-26.6%`** |
| **`1KB (1024 chars)` @ C=4** | `38.1 ms` | `21.2 ms` | **`20.5 ms`** | **`-46.2%`** | **`-3.3%`** | `42.9 ms` | `22.8 ms` | **`23.9 ms`** | **`-44.3%`** | `+4.8%` |
| **`2KB (2048 chars)` @ C=4** | `52.1 ms` | `40.7 ms` | **`27.4 ms`** | **`-47.4%`** | **`-32.7%`** | `58.2 ms` | `42.8 ms` | **`32.2 ms`** | **`-44.7%`** | **`-24.8%`** |
| **`1KB (1024 chars)` @ C=8** | `58.7 ms` | `42.5 ms` | **`31.4 ms`** | **`-46.5%`** | **`-26.1%`** | `66.3 ms` | `44.5 ms` | **`37.5 ms`** | **`-43.4%`** | **`-15.7%`** |
| **`2KB (2048 chars)` @ C=8** | `87.4 ms` | `82.6 ms` | **`50.1 ms`** | **`-42.7%`** | **`-39.3%`** | `97.3 ms` | `86.0 ms` | **`54.4 ms`** | **`-44.1%`** | **`-36.7%`** |

---

### 2.2 RPS Saturation Result (`< 50 ms` `P99` Latency SLA)

| Payload Size | NVIDIA L4 (`g2-standard-8`) | TPU v5e-1 (`FP32`) | TPU v6e-1 (`FP32` Megakernel, `max_len=2048`) | TPU v6e `P50` at Max RPS | TPU v6e `P99` at Max RPS | TPU v6e Gain vs L4 | TPU v6e Gain vs TPU v5e |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1KB (1,024 chars ≈ 1,009 tokens)`** | `70 RPS` | `180 RPS` | **`270 RPS`** | `19.0 ms` | `28.3 ms` | **`3.86x (+285.7%)`** | **`1.50x (+50.0%)`** |
| **`2KB (2,048 chars ≈ 2,016 tokens)`** | `40 RPS` | `90 RPS` | **`150 RPS`** | `13.9 ms` | `31.1 ms` | **`3.75x (+275.0%)`** | **`1.67x (+66.7%)`** |

---

### 2.3 Cost Improvement & Production Fleet TCO (`100%` Max Capacity & `40%` Production Fleet Utilization)

| Payload Size | Platform | On-Demand Price (`$/hr`) | Max Validated RPS (`<50ms P99`) | Throughput per Dollar (`RPS/$/hr`) | Relative Perf/$ vs L4 | Effective RPS @ `40%` Fleet Util | Nodes for `1,000 RPS` (`@40%` Util) | Monthly Cost (`1,000 RPS` @ `40%`) | Monthly TCO Savings vs L4 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1KB`** | **NVIDIA L4 (`g2-standard-8`)** | `\$1.0124` | `70 RPS` | `69.14 RPS/\$` | `1.00x (Baseline)` | `28.0 RPS` | `36` | `\$26,606 / mo` | `Baseline` |
| **`1KB`** | **TPU v5e-1 (`ct5lp-hightpu-1t`, FP32)** | `\$0.7990` | `180 RPS` | `225.28 RPS/\$` | `3.26x (+225.8%)` | `72.0 RPS` | `14` | `\$8,166 / mo` | **`-69.3% (\$18,440/mo saved)`** |
| **`1KB`** | **TPU v6e-1 (`ct6e-standard-1t`, FP32 Megakernel)** | `\$0.9990` | **`270 RPS`** | **`270.27 RPS/\$`** | **`3.91x (+290.9%)`** | **`108.0 RPS`** | **`10`** | **`\$7,293 / mo`** | **`-72.6% (\$19,313/mo saved)`** |
| **`2KB`** | **NVIDIA L4 (`g2-standard-8`)** | `\$1.0124` | `40 RPS` | `39.51 RPS/\$` | `1.00x (Baseline)` | `16.0 RPS` | `63` | `\$46,560 / mo` | `Baseline` |
| **`2KB`** | **TPU v5e-1 (`ct5lp-hightpu-1t`, FP32)** | `\$0.7990` | `90 RPS` | `112.64 RPS/\$` | `2.85x (+185.1%)` | `36.0 RPS` | `28` | `\$16,332 / mo` | **`-64.9% (\$30,228/mo saved)`** |
| **`2KB`** | **TPU v6e-1 (`ct6e-standard-1t`, FP32 Megakernel)** | `\$0.9990` | **`150 RPS`** | **`150.15 RPS/\$`** | **`3.80x (+280.0%)`** | **`60.0 RPS`** | **`17`** | **`\$12,398 / mo`** | **`-73.4% (\$34,162/mo saved)`** |
