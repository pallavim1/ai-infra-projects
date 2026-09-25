# Cloud TPU v6e (`FP32` 4-Layer Fused Megakernel) Benchmark & Cost Report — Strictly `max_model_len = 2048` (`jina-v2-embeddings-clean`)

> [!IMPORTANT]
> **Tested Engine Branch, Optimizations Applied, Strict `max_model_len = 2048` Config & On-Demand Pricing**
> - **Engine Branch Tested**: [`pallavim1/tpu-inference@jina-v2-embeddings-clean`](https://github.com/pallavim1/tpu-inference/tree/jina-v2-embeddings-clean) (`commit 8a096d8c`)
> - **Reports & Docs Branch**: [`pallavim1/tpu-inference@panw-tpu-inference`](https://github.com/pallavim1/tpu-inference/tree/panw-tpu-inference/models/JinaEmbedding/vLLM/megakernel)
> - **Model & Precision**: `jinaai/jina-embeddings-v2-small-en` (`JinaBertForMaskedLM`, `dtype = float32`)
> - **Strict `2048` Token Limit**:
>   1. `JinaBertModel` (`tpu_inference/models/jax/jina_bert.py`) and `jina_v6e_4layer_megakernel` (`tpu_inference/kernels/jina_v6e_megakernel.py`) strictly enforce **`MAX_MODEL_LEN = 2048`** (`T <= 2048`; `4096` and `8192` token buckets are never compiled or executed).
>   2. `vllm serve` runs with **`--max-model-len 2048 --max-num-batched-tokens 2048 --dtype float32`**.
>   3. Adapter proxy (`megakernel_proxy.py`) enforces **`"truncate_prompt_tokens": 2048`** on every request.
> - **Internal Megakernel & Proxy Optimizations Applied (`commit 8a096d8c`)**:
>   1. **Compile-Time NumPy ALiBi `[1, 8, T, T]` Constant in HBM**: Eliminates runtime `jnp.abs(idx[:, None] - idx[None, :]) * slopes` VPU tensor construction on every forward step (`0.00 ms` runtime VPU ALiBi overhead).
>   2. **1-Pass `FP32`-Accumulated MXU Projections**: Uses native 1-pass `preferred_element_type=jnp.float32` MXU dot products matching `tpu_inference`'s `JaxEinsum`/`JaxLinear`, cutting projection MXU passes by `3x` across `QKV`, `O`, `GeGLU`, and `WO`.
>   3. **Semaphore-Before-Drain Pair Coalescing (`1KB`)**: Acquires a pipeline slot before draining up to two `1KB` (`~1,009`-token) requests (`2 * 1,009 = 2,018 <= 2,048` token budget) so `1KB` pairs never fragment across workers.
> - **On-Demand (OD) Hourly Pricing**:
>   - **NVIDIA L4**: `\$0.70 / hr` (`\$511.00 / mo` per GPU)
>   - **Cloud TPU v5e**: `\$1.20 / hr` (`\$876.00 / mo` per chip)
>   - **Cloud TPU v6e**: `\$2.70 / hr` (`\$1,971.00 / mo` per chip)

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

| RPS | TPU v6e Megakernel Achieved | TPU v6e Megakernel `P50` | TPU v6e Megakernel `P99` | TPU v6e Megakernel SLA (`<50ms`) | TPU v5e Achieved (Zhemin) | TPU v5e `P50` (Zhemin) | TPU v5e `P99` (Zhemin) | TPU v5e SLA (Zhemin) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | `99.99` | `11.1 ms` | `14.5 ms` | `✅ PASS` | `100` | `11.5 ms` | `14.0 ms` | `✅ PASS` |
| **120** | `119.96` | `12.3 ms` | `15.0 ms` | `✅ PASS` | `120` | `11.8 ms` | `15.5 ms` | `✅ PASS` |
| **140** | `139.93` | `12.6 ms` | `15.4 ms` | `✅ PASS` | `140` | `11.6 ms` | `18.5 ms` | `✅ PASS` |
| **160** | `159.92` | `12.0 ms` | `14.1 ms` | `✅ PASS` | `160` | `16.8 ms` | `24.2 ms` | `✅ PASS` |
| **180** | `179.92` | `11.3 ms` | `19.9 ms` | `✅ PASS` | `180.1` | `19.9 ms` | `29.6 ms` | `✅ PASS` |
| **190** | `189.90` | `11.3 ms` | `15.3 ms` | `✅ PASS` | `187.8` | `523.7 ms` | `762.7 ms` | `⚠️ SATURATED` |
| **200** | `199.89` | `11.2 ms` | `19.8 ms` | `✅ PASS` | `189` | `1709 ms` | `2818 ms` | `⚠️ SATURATED` |
| **220** | `219.73` | `16.8 ms` | `24.0 ms` | `✅ PASS` | `187.9` | `3738 ms` | `6182 ms` | `⚠️ SATURATED` |
| **240** | `239.74` | `17.6 ms` | `24.4 ms` | `✅ PASS` | — | — | — | `⚠️ SATURATED` |
| **260** | `259.71` | `18.1 ms` | `24.8 ms` | `✅ PASS` | — | — | — | `⚠️ SATURATED` |
| **270** | `269.71` | `18.4 ms` | `27.9 ms` | `✅ PASS` | — | — | — | `⚠️ SATURATED` |
| **280** | **`279.56`** | **`20.3 ms`** | **`36.8 ms`** | **`✅ PASS`** | — | — | — | `⚠️ SATURATED` |
| **290** | `287.98` | `52.2 ms` | `96.4 ms` | `⚠️ SATURATED` | — | — | — | `⚠️ SATURATED` |
| **300** | `278.11` | `262.5 ms` | `934.2 ms` | `⚠️ SATURATED` | — | — | — | `⚠️ SATURATED` |

---

### 1.3 `2KB Dedicated Saturation` (`2,048 chars ≈ 2,016 tokens`, `< 50 ms` `P99` SLA)

| RPS | TPU v6e Megakernel Achieved | TPU v6e Megakernel `P50` | TPU v6e Megakernel `P99` | TPU v6e Megakernel SLA (`<50ms`) | TPU v5e Achieved (Zhemin) | TPU v5e `P50` (Zhemin) | TPU v5e `P99` (Zhemin) | TPU v5e SLA (Zhemin) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **90** | `89.97` | `12.7 ms` | `17.5 ms` | `✅ PASS` | `90` | `17.2 ms` | `26.5 ms` | `✅ PASS` |
| **95** | `94.98` | `13.7 ms` | `18.6 ms` | `✅ PASS` | `94.75` | `218.6 ms` | `346.5 ms` | `⚠️ SATURATED` |
| **100** | `99.98` | `15.7 ms` | `17.4 ms` | `✅ PASS` | `96.32` | `1031 ms` | `2185 ms` | `⚠️ SATURATED` |
| **110** | `109.96` | `15.3 ms` | `16.9 ms` | `✅ PASS` | `96.74` | `3208 ms` | `5170 ms` | `⚠️ SATURATED` |
| **120** | `119.95` | `14.4 ms` | `15.9 ms` | `✅ PASS` | — | — | — | `⚠️ SATURATED` |
| **130** | `129.93` | `13.8 ms` | `15.8 ms` | `✅ PASS` | — | — | — | `⚠️ SATURATED` |
| **140** | `139.93` | `13.3 ms` | `17.6 ms` | `✅ PASS` | — | — | — | `⚠️ SATURATED` |
| **150** | `149.91` | `12.9 ms` | `15.8 ms` | `✅ PASS` | — | — | — | `⚠️ SATURATED` |
| **155** | `154.92` | `12.7 ms` | `17.9 ms` | `✅ PASS` | — | — | — | `⚠️ SATURATED` |
| **160** | `159.91` | `12.7 ms` | `29.7 ms` | `✅ PASS` | — | — | — | `⚠️ SATURATED` |
| **165** | **`164.83`** | **`15.7 ms`** | **`42.9 ms`** | **`✅ PASS`** | — | — | — | `⚠️ SATURATED` |
| **170** | `169.87` | `73.1 ms` | `132.7 ms` | `⚠️ SATURATED` | — | — | — | `⚠️ SATURATED` |

---

## Part 2: Platform Comparison & Revised TCO Analysis Matching `gid=1972899730` (`L4 = $0.70/hr`, `v5e = $1.20/hr`, `v6e = $2.70/hr`)

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

| Payload Size | NVIDIA L4 (`$0.70/hr`) | Cloud TPU v5e (`$1.20/hr`) | Cloud TPU v6e Megakernel (`$2.70/hr`) | TPU v6e `P50` at Max RPS | TPU v6e `P99` at Max RPS | TPU v6e Gain vs L4 | TPU v6e Gain vs TPU v5e |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1KB (1,024 chars ≈ 1,009 tokens)`** | `70 RPS` | `180 RPS` | **`280 RPS`** | `20.3 ms` | `36.8 ms` | **`4.00x (+300.0%)`** | **`1.56x (+55.6%)`** |
| **`2KB (2,048 chars ≈ 2,016 tokens)`** | `40 RPS` | `90 RPS` | **`165 RPS`** | `15.7 ms` | `42.9 ms` | **`4.13x (+312.5%)`** | **`1.83x (+83.3%)`** |

---

### 2.3 Revised Cost Improvement & Fleet TCO Analysis (`OD Prices: L4 = $0.70/hr, TPU v5e = $1.20/hr, TPU v6e = $2.70/hr`)

| Payload Size | Platform | OD Price (`$/hr`) | Max RPS (`<50ms P99`) | Throughput per Dollar (`RPS/$/hr`) | Relative Perf/$ vs L4 | Cost per Request vs L4 (`@40% Util`) | Effective RPS @ `40%` Util | Chips for `1,000 RPS` (`@40%`) | Monthly Cost (`1,000 RPS` @ `40%`) | Fleet TCO Summary vs L4 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`1KB`** | **NVIDIA L4** | `\$0.70` | `70 RPS` | `100.00 RPS/\$` | `1.00x (Baseline)` | `Baseline (\$6.94 / 1M req)` | `28.0 RPS` | `36` | `\$18,396 / mo` (`\$18,250` exact) | `Baseline (36 GPUs)` |
| **`1KB`** | **Cloud TPU v5e (`FP32`)** | `\$1.20` | `180 RPS` | **`150.00 RPS/\$`** | **`1.50x (+50.0%)`** | **`-33.3% (\$4.63 / 1M req)`** | `72.0 RPS` | `14` | **`\$12,264 / mo`** (`\$12,167` exact) | **`-33.3% (\$6,132/mo saved, 14 chips)`** |
| **`1KB`** | **Cloud TPU v6e (`FP32` Megakernel)** | `\$2.70` | **`280 RPS`** | **`103.70 RPS/\$`** | **`1.04x (+3.7% vs L4)`** | **`-3.6% (\$6.70 / 1M req)`** | **`112.0 RPS`** | **`9`** | **`\$17,739 / mo`** (`\$17,600` exact) | **`-3.6% vs L4 (\$657/mo saved, 4.0x fewer nodes: 9 vs 36)`** |
| **`2KB`** | **NVIDIA L4** | `\$0.70` | `40 RPS` | `57.14 RPS/\$` | `1.00x (Baseline)` | `Baseline (\$12.15 / 1M req)` | `16.0 RPS` | `63` | `\$32,193 / mo` (`\$31,938` exact) | `Baseline (63 GPUs)` |
| **`2KB`** | **Cloud TPU v5e (`FP32`)** | `\$1.20` | `90 RPS` | **`75.00 RPS/\$`** | **`1.31x (+31.3%)`** | **`-23.8% (\$9.26 / 1M req)`** | `36.0 RPS` | `28` | **`\$24,528 / mo`** (`\$24,333` exact) | **`-23.8% (\$7,665/mo saved, 28 chips)`** |
| **`2KB`** | **Cloud TPU v6e (`FP32` Megakernel)** | `\$2.70` | **`165 RPS`** | **`61.11 RPS/\$`** | **`1.07x (+7.0% vs L4)`** | **`-6.5% (\$11.36 / 1M req)`** | **`66.0 RPS`** | **`16`** | **`\$31,536 / mo`** (`\$29,864` exact) | **`-6.5% exact (-2.0% fleet, \$657/mo saved, 3.94x fewer nodes: 16 vs 63)`** |

> [!NOTE]
> **Key Takeaways Under `L4 = $0.70/hr`, `TPU v5e = $1.20/hr`, `TPU v6e = $2.70/hr` After Megakernel Optimizations**:
> 1. **Cloud TPU v6e (`\$2.70/hr`) Beats NVIDIA L4 (`\$0.70/hr`) on BOTH Throughput (`4.00x–4.13x`) AND Price-Performance (`+3.7%` to `+7.0%` `RPS/\$`)**:
>    - Although TPU v6e costs `3.86x` more per hour than L4 (`\$2.70` vs `\$0.70`), the optimized 2048-capped `FP32` Megakernel (`commit 8a096d8c`) delivers **`4.00x` the `1KB` throughput (`280 RPS` vs `70 RPS`)** and **`4.13x` the `2KB` throughput (`165 RPS` vs `40 RPS`)** under the `< 50 ms` `P99` SLA.
>    - This yields **`103.70 RPS/\$/hr` (`+3.7%` vs L4)** on `1KB` and **`61.11 RPS/\$/hr` (`+7.0%` vs L4)** on `2KB`, shrinking a `1,000 RPS` (`@40%` utilization) fleet from **`36–63` L4 GPUs down to just `9–16` TPU v6e chips** (`75%` fewer nodes) while cutting `P50`/`P99` latency by **`43%–56%`**.
> 2. **Cloud TPU v5e (`\$1.20/hr`) Remains the Pure TCO Leader (`-23.8%` to `-33.3%` vs L4)**:
>    - Delivers **`150.00 RPS/\$/hr` on `1KB`** (`1.50x` L4) and **`75.00 RPS/\$/hr` on `2KB`** (`1.31x` L4).
>    - Meanwhile, **Cloud TPU v6e (`\$2.70/hr`) delivers `+55.6%` (`1.56x`) higher `1KB` RPS (`280` vs `180 RPS`) and `+83.3%` (`1.83x`) higher `2KB` RPS (`165` vs `90 RPS`) per chip than TPU v5e**, making v6e ideal when maximizing per-node density (`9–16` chips vs `14–28` v5e chips or `36–63` L4 GPUs) or deploying in `v6e`-rich regions (`us-east5`, `us-south1`).
