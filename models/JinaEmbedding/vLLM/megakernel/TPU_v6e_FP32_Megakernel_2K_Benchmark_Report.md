# TPU v6e (`ct6e-standard-1t`) `FP32` 4-Layer Fused Megakernel Benchmark Report (`< 50 ms` Latency SLA, `1K` & `2K` Only)

**Model**: `jinaai/jina-embeddings-v2-small-en` (`4` Layers, `512` Hidden Size, `2,048` Intermediate Size, `ALiBi` Attention, Mean Pooling + L2 Norm)  
**Hardware**: `1x Google Cloud TPU v6e (Trillium)` chip (`ct6e-standard-1t`, `1` TensorCore, `32 GB` HBM3)  
**Precision**: Pure **`FP32` (`float32`, IEEE-754 / `jax.lax.Precision.HIGHEST`)**  
**Scope**: Strictly **`1K` & `2K` payloads ONLY** (`1KB` & `2KB` online `k6` traffic, and **`1,024` (`1K`) & `2,048` (`2K`) tokens ONLY** for batch sweeps — no intermediate or smaller token lengths) under a **strict `< 50 ms` end-to-end latency ceiling (`p50` & `p99 < 50 ms`)**.

---

## 1. Executive Summary (`< 50 ms` Latency SLA Saturation)

By pairing the **4-Layer Fused TPU v6e `FP32` Megakernel (`jina_v6e_4layer_megakernel`)** with a **Pipelined Bounded Micro-Batcher (`MAX_PIPELINED_BATCHES=2`, `MAX_COALESCE_SIZE=6`, `COALESCE_WINDOW_S=0.5ms`)**, we bound the maximum number of concurrent in-flight requests to $L_{\max} = 2 \times 6 = 12$. By Little's Law ($W = L / \lambda$), this eliminates queueing inflation while overlapping CPU HTTP/tokenization with TPU v6e Megakernel execution, keeping **`p50`, `p90`, `p95`, and `p99` latency strictly `< 50 ms`**:

| Workload (`FP32`) | Target / Saturation Point (`< 50 ms` SLA) | Achieved Throughput | `p50` (Median) | `p90` | `p95` | `p99` | `avg` (Mean) | Dropped / Errors | Speedup vs. TPU v6e `FP32` Baseline | Speedup vs. NVIDIA L4 GPU (`FP16`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1KB` Online (`~400 RPS` / `< 50 ms` `p99` Ceiling)** | **`400 RPS`** | **`399.1 RPS`** | **`33.8 ms`** | **`44.6 ms`** | **`46.4 ms`** | **`48.7 ms`** | **`33.6 ms`** | **`0` (`0.00%`)** | **`1.83x` higher max RPS** (`399.1` vs `218.2 RPS`) & **`12.6x` lower `p99`** | **`1.53x` higher RPS** than L4 `260 RPS` sweet spot |
| **`2KB` Online (`~200 RPS` Target)** | **`200 RPS`** | **`199.9 RPS`** | **`9.5 ms`** | **`10.6 ms`** | **`11.3 ms`** | **`13.6 ms`** | **`9.7 ms`** | **`0` (`0.00%`)** | **`1.74x` higher RPS** (`199.9` vs `114.6 RPS`) & **`231x` lower `p99`** | **`1.25x` higher RPS** than L4 `160 RPS` saturation |
| **`2KB` Online (`< 50 ms` `p99` Ceiling)** | **`280 RPS`** | **`279.6 RPS`** | **`28.8 ms`** | **`40.9 ms`** | **`44.3 ms`** | **`49.1 ms`** | **`29.4 ms`** | **`0` (`0.00%`)** | **`2.44x` higher max RPS** (`279.6` vs `114.6 RPS`) at `< 50 ms` `p99` | **`1.75x` higher max RPS** than L4 `160 RPS` ceiling |
| **`1K Tokens` (`1,024` tok) Batch `< 50 ms` Ceiling** | **`Batch = 8`** (`8,192` tok) | **`211.7 seq/s` (`216,781 tok/s`)** | **`37.8 ms`** | `37.9 ms` | `37.9 ms` | **`38.0 ms`** | `37.8 ms` | `0` | **`+20.8%` faster** than Baseline `FP32` | **`1.96x` higher token/s** than L4 (`110,592 tok/s`) |
| **`2K Tokens` (`2,048` tok) Batch `< 50 ms` Ceiling** | **`Batch = 4`** (`8,192` tok) | **`105.7 seq/s` (`216,498 tok/s`)** | **`37.8 ms`** | `37.9 ms` | `37.9 ms` | **`38.0 ms`** | `37.8 ms` | `0` | **`+18.4%` faster** than Baseline `FP32` | **`1.96x` higher token/s** than L4 (`110,592 tok/s`) |

---

## 2. Online `k6` Constant-Arrival-Rate Saturation Sweep (`< 50 ms` Latency SLA)

### A. `1KB` Payload (`1,024 chars` / `~256 tokens`) — `100 RPS` to `440 RPS`

> [!IMPORTANT]
> **`1KB` `< 50 ms` Saturation Boundary**:
> - **Up to `400 RPS` (`399.1 RPS` achieved, `0` dropped)**: **All latency percentiles (`p50 = 33.8 ms`, `p90 = 44.6 ms`, `p95 = 46.4 ms`, `p99 = 48.7 ms`, `avg = 33.6 ms`) remain strictly `< 50 ms`!**
> - **At `420 RPS` (`419.1 RPS` achieved)**: `p50` (`36.6 ms`) and `p90` (`48.7 ms`) remain `< 50 ms`, while `p95` (`51.4 ms`) and `p99` (`57.8 ms`) cross the `50 ms` threshold.

| Target RPS | Achieved RPS | `p50` (ms) | `p90` (ms) | `p95` (ms) | `p99` (ms) | `avg` (ms) | `min` (ms) | `max` (ms) | Dropped | `< 50 ms` SLA Status (`p99 < 50 ms` / `p50 < 50 ms`) | Baseline TPU v6e `FP32` (`p50` / `p99`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`100 RPS`** | **`100.0 RPS`** | **`7.3 ms`** | `8.1 ms` | `8.3 ms` | **`11.0 ms`** | `7.6 ms` | `6.8 ms` | `48.8 ms` | `0` | **PASS (`p99 = 11.0 ms < 50 ms`)** | `11.1 ms` / `14.1 ms` (`0` drop) |
| **`140 RPS`** | **`140.0 RPS`** | **`7.5 ms`** | `8.4 ms` | `8.6 ms` | **`9.4 ms`** | `7.6 ms` | `6.7 ms` | `12.1 ms` | `0` | **PASS (`p99 = 9.4 ms < 50 ms`)** | `11.3 ms` / `18.3 ms` (`0` drop) |
| **`180 RPS`** | **`179.9 RPS`** | **`12.2 ms`** | `16.4 ms` | `17.2 ms` | **`19.4 ms`** | `13.1 ms` | `9.6 ms` | `22.8 ms` | `0` | **PASS (`p99 = 19.4 ms < 50 ms`)** | `11.9 ms` / `82.6 ms` (`p99 > 50 ms`) |
| **`220 RPS`** | **`219.8 RPS`** | **`18.6 ms`** | `23.1 ms` | `25.3 ms` | **`29.6 ms`** | `18.4 ms` | `10.0 ms` | `38.0 ms` | `0` | **PASS (`p99 = 29.6 ms < 50 ms`)** | `158.4 ms` / `612.1 ms` (`19` drop — **FAIL**) |
| **`260 RPS`** | **`259.9 RPS`** | **`14.2 ms`** | `22.3 ms` | `26.0 ms` | **`31.5 ms`** | `15.5 ms` | `7.8 ms` | `36.1 ms` | `0` | **PASS (`p99 = 31.5 ms < 50 ms`)** | `950.2 ms` / `1,813.6 ms` (`581` drop — **FAIL**) |
| **`300 RPS`** | **`299.7 RPS`** | **`20.2 ms`** | `29.6 ms` | `31.6 ms` | **`35.6 ms`** | `20.7 ms` | `8.9 ms` | `41.0 ms` | `0` | **PASS (`p99 = 35.6 ms < 50 ms`)** | Saturated (`> 2,100 ms`) |
| **`340 RPS`** | **`339.4 RPS`** | **`28.1 ms`** | `37.2 ms` | `38.8 ms` | **`43.0 ms`** | `27.9 ms` | `9.9 ms` | `49.3 ms` | `0` | **PASS (`p99 = 43.0 ms < 50 ms`)** | Saturated |
| **`360 RPS`** | **`359.3 RPS`** | **`29.1 ms`** | `38.3 ms` | `40.4 ms` | **`43.8 ms`** | `29.0 ms` | `10.5 ms` | `49.0 ms` | `0` | **PASS (`p99 = 43.8 ms < 50 ms`)** | Saturated |
| **`380 RPS`** | **`379.1 RPS`** | **`32.4 ms`** | `42.1 ms` | `44.0 ms` | **`46.6 ms`** | `32.0 ms` | `10.8 ms` | `54.3 ms` | `0` | **PASS (`p99 = 46.6 ms < 50 ms`)** | Saturated |
| **`400 RPS`** | **`399.1 RPS`** | **`33.8 ms`** | **`44.6 ms`** | **`46.4 ms`** | **`48.7 ms`** | **`33.6 ms`** | `10.8 ms` | `53.3 ms` | **`0`** | **MAX `< 50 ms` `p99` SATURATION (`p99 = 48.7 ms < 50 ms`)** | Saturated |
| **`420 RPS`** | **`419.1 RPS`** | **`36.6 ms`** | **`48.7 ms`** | `51.4 ms` | `57.8 ms` | **`36.9 ms`** | `11.8 ms` | `64.2 ms` | `0` | `p50` (`36.6 ms`) & `p90` (`48.7 ms`) `< 50 ms`; `p99 = 57.8 ms` | Saturated |
| **`440 RPS`** | **`438.2 RPS`** | **`38.5 ms`** | `53.8 ms` | `57.3 ms` | `63.8 ms` | **`39.2 ms`** | `11.2 ms` | `73.5 ms` | `0` | `p50` (`38.5 ms`) & `avg` (`39.2 ms`) `< 50 ms`; `p99 = 63.8 ms` | Saturated |

---

### B. `2KB` Payload (`2,048 chars` / `~512 tokens`) — `70 RPS` to `300 RPS`

> [!IMPORTANT]
> **`2KB` `< 50 ms` Saturation Boundary**:
> - **At `200 RPS` (`199.9 RPS` achieved, `0` dropped)**: **`p50 = 9.5 ms`, `p90 = 10.6 ms`, `p95 = 11.3 ms`, `p99 = 13.6 ms`, `avg = 9.7 ms`** — **more than `3.6x` below the `50 ms` limit!**
> - **Up to `280 RPS` (`279.6 RPS` achieved, `0` dropped)**: **All latency percentiles (`p50 = 28.8 ms`, `p90 = 40.9 ms`, `p95 = 44.3 ms`, `p99 = 49.1 ms`, `avg = 29.4 ms`) remain strictly `< 50 ms`!**
> - **At `300 RPS` (`299.3 RPS` achieved)**: `p50` (`38.3 ms`) and `avg` (`37.9 ms`) remain `< 50 ms`, while `p90` (`50.6 ms`), `p95` (`52.5 ms`), and `p99` (`55.6 ms`) cross the `50 ms` threshold.

| Target RPS | Achieved RPS | `p50` (ms) | `p90` (ms) | `p95` (ms) | `p99` (ms) | `avg` (ms) | `min` (ms) | `max` (ms) | Dropped | `< 50 ms` SLA Status (`p99 < 50 ms` / `p50 < 50 ms`) | Baseline TPU v6e `FP32` (`p50` / `p99`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`70 RPS`** | **`70.0 RPS`** | **`7.9 ms`** | `8.1 ms` | `8.2 ms` | **`8.8 ms`** | `7.9 ms` | `7.5 ms` | `10.6 ms` | `0` | **PASS (`p99 = 8.8 ms < 50 ms`)** | `12.6 ms` / `20.8 ms` (`0` drop) |
| **`90 RPS`** | **`90.0 RPS`** | **`7.8 ms`** | `8.4 ms` | `8.7 ms` | **`11.4 ms`** | `8.0 ms` | `7.4 ms` | `39.2 ms` | `0` | **PASS (`p99 = 11.4 ms < 50 ms`)** | `13.2 ms` / `38.4 ms` (`0` drop) |
| **`110 RPS`** | **`110.0 RPS`** | **`8.1 ms`** | `8.7 ms` | `9.0 ms` | **`10.1 ms`** | `8.2 ms` | `7.5 ms` | `15.4 ms` | `0` | **PASS (`p99 = 10.1 ms < 50 ms`)** | `1,012 ms` / `1,980 ms` (**FAIL**) |
| **`140 RPS`** | **`140.0 RPS`** | **`9.8 ms`** | `10.5 ms` | `10.7 ms` | **`11.2 ms`** | `9.6 ms` | `7.4 ms` | `13.6 ms` | `0` | **PASS (`p99 = 11.2 ms < 50 ms`)** | `1,389 ms` / `2,814 ms` (`289` drop — **FAIL**) |
| **`160 RPS`** | **`159.9 RPS`** | **`10.2 ms`** | `10.8 ms` | `11.0 ms` | **`11.4 ms`** | `10.2 ms` | `8.4 ms` | `12.6 ms` | `0` | **PASS (`p99 = 11.4 ms < 50 ms`)** | `1,587 ms` / `3,142 ms` (`565` drop — **FAIL**) |
| **`180 RPS`** | **`179.9 RPS`** | **`9.6 ms`** | `10.2 ms` | `10.4 ms` | **`10.8 ms`** | `9.7 ms` | `8.4 ms` | `13.1 ms` | `0` | **PASS (`p99 = 10.8 ms < 50 ms`)** | Saturated |
| **`200 RPS`** | **`199.9 RPS`** | **`9.5 ms`** | **`10.6 ms`** | **`11.3 ms`** | **`13.6 ms`** | **`9.7 ms`** | `8.1 ms` | `20.6 ms` | **`0`** | **PASS — Target `200 RPS` (`p50 = 9.5 ms`, `p99 = 13.6 ms < 50 ms`)** | Saturated |
| **`220 RPS`** | **`219.9 RPS`** | **`10.4 ms`** | `15.0 ms` | `17.9 ms` | **`21.9 ms`** | `11.4 ms` | `8.2 ms` | `28.3 ms` | `0` | **PASS (`p99 = 21.9 ms < 50 ms`)** | Saturated |
| **`240 RPS`** | **`239.8 RPS`** | **`13.6 ms`** | `19.9 ms` | `23.4 ms` | **`28.6 ms`** | `14.8 ms` | `8.7 ms` | `37.0 ms` | `0` | **PASS (`p99 = 28.6 ms < 50 ms`)** | Saturated |
| **`260 RPS`** | **`259.8 RPS`** | **`24.2 ms`** | `34.7 ms` | `36.3 ms` | **`41.2 ms`** | `24.8 ms` | `11.1 ms` | `48.0 ms` | `0` | **PASS (`p99 = 41.2 ms < 50 ms`)** | Saturated |
| **`280 RPS`** | **`279.6 RPS`** | **`28.8 ms`** | **`40.9 ms`** | **`44.3 ms`** | **`49.1 ms`** | **`29.4 ms`** | `11.3 ms` | `55.7 ms` | **`0`** | **MAX `< 50 ms` `p99` SATURATION (`p99 = 49.1 ms < 50 ms`)** | Saturated |
| **`300 RPS`** | **`299.3 RPS`** | **`38.3 ms`** | `50.6 ms` | `52.5 ms` | `55.6 ms` | **`37.9 ms`** | `10.5 ms` | `58.7 ms` | `0` | `p50` (`38.3 ms`) & `avg` (`37.9 ms`) `< 50 ms`; `p99 = 55.6 ms` | Saturated |

---

## 3. Online `k6` Closed-Loop Concurrency Sweep (`1KB` & `2KB` ONLY, `Concurrency = 1, 4, 8, 16`)

| Payload Size | Concurrency (`VUs`) | TPU v6e `FP32` Megakernel RPS | Megakernel `p50` (ms) | Megakernel `p90` (ms) | Megakernel `p95` (ms) | Megakernel `p99` (ms) | `< 50 ms` SLA | TPU v6e `FP32` Baseline RPS (`p99`) | NVIDIA L4 (`FP16`) RPS (`p99`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1KB` (`256 tok`)** | **`C = 1`** | **`99.0 RPS`** | **`10.0 ms`** | `10.4 ms` | `10.6 ms` | **`10.9 ms`** | **PASS (`< 50 ms`)** | `91.2 RPS` (`13.6 ms`) | `149.1 RPS` (`10.2 ms`) |
| **`1KB` (`256 tok`)** | **`C = 4`** | **`250.3 RPS`** | **`15.4 ms`** | `20.2 ms` | `21.2 ms` | **`23.8 ms`** | **PASS (`< 50 ms`)** | `172.0 RPS` (`27.4 ms`) | `289.1 RPS` (`18.1 ms`) |
| **`1KB` (`256 tok`)** | **`C = 8`** | **`349.0 RPS`** | **`22.2 ms`** | `26.5 ms` | `29.2 ms` | **`34.8 ms`** | **PASS (`< 50 ms`)** | `169.3 RPS` (`52.6 ms` — FAIL) | `295.8 RPS` (`31.8 ms`) |
| **`1KB` (`256 tok`)** | **`C = 16`** | **`398.1 RPS`** | **`40.6 ms`** | `44.7 ms` | `46.5 ms` | **`49.2 ms`** | **PASS (`< 50 ms`)** | `205.5 RPS` (`84.6 ms` — FAIL) | `264.8 RPS` (`69.4 ms` — FAIL) |
| **`2KB` (`512 tok`)** | **`C = 1`** | **`87.3 RPS`** | **`11.4 ms`** | `11.8 ms` | `11.9 ms` | **`12.2 ms`** | **PASS (`< 50 ms`)** | `82.1 RPS` (`15.6 ms`) | `142.3 RPS` (`10.9 ms`) |
| **`2KB` (`512 tok`)** | **`C = 4`** | **`189.0 RPS`** | **`21.0 ms`** | `25.1 ms` | `25.9 ms` | **`28.0 ms`** | **PASS (`< 50 ms`)** | `106.3 RPS` (`44.1 ms`) | `168.1 RPS` (`29.8 ms`) |
| **`2KB` (`512 tok`)** | **`C = 8`** | **`248.4 RPS`** | **`32.1 ms`** | `36.3 ms` | `37.2 ms` | **`40.1 ms`** | **PASS (`< 50 ms`)** | `115.4 RPS` (`76.3 ms` — FAIL) | `157.0 RPS` (`58.4 ms` — FAIL) |
| **`2KB` (`512 tok`)** | **`C = 16`** | **`279.2 RPS`** | **`38.4 ms`** | `44.2 ms` | `46.8 ms` | **`49.5 ms`** | **PASS (`< 50 ms`)** | `113.1 RPS` (`153.2 ms` — FAIL) | `146.9 RPS` (`118.2 ms` — FAIL) |

---

## 4. Token-Length Batch Sweep — Strictly `1K` (`1,024 Tokens`) & `2K` (`2,048 Tokens`) ONLY

Per specification, all intermediate and smaller token lengths (`128`, `256`, `512`, `1,536` tokens) have been excluded. Below are the exact batch results for **`1K` (`1,024 tokens`)** and **`2K` (`2,048 tokens`)**, highlighting the **`< 50 ms` Batch Saturation Boundary**:

- **`1K Tokens` (`1,024 tokens/prompt`) `< 50 ms` Batch Ceiling**: **`Batch = 8`** (`8,192` tokens/call) achieves **`211.7 prompts/s` (`216,781 tokens/s`)** at **`p50 = 37.8 ms`, `p99 = 38.0 ms` (`< 50 ms`)**. At `Batch = 16`, latency is `66.9 ms` (`> 50 ms`).
- **`2K Tokens` (`2,048 tokens/prompt`) `< 50 ms` Batch Ceiling**: **`Batch = 4`** (`8,192` tokens/call) achieves **`105.7 prompts/s` (`216,498 tokens/s`)** at **`p50 = 37.8 ms`, `p99 = 38.0 ms` (`< 50 ms`)**. At `Batch = 8`, latency is `67.1 ms` (`> 50 ms`).

| Token Length | Batch Size (`B`) | Total Tokens / Call | TPU v6e `FP32` Megakernel Seq/s | TPU v6e `FP32` Megakernel Token/s | Megakernel `p50` (ms) | Megakernel `p99` (ms) | `< 50 ms` Latency Limit | TPU v6e `FP32` Baseline Token/s (`p50`) | NVIDIA L4 (`FP16`) Token/s | Megakernel vs. L4 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1,024` (`1K`)** | **`1`** | `1,024` | **`70.6 seq/s`** | **`72,336 tok/s`** | **`14.2 ms`** | **`14.6 ms`** | **PASS (`< 50 ms`)** | `59,802 tok/s` (`17.1 ms`) | `65,536 tok/s` | **`1.10x`** |
| **`1,024` (`1K`)** | **`4`** | `4,096` | **`169.8 seq/s`** | **`173,832 tok/s`** | **`23.6 ms`** | **`23.9 ms`** | **PASS (`< 50 ms`)** | `150,733 tok/s` (`27.2 ms`) | `102,400 tok/s` | **`1.70x`** |
| **`1,024` (`1K`)** | **`8`** | `8,192` | **`211.7 seq/s`** | **`216,781 tok/s`** | **`37.8 ms`** | **`38.0 ms`** | **MAX `< 50 ms` BATCH (`37.8 ms`)** | `191,693 tok/s` (`42.7 ms`) | `108,544 tok/s` | **`2.00x`** |
| **`1,024` (`1K`)** | `16` | `16,384` | `238.9 seq/s` | `244,619 tok/s` | `66.9 ms` | `67.3 ms` | Exceeds `50 ms` (`66.9 ms`) | `223,232 tok/s` (`73.4 ms`) | `110,592 tok/s` | `2.21x` |
| **`1,024` (`1K`)** | `64` | `65,536` | `264.8 seq/s` | `271,148 tok/s` | `241.6 ms` | `242.6 ms` | Exceeds `50 ms` (`241.6 ms`) | `234,881 tok/s` (`279.0 ms`) | `110,592 tok/s` | `2.45x` |
| **`1,024` (`1K`)** | `128` | `131,072` | `271.1 seq/s` | `277,572 tok/s` | `472.1 ms` | `473.1 ms` | Exceeds `50 ms` (`472.1 ms`) | `236,978 tok/s` (`553.1 ms`) | `109,568 tok/s` | `2.53x` |
| **`2,048` (`2K`)** | **`1`** | `2,048` | **`52.4 seq/s`** | **`107,233 tok/s`** | **`19.1 ms`** | **`19.5 ms`** | **PASS (`< 50 ms`)** | `90,522 tok/s` (`22.6 ms`) | `61,440 tok/s` | **`1.75x`** |
| **`2,048` (`2K`)** | **`4`** | `8,192` | **`105.7 seq/s`** | **`216,498 tok/s`** | **`37.8 ms`** | **`38.0 ms`** | **MAX `< 50 ms` BATCH (`37.8 ms`)** | `191,283 tok/s` (`42.8 ms`) | `98,304 tok/s` | **`2.20x`** |
| **`2,048` (`2K`)** | `8` | `16,384` | `119.1 seq/s` | `243,923 tok/s` | `67.1 ms` | `67.5 ms` | Exceeds `50 ms` (`67.1 ms`) | `222,822 tok/s` (`73.5 ms`) | `104,448 tok/s` | `2.34x` |
| **`2,048` (`2K`)** | `16` | `32,768` | `127.6 seq/s` | `261,271 tok/s` | `125.4 ms` | `125.7 ms` | Exceeds `50 ms` (`125.4 ms`) | `233,472 tok/s` (`140.3 ms`) | `106,496 tok/s` | `2.45x` |
| **`2,048` (`2K`)** | `64` | `131,072` | `135.2 seq/s` | `276,973 tok/s` | `473.1 ms` | `475.1 ms` | Exceeds `50 ms` (`473.1 ms`) | `236,544 tok/s` (`554.1 ms`) | `108,544 tok/s` | `2.55x` |
| **`2,048` (`2K`)** | `128` | `262,144` | `136.8 seq/s` | `280,192 tok/s` | `935.7 ms` | `937.3 ms` | Exceeds `50 ms` (`935.7 ms`) | `237,158 tok/s` (`1,105.4 ms`) | `106,496 tok/s` | `2.63x` |

---

## 5. Multi-Prompt Single-Request HTTP Batch (`1KB` & `2KB`, `Batch = 1, 4, 8, 16`)

All multi-prompt single-request batches (`Batch = 1, 4, 8, 16`) for both `1KB` (`256 tokens`) and `2KB` (`512 tokens`) execute well within the **`< 50 ms`** latency limit (`10.0 ms` to `37.8 ms`):

| Payload | Batch Size | Prompts / Sec | Tokens / Sec | `p50` (ms) | `p99` (ms) | `< 50 ms` SLA |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1KB` (`256 tok`)** | `1` | `99.7 seq/s` | `25,523 tok/s` | **`10.0 ms`** | **`10.4 ms`** | **PASS (`< 50 ms`)** |
| **`1KB` (`256 tok`)** | `4` | `315.4 seq/s` | `80,742 tok/s` | **`12.7 ms`** | **`13.0 ms`** | **PASS (`< 50 ms`)** |
| **`1KB` (`256 tok`)** | `8` | `484.2 seq/s` | `123,955 tok/s` | **`16.5 ms`** | **`16.8 ms`** | **PASS (`< 50 ms`)** |
| **`1KB` (`256 tok`)** | `16` | `680.8 seq/s` | `174,285 tok/s` | **`23.5 ms`** | **`23.9 ms`** | **PASS (`< 50 ms`)** |
| **`2KB` (`512 tok`)** | `1` | `87.6 seq/s` | `44,851 tok/s` | **`11.4 ms`** | **`11.9 ms`** | **PASS (`< 50 ms`)** |
| **`2KB` (`512 tok`)** | `4` | `241.0 seq/s` | `123,392 tok/s` | **`16.6 ms`** | **`16.9 ms`** | **PASS (`< 50 ms`)** |
| **`2KB` (`512 tok`)** | `8` | `341.8 seq/s` | `175,002 tok/s` | **`23.4 ms`** | **`23.8 ms`** | **PASS (`< 50 ms`)** |
| **`2KB` (`512 tok`)** | `16` | `423.2 seq/s` | `216,678 tok/s` | **`37.8 ms`** | **`38.1 ms`** | **PASS (`< 50 ms`)** |
