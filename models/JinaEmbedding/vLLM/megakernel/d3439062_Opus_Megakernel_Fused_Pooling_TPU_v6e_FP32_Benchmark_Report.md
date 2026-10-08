# Empirical Benchmark & Validation Report: Opus Megakernel Commit `d3439062` (`jina-v2-opus-megakernel`) on TPU v6e (FP32)

> [!IMPORTANT]
> **Target Codebase & Commit Validated**
> - **Repository & Branch:** [`pallavim1/tpu-inference` @ `jina-v2-opus-megakernel`](https://github.com/pallavim1/tpu-inference/tree/jina-v2-opus-megakernel)
> - **Commit:** [`d34390621876658cc7d6adc5b5fb5681eb105639`](https://github.com/pallavim1/tpu-inference/commit/d34390621876658cc7d6adc5b5fb5681eb105639) (*"Fuse device mean pooling + L2 norm into forward XLA executable and add TPU encoder input fast path"*)
> - **Hardware:** Single TPU v6e chip (`ct6e-standard-1t`, 1x TPU v6 lite, 32 GB HBM) on GKE pod `jina-opus-v6e-test`; load driven from CPU benchmark pod `panw-v6e-benchmark-runner`.
> - **Model & Precision:** `jinaai/jina-embeddings-v2-small-en`, `--dtype float32 --max-model-len 2048 --max-num-batched-tokens 2048 --no-enable-prefix-caching`
> - **Active Runtime Flags on `d3439062`:**
>   - `USE_JINA_BERT_MEGAKERNEL=1`
>   - `JINA_BERT_MEGAKERNEL_VERSION=v2`
>   - `JINA_BERT_MEGAKERNEL_PRECISION=default`
>   - `TPU_POOLING_FAST_PATH=1`
>   - `JINA_BERT_DEVICE_POOLING=1`
>   - `JINA_BERT_FUSED_POOLING=1` *(default in `d3439062`: fuses masked mean pooling + L2 normalization into the JIT-compiled encoder forward XLA executable)*
>   - `TPU_ENCODER_INPUT_FAST_PATH=1` *(default in `d3439062`: vectorized host-side encoder token/position preparation)*

---

## 1. Correctness & Numerical Validation on TPU v6e (`pytest` Suite on `d3439062`)

All **20 unit and numerical validation tests (`20 passed in 22.90s`)** passed on TPU v6e (`TPU v6 lite`), including:
- `tests/kernels/megakernel/test_jina_bert_megakernel.py` (**10 passed**):
  - Single-layer and 4-layer stacked `v1` & `v2` Pallas megakernel vs. JAX/XLA reference across single-sequence (`1x1024`, `1x2048`) and packed multi-sequence (`2x1024`, variable lengths `[17, 63, 250]`, `[512, 1024, 512]`) batches.
  - Cosine similarity $\ge 0.99999$ (`default` precision) and $\ge 0.9999999$ (`highest` FP32 accumulator precision).
- `tests/models/jax/test_jina_bert.py` (**10 passed**):
  - End-to-end JinaBERT model forward, device mean pooling (`JINA_BERT_DEVICE_POOLING=1`), and **fused device mean pooling + L2 normalization (`JINA_BERT_FUSED_POOLING=1`)** vs. reference PyTorch CPU & XLA execution.

---

## 2. Step & Forward Microbenchmark (`scripts/measure_jina_forward.py --runs 200 --warmup 20`)

Raw file: [`measure_jina_forward_d3439062.json`](https://github.com/pallavim1/ai-infra-projects/blob/main/models/JinaEmbedding/vLLM/megakernel/measure_jina_forward_d3439062.json)

### 2A. Raw Encoder Forward Pass (`forward`, 200 runs, wall-clock dispatch $\rightarrow$ `block_until_ready`)

| Batch Shape | Total Tokens | XLA Baseline (`default`) Median / p90 | Megakernel `v1` (`default`) Median / p90 | **Megakernel `v2` (`default`) Median / p90** | Speedup (`v2` vs XLA) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`1x1024` (1KB single)** | `1,024` | `1.3355 ms` / `1.3480 ms` | `0.4427 ms` / `0.4593 ms` | **`0.4258 ms` / `0.4375 ms`** | **3.14× faster** |
| **`2x1024` (1KB packed pair)** | `2,048` | `5.3492 ms` / `5.3760 ms` | `0.6641 ms` / `0.6772 ms` | **`0.6225 ms` / `0.6382 ms`** | **8.59× faster** |
| **`1x2048` (2KB single)** | `2,048` | `5.3477 ms` / `5.3696 ms` | `0.8507 ms` / `0.8638 ms` | **`0.7520 ms` / `0.7652 ms`** | **7.11× faster** |

### 2B. Full Engine Step (`step` = Forward + Device-to-Host Copy + Pooler + L2 Norm, Megakernel `v2`)

| Batch Shape | `pooling=old` (`torchax` dispatch) Median / p90 | `pooling=fast` (Host PyTorch) Median / p90 | `pooling=device` (`cb460828` 2 XLA dispatches) Median / p90 | **`pooling=fused` (`d3439062` 1 XLA dispatch) Median / p90** | Total Step Speedup (`old` $\rightarrow$ `device`/`fused`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`1x1024` (`1,024` tok)** | `2.9157 ms` / `3.0504 ms` | `1.7212 ms` / `1.7465 ms` | `0.5837 ms` / `0.6035 ms` | **`0.6502 ms` / `0.6733 ms`** | **4.48× – 5.00× faster** |
| **`2x1024` (`2,048` tok)** | `2.9606 ms` / `3.0913 ms` | `1.9540 ms` / `1.9768 ms` | `0.7853 ms` / `0.8093 ms` | **`0.8130 ms` / `0.8370 ms`** | **3.64× – 3.77× faster** |
| **`1x2048` (`2,048` tok)** | `3.1695 ms` / `3.2944 ms` | `2.0951 ms` / `2.1301 ms` | `0.9210 ms` / `0.9402 ms` | **`0.9595 ms` / `0.9926 ms`** | **3.30× – 3.44× faster** |

> [!NOTE]
> While `pooling=device` and `pooling=fused` are within ~`0.03 ms` in isolated single-thread microbenchmarks, under **live `vllm serve` HTTP load** `d3439062` (`pooling=fused` + `TPU_ENCODER_INPUT_FAST_PATH=1`) eliminates the second Python $\rightarrow$ PJRT XLA dispatch and host-side PyTorch L2 normalization per step, increasing live SLA-passing saturation throughput by **+40 RPS on 1KB (`490` $\rightarrow$ `530 RPS`)** and **+30 RPS on 2KB (`320` $\rightarrow$ `350 RPS`)**.

---

## 3. Live `vllm serve` Benchmarks on `d3439062` (Zhemin's Test Suite)

Raw file: [`d3439062_full_eval_results.json`](https://github.com/pallavim1/ai-infra-projects/blob/main/models/JinaEmbedding/vLLM/megakernel/d3439062_full_eval_results.json)

### 3A. Test 1 — Batch Request Testing: Single HTTP Request with $N \in \{1, 4, 8, 16\}$ Prompts (`{"text": [p_1, ..., p_N]}`, Exact `1,024` & `2,048` Tokens)

| Payload Size | Concurrency ($N$) | **TPU v6e `d3439062` (Fused)** Throughput | **TPU v6e `d3439062` (Fused)** `p50` | **TPU v6e `d3439062` (Fused)** `p99` | TPU v6e `cb460828` Throughput / `p50` / `p99` | TPU v5e (Zhemin) Throughput / `p50` / `p99` | L4 GPU (Zhemin) Throughput / `p50` / `p99` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1KB (`1,024` tok)** | **1** | **139.4/s** | **7.1ms** | **8.1ms** | 130.3/s / 7.4ms / 10.4ms | 85.8/s / 11.5ms / 12.7ms | 43.2/s / 20.7ms / 23.5ms |
| **1KB (`1,024` tok)** | **4** | **235.1/s** | **16.9ms** | **18.1ms** | 227.0/s / 17.4ms / 19.4ms | 187.4/s / 21.2ms / 22.8ms | 103.4/s / 38.1ms / 42.9ms |
| **1KB (`1,024` tok)** | **8** | **259.5/s** | **30.6ms** | **32.3ms** | 254.3/s / 31.2ms / 32.7ms | 187.6/s / 42.5ms / 44.5ms | 135.1/s / 58.7ms / 66.3ms |
| **1KB (`1,024` tok)** | **16** | **272.9/s** | **58.5ms** | **59.6ms** | 267.9/s / 59.5ms / 60.8ms | 186.8/s / 85.4ms / 89.9ms | 165.2/s / 96.5ms / 107.6ms |
| **2KB (`2,048` tok)** | **1** | **115.5/s** | **8.5ms** | **9.5ms** | 108.1/s / 9.0ms / 10.6ms | 59.5/s / 16.5ms / 17.7ms | 39.0/s / 24.3ms / 28.2ms |
| **2KB (`2,048` tok)** | **4** | **153.2/s** | **25.9ms** | **27.1ms** | 148.3/s / 26.6ms / 29.3ms | 97.7/s / 40.7ms / 42.8ms | 75.7/s / 52.1ms / 58.2ms |
| **2KB (`2,048` tok)** | **8** | **165.3/s** | **48.1ms** | **50.5ms** | 158.3/s / 50.1ms / 52.4ms | 96.6/s / 82.6ms / 86.0ms | 90.5/s / 87.4ms / 97.3ms |
| **2KB (`2,048` tok)** | **16** | **172.4/s** | **92.1ms** | **98.0ms** | 163.0/s / 97.3ms / 104.0ms | 96.5/s / 165.9ms / 171.2ms | 101.1/s / 156.9ms / 175.5ms |

---

### 3B. Test 2 — Concurrent HTTP Request Testing (`k6 constant-vus`, $\text{VUS} \in \{1, 4, 8, 16\}$, Exact `1,024` & `2,048` Tokens, 30s per step)

| Payload Size | Concurrency (`VUS`) | **TPU v6e `d3439062` (Fused)** Throughput | **TPU v6e `d3439062` (Fused)** `p50` | **TPU v6e `d3439062` (Fused)** `p99` | TPU v6e `cb460828` Throughput / `p50` / `p99` | TPU v5e (Zhemin) Throughput / `p50` / `p99` | L4 GPU (Zhemin) Throughput / `p50` / `p99` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1KB (`1,024` tok)** | **1** | **134.1/s** | **6.6ms** | **7.1ms** | 128.4/s / 6.9ms / 8.0ms | 85.8/s / 11.5ms / 12.7ms | 43.2/s / 20.7ms / 23.5ms |
| **1KB (`1,024` tok)** | **4** | **394.5/s** | **9.3ms** | **12.0ms** | 376.4/s / 9.9ms / 12.3ms | 187.4/s / 21.2ms / 22.8ms | 103.4/s / 38.1ms / 42.9ms |
| **1KB (`1,024` tok)** | **8** | **519.2/s** | **14.3ms** | **19.7ms** | 507.1/s / 14.8ms / 20.2ms | 187.6/s / 42.5ms / 44.5ms | 135.1/s / 58.7ms / 66.3ms |
| **1KB (`1,024` tok)** | **16** | **546.4/s** | **28.4ms** | **37.4ms** | 518.0/s / 30.1ms / 39.3ms | 186.8/s / 85.4ms / 89.9ms | 165.2/s / 96.5ms / 107.6ms |
| **2KB (`2,048` tok)** | **1** | **105.8/s** | **8.0ms** | **9.0ms** | 99.0/s / 8.6ms / 9.5ms | 59.5/s / 16.5ms / 17.7ms | 39.0/s / 24.3ms / 28.2ms |
| **2KB (`2,048` tok)** | **4** | **294.8/s** | **12.3ms** | **15.6ms** | 290.1/s / 12.4ms / 16.2ms | 97.7/s / 40.7ms / 42.8ms | 75.7/s / 52.1ms / 58.2ms |
| **2KB (`2,048` tok)** | **8** | **374.0/s** | **19.7ms** | **26.0ms** | 367.6/s / 20.1ms / 26.8ms | 96.6/s / 82.6ms / 86.0ms | 90.5/s / 87.4ms / 97.3ms |
| **2KB (`2,048` tok)** | **16** | **302.3/s** | **51.2ms** | **58.2ms** | 366.9/s / 42.0ms / 50.0ms | 96.5/s / 165.9ms / 171.2ms | 101.1/s / 156.9ms / 175.5ms |

---

### 3C. Test 3 — Dedicated Open-Loop Saturation (`k6 constant-arrival-rate`, 60s per step, $\text{p99} < 50\text{ ms}$ SLA)

#### 1KB (`1,024` Exact Tokens) Saturation Curve on `d3439062`
| Target RPS | Requests Completed (60s) | Achieved RPS | `min` | `p50` | `avg` | `p90` | `p95` | **`p99`** | `max` | Server Tokens/Req | SLA Status ($\text{p99} < 50\text{ ms}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **480** | 28,801 | **480.30** | 7.4 ms | **11.2 ms** | 11.6 ms | 13.6 ms | 15.1 ms | **21.5 ms** | 52.5 ms | `1024.0` | **PASS** |
| **500** | 30,001 | **500.66** | 7.2 ms | **11.6 ms** | 12.4 ms | 14.8 ms | 16.3 ms | **24.2 ms** | 83.8 ms | `1024.0` | **PASS** |
| **510** | 30,600 | **510.37** | 7.7 ms | **12.2 ms** | 12.8 ms | 15.5 ms | 17.1 ms | **23.4 ms** | 69.7 ms | `1024.0` | **PASS** |
| **520** | 31,201 | **520.39** | 7.4 ms | **12.8 ms** | 13.6 ms | 16.4 ms | 18.5 ms | **36.1 ms** | 57.6 ms | `1024.0` | **PASS (20-RPS Grid Max)** |
| **530** | 31,801 | **530.35** | 7.8 ms | **13.1 ms** | 15.0 ms | 19.7 ms | 29.0 ms | **47.9 ms** | 71.6 ms | `1024.0` | **PASS (`d3439062` Max SLA RPS)** |
| **540** | 32,401 | 540.57 | 8.3 ms | 15.7 ms | 21.7 ms | 36.3 ms | 56.7 ms | 121.1 ms | 154.9 ms | `1024.0` | SATURATED |
| **550** | 33,001 | 546.40 | 9.1 ms | 52.0 ms | 136.8 ms | 352.5 ms | 506.6 ms | 564.0 ms | 656.7 ms | `1024.0` | SATURATED |

#### 2KB (`2,048` Exact Tokens) Saturation Curve on `d3439062`
| Target RPS | Requests Completed (60s) | Achieved RPS | `min` | `p50` | `avg` | `p90` | `p95` | **`p99`** | `max` | Server Tokens/Req | SLA Status ($\text{p99} < 50\text{ ms}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **280** | 16,800 | **280.16** | 7.7 ms | **10.7 ms** | 11.3 ms | 13.6 ms | 14.0 ms | **16.2 ms** | 31.4 ms | `2048.0` | **PASS** |
| **300** | 18,001 | **300.14** | 7.9 ms | **12.4 ms** | 12.0 ms | 13.4 ms | 13.8 ms | **17.0 ms** | 30.9 ms | `2048.0` | **PASS** |
| **320** | 19,200 | **320.14** | 8.1 ms | **12.2 ms** | 12.4 ms | 13.2 ms | 14.0 ms | **18.6 ms** | 46.1 ms | `2048.0` | **PASS** |
| **340** | 20,400 | **340.18** | 7.8 ms | **12.0 ms** | 12.8 ms | 15.3 ms | 17.7 ms | **27.0 ms** | 50.3 ms | `2048.0` | **PASS (20-RPS Grid Max)** |
| **350** | 21,001 | **350.20** | 8.4 ms | **11.7 ms** | 12.5 ms | 15.4 ms | 17.4 ms | **25.2 ms** | 51.4 ms | `2048.0` | **PASS (`d3439062` Max SLA RPS)** |
| **360** | 21,601 | 359.88 | 7.9 ms | 12.6 ms | 17.0 ms | 23.3 ms | 31.3 ms | 89.3 ms | 96.8 ms | `2048.0` | SATURATED |
| **370** | 22,200 | 370.17 | 8.5 ms | 87.4 ms | 121.6 ms | 266.0 ms | 283.3 ms | 325.1 ms | 342.9 ms | `2048.0` | SATURATED |
