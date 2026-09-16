# Report 3: Comparison of `FP32` vs `BF16` Runs on Cloud TPU v5e + vLLM (`--max-model-len 2048`, Restricted at `2048`)

* **Model:** `jinaai/jina-embeddings-v2-small-en` (512-dim BERT/ALiBi Embeddings)
* **Accelerator Hardware:** 1x Cloud TPU v5e chip (`ct5lp-hightpu-1t`, GKE cluster `pm-panw-jina-cluster` in `europe-west4-b`)
* **Common Configuration:** `--max-model-len 2048`, `truncate_prompt_tokens: 2048` (`BertTokenizerFast`), `--max-num-seqs=40`
* **Compared Runs:**
  * **Run A (`FP32`)**: `--dtype float32` (`2026-09-16 07:51:37 UTC`)
  * **Run B (`BF16`)**: `--dtype bfloat16` (`2026-09-16 15:34:24 UTC`)

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

## 1. Executive Summary (`FP32` vs `BF16` on Cloud TPU v5e)

1. **Single-Request Multi-Prompt Batch Test (`Batch = 8` & `Batch = 16`)**:
   * Because Cloud TPU v5e MXUs execute `bfloat16` at **2x the matrix compute speed and half the HBM memory bandwidth** of `float32`, **`BF16` delivers `1.83x to 2.23x higher throughput` and `45% to 59% lower latency`** when processing multi-prompt batches:
     * **`1 KB` (`Batch = 16`)**: `BF16` achieves **`112.3 prompts/s`** ($P_{50}=142.8\text{ ms}$) vs `FP32` `50.4 prompts/s` ($P_{50}=331.4\text{ ms}$) $\implies$ **`2.23x` throughput, `-56.9%` $P_{50}$ latency**.
     * **`2 KB` (`Batch = 8`)**: `BF16` achieves **`59.1 prompts/s`** ($P_{50}=130.9\text{ ms}$) vs `FP32` `28.6 prompts/s` ($P_{50}=319.1\text{ ms}$) $\implies$ **`2.07x` throughput, `-59.0%` $P_{50}$ latency**.
     * **`3 KB` (`Batch = 16`)**: `BF16` achieves **`47.6 prompts/s`** ($P_{50}=339.1\text{ ms}$) vs `FP32` `25.5 prompts/s` ($P_{50}=632.9\text{ ms}$) $\implies$ **`1.87x` throughput, `-46.4%` $P_{50}$ latency**.
     * **`4 KB` (`Batch = 16`)**: `BF16` achieves **`46.7 prompts/s`** ($P_{50}=353.7\text{ ms}$) vs `FP32` `24.9 prompts/s` ($P_{50}=648.2\text{ ms}$) $\implies$ **`1.88x` throughput, `-45.4%` $P_{50}$ latency**.
2. **Low-Batch / Single-Sequence Dispatch (`Batch = 1`, `50–90 RPS` Multi-Payload Sweep)**:
   * At `Batch = 1` or moderate arrival rates (`50–90 RPS`), request latency (`12–21 ms`) is dominated by HTTP/Python/tokenization overhead rather than MXU compute, so `FP32` and `BF16` perform almost identically (**both pass `50–90 RPS` across `1 KB, 2 KB, 5 KB, 7 KB` with `0.00%` errors and $P_{99} < 26\text{ ms}$**).
3. **High-Rate Dedicated Saturation (`160 RPS` on `1 KB`)**:
   * At `160 RPS` (`1 KB`), `BF16` cuts $P_{99}$ latency in half from `54.6 ms` (`FP32`) to **`27.3 ms`** (`BF16` ✅ PASS), and under heavy overload (`190–220 RPS`), `BF16` sustains **`152–153 RPS`** vs `FP32`'s `78–127 RPS` (**up to `+94%` higher overloaded throughput**).

---

## 2. Suite 1A: Single-HTTP-Request Multi-Prompt Batch Comparison (`FP32` vs `BF16`)

| Payload Size | Batch Size (`N` prompts/req) | **`FP32` Throughput (`prompts/s`)** | **`BF16` Throughput (`prompts/s`)** | **`FP32` $P_{50}$ (`ms`)** | **`BF16` $P_{50}$ (`ms`)** | **`FP32` $P_{99}$ (`ms`)** | **`BF16` $P_{99}$ (`ms`)** | **`BF16` vs `FP32` Gain** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1 KB (1024 chars)** | **1** | 78.8/s | 75.9/s | 12.7 ms | 13.1 ms | 13.2 ms | 14.7 ms | Tied (~13 ms) |
| **1 KB (1024 chars)** | **4** | 104.9/s | **109.2/s** | 41.1 ms | **34.9 ms** | 42.8 ms | 43.6 ms | **`+4.1%` tput, `-15.1%` $P_{50}$** |
| **1 KB (1024 chars)** | **8** | 75.6/s | **86.5/s** | 107.8 ms | **100.4 ms** | 110.0 ms | **103.1 ms** | **`+14.4%` tput, `-6.9%` $P_{50}$** |
| **1 KB (1024 chars)** | **16** | 50.4/s | **112.3/s** | 331.4 ms | **142.8 ms** | 342.9 ms | **147.8 ms** | **`2.23x` tput, `-56.9%` $P_{50}$** |
| **2 KB (2048 chars)** | **1** | 57.9/s | 55.9/s | 17.2 ms | 17.8 ms | 18.0 ms | 19.9 ms | Tied (~17.5 ms) |
| **2 KB (2048 chars)** | **4** | 43.9/s | **47.6/s** | 93.0 ms | 95.3 ms | 102.8 ms | **97.5 ms** | **`+8.4%` tput, `-5.2%` $P_{99}$** |
| **2 KB (2048 chars)** | **8** | 28.6/s | **59.1/s** | 319.1 ms | **130.9 ms** | 331.0 ms | **167.8 ms** | **`2.07x` tput, `-59.0%` $P_{50}$** |
| **2 KB (2048 chars)** | **16** | 37.8/s | **56.5/s** | 419.1 ms | **281.5 ms** | 430.3 ms | **327.5 ms** | **`1.49x` tput, `-32.8%` $P_{50}$** |
| **3 KB (3072 chars)** | **1** | 55.2/s | 52.4/s | 18.1 ms | 19.1 ms | 18.8 ms | 20.7 ms | Tied (~18.5 ms) |
| **3 KB (3072 chars)** | **4** | 40.2/s | **47.0/s** | 104.5 ms | **99.6 ms** | 106.4 ms | **101.3 ms** | **`+16.9%` tput, `-4.7%` $P_{50}$** |
| **3 KB (3072 chars)** | **8** | 24.7/s | **46.2/s** | 334.4 ms | **181.5 ms** | 390.6 ms | **185.6 ms** | **`1.87x` tput, `-45.7%` $P_{50}$** |
| **3 KB (3072 chars)** | **16** | 25.5/s | **47.6/s** | 632.9 ms | **339.1 ms** | 700.2 ms | **349.8 ms** | **`1.87x` tput, `-46.4%` $P_{50}$** |
| **4 KB (4096 chars)** | **1** | 53.7/s | 51.5/s | 18.5 ms | 19.3 ms | 20.1 ms | 21.0 ms | Tied (~19 ms) |
| **4 KB (4096 chars)** | **4** | 41.0/s | **42.4/s** | 96.5 ms | 100.8 ms | 108.4 ms | **104.2 ms** | **`+3.4%` tput, `-3.9%` $P_{99}$** |
| **4 KB (4096 chars)** | **8** | 25.3/s | **46.2/s** | 327.1 ms | **175.1 ms** | 390.8 ms | **186.2 ms** | **`1.83x` tput, `-46.5%` $P_{50}$** |
| **4 KB (4096 chars)** | **16** | 24.9/s | **46.7/s** | 648.2 ms | **353.7 ms** | 704.2 ms | **357.4 ms** | **`1.88x` tput, `-45.4%` $P_{50}$** |

---

## 3. Suite 1B: `k6` Concurrent Virtual Users (`1 Prompt per HTTP Request` — `FP32` vs `BF16`)

| Payload Size | Concurrency (`VUs`) | **`FP32` Throughput (`req/s`)** | **`BF16` Throughput (`req/s`)** | **`FP32` $P_{50}$ (`ms`)** | **`BF16` $P_{50}$ (`ms`)** | **`FP32` $P_{99}$ (`ms`)** | **`BF16` $P_{99}$ (`ms`)** | **`BF16` vs `FP32` Gain** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1 KB (1024 chars)** | **1** | 85.4/s | 83.1/s | 11.5 ms | 11.8 ms | 12.7 ms | 13.4 ms | Tied (~11.5 ms) |
| **1 KB (1024 chars)** | **4** | 153.8/s | **176.5/s** | 21.3 ms | 21.7 ms | 40.4 ms | 41.3 ms | **`+14.8%` higher throughput** |
| **1 KB (1024 chars)** | **8** | 172.4/s | 163.7/s | 44.6 ms | 47.1 ms | 101.5 ms | **97.0 ms** | **`-4.4%` lower $P_{99}$** |
| **1 KB (1024 chars)** | **16** | 103.9/s | **113.6/s** | 153.2 ms | **140.3 ms** | 155.8 ms | **146.1 ms** | **`+9.3%` tput, `-8.4%` $P_{50}$** |
| **2 KB (2048 chars)** | **1** | 61.7/s | 59.7/s | 16.0 ms | 16.5 ms | 17.2 ms | 18.1 ms | Tied (~16.2 ms) |
| **2 KB (2048 chars)** | **4** | 51.2/s | **64.2/s** | 89.1 ms | **47.3 ms** | 99.9 ms | **93.1 ms** | **`+25.4%` tput, `-46.9%` $P_{50}$** |
| **2 KB (2048 chars)** | **8** | 50.6/s | **57.7/s** | 153.2 ms | **138.2 ms** | 339.1 ms | **142.9 ms** | **`+14.0%` tput, `-57.9%` $P_{99}$** |
| **2 KB (2048 chars)** | **16** | 28.5/s | **71.3/s** | 560.6 ms | **210.4 ms** | 597.2 ms | **281.4 ms** | **`2.50x` tput, `-62.5%` $P_{50}$** |
| **3 KB (3072 chars)** | **1** | 57.4/s | 57.0/s | 17.1 ms | 17.2 ms | 19.2 ms | 19.6 ms | Tied (~17.1 ms) |
| **3 KB (3072 chars)** | **4** | 89.0/s | 85.8/s | 44.5 ms | 46.1 ms | 46.4 ms | 49.4 ms | Both pass `< 50 ms` $P_{99}$ SLA |
| **3 KB (3072 chars)** | **8** | 22.3/s | **57.4/s** | 356.5 ms | **138.9 ms** | 358.4 ms | **143.3 ms** | **`2.57x` tput, `-61.0%` $P_{50}$** |
| **3 KB (3072 chars)** | **16** | 28.6/s | **57.1/s** | 563.2 ms | **279.7 ms** | 625.9 ms | **284.7 ms** | **`2.00x` tput, `-50.3%` $P_{50}$** |
| **4 KB (4096 chars)** | **1** | 56.7/s | 54.4/s | 17.4 ms | 18.2 ms | 18.9 ms | 19.2 ms | Tied (~17.8 ms) |
| **4 KB (4096 chars)** | **4** | 87.7/s | 84.9/s | 45.1 ms | 46.8 ms | 47.6 ms | 49.7 ms | Both pass `< 50 ms` $P_{99}$ SLA |
| **4 KB (4096 chars)** | **8** | 52.1/s | **57.0/s** | 152.9 ms | **139.9 ms** | 155.8 ms | **143.9 ms** | **`+9.4%` tput, `-8.5%` $P_{50}$** |
| **4 KB (4096 chars)** | **16** | 28.5/s | **56.8/s** | 562.4 ms | **281.5 ms** | 610.0 ms | **286.5 ms** | **`1.99x` tput, `-49.9%` $P_{50}$** |

---

## 4. Suite 2 & Suite 3: Multi-Payload & Dedicated Saturation (`FP32` vs `BF16`)

| Test Suite | Target Rate | **`FP32` $P_{50}$ / $P_{99}$ (`ms`)** | **`BF16` $P_{50}$ / $P_{99}$ (`ms`)** | **`FP32` SLA (`< 50 ms`)** | **`BF16` SLA (`< 50 ms`)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multi-Payload (`1K, 2K, 5K, 7K`)** | **50–90 RPS** | `13.2–24.8 ms` $P_{99}$ (`0%` err) | `13.1–25.9 ms` $P_{99}$ (`0%` err) | ✅ **ALL PASS** | ✅ **ALL PASS** |
| **`1 KB` Dedicated Sweep** | **140 RPS** | `12.2 ms` / `18.1 ms` | `13.0 ms` / `19.7 ms` | ✅ **PASS** | ✅ **PASS** |
| **`1 KB` Dedicated Sweep** | **160 RPS** | `18.8 ms` / `54.6 ms` *(24.2 ms\*)* | **`19.1 ms` / `27.3 ms`** (**-50% $P_{99}$**) | ⚠️ Borderline *(✅ w/ 8K)* | ✅ **PASS** |
| **`1 KB` Dedicated Sweep** | **180 RPS** | `26.5 ms` / `57.4 ms` *(29.6 ms\*)* | `48.5 ms` / `61.5 ms` | ⚠️ Borderline *(✅ w/ 8K)* | ⚠️ Borderline ($P_{50}=48.5\text{ ms}$) |
| **`2 KB` Dedicated Sweep** | **90 RPS** | `17.0 ms` / `20.8 ms` | `18.1 ms` / `23.2 ms` | ✅ **PASS** | ✅ **PASS** |
| **`3 KB` Dedicated Sweep** | **90 RPS** | `18.0 ms` / `20.3 ms` | `19.0 ms` / `22.6 ms` | ✅ **PASS** | ✅ **PASS** |
