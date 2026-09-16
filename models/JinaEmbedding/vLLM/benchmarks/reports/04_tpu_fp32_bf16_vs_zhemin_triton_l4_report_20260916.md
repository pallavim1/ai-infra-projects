# Report 4: 3-Way Comparison — Cloud TPU v5e (`FP32` & `BF16`) vs Zhemin's NVIDIA L4 GPU (`Triton TensorRT`)

* **Model:** `jinaai/jina-embeddings-v2-small-en` (512-dim BERT/ALiBi Embeddings)
* **Tokenization & Truncation Policy (Identical Across All 3 Setups):**
  * **Cloud TPU v5e + vLLM (`FP32` & `BF16`)**: `BertTokenizerFast` + `--max-model-len 2048` + `truncate_prompt_tokens: 2048`
  * **Zhemin's NVIDIA L4 + Triton TensorRT (`FP16` & `FP32`)**: `BertTokenizerFast` + `tokenizer(text, truncation=True, max_length=2048)`
* **Reference Customer Worksheet:** [ATP AIC2 Benchmarks (`gid=1161755388` & `gid=1972899730`)](https://docs.google.com/spreadsheets/d/1fGgqjRp4giG0MD6NjSIzAbpDbeJoq_Aao7B2MM78QgU/edit?gid=1161755388#gid=1161755388)

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

## 🧭 Understanding the 2 Separate Client Tests (`Single-Request Multi-Prompt Batch` vs `k6` Load Generator)

1. **Test #1 — Single HTTP Request with Multiple Prompts ([`single_req_batch_bench.py`](https://github.com/pallavim1/ai-infra-projects/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/single_req_batch_bench.py))**:
   * Sends **1 single HTTP request** containing an array of `N` prompts (`{"text": ["prompt_1", ..., "prompt_N"]}`, where $N = 1, 4, 8, 16$) and measures the response latency in milliseconds. Matches the top table in Zhemin's worksheet (*"Batch Request Testing: A single HTTP request contains multiple prompts"*).
2. **Test #2 — `k6` Load Generator Script ([`k6_ray_serve_test.js`](https://github.com/pallavim1/ai-infra-projects/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/k6_ray_serve_test.js))**:
   * Sends **50 to 200 separate HTTP requests per second**, where each HTTP request contains **1 single prompt** (`{"text": "single_prompt"}`), and `vLLM` dynamically groups them on the server (*continuous batching*). Matches the bottom saturation tables in Zhemin's worksheet.
3. **Server-Side `vLLM` `BertTokenizerFast` (`truncate_prompt_tokens: 2048`)**:
   * Used by **both** client tests above to convert raw character strings into token IDs (`<= 2,048` tokens) on the TPU v5e server before running the embedding model forward pass.

---

## 1. Executive Summary: Cloud TPU v5e (`FP32` & `BF16`) vs Zhemin's NVIDIA L4 Triton (`FP16` & `FP32`)

| Benchmark Dimension | **Cloud TPU v5e (`FP32`, `max=2048`)** | **Cloud TPU v5e (`BF16`, `max=2048`)** | **Zhemin's NVIDIA L4 Triton (`FP16`, `max=2048`)** | **Zhemin's NVIDIA L4 Triton (`FP32`, `max=2048`)** | **Winner & Key Takeaway** |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. `1 KB` Max Sustained RPS (`P99 < 50 ms`)** | **`180 RPS`**<br>($P_{50}=19.9\text{ ms}, P_{99}=29.6\text{ ms}$) | **`160–180 RPS`**<br>($P_{50}=19.1\text{ ms}, P_{99}=27.3\text{ ms}$ @ 160) | **`70 RPS`** *(Dedicated)*<br>**`40 RPS`** *(Ray Serve `22.8 ms`)* | **`~30–40 RPS`** | 🏆 **Cloud TPU v5e (`2.57x` higher RPS than L4 `FP16`)** |
| **2. `2 KB` Max Sustained RPS (`P99 < 50 ms`)** | **`100 RPS`**<br>($P_{50}=18.9\text{ ms}, P_{99}=25.6\text{ ms}$) | **`90 RPS`**<br>($P_{50}=18.1\text{ ms}, P_{99}=23.2\text{ ms}$) | **`40 RPS`**<br>($P_{50}=23.3\text{ ms}, P_{99}=46.5\text{ ms}$) | **`~20 RPS`** | 🏆 **Cloud TPU v5e (`2.25x–2.50x` higher RPS than L4 `FP16`)** |
| **3. `3 KB` Max Sustained RPS (`P99 < 50 ms`)** | **`90 RPS`**<br>($P_{50}=18.0\text{ ms}, P_{99}=20.3\text{ ms}$) | **`90 RPS`**<br>($P_{50}=19.0\text{ ms}, P_{99}=22.6\text{ ms}$) | **`~30 RPS`** | **`~15 RPS`** | 🏆 **Cloud TPU v5e (`3.0x` higher RPS than L4 `FP16`, `0%` errors)** |
| **4. `5 KB` & `7 KB` Max Sustained RPS (`P99 < 50 ms`)** | **`90 RPS` (Multi-Payload)**<br>($P_{99}=20.0\text{–}20.4\text{ ms}$, `0%` err) | **`90 RPS` (Multi-Payload)**<br>($P_{99}=21.2\text{–}24.0\text{ ms}$, `0%` err) | **`20 RPS` (`5 KB`, $49.1\text{ ms}$)**<br>**`10 RPS` (`7 KB`, $46.4\text{ ms}$)** | **`< 10 RPS`** | 🏆 **Cloud TPU v5e (`4.5x–9.0x` higher RPS on `5 KB` & `7 KB`)** |
| **5. Low Concurrency (`Conc = 1 & 4`, `1 KB–4 KB`)** | **`11.5 ms–45.1 ms` $P_{50}$**<br>(`56.7–187.4 req/s`) | **`11.8 ms–46.8 ms` $P_{50}$**<br>(`54.4–176.5 req/s`) | **`20.7 ms–60.6 ms` $P_{50}$**<br>(`32.6–103.4 req/s`) | **`31.5 ms–175.0 ms` $P_{50}$**<br>(`18.1–52.3 req/s`) | 🏆 **Cloud TPU v5e (`1.3x–1.98x` faster than L4 `FP16`; `3x` faster than L4 `FP32`)** |

---

## 2. Suite 1: Concurrency / Batch Request Testing (`1 KB, 2 KB, 3 KB, 4 KB` @ `1, 4, 8, 16`) — TPU (`FP32` & `BF16`) vs Zhemin's Triton L4 (`FP16` & `FP32`)

| Payload Size | Concurrency / Batch | **TPU v5e `FP32`**<br>Tput / $P_{50}$ / $P_{99}$ | **TPU v5e `BF16`**<br>Tput / $P_{50}$ / $P_{99}$ | **Zhemin Triton L4 `FP16`**<br>Tput / $P_{50}$ / $P_{99}$ | **Zhemin Triton L4 `FP32`**<br>Tput / $P_{50}$ / $P_{99}$ | **Faster Accelerator Setup** |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1 KB (1024 chars)** | **1** | **85.4/s** \| `11.5ms` \| `12.7ms` | **83.1/s** \| `11.8ms` \| `13.4ms` | 43.2/s \| `20.7ms` \| `23.5ms` | 29.8/s \| `31.5ms` \| `35.4ms` | 🏆 **TPU (`1.98x` vs L4 `FP16`; `2.86x` vs L4 `FP32`)** |
| **1 KB (1024 chars)** | **4** | **153.8/s** *(187.4/s\*)* \| `21.3ms` \| `40.4ms` | **176.5/s** \| `21.7ms` \| `41.3ms` | 103.4/s \| `38.1ms` \| `42.9ms` | 52.3/s \| `75.2ms` \| `83.1ms` | 🏆 **TPU `BF16`/`FP32` (`1.71x–1.81x` vs L4 `FP16`; `3.37x` vs L4 `FP32`)** |
| **1 KB (1024 chars)** | **8** | **172.4/s** *(187.6/s\*)* \| `44.6ms` \| `101.5ms` *(44.5ms\*)* | **163.7/s** \| `47.1ms` \| `97.0ms` | 135.1/s \| `58.7ms` \| `66.3ms` | 56.1/s \| `141.2ms` \| `154.8ms` | 🏆 **TPU (`1.28x–1.39x` vs L4 `FP16`; `3.07x` vs L4 `FP32`)** |
| **1 KB (1024 chars)** | **16** | 103.9/s *(186.8/s\*)* \| `153.2ms` *(85.4ms\*)* | **113.6/s** *(Batch: `112.3/s`)* \| `140.3ms` | 165.2/s \| `96.5ms` \| `107.6ms` | 57.4/s \| `276.4ms` \| `298.0ms` | 🏆 **TPU w/ 8K bucket (`186.8/s` vs `165.2/s`)**; **TPU `BF16` `2.0x` vs L4 `FP32`** |
| **2 KB (2048 chars)** | **1** | **61.7/s** \| `16.0ms` \| `17.2ms` | **59.7/s** \| `16.5ms` \| `18.1ms` | 39.0/s \| `24.3ms` \| `28.2ms` | 24.1/s \| `39.8ms` \| `44.6ms` | 🏆 **TPU (`1.58x` vs L4 `FP16`; `2.56x` vs L4 `FP32`)** |
| **2 KB (2048 chars)** | **4** | 51.2/s *(97.7/s\*)* \| `89.1ms` *(40.7ms\*)* | **64.2/s** \| `47.3ms` \| `93.1ms` | 75.7/s \| `52.1ms` \| `58.2ms` | 33.4/s \| `118.4ms` \| `131.0ms` | 🏆 **TPU `BF16` (`47.3ms` $P_{50}$ vs L4 `52.1ms`); `1.92x` vs L4 `FP32`** |
| **2 KB (2048 chars)** | **8** | 50.6/s *(96.6/s\*)* \| `153.2ms` *(82.6ms\*)* | **59.1/s** *(Batch)* \| `130.9ms` \| `142.9ms` | 90.5/s \| `87.4ms` \| `97.3ms` | 35.2/s \| `225.8ms` \| `246.2ms` | 🏆 **TPU w/ 8K bucket (`96.6/s` vs `90.5/s`); TPU `BF16` `1.68x` vs L4 `FP32`** |
| **2 KB (2048 chars)** | **16** | 28.5/s *(96.5/s\*)* \| `560.6ms` *(165.9ms\*)* | **71.3/s** \| `210.4ms` \| `281.4ms` | 101.1/s \| `156.9ms` \| `175.5ms` | 35.8/s \| `445.0ms` \| `482.1ms` | **Tied w/ 8K bucket (`~96.5/s` vs `101.1/s`); TPU `BF16` `2.0x` vs L4 `FP32`** |
| **3 KB (3072 chars)** | **1** | **57.4/s** \| `17.1ms` \| `19.2ms` | **57.0/s** \| `17.2ms` \| `19.6ms` | 34.8/s \| `26.6ms` \| `31.3ms` | 21.2/s \| `45.2ms` \| `51.0ms` | 🏆 **TPU (`1.65x` vs L4 `FP16`; `2.71x` vs L4 `FP32`)** |
| **3 KB (3072 chars)** | **4** | **89.0/s** \| `44.5ms` \| `46.4ms` | **85.8/s** \| `46.1ms` \| `49.4ms` | 68.8/s \| `57.0ms` \| `63.7ms` | 29.5/s \| `134.1ms` \| `148.5ms` | 🏆 **TPU (`1.29x` vs L4 `FP16`, passes `< 50ms` $P_{99}$; `3.02x` vs L4 `FP32`)** |
| **3 KB (3072 chars)** | **8** | 22.3/s \| `356.5ms` \| `358.4ms` | **57.4/s** \| `138.9ms` \| `143.3ms` | 86.9/s \| `90.8ms` \| `100.2ms` | 31.0/s \| `256.4ms` \| `279.0ms` | **L4 `FP16` faster at Conc 8; TPU `BF16` `1.85x` faster than L4 `FP32`** |
| **3 KB (3072 chars)** | **16** | 28.6/s \| `563.2ms` \| `625.9ms` | **57.1/s** \| `279.7ms` \| `284.7ms` | 94.7/s \| `167.8ms` \| `189.0ms` | 31.4/s \| `508.2ms` \| `549.1ms` | **L4 `FP16` faster at Conc 16; TPU `BF16` `1.82x` faster than L4 `FP32`** |
| **4 KB (4096 chars)** | **1** | **56.7/s** \| `17.4ms` \| `18.9ms` | **54.4/s** \| `18.2ms` \| `19.2ms` | 32.6/s \| `28.6ms` \| `34.2ms` | 18.1/s \| `53.2ms` \| `59.8ms` | 🏆 **TPU (`1.74x` vs L4 `FP16`; `3.13x` vs L4 `FP32`)** |
| **4 KB (4096 chars)** | **4** | **87.7/s** \| `45.1ms` \| `47.6ms` | **84.9/s** \| `46.8ms` \| `49.7ms` | 64.5/s \| `60.6ms` \| `68.5ms` | 22.8/s \| `175.0ms` \| `192.4ms` | 🏆 **TPU (`1.36x` vs L4 `FP16`, passes `< 50ms` $P_{99}$; `3.85x` vs L4 `FP32`)** |
| **4 KB (4096 chars)** | **8** | 52.1/s \| `152.9ms` \| `155.8ms` | **57.0/s** \| `139.9ms` \| `143.9ms` | 78.7/s \| `100.2ms` \| `113.9ms` | 24.1/s \| `331.0ms` \| `358.2ms` | **L4 `FP16` faster at Conc 8; TPU `BF16` `2.37x` faster than L4 `FP32`** |
| **4 KB (4096 chars)** | **16** | 28.5/s \| `562.4ms` \| `610.0ms` | **56.8/s** \| `281.5ms` \| `286.5ms` | 88.1/s \| `180.5ms` \| `204.0ms` | 24.5/s \| `651.2ms` \| `701.5ms` | **L4 `FP16` faster at Conc 16; TPU `BF16` `2.32x` faster than L4 `FP32`** |

---

## 3. Suite 2 & Suite 3: Open-Loop RPS & SLA (`P99 < 50 ms`) Comparison — TPU (`FP32` & `BF16`) vs Zhemin's Triton L4 (`FP16`)

In production traffic (open-loop `k6` RPS arrival where `P99 < 50 ms` is required), **Cloud TPU v5e (`FP32` and `BF16`) outperforms Zhemin's NVIDIA L4 Triton (`FP16`) across every payload size from `1 KB` to `7 KB`**:

| Payload Tier | **Zhemin Triton L4 (`FP16`)**<br>Max RPS Passing `P99 < 50 ms` | **Cloud TPU v5e (`FP32`)**<br>Max RPS Passing `P99 < 50 ms` | **Cloud TPU v5e (`BF16`)**<br>Max RPS Passing `P99 < 50 ms` | **TPU v5e Advantage Over L4 Triton (`FP16`)** |
| :--- | :---: | :---: | :---: | :--- |
| **`1 KB` (`1024 B`)** | **`70 RPS`** *(Dedicated)* / **`40 RPS`** *(Ray Serve `22.8 ms`)* | **`180 RPS`** ($P_{50}=19.9\text{ ms}, P_{99}=29.6\text{ ms}$) | **`160–180 RPS`** ($P_{99}=27.3\text{ ms}$ @ `160 RPS`) | 🏆 **`2.57x` higher sustained RPS** (`180 RPS` vs `70 RPS`) |
| **`2 KB` (`2048 B`)** | **`40 RPS`** ($P_{50}=23.3\text{ ms}, P_{99}=46.5\text{ ms}$) | **`100 RPS`** ($P_{50}=18.9\text{ ms}, P_{99}=25.6\text{ ms}$) | **`90 RPS`** ($P_{50}=18.1\text{ ms}, P_{99}=23.2\text{ ms}$) | 🏆 **`2.25x–2.50x` higher sustained RPS** (`90–100 RPS` vs `40 RPS`) |
| **`3 KB` (`3072 B`, truncated at `2048`)** | **`~30 RPS`** | **`90 RPS`** ($P_{50}=18.0\text{ ms}, P_{99}=20.3\text{ ms}$) | **`90 RPS`** ($P_{50}=19.0\text{ ms}, P_{99}=22.6\text{ ms}$) | 🏆 **`3.00x` higher sustained RPS** (`90 RPS` vs `30 RPS`) |
| **`5 KB` (`5120 B`, truncated at `2048`)** | **`20 RPS`** ($P_{50}=31.2\text{ ms}, P_{99}=49.1\text{ ms}$; fails @ `30 RPS` w/ `61.5 ms`) | **`90 RPS`** ($P_{99}=20.4\text{ ms}$, `0.00%` err) | **`90 RPS`** ($P_{99}=21.2\text{ ms}$, `0.00%` err) | 🏆 **`4.50x` higher sustained RPS** (`90 RPS` vs `20 RPS`) |
| **`7 KB` (`7168 B`, truncated at `2048`)** | **`10 RPS`** ($P_{50}=36.6\text{ ms}, P_{99}=46.4\text{ ms}$; fails @ `20 RPS` w/ `70.2 ms`) | **`90 RPS`** ($P_{99}=20.0\text{ ms}$, `0.00%` err) | **`90 RPS`** ($P_{99}=24.0\text{ ms}$, `0.00%` err) | 🏆 **`9.00x` higher sustained RPS** (`90 RPS` vs `10 RPS`) |
