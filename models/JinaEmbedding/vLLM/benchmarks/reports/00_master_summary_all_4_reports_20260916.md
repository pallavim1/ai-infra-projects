# Master Summary & Index: Jina AI Embedding Model (`jina-embeddings-v2-small-en`) on Cloud TPU v5e + vLLM (`--max-model-len 2048`)

* **Generated (UTC):** `2026-09-16`
* **Model:** `jinaai/jina-embeddings-v2-small-en` (512-dim BERT/ALiBi Embeddings)
* **Accelerator Hardware:** 1x Cloud TPU v5e chip (`ct5lp-hightpu-1t`, GKE cluster `pm-panw-jina-cluster` in `europe-west4-b`)
* **Serving Engine:** `vLLM v0.26.0` (`--runner pooling --convert embed --trust-remote-code --max-model-len 2048 --max-num-seqs=40`)
* **Tokenization & Truncation Policy:** `vLLM` built-in `BertTokenizerFast` + `truncate_prompt_tokens: 2048` (matching customer's Triton TensorRT `max_length=2048` baseline)

---

## 🔗 Quick Links to All 4 Clean Reports

| # | Report Name | Local Artifact | GitHub (`ai-infra-projects`) | GitHub (`tpu-inference`) |
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

---

### 2. 3-Way Production SLA Comparison (`P99 < 50 ms`): TPU `FP32` vs TPU `BF16` vs Zhemin's Triton (`L4 GPU`)

| Payload Size | **Zhemin Triton L4 (`FP16`)**<br>Max Sustained RPS (`P99 < 50 ms`) | **Cloud TPU v5e (`FP32`)**<br>Max Sustained RPS (`P99 < 50 ms`) | **Cloud TPU v5e (`BF16`)**<br>Max Sustained RPS (`P99 < 50 ms`) | **Cloud TPU v5e Advantage Over Triton L4** |
| :--- | :---: | :---: | :---: | :--- |
| **`1 KB` (`1024 B`)** | **`70 RPS`** *(Dedicated)* / **`40 RPS`** *(Ray Serve)* | **`180 RPS`** ($P_{99}=29.6\text{ ms}$) | **`160–180 RPS`** ($P_{99}=27.3\text{ ms}$ @ 160) | 🏆 **`2.57x` higher sustained RPS** |
| **`2 KB` (`2048 B`)** | **`40 RPS`** ($P_{99}=46.5\text{ ms}$) | **`100 RPS`** ($P_{99}=25.6\text{ ms}$) | **`90 RPS`** ($P_{99}=23.2\text{ ms}$) | 🏆 **`2.25x–2.50x` higher sustained RPS** |
| **`3 KB` (`3072 B`, truncated @ 2048)** | **`~30 RPS`** | **`90 RPS`** ($P_{99}=20.3\text{ ms}$) | **`90 RPS`** ($P_{99}=22.6\text{ ms}$) | 🏆 **`3.00x` higher sustained RPS (`0%` errors)** |
| **`5 KB` (`5120 B`, truncated @ 2048)** | **`20 RPS`** ($P_{99}=49.1\text{ ms}$; fails @ 30 RPS) | **`90 RPS`** ($P_{99}=20.4\text{ ms}$) | **`90 RPS`** ($P_{99}=21.2\text{ ms}$) | 🏆 **`4.50x` higher sustained RPS (`0%` errors)** |
| **`7 KB` (`7168 B`, truncated @ 2048)** | **`10 RPS`** ($P_{99}=46.4\text{ ms}$; fails @ 20 RPS) | **`90 RPS`** ($P_{99}=20.0\text{ ms}$) | **`90 RPS`** ($P_{99}=24.0\text{ ms}$) | 🏆 **`9.00x` higher sustained RPS (`0%` errors)** |

---

## 🧭 Architecture & Methodology: Why There Are 2 Separate Client Tests + `vLLM` `BertTokenizerFast`

There are **2 separate client tests** in these benchmark reports because the customer's benchmark spreadsheet ([`ATP AIC2 Benchmarks`](https://docs.google.com/spreadsheets/d/1fGgqjRp4giG0MD6NjSIzAbpDbeJoq_Aao7B2MM78QgU/edit?gid=1161755388#gid=1161755388)) evaluates two distinct traffic patterns:

### 1. Test #1: Single HTTP Request with Multiple Prompts ([`single_req_batch_bench.py`](https://github.com/pallavim1/ai-infra-projects/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/single_req_batch_bench.py))
* **Where it comes from**: Matches the **Top Table** in Zhemin's document (*"Batch Request Testing: A single HTTP request contains multiple prompts — Concurrently sends a batch of random characters in a single vLLM inference request and returns the response latency in milliseconds"*).
* **How it works**:
  * The Python client script packs **`N` prompts (`N = 1, 4, 8, 16`) into ONE single HTTP request**:
    ```json
    POST /prompt_c2
    {"text": ["prompt_1", "prompt_2", "prompt_3", "prompt_4"]}
    ```
  * It sends that 1 HTTP request, waits for `vLLM` to return all `N` embeddings in a single response, and records the response latency in milliseconds.
* **What it tests**: Offline/client-bundled batch latency (`batch_size = 1, 4, 8, 16`).

### 2. Test #2: `k6` Load Generator Script ([`k6_ray_serve_test.js`](https://github.com/pallavim1/ai-infra-projects/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/k6_ray_serve_test.js) / [`k6_customer_match.js`](https://github.com/pallavim1/ai-infra-projects/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/k6_customer_match.js))
* **Where it comes from**: Matches the **Sustained RPS & Saturation Tables** in Zhemin's document (*50–90 RPS Multi-Payload Sweep*, *1KB Dedicated 100–220 RPS*, *2KB/3KB Dedicated 40–110 RPS*).
* **How it works**:
  * The `k6` script fires **50 to 200 separate HTTP requests per second**, where **each HTTP request contains only 1 single prompt**:
    ```json
    POST /prompt_c2
    {"text": "single_prompt_string"}
    ```
  * Because dozens of requests arrive at the server simultaneously, **`vLLM` automatically groups them together on the server** (*continuous batching*) and runs them on the TPU.
* **What it tests**: Real-world production serving capacity—how many concurrent requests per second (`RPS`) the server can sustain while keeping end-to-end $P_{99}$ latency under **`< 50 ms`**.

### 3. Role of `vLLM` `BertTokenizerFast` (Server-Side Preprocessor Used by BOTH Tests)
* **`vLLM` `BertTokenizerFast`** runs **inside the `vLLM` server on the TPU v5e pod** (configured in [`jina_v5e_deployment.yaml#L33-L45`](https://github.com/pallavim1/ai-infra-projects/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/deploy/jina_v5e_deployment.yaml#L33-L45) via `"truncate_prompt_tokens": 2048`).
* Every time either **Test #1 (`single_req_batch_bench.py`)** or **Test #2 (`k6`)** sends raw text strings to `/prompt_c2`, `vLLM` uses `BertTokenizerFast` in Rust to convert the text strings into integer token IDs (`[101, 4821, ...]`) and truncates any sequence $> 2,048$ tokens before executing the TPU v5e forward pass:

![k6 to vLLM Fast Tokenizer to TPU v5e Flow Diagram](/usr/local/google/home/pallaviam/.gemini/jetski/brain/292ebc90-dbb8-4a33-8f1a-ca6dee63d8ff/k6_vllm_tpu_flow_diagram.png)
