# PANW Jina Embeddings (`jina-embeddings-v2-small-en`) on Google Cloud TPU v6e & TPU v5e (`vLLM`) vs NVIDIA L4 (`Triton + TensorRT`)

This directory (`models/JinaEmbedding/vLLM` on the `panw-tpu-inference` branch of `pallavim1/tpu-inference`) contains the complete benchmark suites (`Online k6` + `Batch`), Performance & TCO Comparison Reports (`FP32` and `BF16`), consolidated Excel workbooks (`.xlsx`), raw JSON datasets, and GKE deployment manifests for serving **`jinaai/jina-embeddings-v2-small-en`** (`--max-model-len 2048`) on **Cloud TPU v6e (`ct6e-standard-1t`)** and **Cloud TPU v5e (`ct5lp-hightpu-1t`)** vs **NVIDIA L4 (`g2-standard-4`)**.

---

## 1. Consolidated Excel Workbooks (`.xlsx`)

* **[Consolidated 8-Tab Workbook — TPU v6e (`FP32` & `BF16`) + TPU v5e (`FP32` & `BF16`) + NVIDIA L4 (`ATP_AIC2_Benchmarks_TPU_v6e_v5e_FP32_and_BF16_vs_L4.xlsx`)](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/ATP_AIC2_Benchmarks_TPU_v6e_v5e_FP32_and_BF16_vs_L4.xlsx)**
  * **Tab 1**: `TPU V6e (FP32) vs L4 & V5e`
  * **Tab 2**: `TPU V6e (BF16) vs L4 & V5e`
  * **Tab 3**: `Jina + TPU V6e + vLLM (FP32)` *(All Online `k6` Suites + Multi-Prompt Batch + High-Batch Token Sweep)*
  * **Tab 4**: `Jina + TPU V6e + vLLM (BF16)` *(All Online `k6` Suites + Multi-Prompt Batch + High-Batch Token Sweep)*
  * **Tab 5**: `TPU V5e (FP32) vs L4`
  * **Tab 6**: `TPU V5e (BF16) vs L4`
  * **Tab 7**: `Jina + TPU V5e + vLLM (FP32)`
  * **Tab 8**: `Jina + TPU V5e + vLLM (BF16)`
* **[TPU v5e (`FP32` & `BF16`) vs NVIDIA L4 Workbook (`ATP_AIC2_Benchmarks_TPU_v5e_FP32_and_BF16_vs_L4.xlsx`)](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/ATP_AIC2_Benchmarks_TPU_v5e_FP32_and_BF16_vs_L4.xlsx)**

---

## 2. Performance & TCO Comparison Reports (`TPU v6e`, `TPU v5e`, and `NVIDIA L4`)

### **Cloud TPU v6e (`ct6e-standard-1t`) Reports & Megakernel Design**
* **[`models/JinaEmbedding/vLLM/megakernel/` — Approaching Megakernels on TPU v6e & `FP32` `<= 2K` Benchmark Results](https://github.com/pallavim1/tpu-inference/tree/panw-tpu-inference/models/JinaEmbedding/vLLM/megakernel)**:
  * **[`megakernel/README.md` (Architecture & Asset Index)](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/megakernel/README.md)**
  * **[`megakernel/TPU_v6e_FP32_Megakernel_2K_Benchmark_Report.md` (`FP32` `<= 2K` Benchmark & TCO Report)](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/megakernel/TPU_v6e_FP32_Megakernel_2K_Benchmark_Report.md)**
* **[`06A` — TPU v6e (`FP32`) vs NVIDIA L4 & TPU v5e (`Performance & TCO Comparison`)](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/06A_tpu_v6e_fp32_vs_l4_and_v5e_performance_comparison.md)**
* **[`06B` — TPU v6e (`BF16`) vs NVIDIA L4 & TPU v5e (`Performance & TCO Comparison`)](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/06B_tpu_v6e_bf16_vs_l4_and_v5e_performance_comparison.md)**

### **Cloud TPU v5e (`ct5lp-hightpu-1t`) Reports**
* **[`05A` — TPU v5e (`FP32`) vs NVIDIA L4 (`Performance & TCO Comparison`)](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/05A_tpu_v5e_fp32_vs_l4_performance_comparison.md)**
* **[`05B` — TPU v5e (`BF16`) vs NVIDIA L4 (`Performance & TCO Comparison`)](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/05B_tpu_v5e_bf16_vs_l4_performance_comparison.md)**
* **[`00` — Master Summary Report (`All 4 Reports`)](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/00_master_summary_all_4_reports_20260916.md)**
* **[`01` — TPU v5e `FP32` (`max-model-len=2048`) Detailed Report](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/01_fp32_max2048_truncated_report_20260916.md)**
* **[`02` — TPU v5e `BF16` (`max-model-len=2048`) Detailed Report](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/02_bf16_max2048_truncated_report_20260916.md)**
* **[`03` — TPU v5e `FP32` vs `BF16` Comparison Report](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/03_fp32_vs_bf16_comparison_report_20260916.md)**
* **[`04` — TPU (`FP32` & `BF16`) vs Zhemin's Triton L4 Report](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/04_tpu_fp32_bf16_vs_zhemin_triton_l4_report_20260916.md)**
* **[Architecture & Flow Diagram (`k6_vllm_tpu_flow_diagram.png`)](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/k6_vllm_tpu_flow_diagram.png)**

---

## 3. Raw Benchmark JSON Datasets (`TPU v6e` & `TPU v5e`)

* **[TPU v6e `FP32` Complete Results (`Online k6` + `Multi-Prompt Batch` + `Token Sweep`)](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/results/v6e_float32_complete_results_20260924_184725.json)**
* **[TPU v6e `BF16` Complete Results (`Online k6` + `Multi-Prompt Batch` + `Token Sweep`)](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/results/v6e_bfloat16_complete_results_20260924_175613.json)**
* **[TPU v5e `BF16` Customer `k6` Results (`20260916_153424`)](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/customer_k6_results_20260916_153424.json)**
* **[TPU v5e `FP32` Customer `k6` Results (`20260916_075137`)](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/customer_k6_results_20260916_075137.json)**
* **[TPU v5e `BF16` Single-Request Multi-Prompt Batch Results](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports/bf16_single_req_batch_20260916_152811.json)**

---

## 4. Benchmark Scripts (`k6` & Python Suites) & GKE Deployment Manifests

* **[Complete TPU v6e Benchmark Runner (`run_v6e_complete_suite.py`)](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/run_v6e_complete_suite.py)**
* **[Customer-Matching `k6` Load Script (`k6_customer_match.js`)](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/k6_customer_match.js)**
* **[Customer-Matching `k6` Python Orchestrator (`run_customer_matching_k6.py`)](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/run_customer_matching_k6.py)**
* **[Single-Request Batch Benchmark (`single_req_batch_bench.py`)](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/single_req_batch_bench.py)**
* **[Benchmarking Guide (`BENCHMARKING_GUIDE.md`)](https://github.com/pallavim1/tpu-inference/blob/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/BENCHMARKING_GUIDE.md)**
* **[GKE Deployment Manifests (`models/JinaEmbedding/vLLM/deploy/`)](https://github.com/pallavim1/tpu-inference/tree/panw-tpu-inference/models/JinaEmbedding/vLLM/deploy)**
* **[Custom ALiBi FlashAttention Pallas Kernel Branch (`jina-v2-alibi-kernel`)](https://github.com/pallavim1/tpu-inference/tree/jina-v2-alibi-kernel)**

---

## 5. Executive Summary: TPU v6e vs TPU v5e vs NVIDIA L4 ($P_{99} \le 50\text{ ms}$ SLA)

| Payload Size | NVIDIA L4 (`g2-standard-4`) Max SLA RPS | TPU v5e (`FP32` / `BF16`) Max SLA RPS | **TPU v6e (`FP32` / `BF16`) Max SLA RPS** | **TPU v6e vs L4 Throughput Multiplier** | **TPU v6e Cost Reduction vs L4 (`1-Yr` / `3-Yr CUD`)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1 KB (`1024 B`)** | **70 RPS** | 140 / 160 RPS | **160–180 (`FP32`) / 160–200 (`BF16`) RPS** | **2.29x – 2.86x Higher** | **15.0% (`1-Yr`) – 38.9% (`3-Yr`) Cheaper** |
| **2 KB (`2048 B`)** | **40 RPS** | 70 / 90 RPS | **100 (`FP32`) / 110 (`BF16`) RPS** | **2.50x – 2.75x Higher** | **11.7% (`1-Yr`) – 36.6% (`3-Yr`) Cheaper** |
| **3 KB (`3072 B`)** | **30 RPS** | 70 / 90 RPS | **110 (`FP32`) / 100 (`BF16`) RPS** | **3.33x – 3.67x Higher** | **27.1% (`1-Yr`) – 52.5% (`3-Yr`) Cheaper** |
| **5 KB (`5120 B`)** | **20 RPS** | 70 / 90 RPS | **100 (`FP32`) / 110 (`BF16`) RPS** | **5.00x – 5.50x Higher** | **55.9% (`1-Yr`) – 68.3% (`3-Yr`) Cheaper** *(29.9% On-Demand)* |
| **7 KB (`7168 B`)** | **10 RPS** | 70 / 90 RPS | **100 (`FP32`) / 110 (`BF16`) RPS** | **10.0x – 11.0x Higher** | **77.9% (`1-Yr`) – 84.2% (`3-Yr`) Cheaper** *(64.9% On-Demand)* |
