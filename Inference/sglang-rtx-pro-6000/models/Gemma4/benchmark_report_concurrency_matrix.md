# Gemma 4 26B Full-Matrix Benchmark Report (Max Concurrency Mode)

## Executive Summary
This report presents benchmark results for **Gemma 4 26B** (`google/gemma-4-26B-A4B`) served via **vLLM** on a Google Cloud GCE **G4 instance** (`g4-standard-48` equipped with **1x NVIDIA RTX PRO 6000 Blackwell Edition GPU**, 96 GB VRAM).

Performance was evaluated using **infinite request rate with fixed concurrency caps** (`--request-rate inf --max-concurrency $C`) across 4 variations of **Input Sequence Length (ISL)** and **Output Sequence Length (OSL)** at 5 concurrency levels (64, 128, 256, 512, 1024), totaling **20 benchmark runs** and **20,480 completed requests** with 0 failures.

### Serving Configuration
* **Model**: `google/gemma-4-26B-A4B`
* **Precision / Quantization**: **QWIX FP8 Quantization** (`weight_qtype: float8_e4m3fn`, `act_qtype: float8_e4m3fn`)
* **KV Cache**: **FP8 KV Cache** (`--kv-cache-dtype fp8`)
* **Max Model Length**: 16,384 tokens (`--max-model-len 16384`)
* **Max Batched Tokens**: 16,384 tokens (`--max-num-batched-tokens 16384`)
* **Max Sequences**: 1,024 sequences (`--max-num-seqs 1024`)
* **Block Size**: 256
* **Multimodal Limits**: `{"image": 1, "video": 0, "audio": 0}`
* **Execution Date**: August 8, 2026

---

## Benchmark Results Data Tables

### 1. Workload: `1k_8k` (ISL = 1000, OSL = 8000)
*Stress test for long-generation decoding memory footprint and continuous KV cache allocation.*

| Concurrency ($C$) | Prompts | Duration (s) | Output TPS (tok/s) | Peak Output TPS | Total TPS (tok/s) | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Success |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **64** | 128 | 613.05 | **1,670.33** | 1,984.00 | 1,879.12 | 1,638.12 | 1,462.81 | 3,113.25 | **38.08** | 38.07 | 39.11 | 128 / 128 |
| **128** | 256 | 793.41 | **2,581.25** | 3,072.00 | 2,903.91 | 1,743.92 | 1,442.97 | 3,855.32 | **49.31** | 49.41 | 49.58 | 256 / 256 |
| **256** | 512 | 1,402.80 | **2,919.87** | 4,352.00 | 3,284.86 | 46,029.11 | 7,784.63 | 370,523.85 | **67.65** | 62.82 | 118.88 | 512 / 512 |
| **512** | 1,024 | 2,653.54 | **3,087.20** | 4,418.00 | 3,473.10 | 451,200.98 | 519,162.20 | 1,019,809.78 | **77.15** | 63.31 | 126.26 | 1,024 / 1,024 |
| **1024** | 2,048 | 5,224.61 | **3,135.93** | 4,415.00 | 3,527.92 | 1,398,691.17 | 1,561,293.58 | 2,042,784.82 | **80.05** | 63.51 | 127.07 | 2,048 / 2,048 |

---

### 2. Workload: `8k_1k` (ISL = 8000, OSL = 1000)
*Stress test for long-context prefill compute, activation memory, and prompt throughput.*

| Concurrency ($C$) | Prompts | Duration (s) | Output TPS (tok/s) | Peak Output TPS | Total TPS (tok/s) | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Success |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **64** | 128 | 99.92 | **1,280.99** | 2,176.00 | 11,528.94 | 5,471.17 | 2,236.35 | 17,534.74 | **44.35** | 48.05 | 49.03 | 128 / 128 |
| **128** | 256 | 161.25 | **1,587.64** | 2,944.00 | 14,288.78 | 10,020.13 | 2,293.11 | 35,118.27 | **70.23** | 77.72 | 78.98 | 256 / 256 |
| **256** | 512 | 317.69 | **1,611.64** | 3,380.00 | 14,504.79 | 51,219.17 | 30,894.39 | 121,633.93 | **86.56** | 90.88 | 96.86 | 512 / 512 |
| **512** | 1,024 | 615.27 | **1,664.31** | 3,380.00 | 14,978.81 | 165,571.19 | 203,216.58 | 253,531.66 | **89.30** | 92.65 | 98.21 | 1,024 / 1,024 |
| **1024** | 2,048 | 1,205.34 | **1,699.10** | 3,549.00 | **15,291.94** | 387,376.42 | 492,295.16 | 547,719.60 | **90.50** | 91.89 | 98.65 | 2,048 / 2,048 |

---

### 3. Workload: `1k_1k` (ISL = 1000, OSL = 1000)
*Balanced conversational / agentic workload.*

| Concurrency ($C$) | Prompts | Duration (s) | Output TPS (tok/s) | Peak Output TPS | Total TPS (tok/s) | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Success |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **64** | 128 | 77.57 | **1,650.11** | 1,984.00 | 3,300.22 | 1,125.54 | 1,252.89 | 1,916.06 | **37.67** | 37.68 | 38.54 | 128 / 128 |
| **128** | 256 | 98.08 | **2,610.18** | 3,072.00 | 5,220.35 | 1,739.89 | 1,430.40 | 3,790.12 | **47.29** | 47.57 | 48.54 | 256 / 256 |
| **256** | 512 | 147.66 | **3,467.45** | 4,352.00 | 6,934.89 | 2,801.99 | 1,477.54 | 7,750.61 | **66.79** | 68.09 | 68.81 | 512 / 512 |
| **512** | 1,024 | 282.13 | **3,629.53** | 4,709.00 | 7,259.06 | 48,233.41 | 72,201.90 | 78,580.55 | **75.32** | 69.79 | 132.97 | 1,024 / 1,024 |
| **1024** | 2,048 | 566.81 | **3,613.21** | 4,590.00 | 7,226.43 | 153,437.39 | 213,962.97 | 222,100.51 | **76.79** | 70.37 | 133.88 | 2,048 / 2,048 |

---

### 4. Workload: `1k_512` (ISL = 1000, OSL = 512)
*Short generation / classification / extraction workload.*

| Concurrency ($C$) | Prompts | Duration (s) | Output TPS (tok/s) | Peak Output TPS | Total TPS (tok/s) | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Success |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **64** | 128 | 40.67 | **1,611.49** | 1,984.00 | 4,758.93 | 1,142.98 | 1,076.72 | 1,934.73 | **37.51** | 37.32 | 39.77 | 128 / 128 |
| **128** | 256 | 53.54 | **2,448.05** | 2,944.00 | 7,229.40 | 1,743.93 | 1,431.61 | 3,852.44 | **48.86** | 49.34 | 51.54 | 256 / 256 |
| **256** | 512 | 77.86 | **3,367.07** | 4,352.00 | 9,943.36 | 2,827.70 | 1,487.93 | 7,799.92 | **70.34** | 72.59 | 74.48 | 512 / 512 |
| **512** | 1,024 | 158.89 | **3,299.66** | 4,440.00 | 9,744.30 | 28,242.73 | 40,890.38 | 48,345.44 | **80.67** | 77.39 | 140.67 | 1,024 / 1,024 |
| **1024** | 2,048 | 317.62 | **3,301.33** | 4,694.00 | 9,749.24 | 86,632.10 | 90,315.21 | 128,099.41 | **81.99** | 77.18 | 140.23 | 2,048 / 2,048 |

---

## Key Performance Insights

1. **Peak Generation Throughput**:
   * Peak output token throughput reached **3,629.53 tok/s** at Concurrency 512 under the balanced `1k_1k` workload.
2. **Prefill Throughput**:
   * Total token throughput scaled up to **15,291.94 tok/s** under the prefill-heavy `8k_1k` workload.
3. **Low Concurrency Latencies (C=64 to 128)**:
   * Decoding TPOT remained between **~37.5 ms and 49.3 ms** across all 1k-input workloads with TTFT < 1.75 s.
4. **Saturation Dynamics**:
   * Throughput plateaued beyond Concurrency 256–512 as GPU execution cores and KV cache memory bandwidth became fully saturated.
   * Queueing delay caused TTFT to scale proportionally with batch queue depth at C=512 and C=1024.
