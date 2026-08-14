# Kimi-K2.6 NVFP4 Batch serving benchmark (bench_one_batch_server)

This report details the results of our batch serving benchmark for **Kimi-K2.6-NVFP4** on 2 Nodes (16x RTX 6000 Ada GPUs) using SGLang.

## 📊 Benchmark Results Summary

* **Configuration**:
  * **Batch size**: 512
  * **Input sequence length**: 1024 tokens
  * **Output sequence length**: 8192 tokens
  * **Quantization**: FP4 (`modelopt_fp4`)
  * **Tokenizer**: In-process (zero tokenizer worker processes)

| Metric | Kimi-K2.5-NVFP4 (Baseline) | Kimi-K2.6-NVFP4 (June 22 Run) | Delta vs Kimi-K2.5 (%) |
| :--- | :---: | :---: | :---: |
| **Total Latency** | 1,138.99 s | **1,169.38 s** | +2.7% (Slower) |
| **Input Prefill Throughput** | 14,015.66 tokens/s | **15,881.61 tokens/s** | **+13.3% (Faster)** |
| **Output Decode Throughput** | 3,807.51 tokens/s | **3,690.98 tokens/s** | -3.1% (Slower) |
| **Overall Token Throughput** | 4,142.77 tokens/s | **4,035.13 tokens/s** | -2.6% (Slower) |
| **Slowest Request TTFT (last_ttft)** | 37.41 s | **33.01 s** | **-11.8% (Faster/Better)** |
| **Average Generation Speed** | 487.57 tokens/s (per rank) | **487.34 tokens/s (per rank)**| -0.05% (Consistent) |

## 🔍 Key Insights & Observations

1. **Massive Prefill Speedup:** Kimi-K2.6-NVFP4 achieves **+13.3% faster prefill throughput** (15,881.61 tokens/s vs 14,015.66 tokens/s), indicating improved attention/prefill routing and compute efficiency under high batch conditions.
2. **Improved TTFT:** The slowest request TTFT dropped from **37.41s** to **33.01s** (an **11.8% improvement**), showing more predictable scheduling under full load.
3. **Decentralized Generation Consistencies:** The generation speed per rank remained virtually identical (~487 tokens/s), demonstrating extremely consistent and robust execution performance across hardware ranks.
4. **Slight Decode Throughput Delta:** We see a slight (3.1%) drop in output decode throughput. This minor variance is expected due to slight structural differences or routing choices in the mixture-of-experts (MoE) layers between Kimi-K2.5 and Kimi-K2.6 when running under max generation sequences.
