# Kimi-K2.6 on GCP G4

Optimized configurations and benchmarks for [Moonshot AI's Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) on GCP G4 GKE instances using SGLang. For H100 configurations, see the H100 tests directory.

## Model Overview
Kimi-K2.6 is the latest iteration of the large-scale Mixture-of-Experts (MoE) model, optimized for advanced reasoning and tool-use capabilities.

## Serving Configuration

### 1-Node Setup (Optimized for G4 with Speculative Decoding)
- **Model**: `moonshotai/Kimi-K2.6`
- **Tensor Parallelism**: 8
- **KV Cache**: FP8 (e4m3)
- **Speculative Algorithm**: EAGLE3 (`lightseekorg/kimi-k2.5-eagle3`)
- **Serving Image**: `lmsysorg/sglang:dev-cu13`
- **Key Features**: Hierarchical cache enabled, mixed-chunk scheduling, and specialized Kimi-K2 parsers.

### 2-Node Setup (G4 Distributed)
- **Model**: `moonshotai/Kimi-K2.6`
- **Tensor Parallelism**: 8
- **Pipeline Parallelism**: 2
- **KV Cache**: INT4
- **Serving Image**: `lmsysorg/sglang:v0.5.10.post1`

## Benchmark Results
The following benchmarks were conducted using the high-concurrency (Max 512) sustained load benchmark.

| Metric | G4 Cluster (1-Node Speculative) | H100 Cluster (2-Node Distributed - Optimized) | Improvement |
| :--- | :---: | :---: | :---: |
| **Model Precision** | FP4 Quantized | **BF16 Full Weights** | *Higher Quality* |
| **Request Throughput** | 0.35 req/s | **0.77 req/s** | **2.2x** |
| **Output Token Throughput** | 1,459.26 tok/s | **3,216.78 tok/s** | **2.2x** |
| **Peak Output Throughput** | 850.00 tok/s | **4,554.00 tok/s** | **5.3x** |
| **Total Token Throughput** | 1,637.28 tok/s | **3,609.18 tok/s** | **2.2x** |
| **Median TTFT (First Token)** | 1,087.88 s | **218.43 s** | **5.0x** |
| **Median End-to-End Latency** | 1,274.65 s | **582.91 s** | **2.2x** |
| **Mean TPOT** | **82.43 ms** | 105.86 ms | - |

> **Note**: The high TTFT and E2E latencies are reflective of the extreme 512-concurrency load testing (1,536 total prompts).

### Raw H100 Serving Benchmark Output
```
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf
Max request concurrency:                 512
Successful requests:                     1536
Benchmark duration (s):                  2000.41
Total input tokens:                      784969
Total input text tokens:                 784969
Total generated tokens:                  6434886
Total generated tokens (retokenized):    6429763
Request throughput (req/s):              0.77
Input token throughput (tok/s):          392.40
Output token throughput (tok/s):         3216.78
Peak output token throughput (tok/s):    4554.00
Peak concurrent requests:                517
Total token throughput (tok/s):          3609.18
Concurrency:                             450.87
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   587187.91
Median E2E Latency (ms):                 582905.56
P90 E2E Latency (ms):                    869020.99
P99 E2E Latency (ms):                    1072702.10
---------------Time to First Token----------------
Mean TTFT (ms):                          177355.31
Median TTFT (ms):                        218431.30
P99 TTFT (ms):                           418795.92
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          105.86
Median TPOT (ms):                        95.82
P99 TPOT (ms):                           191.74
---------------Inter-Token Latency----------------
Mean ITL (ms):                           97.95
Median ITL (ms):                         89.54
P95 ITL (ms):                            152.78
P99 ITL (ms):                            161.23
Max ITL (ms):                            743.46
==================================================
```

## Usage

### 1-Node G4 Deployment
To deploy the single-node optimized setup:
```bash
kubectl apply -f sglang-kimi-26-1node.yaml
```

### 2-Node G4 Deployment
To deploy the distributed 2-node setup on G4:
```bash
kubectl apply -f sglang-kimi-26-2nd.yaml
```

### Run Benchmark Client
```bash
kubectl apply -f benchmarking_scripts/benchmark-kimik26.yaml
```

