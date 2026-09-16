# SGLang Benchmark Report: `google/gemma-4-26B-A4B` on 2x NVIDIA H100 GPUs (TP=2, 10K Input / 500 Output)

Comprehensive performance sweep across concurrency levels **1 to 512** for **`google/gemma-4-26B-A4B`** served via **vLLM** on **2x NVIDIA H100-SXM5-80GB GPUs (`TP=2`)** from a single H100 node (`pallaviam-h100-tcpx-pool`, GKE cluster `pallaviam-gke-h100-tcpx-cluster`, `us-east5-a`).

- **Hardware**: 2x NVIDIA H100-SXM5-80GB (160 GB total VRAM, NVLink P2P intra-node, `TP=2`)
- **Model**: `google/gemma-4-26B-A4B` (`--quantization fp8 --kv-cache-dtype fp8_e4m3 --attention-config '{"backend": "TRITON_ATTN"}'`)
- **Input Prompt Length (ISL)**: **10,240 tokens** (`--random-input-len 10240 --random-range-ratio 1.0`)
- **Output Length (OSL)**: **500 tokens** (`--random-output-len 500 --random-range-ratio 1.0 --ignore-eos`)
- **Backend**: vLLM OpenAI API Server (`vllm-gemma4-1node-2gpu-0`)
- **Client**: `sglang.bench_serving --backend vllm` executed from `benchmark-client-pool-tcpx`
- **Raw JSON Outputs**: Stored in `results/result_c{1,8,16,32,64,128,256,512}.json`

---

## 1. Concurrency Sweep Summary Table (Median & P99 Metrics)

All figures below are parsed directly from the raw JSON output files (`result_c*.json`) produced on `pallaviam-gke-h100-tcpx-cluster`.

| Concurrency | Completed Reqs | Output Tok/s | Input Tok/s | Total Tok/s | Req/s | Median TTFT (ms) | P99 TTFT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Median ITL (ms) | P99 ITL (ms) | Median E2E (s) | P99 E2E (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 8 | **18.88** | 386.65 | 405.53 | 0.04 | **267.47** | 531.46 | **52.39** | 53.19 | **52.19** | 57.80 | **26.41** | 27.07 |
| **8** | 16 | **135.90** | 2,782.91 | 2,918.81 | 0.27 | **961.33** | 2,703.29 | **55.56** | 57.90 | **53.08** | 59.50 | **28.89** | 30.32 |
| **16** | 32 | **261.46** | 5,354.13 | 5,615.58 | 0.52 | **779.69** | 3,346.99 | **59.47** | 60.97 | **53.74** | 61.26 | **30.34** | 30.96 |
| **32** | 64 | **505.75** | 10,356.80 | 10,862.56 | 1.01 | **1,223.96** | 6,182.36 | **56.60** | 66.50 | **54.73** | 331.10 | **31.58** | 34.16 |
| **64** | 128 | **921.04** | 18,860.99 | 19,782.03 | 1.84 | **1,298.26** | 12,646.08 | **56.80** | 77.64 | **54.07** | 368.68 | **34.28** | 39.83 |
| **128** ⚡ | 256 | **1,533.44** | 31,401.72 | 32,935.16 | 3.07 | **1,759.33** | 20,239.88 | **64.78** | 101.39 | **54.18** | 375.93 | **41.71** | 51.86 |
| **256** 🏆 | 512 | **2,223.66** | 45,536.10 | 47,759.76 | 4.45 | **3,848.03** | 43,243.34 | **78.50** | 152.64 | **54.80** | 386.87 | **56.80** | 78.12 |
| **512** | 1,024 | **1,765.25** | 36,148.79 | 37,914.04 | 3.53 | **5,629.38** | 118,099.70 | **262.00** | 274.67 | **59.18** | 416.94 | **136.02** | 253.17 |

---

## 2. Detailed Performance Comparison: Mean vs. Median Metrics

| Concurrency | Output Tok/s | Total Tok/s | Req/s | Mean TTFT (ms) | Median TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | Mean ITL (ms) | Median ITL (ms) | Mean E2E (s) | Median E2E (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 18.88 | 405.53 | 0.04 | 300.74 | **267.47** | 52.46 | **52.39** | 52.49 | **52.19** | 26.48 | **26.41** |
| **8** | 135.90 | 2,918.81 | 0.27 | 1,110.78 | **961.33** | 56.29 | **55.56** | 56.29 | **53.08** | 29.01 | **28.89** |
| **16** | 261.46 | 5,615.58 | 0.52 | 1,266.60 | **779.69** | 58.52 | **59.47** | 58.55 | **53.74** | 30.39 | **30.34** |
| **32** | 505.75 | 10,862.56 | 1.01 | 2,246.97 | **1,223.96** | 58.36 | **56.60** | 58.69 | **54.73** | 31.37 | **31.58** |
| **64** | 921.04 | 19,782.03 | 1.84 | 3,710.70 | **1,298.26** | 61.10 | **56.80** | 61.50 | **54.07** | 34.23 | **34.28** |
| **128** | 1,533.44 | 32,935.16 | 3.07 | 6,086.19 | **1,759.33** | 69.33 | **64.78** | 69.81 | **54.18** | 40.76 | **41.71** |
| **256** | 2,223.66 | 47,759.76 | 4.45 | 11,846.74 | **3,848.03** | 87.24 | **78.50** | 87.84 | **54.80** | 55.43 | **56.80** |
| **512** | 1,765.25 | 37,914.04 | 3.53 | 31,709.38 | **5,629.38** | 217.47 | **262.00** | 218.19 | **59.18** | 140.23 | **136.02** |

---

## 3. Side-by-Side Comparison: 2x H100 GPUs (`TP=2`) vs. 1x G4 GPU (`TP=1`) on `10K / 500`

| Concurrency | 2x H100 (`TP=2`) Output Tok/s | 1x G4 (`TP=1`) Output Tok/s | 2x H100 (`TP=2`) Total Tok/s | 1x G4 (`TP=1`) Total Tok/s | 2x H100 Median TTFT (ms) | 1x G4 Median TTFT (ms) | 2x H100 Median ITL (ms) | 1x G4 Median ITL (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 18.88 | 118.03 | 405.53 | 2,435.80 | 267.47 | 177.51 | 52.19 | 7.86 |
| **8** | 135.90 | 321.25 | 2,918.81 | 5,779.40 | 961.33 | 275.49 | 53.08 | 16.87 |
| **16** | 261.46 | 436.33 | 5,615.58 | 8,674.37 | 779.69 | 363.13 | 53.74 | 22.40 |
| **32** | 505.75 | 565.62 | 10,862.56 | 11,995.77 | 1,223.96 | 592.67 | 54.73 | 30.79 |
| **64** | **921.04** | 733.83 | **19,782.03** | 17,041.97 | 1,298.26 | 680.76 | 54.07 | 37.59 |
| **128** | **1,533.44** | 941.48 | **32,935.16** | 20,620.57 | 1,759.33 | 1,039.43 | 54.18 | 47.43 |
| **256** | **2,223.66** | 1,056.21 | **47,759.76** | 23,906.38 | **3,848.03** | 14,167.12 | **54.80** | 56.42 |
| **512** | **1,765.25** | 1,154.84 | **37,914.04** | 24,515.32 | **5,629.38** | 67,969.69 | **59.18** | 57.93 |

---

## 4. Key Architectural & Sizing Insights

1. **Peak Throughput Achieved at Concurrency 256 (`2,223.66 tok/s` Output, `47,759.76 tok/s` Total)**:
   - On 2x H100 GPUs (`TP=2`), peak output throughput reaches **2,223.66 tok/s** (**2.11x higher** than the 1x G4 GPU peak of `1,056.21 tok/s` at C=256) and total token throughput reaches **47,759.76 tok/s** (**2.00x higher** than G4's `23,906.38 tok/s`).
2. **Dramatically Lower TTFT at High Concurrency (`C=256` & `C=512`)**:
   - At `C=256`, 2x H100 GPUs (`TP=2`) deliver a **3.85s (`3,848.03 ms`) Median TTFT** compared to **14.17s (`14,167.12 ms`)** on 1x G4 GPU (**3.68x faster** prefill turnaround).
   - At `C=512`, 2x H100 GPUs (`TP=2`) maintain a **5.63s (`5,629.38 ms`) Median TTFT** compared to **67.97s (`67,969.69 ms`)** on 1x G4 GPU (**12.07x faster** prefill turnaround), thanks to 2x HBM3 memory bandwidth and distributed KV cache capacity (`160 GB` total VRAM).
3. **Host TCPXO Plugin Isolation for Single-Node 2-GPU Pods**:
   - On GKE H100 nodes with `nccl-tcpxo-installer`, isolating the container's `/usr/local/nvidia/lib64` without `libnccl-net.so` / `libnccl-tuner.so` (`NCCL_NET="Socket"`, `NCCL_TUNER_PLUGIN="none"`) enables direct NVLink P2P (`P2P/CUMEM`) between the 2 intra-node H100 GPUs without requiring `rxdm`.
