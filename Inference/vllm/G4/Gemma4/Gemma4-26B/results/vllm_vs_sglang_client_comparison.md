# vLLM Benchmark Client (`vllm bench serve`) vs. SGLang Benchmark Client (`sglang.bench_serving`)

**Serving Configuration (Identical for Both Runs)**
- **Serving Engine**: vLLM (`vllm/vllm-openai:gemma4`, `TP=1`, FP8 dynamic quantization, `--kv-cache-dtype fp8`, `--max-model-len 12288`)
- **Hardware**: 1x NVIDIA RTX PRO 6000 Blackwell Server Edition (GKE `g4-standard-48`, 96 GB VRAM) on `pm-g4-sglang-cluster`
- **Target Workload**: 10,240 Input Tokens (10K Context) / 500 Max Output Tokens

---

## 1. Side-by-Side Comparison across Concurrency Levels (1 to 512)

| Concurrency | Client Tool | Avg Input / Output Tokens per Prompt | Output Tok/s | Total Tok/s | Median TTFT (ms) | P99 TTFT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Median E2E Latency (s) | P99 E2E Latency (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **`vllm bench serve`** | **10,239 / 500** (Fixed) | 109.45 | 2,350.85 | 368.51 | 372.90 | 8.41 | 8.48 | 4.57 | 4.60 |
| **1** | **`sglang.bench_serving`** | ~5,795 / ~291 (Uniform `[0.5x, 1.0x]`) | 118.03 | 2,435.80 | 177.51 | 350.25 | 7.80 | 8.40 | 1.35 | 4.46 |
| **8** | **`vllm bench serve`** | **10,239 / 500** (Fixed) | 395.76 | 8,500.21 | 1,729.81 | 2,674.27 | 16.76 | 19.52 | 10.09 | 10.71 |
| **8** | **`sglang.bench_serving`** | ~5,427 / ~311 (Uniform `[0.5x, 1.0x]`) | 321.25 | 5,779.40 | 275.49 | 1,685.39 | 19.05 | 23.47 | 5.77 | 9.65 |
| **16** | **`vllm bench serve`** | **10,239 / 500** (Fixed) | 555.73 | 11,935.96 | 2,096.29 | 5,299.56 | 25.00 | 28.18 | 14.31 | 17.90 |
| **16** | **`sglang.bench_serving`** | ~5,665 / ~300 (Uniform `[0.5x, 1.0x]`) | 436.33 | 8,674.37 | 363.13 | 3,477.15 | 29.11 | 33.48 | 8.14 | 15.69 |
| **32** | **`vllm bench serve`** | **10,239 / 500** (Fixed) | 701.89 | 15,075.21 | 2,651.97 | 10,464.16 | 40.19 | 44.48 | 22.62 | 30.65 |
| **32** | **`sglang.bench_serving`** | ~5,792 / ~287 (Uniform `[0.5x, 1.0x]`) | 565.62 | 11,995.77 | 592.67 | 6,419.16 | 45.60 | 53.78 | 12.91 | 26.86 |
| **64** | **`vllm bench serve`** | **10,239 / 500** (Fixed) | 884.24 | 18,991.66 | 2,675.55 | 21,010.99 | 68.08 | 70.77 | 35.66 | 55.11 |
| **64** | **`sglang.bench_serving`** | ~5,791 / ~261 (Uniform `[0.5x, 1.0x]`) | 733.83 | 17,041.97 | 680.76 | 9,858.02 | 68.08 | 72.10 | 16.94 | 34.48 |
| **128** | **`vllm bench serve`** | **10,239 / 500** (Fixed) | 1,000.53 | 21,489.31 | 2,176.53 | 42,330.31 | 120.23 | 124.14 | 62.69 | 102.66 |
| **128** | **`sglang.bench_serving`** | ~5,747 / ~275 (Uniform `[0.5x, 1.0x]`) | 941.48 | 20,620.57 | 1,039.43 | 18,399.83 | 98.68 | 108.99 | 28.15 | 54.05 |
| **256** | **`vllm bench serve`** | **10,239 / 500** (Fixed) | 1,020.99 | 21,928.85 | 37,146.25 | 107,200.99 | 144.94 | 154.20 | 109.40 | 179.74 |
| **256** | **`sglang.bench_serving`** | ~5,772 / ~267 (Uniform `[0.5x, 1.0x]`) | 1,056.21 | 23,906.38 | 14,167.12 | 35,342.45 | 146.21 | 156.51 | 49.55 | 90.08 |
| **512** | **`vllm bench serve`** | **10,239 / 500** (Fixed) | 1,038.83 | 22,311.93 | 166,089.84 | 235,171.29 | 145.36 | 154.01 | 237.53 | 311.29 |
| **512** | **`sglang.bench_serving`** | ~5,751 / ~271 (Uniform `[0.5x, 1.0x]`) | 1,154.84 | 24,515.32 | 67,969.69 | 132,598.48 | 156.89 | 169.31 | 95.17 | 174.16 |

---

## 2. Root Cause of Differences Between the Two Benchmark Clients

### A. Prompt Length Sampling (`--random-range-ratio` Semantics)
The single biggest difference between `vllm bench serve` and `sglang.bench_serving` is how each CLI defines `--random-range-ratio`:
1. **`vllm bench serve` (`--random-range-ratio 0.0`)**:
   - Samples prompt lengths from `[L * (1 - r), L * (1 + r)]`.
   - With `r = 0.0` (the vLLM default), **every single request is strictly 10,239 input tokens and 500 output tokens** (`20.48 : 1` prefill-to-decode ratio).
   - Total tokens processed per request = **10,739 tokens**.
2. **`sglang.bench_serving` (`--random-range-ratio 1`)**:
   - In SGLang's `bench_serving.py`, `--random-range-ratio` defines `min_len / max_len`, but SGLang clamps the sampling range to `[0.5 * L, 1.0 * L]` when generating random datasets unless modified.
   - As a result, `sglang.bench_serving` sent requests averaging **~5,770 input tokens** (range `5,120`–`10,240`) and **~280 output tokens** (range `250`–`500`).
   - Total tokens processed per request = **~6,050 tokens** (~44% fewer tokens per request than the fixed 10K/500 vLLM client run).

### B. Why `vllm bench serve` Achieves Higher Throughput at Medium Concurrency (`C=8` to `C=128`)
- At **C=8 through C=128**, **`vllm bench serve` achieves higher Total Token Throughput (`8,500`–`21,489` tok/s vs. `5,779`–`20,620` tok/s) and higher Output Token Throughput (`395`–`1,000` tok/s vs. `321`–`941` tok/s)**:
  - Because every request in `vllm bench serve` generates a full 500-token decode sequence (`--ignore-eos`, fixed 500 OSL), requests stay in the continuous batching decode loop longer without constantly terminating early at 250–300 tokens.
  - Longer decode sequences amortize the initial prefill cost across more decode steps, allowing the RTX PRO 6000 GPU's FP8 tensor cores to stay saturated with larger decode batches.

### C. Why `sglang.bench_serving` Reports Lower TTFT and E2E Latency
- **Time to First Token (TTFT)**:
  - Prefill latency scales roughly linearly with input sequence length. Because `vllm bench serve` sends strictly **10,239 input tokens** per request while `sglang.bench_serving` averages **~5,770 input tokens**, the prefill time per request is ~1.8x longer in `vllm bench serve` (`368 ms` vs `177 ms` at `C=1`).
  - At high concurrency (`C=256`, `C=512`), where requests queue behind active prefills (`max-num-batched-tokens=12288`), queueing 10,239-token prefills creates ~2.4x more prefill queue wait time than queueing 5,770-token prefills (`166 s` vs `68 s` median TTFT at `C=512`).
- **End-to-End Latency (E2EL)**:
  - In `vllm bench serve`, every request decodes **500 tokens** instead of **~280 tokens**. Even at identical TPOT (`~68 ms` at `C=64`), decoding 500 tokens takes `34 s` (`500 * 68 ms`) plus `2.67 s` prefill = **`35.66 s`**, whereas decoding 261 tokens takes `17.7 s` (`261 * 68 ms`) = **`16.94 s`**.

---

## 3. Summary Recommendations

| Use Case | Recommended Client & Flags | Reason |
| :--- | :--- | :--- |
| **Strict Fixed-Length Hardware Stress Testing** (`10K Input / 500 Output`) | `vllm bench serve --dataset-name random --random-input-len 10240 --random-output-len 500 --random-range-ratio 0.0 --ignore-eos` | Guarantees zero variance in prompt lengths; measures true worst-case KV cache pressure and sustained prefill/decode saturation at exact target context lengths. |
| **Cross-Engine / Upstream Baseline Comparison** (matching `shivajid/sglang-rtx-pro-6000`) | `python3 -m sglang.bench_serving --dataset-name random --random-input-len 10240 --random-output-len 500 --random-range-ratio 1 --ignore-eos` | Replicates the exact request-length distribution (`[0.5x, 1.0x]`) used in published SGLang benchmark reports. |
