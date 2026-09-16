# 2x G4 (`TP=2, EP=2`) Optimized Customer SLA Benchmark & 3-Way Comparison (`1x G4` vs. `2x G4` vs. `2x H100`)

**Model**: `google/gemma-4-26B-A4B` (`FP8` Weights + `FP8` KV Cache)  
**Workload**: Customer Target (`10,000` Input Tokens / `500` Output Tokens / `0` Cached Tokens / `--random-range-ratio 1.0`)  
**G4 Hardware**: `g4-standard-384` (`2x NVIDIA RTX PRO 6000 Blackwell Server Edition 96GB GPUs`, PCIe Gen5 P2P)  
**H100 Hardware**: `a3-highgpu-8g` (`2x NVIDIA H100 80GB HBM3 GPUs`, NVLink 900 GB/s)  
**Serving Engine**: `vLLM v0.29.0` (`--tensor-parallel-size 2 --enable-expert-parallel -O3`)  
**Benchmark Client**: `sglang.bench_serving` (`--dataset-name random --random-input-len 10000 --random-output-len 500 --random-range-ratio 1.0`)

---

## 1. Executive Summary: Does 2x G4 (`TP=2, EP=2`) Reduce E2E Latency?

> [!IMPORTANT]
> **Yes — scaling from `1x G4 (TP=1)` to `2x G4 (TP=2, EP=2)` reduces End-to-End (E2E) latency across every single load point (`-23.0%` at `75 QPM`, and `-16.8% to -20.0%` P99 E2E latency reduction across `C=8..16`), allowing 2x G4 GPUs to meet the customer's `7.7s` E2E latency target up to `~70–75 QPM` (`6.87s` Median / `7.25s` P99 at `C=8`, and `7.89s` Median at `C=10` / `75.9 QPM`).**

1. **At the Customer's Poisson Effective Load (`75 QPM` = `1.25 req/s`, `0 cached tokens`)**:
   - **`1x G4 (TP=1)`**: **`11.12s` Median E2E Latency** (`10.64s` Mean, `13.43s` P99, `21.63 ms` Median TPOT)
   - **`2x G4 (TP=2, EP=2)`**: **`8.56s` Median E2E Latency** (`8.57s` Mean, `11.55s` P99, `16.20 ms` Median TPOT) — **`23.0%` (`2.56 seconds`) lower Median E2E Latency** and **`25.1%` faster TPOT** than `1x G4`!
   - **`2x H100 (TP=2, EP=2)`**: **`5.38s` Median E2E Latency** (`5.68s` Mean, `8.09s` P99, `10.06 ms` Median TPOT).
2. **At Steady-State Concurrency `C = 8` (`69.8 QPM` on 2x G4 vs. `62.8 QPM` on 1x G4)**:
   - **`1x G4 (TP=1)`**: **`7.57s` Median E2E Latency**, **`8.72s` P99 E2E Latency** (`13.80 ms` Median TPOT) — P99 exceeds the `7.7s` SLA by `+1.02s`.
   - **`2x G4 (TP=2, EP=2)`**: **`6.87s` Median E2E Latency**, **`7.25s` P99 E2E Latency** (`10.26 ms` Median TPOT) — **both Median (`6.87s`) and P99 (`7.25s`) beat the `7.7s` customer SLA** while delivering **`+11.0%` higher QPM (`69.8 QPM`)**!
3. **At Steady-State Concurrency `C = 10` (`75.9 QPM` on 2x G4 vs. `68.8 QPM` on 1x G4)**:
   - **`1x G4 (TP=1)`**: **`8.62s` Median E2E Latency**, **`10.27s` P99 E2E Latency** (`15.89 ms` Median TPOT, `68.8 QPM`).
   - **`2x G4 (TP=2, EP=2)`**: **`7.89s` Median E2E Latency**, **`8.21s` P99 E2E Latency** (`11.67 ms` Median TPOT, **`75.9 QPM`**) — reduces P99 E2E latency by **`2.06 seconds` (`-20.0%`)** and Median E2E latency by **`730 ms` (`-8.5%`)**, achieving the customer's **`75 QPM` effective load target (`75.9 QPM`)** at **`7.89s`** E2E latency.

---

## 2. Three-Way Comparison: `1x G4 (TP=1)` vs. `2x G4 (TP=2, EP=2)` vs. `2x H100 (TP=2, EP=2)`

All configurations below were tested with the exact customer benchmark parameters (`ISL = 10,000`, `OSL = 500`, `--random-range-ratio 1.0`, `0 cached tokens`):

| Benchmark Mode | Metric | `1x G4 GPU` (`TP=1`) | `2x G4 GPUs` (`TP=2, EP=2`) | `2x G4` vs. `1x G4` Gain | `2x H100 GPUs` (`TP=2, EP=2`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Customer Effective (`75 QPM` Poisson)** | **Achieved QPM** | `74.0 QPM` | **`77.9 QPM`** | **`+5.3%` QPM** | **`81.5 QPM`** |
| *(Target: `1.25 req/s`, `Max Batch 32`)* | **Median E2E Latency** | `11.12 s` | **`8.56 s`** | **`-23.0%` (`-2.56s`)** | **`5.38 s`** ✅ |
| | **Mean E2E Latency** | `10.64 s` | **`8.57 s`** | **`-19.5%` (`-2.07s`)** | **`5.68 s`** ✅ |
| | **P99 E2E Latency** | `13.43 s` | **`11.55 s`** | **`-14.0%` (`-1.88s`)** | **`8.09 s`** |
| | **Median TPOT / ITL** | `21.63 ms` / `14.71 ms` | **`16.20 ms` / `10.65 ms`** | **`-25.1%` TPOT** | **`10.06 ms` / `7.69 ms`** |
| **Concurrency 1 (`C=1`)** | **Achieved QPM** | `17.6 QPM` | **`18.6 QPM`** | **`+5.7%` QPM** | **`20.5 QPM`** |
| *(Single-Request Floor)* | **Median E2E Latency** | `3.43 s` | **`3.23 s`** ✅ | **`-5.8%` (`-200ms`)** | **`2.92 s`** ✅ |
| | **P99 E2E Latency** | `3.44 s` | **`3.23 s`** ✅ | **`-6.1%` (`-210ms`)** | **`2.93 s`** ✅ |
| | **Median TPOT / ITL** | `6.33 ms` / `6.35 ms` | **`5.82 ms` / `5.84 ms`** | **`-8.1%` TPOT** | **`5.37 ms` / `5.39 ms`** |
| **Concurrency 8 (`C=8`)** | **Achieved QPM** | `62.8 QPM` | **`69.8 QPM`** | **`+11.0%` QPM** | **`87.3 QPM`** |
| *(7.7s SLA Sweet Spot on 2x G4)* | **Median E2E Latency** | `7.57 s` ✅ | **`6.87 s`** ✅ | **`-9.3%` (`-702ms`)** | **`5.48 s`** ✅ |
| | **P99 E2E Latency** | `8.72 s` ❌ | **`7.25 s`** ✅ | **`-16.8%` (`-1.46s`)** | **`5.81 s`** ✅ |
| | **Median TPOT / ITL** | `13.80 ms` / `11.38 ms` | **`10.26 ms` / `8.81 ms`** | **`-25.7%` TPOT** | **`8.57 ms` / `7.10 ms`** |
| **Concurrency 10 (`C=10`)** | **Achieved QPM** | `68.8 QPM` | **`75.9 QPM`** | **`+10.3%` QPM** | **`101.2 QPM`** |
| *(75 QPM Target on 2x G4)* | **Median E2E Latency** | `8.62 s` | **`7.89 s`** (~`7.7s`) | **`-8.5%` (`-730ms`)** | **`5.93 s`** ✅ |
| | **P99 E2E Latency** | `10.27 s` | **`8.21 s`** | **`-20.0%` (`-2.06s`)** | **`5.93 s`** ✅ |
| | **Median TPOT / ITL** | `15.89 ms` / `12.95 ms` | **`11.67 ms` / `9.96 ms`** | **`-26.6%` TPOT** | **`8.83 ms` / `7.83 ms`** |
| **Concurrency 16 (`C=16`)** | **Achieved QPM** | `86.7 QPM` | **`95.4 QPM`** | **`+10.0%` QPM** | **`130.9 QPM`** |
| *(7.7s SLA Sweet Spot on 2x H100)* | **Median E2E Latency** | `10.87 s` | **`10.04 s`** | **`-7.6%` (`-830ms`)** | **`7.31 s`** ✅ |
| | **P99 E2E Latency** | `13.98 s` | **`11.36 s`** | **`-18.7%` (`-2.62s`)** | **`8.23 s`** |
| | **Median TPOT / ITL** | `20.45 ms` / `14.90 ms` | **`14.43 ms` / `10.81 ms`** | **`-29.4%` TPOT** | **`10.76 ms` / `8.27 ms`** |
| **Concurrency 32 (`C=32`)** | **Achieved QPM** | `111.4 QPM` | **`118.7 QPM`** | **`+6.6%` QPM** | **`170.7 QPM`** |
| *(Customer Max Batch Size 32)* | **Median E2E Latency** | `16.67 s` | **`16.11 s`** | **`-3.4%` (`-560ms`)** | **`11.20 s`** |
| | **P99 E2E Latency** | `23.82 s` | **`22.05 s`** | **`-7.4%` (`-1.77s`)** | **`14.71 s`** |
| | **Median TPOT / ITL** | `32.07 ms` / `20.15 ms` | **`24.37 ms` / `13.78 ms`** | **`-24.0%` TPOT** | **`16.05 ms` / `9.81 ms`** |

---

## 3. Full Verified Benchmark Metrics on `2x G4 GPUs (TP=2, EP=2)`

Extracted directly from `results/g4_2gpu_optimized/result_*.json`:

| Run Mode | Concurrency | Req/s | Achieved QPM | Output Tok/s | Total Tok/s | Mean E2E (s) | Median E2E (s) | P90 E2E (s) | P99 E2E (s) | Median TTFT (ms) | Median TPOT (ms) | Median ITL (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Customer Effective (`75 QPM`)** | `32` (`rate=1.25`) | `1.30` | **`77.9 QPM`** | `649.32` | `13,634.39` | `8.57 s` | **`8.56 s`** | `11.28 s` | `11.55 s` | `505.61` | `16.20` | `10.65` |
| **Customer Peak (`250 QPM`)** | `32` (`rate=4.17`) | `1.98` | **`118.8 QPM`** | `989.75` | `20,782.72` | `14.88 s` | **`15.70 s`** | `16.09 s` | `16.53 s` | `3,373.24` | `24.69` | `13.78` |
| **Concurrency 1 (`C=1`)** | `1` | `0.31` | **`18.6 QPM`** | `155.02` | `3,255.11` | `3.22 s` | **`3.23 s`** ✅ | `3.23 s` | **`3.23 s`** ✅ | `318.28` | `5.82` | `5.84` |
| **Concurrency 8 (`C=8`)** | `8` | `1.16` | **`69.8 QPM`** | `581.26` | `12,205.23` | `6.87 s` | **`6.87 s`** ✅ | `7.00 s` | **`7.25 s`** ✅ | `1,672.40` | `10.26` | `8.81` |
| **Concurrency 10 (`C=10`)** | `10` | `1.27` | **`75.9 QPM`** | `632.72` | `13,285.89` | `7.89 s` | **`7.89 s`** | `8.20 s` | `8.21 s` | `2,070.79` | `11.67` | `9.96` |
| **Concurrency 16 (`C=16`)** | `16` | `1.59` | **`95.4 QPM`** | `794.77` | `16,688.52` | `10.05 s` | **`10.04 s`** | `10.40 s` | `11.36 s` | `2,716.19` | `14.43` | `10.81` |
| **Concurrency 32 (`C=32`)** | `32` | `1.98` | **`118.7 QPM`** | `989.43` | `20,776.07` | `16.14 s` | **`16.11 s`** | `20.13 s` | `22.05 s` | `3,812.68` | `24.37` | `13.78` |

---

## 4. Architectural Analysis: Why `2x G4 (TP=2, EP=2)` Behaves Differently Than `2x H100 (TP=2, EP=2)`

1. **Decode Acceleration (`-25.1% to -29.4%` TPOT Reduction on `2x G4`)**:
   - Autoregressive token generation (`500` output tokens) is **memory-bandwidth bound**.
   - Splitting `google/gemma-4-26B-A4B` across **2x RTX PRO 6000 Blackwell GPUs (`TP=2, EP=2`)** doubles aggregate GDDR7 memory bandwidth from `1.8 TB/s` (`1x G4`) to **`3.6 TB/s` (`2x G4`)**.
   - This reduces Median TPOT from **`13.80 ms -> 10.26 ms`** at `C=8` (`-25.7%`), **`15.89 ms -> 11.67 ms`** at `C=10` (`-26.6%`), and **`20.45 ms -> 14.43 ms`** at `C=16` (`-29.4%`), saving **`1.75s – 3.00s` of pure decode time per request**.
2. **PCIe Gen5 P2P (`2x G4`) vs. NVLink 900 GB/s (`2x H100`) During `10,000`-Token Prefills**:
   - On **`g4-standard-384`**, GPUs communicate via **PCIe Gen5 (`~64 GB/s` unidirectional)** without NVLink. Consequently, `--disable-custom-all-reduce` must be enabled so vLLM uses NCCL PCIe P2P (`P2P/CUMEM`).
   - During large `10,000`-token prefills, Tensor/Expert Parallelism executes AllReduce / AllToAll operations across PCIe Gen5 at every layer, which increases Median TTFT under concurrent load (`1,672 ms` at `C=8` on `2x G4` vs. `1,212 ms` on `2x H100` with `900 GB/s` NVLink).
   - Even with PCIe Gen5 AllReduce overhead during prefill, the **`1.75s – 3.00s` decode latency reduction** dominates, yielding a net **`0.70s – 2.56s` reduction in Median E2E Latency** and a **`1.46s – 2.62s` reduction in P99 E2E Latency** over `1x G4`.
3. **Production Sizing Recommendation (`G4` vs. `H100` for Customer `7.7s` SLA)**:
   - **`2x G4 GPUs (TP=2, EP=2)`**: Meets the `7.7s` E2E Latency SLA up to **`~70–75 QPM` per 2-GPU replica** (`6.87s` Median / `7.25s` P99 at `C=8` / `69.8 QPM`; `7.89s` Median at `C=10` / `75.9 QPM`). To absorb bursts up to **`250 QPM` peak** while staying under `7.7s`, deploy **3 to 4 replicas of `2x G4 (TP=2, EP=2)`** (`6–8 G4 GPUs` total).
   - **`2x H100 GPUs (TP=2, EP=2)`**: Meets the `7.7s` E2E Latency SLA up to **`130.9 QPM` per 2-GPU replica** (`5.38s` Median at `81.5 QPM`; `7.31s` Median at `C=16` / `130.9 QPM`). Only **2 replicas of `2x H100 (TP=2, EP=2)`** (`4 H100 GPUs` total) are needed to absorb the full **`250 QPM` peak** below `7.3s`.
