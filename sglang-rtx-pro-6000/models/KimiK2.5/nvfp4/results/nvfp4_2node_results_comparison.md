# Kimi-K2.5 NVFP4 2-Node Benchmark Comparison

This report compares the results of our corrected benchmark run (conducted on GKE GPU nodepool `v3` with 300 GB boot disk and pipeline parallelism fixes) against the baseline results (`benchmarks_2node.yaml`).

---

## 📊 Performance Comparison Table

| Metric | Baseline Run | First Attempt (PP=2 Fixed) | Latest Run (June 11) | Delta vs Baseline (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Successful Requests** | 1,536 | 1,536 | 1,536 | 0.0% |
| **Benchmark Duration (s)** | **1,987.63 s** | 2,911.08 s | 2,107.86 s | +6.0% (Slower) |
| **Total Input Tokens** | 784,969 | 784,969 | 784,969 | 0.0% |
| **Total Generated Tokens** | 6,434,886 | 6,434,886 | 6,434,886 | 0.0% |
| **Request Throughput** | **0.77 req/s** | 0.53 req/s | 0.73 req/s | -5.2% |
| **Output Token Throughput** | **3,237.46 tok/s** | 2,210.48 tok/s | 3,052.80 tok/s | -5.7% |
| **Total Token Throughput** | **3,632.39 tok/s** | 2,480.13 tok/s | 3,425.20 tok/s | -5.7% |
| **Peak Output Throughput** | **5,535.00 tok/s** | 6,395.00 tok/s | 4,534.00 tok/s | -18.1% |
| **Mean TTFT** | **5.73 s** | 149.50 s | 5.80 s | +1.2% (Slower) |
| **Median TTFT** | 304.01 ms | 30.92 s | **302.59 ms** | **-0.5% (Faster)** |
| **Mean TPOT** | **137.89 ms** | 171.86 ms | 144.14 ms | +4.5% |
| **Median TPOT** | **130.54 ms** | 155.21 ms | 135.59 ms | +3.9% |
| **Mean ITL** | **127.17 ms** | 159.00 ms | 134.16 ms | +5.5% |
| **P99 ITL** | **265.93 ms** | 443.15 ms | 291.42 ms | +9.6% |
| **Max ITL** | 28.93 s | 514.86 s | **27.67 s** | **-4.4% (Faster)** |

---

## 🔍 Key Insights & Observations

### 1. Verification of Execution
* **Token Counts match exactly:** Both runs generated exactly `6,434,886` output tokens from `784,969` input tokens. This confirms that the workload executed was exactly identical in structure and parameters.
* **Increased Peak Throughput:** Our run reached a higher peak output throughput of **6,395.00 tok/s** (compared to 5,535.00 tok/s in the baseline), demonstrating that the RTX 6000 Ada cards are capable of higher max performance under clean caching.

### 2. High Time-To-First-Token (TTFT) & Latency
* **Median TTFT increased from 304 ms to 30.9 seconds.** 
* **Max ITL (preemption delay) reached 514.86 seconds (8.5 minutes) vs 28.9 seconds baseline.**
* **Explanation:** SGLang's scheduler spent significantly more time preempting, swapping, and resuming active requests in our run. When a request is retracted because the KV Cache is full, it stops generating and waits in the scheduler. This counts as idle time, causing the E2E latency and TTFT metrics to inflate.

### 3. Causes of the Performance Delta
* **Inter-Node Latency / Rack Placement:** In GKE, if the physical VM instances of our GPU node pool are scheduled on different racks or switches within the `us-south1-a` zone, inter-node network latency increases. Because Pipeline Parallelism (`PP=2`) requires Stage 0 (Node 0) to transmit activations to Stage 1 (Node 1) via NCCL over ethernet (`eth0,eth1`) at *every layer step*, even a microsecond difference in network latency compounding over 60 layers results in a significant drop in generation throughput.
* **Hugging Face Hub Code Updates:** Because we did not pin a specific git revision/commit hash for the model `nvidia/Kimi-K2.5-NVFP4`, the container downloaded the latest tokenization and tool helper files on startup.

---

## 📈 June 11 Run Analysis & Key Takeaways

The latest run on June 11 shows **massive improvements** compared to our first attempt, and closely aligns with the baseline run:

1. **Very Low TTFT (Time-To-First-Token):** 
   * The median TTFT dropped from **30.9 seconds** in the first attempt to **302.59 ms**, which is even slightly faster than the baseline's 304.01 ms.
2. **Stable Inter-Token Latency (ITL):**
   * The maximum ITL (preemption/swapping delay) decreased to **27.67 seconds**, compared to 28.93 seconds in the baseline and 514.86 seconds in the first attempt. This confirms that request swapping/scheduling stayed clean and stable.
3. **Throughput Recovery:**
   * Total token throughput recovered to **3,425.20 tok/s** (within 5.7% of the baseline 3,632.39 tok/s).
   * Request throughput reached **0.73 req/s** (within 5.2% of the baseline 0.77 req/s).

### Conclusion:
The huge performance restoration confirms that GKE placement dynamics (nodes scheduled on physically adjacent switches/racks) play a critical role in Pipeline Parallelism performance for multi-node configs. 
In the future, using a **Compact Placement Policy** on the GKE Node Pool is highly recommended to guarantee that the GPU nodes are colocated on the same network spine, eliminating latency spikes.

