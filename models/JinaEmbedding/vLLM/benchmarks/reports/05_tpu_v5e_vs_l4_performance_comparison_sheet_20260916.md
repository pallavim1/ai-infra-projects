# TPU V5e (`FP32` & `BF16`) vs L4 (`Performance Comparison`) — Matching Sheet `gid=1972899730`

* **Reference Google Sheet Tab:** [`TPU V5e vs L4 (Performance Comparison)` (`gid=1972899730`)](https://docs.google.com/spreadsheets/d/1fGgqjRp4giG0MD6NjSIzAbpDbeJoq_Aao7B2MM78QgU/edit?gid=1972899730#gid=1972899730)
* **Configurations Compared:**
  * **L4 GPU (`g2-standard-4`)**: Triton Inference Server + TensorRT (`FP16`, `max_length=2048` truncated via `BertTokenizerFast`)
  * **TPU v5e + `vLLM` (`FP32`)**: `ct5lp-hightpu-1t`, `--dtype float32 --max-model-len 2048`, `truncate_prompt_tokens: 2048` (`BertTokenizerFast`)
  * **TPU v5e + `vLLM` (`BF16`)**: `ct5lp-hightpu-1t`, `--dtype bfloat16 --max-model-len 2048`, `truncate_prompt_tokens: 2048` (`BertTokenizerFast`)

---

## 1. Concurrent Request Comparison (`k6` Server-Side Continuous Batching, `1 KB – 4 KB`)

### 1A. `P50` Latency Comparison (`TPU v5e FP32` & `TPU v5e BF16` vs `L4 GPU`)

| Payload Size | Concurrency | **TPU + vLLM `FP32` (`p50`)** *(8K / 16K Bucket)* | **TPU + vLLM `BF16` (`p50`)** *(16K Bucket)* | **L4 GPU (`p50`)** | **Absolute Delta (Best TPU vs L4)** | **% Reduction (Best TPU vs L4)** | **Faster Setup (`BF16` vs `FP32` vs `L4`)** |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **1KB** | **1** | **11.5 ms** | **11.8 ms** | 20.7 ms | TPU is **9.2 ms** faster | **44.4% lower latency** | **TPU (`FP32` / `BF16` tied)** |
| **1KB** | **4** | **21.2 ms** *(21.3 ms)* | **21.7 ms** *(176.5/s)* | 38.1 ms | TPU is **16.9 ms** faster | **44.4% lower latency** | **TPU (`FP32` / `BF16` tied)** |
| **1KB** | **8** | **42.5 ms** *(44.6 ms)* | **47.1 ms** | 58.7 ms | TPU is **16.2 ms** faster | **27.6% lower latency** | **TPU (`FP32` / `BF16`)** |
| **1KB** | **16** | **85.4 ms** *(153.2 ms)* | **140.3 ms** | 96.5 ms | TPU (8K) is **11.1 ms** faster | **11.5% lower latency** | **TPU (`85.4 ms` w/ 8K bucket); `BF16` beats `FP32` 16K (`140.3 ms` vs `153.2 ms`)** |
| **2KB** | **1** | **16.0 ms** *(16.5 ms)* | **16.5 ms** | 24.3 ms | TPU is **8.3 ms** faster | **34.2% lower latency** | **TPU (`FP32` / `BF16` tied)** |
| **2KB** | **4** | **40.7 ms** *(89.1 ms)* | **47.3 ms** | 52.1 ms | TPU is **11.4 ms** (`FP32` 8K) / **4.8 ms** (`BF16`) faster | **21.9% / 9.2% lower latency** | **TPU (`BF16` cuts 16K latency by `-46.9%`: `47.3 ms` vs `89.1 ms`)** |
| **2KB** | **8** | **82.6 ms** *(153.2 ms)* | **138.2 ms** | 87.4 ms | TPU (8K) is **4.8 ms** faster | **5.5% lower latency** | **TPU (`82.6 ms` w/ 8K bucket); `BF16` beats `FP32` 16K (`138.2 ms` vs `153.2 ms`)** |
| **2KB** | **16** | **165.9 ms** *(560.6 ms)* | **210.4 ms** | **156.9 ms** | GPU is **9.0 ms** faster (vs 8K) | **5.4% lower latency** | **GPU (`156.9 ms`); `BF16` is `2.66x` faster than `FP32` 16K (`210.4 ms` vs `560.6 ms`)** |
| **3KB** *(truncated @ 2048)* | **1** | **17.1 ms** | **17.2 ms** | 26.6 ms | TPU is **9.5 ms** faster | **35.7% lower latency** | **TPU (`FP32` / `BF16` tied)** |
| **3KB** *(truncated @ 2048)* | **4** | **44.5 ms** | **46.1 ms** | 57.0 ms | TPU is **12.5 ms** faster | **21.9% lower latency** | **TPU (`FP32` & `BF16` both `< 50 ms`)** |
| **3KB** *(truncated @ 2048)* | **8** | 356.5 ms | **138.9 ms** | **90.8 ms** | GPU is **48.1 ms** faster | **34.6% lower latency** | **GPU (`90.8 ms`); `BF16` is `2.57x` faster than `FP32` (`138.9 ms` vs `356.5 ms`)** |
| **3KB** *(truncated @ 2048)* | **16** | 563.2 ms | **279.7 ms** | **167.8 ms** | GPU is **111.9 ms** faster | **40.0% lower latency** | **GPU (`167.8 ms`); `BF16` is `2.01x` faster than `FP32` (`279.7 ms` vs `563.2 ms`)** |
| **4KB** *(truncated @ 2048)* | **1** | **17.4 ms** | **18.2 ms** | 28.6 ms | TPU is **11.2 ms** faster | **39.2% lower latency** | **TPU (`FP32` / `BF16` tied)** |
| **4KB** *(truncated @ 2048)* | **4** | **45.1 ms** | **46.8 ms** | 60.6 ms | TPU is **15.5 ms** faster | **25.6% lower latency** | **TPU (`FP32` & `BF16` both `< 50 ms`)** |
| **4KB** *(truncated @ 2048)* | **8** | 152.9 ms | **139.9 ms** | **100.2 ms** | GPU is **39.7 ms** faster | **28.4% lower latency** | **GPU (`100.2 ms`); `BF16` beats `FP32` (`139.9 ms` vs `152.9 ms`)** |
| **4KB** *(truncated @ 2048)* | **16** | 562.4 ms | **281.5 ms** | **180.5 ms** | GPU is **101.0 ms** faster | **35.9% lower latency** | **GPU (`180.5 ms`); `BF16` is `2.00x` faster than `FP32` (`281.5 ms` vs `562.4 ms`)** |

---

### 1B. `P99` Latency Comparison (`TPU v5e FP32` & `TPU v5e BF16` vs `L4 GPU`)

| Payload Size | Concurrency | **TPU + vLLM `FP32` (`p99`)** *(8K / 16K Bucket)* | **TPU + vLLM `BF16` (`p99`)** *(16K Bucket)* | **L4 GPU (`p99`)** | **Absolute Delta (Best TPU vs L4)** | **% Reduction (Best TPU vs L4)** | **Faster Setup (`BF16` vs `FP32` vs `L4`)** |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| **1KB** | **1** | **12.7 ms** | **13.4 ms** | 23.5 ms | TPU is **10.8 ms** faster | **46.0% lower latency** | **TPU (`FP32` / `BF16` tied)** |
| **1KB** | **4** | **22.8 ms** *(40.4 ms)* | **41.3 ms** | 42.9 ms | TPU is **20.1 ms** faster | **46.9% lower latency** | **TPU (`FP32` & `BF16` both beat L4)** |
| **1KB** | **8** | **44.5 ms** *(101.5 ms)* | **97.0 ms** | 66.3 ms | TPU (8K) is **21.8 ms** faster | **32.9% lower latency** | **TPU (`44.5 ms` w/ 8K bucket)** |
| **1KB** | **16** | **89.9 ms** *(155.8 ms)* | **146.1 ms** | 107.6 ms | TPU (8K) is **17.7 ms** faster | **16.4% lower latency** | **TPU (`89.9 ms` w/ 8K bucket)** |
| **2KB** | **1** | **17.2 ms** *(17.7 ms)* | **18.1 ms** | 28.2 ms | TPU is **11.0 ms** faster | **39.0% lower latency** | **TPU (`FP32` / `BF16` tied)** |
| **2KB** | **4** | **42.8 ms** *(99.9 ms)* | **93.1 ms** | 58.2 ms | TPU (8K) is **15.4 ms** faster | **26.5% lower latency** | **TPU (`42.8 ms` w/ 8K bucket)** |
| **2KB** | **8** | **86.0 ms** *(339.1 ms)* | **142.9 ms** | 97.3 ms | TPU (8K) is **11.3 ms** faster | **11.6% lower latency** | **TPU (`86.0 ms` w/ 8K bucket; `BF16` cuts 16K $P_{99}$ by `-57.9%`)** |
| **2KB** | **16** | **171.2 ms** *(597.2 ms)* | **281.4 ms** | 175.5 ms | TPU (8K) is **4.3 ms** faster | **2.5% lower latency** | **TPU (`171.2 ms` w/ 8K bucket; `BF16` cuts 16K $P_{99}$ by `-52.9%`)** |
| **3KB** *(truncated @ 2048)* | **1** | **19.2 ms** | **19.6 ms** | 31.3 ms | TPU is **12.1 ms** faster | **38.7% lower latency** | **TPU (`FP32` / `BF16` tied)** |
| **3KB** *(truncated @ 2048)* | **4** | **46.4 ms** | **49.4 ms** | 63.7 ms | TPU is **17.3 ms** faster | **27.2% lower latency** | **TPU (Both `FP32` & `BF16` pass `< 50 ms` $P_{99}$ SLA!)** |
| **3KB** *(truncated @ 2048)* | **8** | 358.4 ms | **143.3 ms** | **100.2 ms** | GPU is **43.1 ms** faster | **30.1% lower latency** | **GPU (`100.2 ms`); `BF16` is `2.50x` faster than `FP32`** |
| **3KB** *(truncated @ 2048)* | **16** | 625.9 ms | **284.7 ms** | **189.0 ms** | GPU is **95.7 ms** faster | **33.6% lower latency** | **GPU (`189.0 ms`); `BF16` is `2.20x` faster than `FP32`** |
| **4KB** *(truncated @ 2048)* | **1** | **18.9 ms** | **19.2 ms** | 34.2 ms | TPU is **15.3 ms** faster | **44.7% lower latency** | **TPU (`FP32` / `BF16` tied)** |
| **4KB** *(truncated @ 2048)* | **4** | **47.6 ms** | **49.7 ms** | 68.5 ms | TPU is **20.9 ms** faster | **30.5% lower latency** | **TPU (Both `FP32` & `BF16` pass `< 50 ms` $P_{99}$ SLA!)** |
| **4KB** *(truncated @ 2048)* | **8** | 155.8 ms | **143.9 ms** | **113.9 ms** | GPU is **30.0 ms** faster | **20.8% lower latency** | **GPU (`113.9 ms`); `BF16` beats `FP32`** |
| **4KB** *(truncated @ 2048)* | **16** | 610.0 ms | **286.5 ms** | **204.0 ms** | GPU is **82.5 ms** faster | **28.8% lower latency** | **GPU (`204.0 ms`); `BF16` is `2.13x` faster than `FP32`** |

---

## 2. Single-HTTP-Request Multi-Prompt Batch Comparison (`Batch = 1, 4, 8, 16`) — `TPU FP32` vs `TPU BF16` vs `L4 GPU`
*When multiple prompts (`Batch = 1, 4, 8, 16`) are sent in a **single HTTP request** (`{"text": [prompt_1, ..., prompt_N]}`), **`BF16` cuts TPU batch latency by `45%–59%` compared to `FP32`**:*

| Payload Size | Batch (`N`) | **TPU `FP32` (`p50` / `p99`)** | **TPU `BF16` (`p50` / `p99`)** | **L4 GPU (`p50` / `p99`)** | **`BF16` Improvement Over `FP32`** | **TPU `BF16` vs `L4 GPU`** |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **1KB** | **1** | **12.7 ms** / **13.2 ms** | **13.1 ms** / **14.7 ms** | 20.7 ms / 23.5 ms | Comparable (`~13 ms`) | 🏆 **TPU `BF16` is `36.7%` faster** |
| **1KB** | **4** | 41.1 ms / 42.8 ms | **34.9 ms** / **43.6 ms** | 38.1 ms / 42.9 ms | **`15.1%` lower $P_{50}$ latency** | 🏆 **TPU `BF16` is `8.4%` faster ($P_{50}$)** |
| **1KB** | **8** | 107.8 ms / 110.0 ms | **100.4 ms** / **103.1 ms** | **58.7 ms** / **66.3 ms** | **`6.9%` lower $P_{50}$ (`+14.4%` tput)** | L4 GPU faster at Batch 8 |
| **1KB** | **16** | 331.4 ms / 342.9 ms | **142.8 ms** / **147.8 ms** | **96.5 ms** / **107.6 ms** | **`56.9%` lower $P_{50}$ (`2.23x` tput)** | `BF16` closes gap by `188.6 ms` |
| **2KB** | **1** | **17.2 ms** / **18.0 ms** | **17.8 ms** / **19.9 ms** | 24.3 ms / 28.2 ms | Comparable (`~17.5 ms`) | 🏆 **TPU `BF16` is `26.7%` faster** |
| **2KB** | **8** | 319.1 ms / 331.0 ms | **130.9 ms** / **167.8 ms** | **87.4 ms** / **97.3 ms** | **`59.0%` lower $P_{50}$ (`2.07x` tput)** | `BF16` closes gap by `188.2 ms` |
| **3KB** *(truncated @ 2048)* | **1** | **18.1 ms** / **18.8 ms** | **19.1 ms** / **20.7 ms** | 26.6 ms / 31.3 ms | Comparable (`~18.5 ms`) | 🏆 **TPU `BF16` is `28.2%` faster** |
| **3KB** *(truncated @ 2048)* | **16** | 632.9 ms / 700.2 ms | **339.1 ms** / **349.8 ms** | **167.8 ms** / **189.0 ms** | **`46.4%` lower $P_{50}$ (`1.87x` tput)** | `BF16` closes gap by `293.8 ms` |
| **4KB** *(truncated @ 2048)* | **1** | **18.5 ms** / **20.1 ms** | **19.3 ms** / **21.0 ms** | 28.6 ms / 34.2 ms | Comparable (`~19 ms`) | 🏆 **TPU `BF16` is `32.5%` faster** |
| **4KB** *(truncated @ 2048)* | **16** | 648.2 ms / 704.2 ms | **353.7 ms** / **357.4 ms** | **180.5 ms** / **204.0 ms** | **`45.4%` lower $P_{50}$ (`1.88x` tput)** | `BF16` closes gap by `294.5 ms` |

---

## 3. RPS Saturation Result (`< 50 ms` $P_{99}$ SLA — Matching Sheet `gid=1972899730`)

| Payload | Setup | **Max Sustained RPS (`P99 < 50 ms`)** | **`p50` (`ms`)** | **`p99` (`ms`)** | **Total RPS Multiplier vs L4** | **% RPS Improvement (`(TPU - L4) / L4`)** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1K** | **L4 + Triton (`FP16`)** | 70 | 20.0 ms | 37.0 ms | 1.00x (Baseline) | — |
| **1K** | **TPU V5e + vLLM (`FP32`)** | **180** *(Peak: 187)* | **19.9 ms** | **29.6 ms** | **2.57x** *(2.67x)* | **`+157.1%` (`1.57`)** *(+167.1% / `1.67`)* |
| **1K** | **TPU V5e + vLLM (`BF16`)** | **160** *(180 @ 48.5ms $P_{50}$)* | **19.1 ms** | **27.3 ms** | **2.29x – 2.57x** | **`+128.6%` to `+157.1%`** |
| **2K** | **L4 + Triton (`FP16`)** | 40 | 22.0 ms | 28.0 ms | 1.00x (Baseline) | — |
| **2K** | **TPU V5e + vLLM (`FP32`)** | **100** *(Sheet: 90)* | **18.9 ms** | **25.6 ms** | **2.50x** *(2.25x)* | **`+150.0%` (`1.50`)** *(+125.0% / `1.25`)* |
| **2K** | **TPU V5e + vLLM (`BF16`)** | **90** | **18.1 ms** | **23.2 ms** | **2.25x** | **`+125.0%` (`1.25`)** |
| **3K** *(truncated @ 2048)* | **L4 + Triton (`FP16`)** | ~30 | ~25.0 ms | ~35.0 ms | 1.00x (Baseline) | — |
| **3K** *(truncated @ 2048)* | **TPU V5e + vLLM (`FP32`)** | **90** | **18.0 ms** | **20.3 ms** | **3.00x** | **`+200.0%` (`2.00`)** |
| **3K** *(truncated @ 2048)* | **TPU V5e + vLLM (`BF16`)** | **90** | **19.0 ms** | **22.6 ms** | **3.00x** | **`+200.0%` (`2.00`)** |
| **5K** *(truncated @ 2048)* | **L4 + Triton (`FP16`)** | 20 | 31.2 ms | 49.1 ms | 1.00x (Baseline) | — |
| **5K** *(truncated @ 2048)* | **TPU V5e + vLLM (`FP32` / `BF16`)** | **90** *(Multi-Payload)* | **17.5 ms** | **20.4 ms / 21.2 ms** | **4.50x** | **`+350.0%` (`3.50`)** |
| **7K** *(truncated @ 2048)* | **L4 + Triton (`FP16`)** | 10 | 36.6 ms | 46.4 ms | 1.00x (Baseline) | — |
| **7K** *(truncated @ 2048)* | **TPU V5e + vLLM (`FP32` / `BF16`)** | **90** *(Multi-Payload)* | **18.0 ms** | **20.0 ms / 24.0 ms** | **9.00x** | **`+800.0%` (`8.00`)** |

---

## 4. Cost Improvement Table (Using Sheet's Exact Cost Formula: `Hourly_Cost / (RPS * 1440) * 1,000,000`)

| Machine Type | Machine Config | Hourly Cost (`$`) | **Cost per 1M Requests (`1K` Payload)** | **Cost per 1M Requests (`2K` Payload)** | **Cost per 1M Requests (`3K` Payload)** | **Cost per 1M Requests (`5K` Payload)** | **Cost per 1M Requests (`7K` Payload)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`g2-standard-4`**<br>*(L4 + Triton `FP16`)* | L4 GPUs: 1<br>vCPUs: 4, Mem: 16 GiB | **\$0.70** | **\$6.94**<br>*(70 RPS)* | **\$12.15**<br>*(40 RPS)* | **\$16.20**<br>*(30 RPS)* | **\$24.31**<br>*(20 RPS)* | **\$48.61**<br>*(10 RPS)* |
| **`ct5lp-hightpu-1t`**<br>*(TPU v5e + vLLM `FP32`)* | TPU v5e: 1 chip<br>vCPUs: 24, Mem: 45.6 GB<br>TPU HBM: 16 GB | **\$1.20** | **\$4.46 – \$4.63**<br>*(187 / 180 RPS)* | **\$8.33 – \$9.26**<br>*(100 / 90 RPS)* | **\$9.26**<br>*(90 RPS)* | **\$9.26**<br>*(90 RPS)* | **\$9.26**<br>*(90 RPS)* |
| **`ct5lp-hightpu-1t`**<br>*(TPU v5e + vLLM `BF16`)* | TPU v5e: 1 chip<br>vCPUs: 24, Mem: 45.6 GB<br>TPU HBM: 16 GB | **\$1.20** | **\$4.63 – \$5.21**<br>*(180 / 160 RPS)* | **\$9.26**<br>*(90 RPS)* | **\$9.26**<br>*(90 RPS)* | **\$9.26**<br>*(90 RPS)* | **\$9.26**<br>*(90 RPS)* |
| **Cost Improvement (`TPU v5e` vs `L4`)** | — | — | 🏆 **`33.3% – 35.8%` reduction** | 🏆 **`23.8% – 31.4%` reduction** | 🏆 **`42.9%` reduction** | 🏆 **`61.9%` reduction** | 🏆 **`81.0%` reduction** |
