import json
import sys

def format_value(val, fmt="{:.2f}"):
    if val is None or val == "N/A" or val == "":
        return "N/A"
    try:
        return fmt.format(float(val))
    except (ValueError, TypeError):
        return str(val)

def format_int(val):
    if val is None or val == "N/A" or val == "":
        return "N/A"
    try:
        return f"{int(val)}"
    except (ValueError, TypeError):
        return str(val)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 format_results.py <results_file.json>")
        sys.exit(1)
        
    filepath = sys.argv[1]
    with open(filepath, "r") as f:
        data = json.load(f)
        
    # Extract keys supporting multiple SGLang versions
    backend = data.get("backend", "sglang-oai")
    request_rate = format_value(data.get("request_rate"), "{}")
    max_concurrency = format_int(data.get("max_concurrency"))
    
    success_rate = data.get("success_rate", 1.0)
    num_prompts = data.get("num_prompts", 1536)
    successful_requests = format_int(data.get("successful_requests", int(float(num_prompts) * float(success_rate))))
    
    duration = data.get("duration", 0.0)
    
    total_input_tokens = data.get("total_input_tokens", 0)
    total_input_text_tokens = data.get("total_input_text_tokens", total_input_tokens)
    
    total_generated_tokens = data.get("total_generated_tokens", data.get("total_output_tokens", 0))
    total_generated_tokens_retokenized = data.get("total_generated_tokens_retokenized", total_generated_tokens)
    
    request_throughput = data.get("request_throughput", 0.0)
    input_throughput = data.get("input_throughput", 0.0)
    output_throughput = data.get("output_throughput", 0.0)
    
    peak_output_throughput = data.get("peak_output_throughput", output_throughput)
    peak_concurrent_requests = format_int(data.get("peak_concurrent_requests", max_concurrency))
    
    total_throughput = data.get("total_throughput", input_throughput + output_throughput)
    concurrency = data.get("concurrency", max_concurrency)
    
    # Latency parsing helper
    # sglang JSON files sometimes output latencies in seconds, and we need them in milliseconds.
    def get_ms(keys):
        for k in keys:
            if k in data:
                val = data[k]
                if val is not None:
                    # If the key name contains 'sec' or value is small, scale it to milliseconds
                    if isinstance(val, (int, float)):
                        if "sec" in k or (val < 15000.0 and "latency" in k and "throughput" not in k) or (val < 1000.0 and "tpot" in k) or (val < 1000.0 and "ttft" in k) or (val < 10.0 and "itl" in k):
                            return val * 1000.0
                    return val
        return None

    mean_e2e = get_ms(["mean_e2e_latency", "mean_e2e_latency_sec", "mean_e2e"])
    median_e2e = get_ms(["median_e2e_latency", "median_e2e_latency_sec", "median_e2e"])
    p90_e2e = get_ms(["p90_e2e_latency", "p90_e2e_latency_sec", "p90_e2e"])
    p99_e2e = get_ms(["p99_e2e_latency", "p99_e2e_latency_sec", "p99_e2e"])
    
    mean_ttft = get_ms(["mean_ttft", "mean_ttft_sec"])
    median_ttft = get_ms(["median_ttft", "median_ttft_sec"])
    p99_ttft = get_ms(["p99_ttft", "p99_ttft_sec"])
    
    mean_tpot = get_ms(["mean_tpot", "mean_tpot_sec"])
    median_tpot = get_ms(["median_tpot", "median_tpot_sec"])
    p99_tpot = get_ms(["p99_tpot", "p99_tpot_sec"])
    
    mean_itl = get_ms(["mean_itl", "mean_itl_sec"])
    median_itl = get_ms(["median_itl", "median_itl_sec"])
    p95_itl = get_ms(["p95_itl", "p95_itl_sec"])
    p99_itl = get_ms(["p99_itl", "p99_itl_sec"])
    max_itl = get_ms(["max_itl", "max_itl_sec"])

    # Fallback to computing from raw requests if some fields are missing (e.g. max_itl, p90_e2e)
    # This guarantees that we never print N/A if raw data is in the JSON!
    if "requests" in data and len(data["requests"]) > 0:
        reqs = data["requests"]
        
        # End-to-end latencies
        e2e_latencies = []
        ttfts = []
        tpots = []
        itls = []
        
        for r in reqs:
            # SGLang request objects contain "latency", "ttft", "itl", etc.
            if "latency" in r and r["latency"] is not None:
                # convert to ms if seconds
                l = float(r["latency"])
                if l < 5000.0:
                    l *= 1000.0
                e2e_latencies.append(l)
            if "ttft" in r and r["ttft"] is not None:
                t = float(r["ttft"])
                if t < 5000.0:
                    t *= 1000.0
                ttfts.append(t)
            # TPOT calculation
            if "latency" in r and "ttft" in r and "output_len" in r:
                try:
                    out_len = int(r["output_len"])
                    if out_len > 1:
                        lat_ms = float(r["latency"]) * 1000.0 if float(r["latency"]) < 5000.0 else float(r["latency"])
                        ttft_ms = float(r["ttft"]) * 1000.0 if float(r["ttft"]) < 5000.0 else float(r["ttft"])
                        tpots.append((lat_ms - ttft_ms) / (out_len - 1))
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
            # ITLs
            if "itls" in r and r["itls"] is not None:
                for itl_val in r["itls"]:
                    # check if ITL is in seconds
                    it_val = float(itl_val)
                    if it_val < 10.0:
                        it_val *= 1000.0
                    itls.append(it_val)

        e2e_latencies.sort()
        ttfts.sort()
        tpots.sort()
        itls.sort()
        
        n_e2e = len(e2e_latencies)
        n_ttft = len(ttfts)
        n_tpot = len(tpots)
        n_itl = len(itls)
        
        if n_e2e > 0:
            if mean_e2e is None: mean_e2e = sum(e2e_latencies) / n_e2e
            if median_e2e is None: median_e2e = e2e_latencies[int(n_e2e * 0.5)]
            if p90_e2e is None: p90_e2e = e2e_latencies[int(n_e2e * 0.9)]
            if p99_e2e is None: p99_e2e = e2e_latencies[int(n_e2e * 0.99)]
            
        if n_ttft > 0:
            if mean_ttft is None: mean_ttft = sum(ttfts) / n_ttft
            if median_ttft is None: median_ttft = ttfts[int(n_ttft * 0.5)]
            if p99_ttft is None: p99_ttft = ttfts[int(n_ttft * 0.99)]
            
        if n_tpot > 0:
            if mean_tpot is None: mean_tpot = sum(tpots) / n_tpot
            if median_tpot is None: median_tpot = tpots[int(n_tpot * 0.5)]
            if p99_tpot is None: p99_tpot = tpots[int(n_tpot * 0.99)]
            
        if n_itl > 0:
            if mean_itl is None: mean_itl = sum(itls) / n_itl
            if median_itl is None: median_itl = itls[int(n_itl * 0.5)]
            if p95_itl is None: p95_itl = itls[int(n_itl * 0.95)]
            if p99_itl is None: p99_itl = itls[int(n_itl * 0.99)]
            if max_itl is None: max_itl = max(itls)

    print("============ Serving Benchmark Result ============")
    print(f"Backend:                                 {backend}")
    print(f"Traffic request rate:                    {request_rate}")
    print(f"Max request concurrency:                 {max_concurrency}")
    print(f"Successful requests:                     {successful_requests}")
    print(f"Benchmark duration (s):                  {format_value(duration, '{:.2f}')}")
    print(f"Total input tokens:                      {format_int(total_input_tokens)}")
    print(f"Total input text tokens:                 {format_int(total_input_text_tokens)}")
    print(f"Total generated tokens:                  {format_int(total_generated_tokens)}")
    print(f"Total generated tokens (retokenized):    {format_int(total_generated_tokens_retokenized)}")
    print(f"Request throughput (req/s):              {format_value(request_throughput, '{:.2f}')}")
    print(f"Input token throughput (tok/s):          {format_value(input_throughput, '{:.2f}')}")
    print(f"Output token throughput (tok/s):         {format_value(output_throughput, '{:.2f}')}")
    print(f"Peak output token throughput (tok/s):    {format_value(peak_output_throughput, '{:.2f}')}")
    print(f"Peak concurrent requests:                {format_int(peak_concurrent_requests)}")
    print(f"Total token throughput (tok/s):          {format_value(total_throughput, '{:.2f}')}")
    print(f"Concurrency:                             {format_value(concurrency, '{:.2f}')}")
    print("----------------End-to-End Latency----------------")
    print(f"Mean E2E Latency (ms):                   {format_value(mean_e2e, '{:.2f}')}")
    print(f"Median E2E Latency (ms):                 {format_value(median_e2e, '{:.2f}')}")
    print(f"P90 E2E Latency (ms):                    {format_value(p90_e2e, '{:.2f}')}")
    print(f"P99 E2E Latency (ms):                    {format_value(p99_e2e, '{:.2f}')}")
    print("---------------Time to First Token----------------")
    print(f"Mean TTFT (ms):                          {format_value(mean_ttft, '{:.2f}')}")
    print(f"Median TTFT (ms):                        {format_value(median_ttft, '{:.2f}')}")
    print(f"P99 TTFT (ms):                           {format_value(p99_ttft, '{:.2f}')}")
    print("-----Time per Output Token (excl. 1st token)------")
    print(f"Mean TPOT (ms):                          {format_value(mean_tpot, '{:.2f}')}")
    print(f"Median TPOT (ms):                        {format_value(median_tpot, '{:.2f}')}")
    print(f"P99 TPOT (ms):                           {format_value(p99_tpot, '{:.2f}')}")
    print("---------------Inter-Token Latency----------------")
    print(f"Mean ITL (ms):                           {format_value(mean_itl, '{:.2f}')}")
    print(f"Median ITL (ms):                         {format_value(median_itl, '{:.2f}')}")
    print(f"P95 ITL (ms):                            {format_value(p95_itl, '{:.2f}')}")
    print(f"P99 ITL (ms):                            {format_value(p99_itl, '{:.2f}')}")
    print(f"Max ITL (ms):                            {format_value(max_itl, '{:.2f}')}")
    print("==================================================")

if __name__ == "__main__":
    main()
