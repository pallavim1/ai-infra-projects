#!/bin/bash
set -eo pipefail

# ==============================================================================
# Gemma 4 26B (google/gemma-4-26B-A4B) vLLM Serving Benchmark Sweep (10K / 500)
# Uses `vllm bench serve` (native vLLM benchmark CLI) instead of SGLang bench
# ==============================================================================
# Usage:
#   # 1. Against an already-running vLLM server (e.g. GKE Service or localhost):
#   VLLM_HOST="vllm-gemma4-1node-1gpu-service" VLLM_PORT="8000" ./run_vllm_benchmark_10k_500.sh
#
#   # 2. Or inside a running Docker container on GCE:
#   CONTAINER_NAME="vllm-gemma4-1node-1gpu" ./run_vllm_benchmark_10k_500.sh
# ==============================================================================

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL_NAME="${MODEL_NAME:-google/gemma-4-26B-A4B}"
INPUT_LEN="${INPUT_LEN:-10240}"
OUTPUT_LEN="${OUTPUT_LEN:-500}"
CONCURRENCIES=(1 8 16 32 64 128 256 512)

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${OUT_DIR:-./results}"
mkdir -p "$OUT_DIR"

REPORT_FILE="${OUT_DIR}/vllm_benchmark_10k_500_report.md"
RAW_LOG="${OUT_DIR}/vllm_benchmark_10k_500_${TIMESTAMP}.log"

# Determine whether to invoke `vllm bench serve` directly or via `docker exec`
if command -v vllm &>/dev/null; then
  BENCH_CMD=(vllm bench serve)
elif [ -n "${CONTAINER_NAME:-}" ]; then
  BENCH_CMD=(sudo docker exec "$CONTAINER_NAME" vllm bench serve)
else
  echo "[ERROR] Neither 'vllm' CLI nor CONTAINER_NAME is available in environment."
  echo "        Run inside a vLLM container or set CONTAINER_NAME=<docker-container-name>."
  exit 1
fi

echo "==============================================================================" | tee -a "$RAW_LOG"
echo " Starting Gemma 4 26B ($MODEL_NAME) vLLM Native Benchmark Sweep" | tee -a "$RAW_LOG"
echo " Target Endpoint: http://${VLLM_HOST}:${VLLM_PORT}" | tee -a "$RAW_LOG"
echo " Workload: Input=${INPUT_LEN} tokens, Output=${OUTPUT_LEN} tokens" | tee -a "$RAW_LOG"
echo " Concurrencies: ${CONCURRENCIES[*]}" | tee -a "$RAW_LOG"
echo "==============================================================================" | tee -a "$RAW_LOG"

for C in "${CONCURRENCIES[@]}"; do
  if [ "$C" -eq 1 ]; then
    NUM_PROMPTS=8
  else
    NUM_PROMPTS=$((C * 2))
  fi

  JSON_FILE="${OUT_DIR}/result_vllm_10k_500_c${C}.json"
  echo "" | tee -a "$RAW_LOG"
  echo "------------------------------------------------------------------------------" | tee -a "$RAW_LOG"
  echo " Running vllm bench serve: Concurrency=${C} (Prompts=${NUM_PROMPTS})" | tee -a "$RAW_LOG"
  echo "------------------------------------------------------------------------------" | tee -a "$RAW_LOG"

  "${BENCH_CMD[@]}" \
    --backend vllm \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --model "$MODEL_NAME" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --random-range-ratio 0.0 \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$C" \
    --request-rate inf \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --save-result \
    --result-dir "$OUT_DIR" \
    --result-filename "result_vllm_10k_500_c${C}.json" \
    2>&1 | tee -a "$RAW_LOG"
done

# Generate Markdown Report matching the exact table format
python3 - "$OUT_DIR" "$REPORT_FILE" << 'EOF'
import json, os, sys

out_dir = sys.argv[1]
report_file = sys.argv[2]
concurrencies = [1, 8, 16, 32, 64, 128, 256, 512]
rows = {}

for c in concurrencies:
    path = os.path.join(out_dir, f"result_vllm_10k_500_c{c}.json")
    if os.path.exists(path):
        with open(path) as f:
            rows[c] = json.load(f)

lines = [
    "# Gemma 4 26B vLLM Benchmark Report (10K Input / 500 Output)",
    "",
    "**Configuration**",
    "- **Model**: `google/gemma-4-26B-A4B`",
    "- **Input Prompt Length**: 10,240 tokens (10K Context)",
    "- **Max Output Tokens**: 500 tokens",
    "- **Framework**: vLLM Serving + vLLM Benchmark CLI (`vllm bench serve`)",
    "- **Hardware**: 1x NVIDIA RTX PRO 6000 Blackwell Server Edition (GKE `g4-standard-48`, TP=1)",
    "",
    "---",
    "",
    "## 1. Concurrency Sweep Summary Table",
    "",
    "| Concurrency | Prompts | Output Tok/s | Total Tok/s | Mean TTFT (ms) | Median TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Mean Latency (s) | Median Latency (s) | P99 Latency (s) |",
    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
]

for c in concurrencies:
    d = rows.get(c, {})
    prompts = d.get("completed", c * 2 if c > 1 else 8)
    out_tps = d.get("output_throughput", 0.0)
    tot_tps = d.get("total_token_throughput", 0.0)
    mean_ttft = d.get("mean_ttft_ms", 0.0)
    med_ttft = d.get("median_ttft_ms", 0.0)
    p99_ttft = d.get("p99_ttft_ms", 0.0)
    mean_tpot = d.get("mean_tpot_ms", 0.0)
    med_tpot = d.get("median_tpot_ms", 0.0)
    p99_tpot = d.get("p99_tpot_ms", 0.0)
    mean_lat = d.get("mean_e2el_ms", 0.0) / 1000.0
    med_lat = d.get("median_e2el_ms", 0.0) / 1000.0
    p99_lat = d.get("p99_e2el_ms", 0.0) / 1000.0
    lines.append(
        f"| **{c}** | {prompts} | {out_tps:.2f} | {tot_tps:.2f} | {mean_ttft:.2f} | {med_ttft:.2f} | {p99_ttft:.2f} | {mean_tpot:.2f} | {med_tpot:.2f} | {p99_tpot:.2f} | {mean_lat:.2f} | {med_lat:.2f} | {p99_lat:.2f} |"
    )

with open(report_file, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nReport generated at: {report_file}")
EOF

# Remove temporary per-concurrency JSON files to keep results/ clean
rm -f "${OUT_DIR}"/result_vllm_10k_500_c*.json

echo "=== Benchmark Sweep Completed ==="
