#!/bin/bash

CONTAINER_NAME="vllm-gemma4-opt26b"
MODEL_NAME="google/gemma-4-26B-A4B-it"
RESULTS_FILE="$HOME/gemma4_26b_manual_results.txt"
PORT=8000

export HF_TOKEN="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"

echo "=============================================================================="
echo " Starting Gemma 4 26B ($MODEL_NAME) Benchmark Setup"
echo " Date: $(date)"
echo "=============================================================================="

echo "==> Stopping any existing containers..."
sudo docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "==> Launching vLLM Server container with optimization flags..."
sudo docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  --net=host \
  -e HF_TOKEN="$HF_TOKEN" \
  -v /tmp/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --seed 42 \
  --max-model-len 16384 \
  --max-num-seqs 1024 \
  --tensor-parallel-size 1 \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 1, "video": 0, "audio": 0}' \
  --block-size 256 \
  --additional-config '{"quantization": { "qwix": { "rules": [{ "module_path": ".*", "weight_qtype": "float8_e4m3fn", "act_qtype": "float8_e4m3fn"}]}}}' \
  --trust-remote-code \
  --enforce-eager

echo "==> Waiting for vLLM server readiness at http://localhost:$PORT/v1/models..."
while true; do
  STATUS=$(sudo docker exec "$CONTAINER_NAME" python3 -c '
import urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/v1/models") as r:
        print("READY")
except Exception:
    print("NOT_READY")
' 2>/dev/null || echo "NOT_READY")

  if [ "$STATUS" = "READY" ]; then
    break
  fi
  echo "    Server still initializing model weights... (sleeping 10s)"
  sleep 10
done

echo "==> vLLM Server is LIVE and READY!"

echo "=== Gemma 4 26B ($MODEL_NAME) Full Matrix Benchmark Suite ===" | tee "$RESULTS_FILE"
echo "Run Date: $(date)" | tee -a "$RESULTS_FILE"
echo "Model: $MODEL_NAME" | tee -a "$RESULTS_FILE"
echo "----------------------------------------------------------------------------------" | tee -a "$RESULTS_FILE"

WORKLOADS=(
  "1k_8k 1000 8000"
  "8k_1k 8000 1000"
  "1k_1k 1000 1000"
  "1k_512 1000 512"
)

CONCURRENCIES=(64 128 256 512 1024)

for workload in "${WORKLOADS[@]}"; do
  set -- $workload
  W_NAME=$1
  ISL=$2
  OSL=$3

  for C in "${CONCURRENCIES[@]}"; do
    PROMPTS=$((C * 2))
    
    echo "" | tee -a "$RESULTS_FILE"
    echo "==========================================================================" | tee -a "$RESULTS_FILE"
    echo " Running Workload: $W_NAME (ISL=$ISL, OSL=$OSL) @ Concurrency=$C (Total Prompts=$PROMPTS)" | tee -a "$RESULTS_FILE"
    echo "==========================================================================" | tee -a "$RESULTS_FILE"

    sudo docker exec "$CONTAINER_NAME" vllm bench serve \
      --backend vllm \
      --model "$MODEL_NAME" \
      --dataset-name random \
      --random-input-len "$ISL" \
      --random-output-len "$OSL" \
      --num-prompts "$PROMPTS" \
      --request-rate "$C" \
      --ready-check-timeout-sec 300 \
      --port "$PORT" 2>&1 | tee -a "$RESULTS_FILE" || true
  done
done

echo "" | tee -a "$RESULTS_FILE"
echo "=== All Benchmark Runs Completed! ===" | tee -a "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE"
