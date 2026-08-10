#!/bin/bash
set -eo pipefail

# ==============================================================================
# Gemma 4 26B (google/gemma-4-26B-A4B) Benchmark Script for GCE (1x RTX PRO 6000)
# Replicating GKE StatefulSet Configuration:
#   - Model: google/gemma-4-26B-A4B
#   - Max Model Length: 8192 (--max-model-len 8192)
#   - Max Sequences: 256 (--max-num-seqs 256)
#   - Max Batched Tokens: 8192 (--max-num-batched-tokens 8192)
#   - KV Cache: FP8 (--kv-cache-dtype fp8)
#   - Block Size: 16 (--block-size 16)
#   - Scheduling: --async-scheduling
#   - Memory config: OMP_NUM_THREADS=16, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ==============================================================================

CONTAINER_NAME="vllm-gemma4-1node-1gpu"
MODEL_NAME="google/gemma-4-26B-A4B"
MAX_MODEL_LEN=8192
PORT=8000
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$HOME/gemma4_26b_benchmark_results_${TIMESTAMP}.txt"
LOG_FILE="$HOME/gemma4_26b_benchmark_run_${TIMESTAMP}.log"

# HuggingFace Token Check
HF_TOKEN="${HF_TOKEN:-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}"

if [ -z "$HF_TOKEN" ]; then
  echo "[ERROR] HF_TOKEN is not set. Please export your HuggingFace token:"
  echo "        export HF_TOKEN='your_hf_token_here'"
  exit 1
fi

echo "=============================================================================="
echo " Starting Gemma 4 26B ($MODEL_NAME) Benchmark Setup"
echo " Date: $(date)"
echo " Results will be saved to: $RESULTS_FILE"
echo " Log file: $LOG_FILE"
echo "=============================================================================="

# 1. Stop any existing containers using port 8000 or running vLLM
echo "==> Cleaning up any existing vLLM containers..."
sudo docker rm -f "$CONTAINER_NAME" vllm-gemma4-opt26b 2>/dev/null || true

# 2. Launch vLLM Server container matching the exact YAML spec
echo "==> Launching vLLM Server container ($CONTAINER_NAME)..."
sudo docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  --net=host \
  --shm-size=32g \
  -e HF_TOKEN="$HF_TOKEN" \
  -e OMP_NUM_THREADS="16" \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  -e VLLM_LOGGING_LEVEL="INFO" \
  -v /tmp/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --host 0.0.0.0 \
  --port "$PORT" \
  --seed 42 \
  --model "$MODEL_NAME" \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90 \
  --async-scheduling \
  --block-size 16 \
  --trust-remote-code

# 3. Health check: Wait until vLLM is ready to serve
echo "==> Waiting for vLLM server readiness inside $CONTAINER_NAME..."
while true; do
  # Check if container is still running
  IS_RUNNING=$(sudo docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || echo "false")
  if [ "$IS_RUNNING" != "true" ]; then
    echo "[ERROR] Container $CONTAINER_NAME died during startup! Logs:"
    sudo docker logs "$CONTAINER_NAME"
    exit 1
  fi

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

# 4. Prepare Benchmark Header
{
  echo "=================================================================================="
  echo "=== Gemma 4 26B ($MODEL_NAME) G4 1-GPU Benchmark Suite ==="
  echo "Run Date: $(date)"
  echo "Model: $MODEL_NAME"
  echo "Serving Spec: max-model-len=8192, async-scheduling, block-size=16, kv-cache=fp8, max-num-seqs=256, max-batched=8192"
  echo "=================================================================================="
} | tee -a "$RESULTS_FILE"

# 5. Workloads: 1k/1k, 1k/512, 1k/8k, 8k/1k
WORKLOADS=(
  "1k_1k 1000 1000"
  "1k_512 1000 512"
  "1k_8k 1000 8000"
  "8k_1k 8000 1000"
)

# Concurrency levels
CONCURRENCIES=(64 128 256 512 1024)

# 6. Execute Benchmark Matrix
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
      --request-rate inf \
      --max-concurrency "$C" \
      --ready-check-timeout-sec 300 \
      --port "$PORT" 2>&1 | tee -a "$RESULTS_FILE" || true
  done
done

echo "" | tee -a "$RESULTS_FILE"
echo "=== All Benchmark Runs Completed! ===" | tee -a "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE"
