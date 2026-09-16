#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="northam-ce-mlai-tpu"
ZONE="us-east5-a"
CLUSTER_NAME="pallaviam-gke-h100-tcpx-cluster"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Getting credentials for GKE cluster ${CLUSTER_NAME} (${ZONE}) ==="
gcloud container clusters get-credentials "${CLUSTER_NAME}" \
  --zone="${ZONE}" \
  --project="${PROJECT_ID}" \
  --dns-endpoint

echo "=== Deploying vLLM server on 2x H100 GPUs (TP=2) ==="
kubectl apply -f "${SCRIPT_DIR}/vllm_gemma4_26b_h100_2GPU.yaml"

echo "=== Deploying SGLang benchmark client runner pod (10K / 500 sweep) ==="
kubectl delete pod sglang-gemma4-h100-2gpu-10k-500-benchmark-runner --ignore-not-found
kubectl apply -f "${SCRIPT_DIR}/sglang-gemma4-h100-2gpu-10k-500-benchmark-sweep.yaml"

echo "=== Waiting for vllm-gemma4-1node-2gpu-0 pod to start running ==="
kubectl wait --for=condition=Ready pod/vllm-gemma4-1node-2gpu-0 --timeout=600s

echo "=== Streaming benchmark runner logs ==="
kubectl logs -f sglang-gemma4-h100-2gpu-10k-500-benchmark-runner
