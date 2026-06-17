#!/bin/bash
# Script to create a 2-node H200 GKE cluster with GPUDirect TCPX (RoCE) enabled in europe-west4-a

set -euo pipefail

# Configuration variables
export PROJECT_NAME="northam-ce-mlai-tpu"
export REGION="europe-west4"
export ZONE="europe-west4-a"
export NETWORK_NAME="a3-mega-h100-cluster-gpunet-0"
export CLUSTER_NAME="h200-tcpx-gke-cluster"
export GPU_POOL_NAME="h200-tcpx-pool"
export RESERVATION_NAME="pm-h200-testing"
export DISK_SIZE="300"

echo "========================================================================="
echo "Creating H200 Node Pool: ${GPU_POOL_NAME} from reservation..."
echo "========================================================================="

gcloud beta container node-pools create "${GPU_POOL_NAME}" \
     --cluster="${CLUSTER_NAME}" \
     --zone="${ZONE}" \
     --machine-type="a3-ultragpu-8g" \
     --num-nodes=2 \
     --accelerator=type=nvidia-h200-141gb,count=8,gpu-driver-version=LATEST \
     --ephemeral-storage-local-ssd=count=32 \
     --enable-image-streaming \
     --workload-metadata=GKE_METADATA \
     --disk-size="${DISK_SIZE}" \
     --scopes=https://www.googleapis.com/auth/cloud-platform \
     --reservation-affinity=specific \
     --reservation="${RESERVATION_NAME}" \
     --project="${PROJECT_NAME}" \
     --additional-node-network=network="nemo-gvnic-net",subnetwork="nemo-gvnic-sub" \
     --additional-node-network=network="nemo-rdma-net",subnetwork="nemo-rdma-sub-0" \
     --additional-node-network=network="nemo-rdma-net",subnetwork="nemo-rdma-sub-1" \
     --additional-node-network=network="nemo-rdma-net",subnetwork="nemo-rdma-sub-2" \
     --additional-node-network=network="nemo-rdma-net",subnetwork="nemo-rdma-sub-3" \
     --additional-node-network=network="nemo-rdma-net",subnetwork="nemo-rdma-sub-4" \
     --additional-node-network=network="nemo-rdma-net",subnetwork="nemo-rdma-sub-5" \
     --additional-node-network=network="nemo-rdma-net",subnetwork="nemo-rdma-sub-6" \
     --additional-node-network=network="nemo-rdma-net",subnetwork="nemo-rdma-sub-7"

echo "========================================================================="
echo "H200 GPU Node Pool successfully created!"
echo "========================================================================="
