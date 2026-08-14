#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# End-to-End GKE Cluster, TPU v5e Pool & CPU Benchmark Pool Provisioning
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ID="northam-ce-mlai-tpu"
REGION="europe-west4"
ZONE="europe-west4-b"
CLUSTER_NAME="pm-panw-jina-cluster"
VPC_NAME="pm-panw-jina-vpc"
SUBNET_NAME="pm-panw-jina-subnet"
TPU_POOL_NAME="pm-panw-jina-tpu-pool"
CPU_POOL_NAME="cpu-benchmark-pool"

echo "=== [1/7] Setting active GCP project: $PROJECT_ID ==="
gcloud config set project "$PROJECT_ID"

echo "=== [2/7] Enabling required Google Cloud APIs ==="
gcloud services enable container.googleapis.com tpu.googleapis.com compute.googleapis.com

echo "=== [3/7] Creating VPC, Subnet and Internal Firewall Rules ==="
if ! gcloud compute networks describe "$VPC_NAME" --project="$PROJECT_ID" &>/dev/null; then
    gcloud compute networks create "$VPC_NAME" \
        --project="$PROJECT_ID" \
        --subnet-mode=custom
fi

if ! gcloud compute networks subnets describe "$SUBNET_NAME" --region="$REGION" --project="$PROJECT_ID" &>/dev/null; then
    gcloud compute networks subnets create "$SUBNET_NAME" \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --network="$VPC_NAME" \
        --range=10.240.0.0/20 \
        --secondary-range=pm-panw-jina-pods=10.241.0.0/16,pm-panw-jina-services=10.242.0.0/20
fi

if ! gcloud compute firewall-rules describe pm-panw-jina-allow-internal --project="$PROJECT_ID" &>/dev/null; then
    gcloud compute firewall-rules create pm-panw-jina-allow-internal \
        --project="$PROJECT_ID" \
        --network="$VPC_NAME" \
        --allow=tcp,udp,icmp \
        --source-ranges=10.240.0.0/20,10.241.0.0/16,10.242.0.0/20
fi

echo "=== [4/7] Creating GKE Cluster: $CLUSTER_NAME ==="
if ! gcloud container clusters describe "$CLUSTER_NAME" --zone="$ZONE" --project="$PROJECT_ID" &>/dev/null; then
    gcloud container clusters create "$CLUSTER_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --release-channel=rapid \
        --network="$VPC_NAME" \
        --subnetwork="$SUBNET_NAME" \
        --cluster-secondary-range-name=pm-panw-jina-pods \
        --services-secondary-range-name=pm-panw-jina-services \
        --num-nodes=1 \
        --machine-type=e2-standard-4 \
        --enable-ip-alias
fi

echo "=== [5/7] Creating Cloud TPU v5e Node Pool: $TPU_POOL_NAME ==="
if ! gcloud container node-pools describe "$TPU_POOL_NAME" --cluster="$CLUSTER_NAME" --zone="$ZONE" --project="$PROJECT_ID" &>/dev/null; then
    gcloud container node-pools create "$TPU_POOL_NAME" \
        --project="$PROJECT_ID" \
        --cluster="$CLUSTER_NAME" \
        --zone="$ZONE" \
        --node-locations="$ZONE" \
        --machine-type=ct5lp-hightpu-1t \
        --tpu-topology=1x1 \
        --num-nodes=1
fi

echo "=== [6/7] Creating Dedicated CPU Benchmark Node Pool: $CPU_POOL_NAME ==="
if ! gcloud container node-pools describe "$CPU_POOL_NAME" --cluster="$CLUSTER_NAME" --zone="$ZONE" --project="$PROJECT_ID" &>/dev/null; then
    gcloud container node-pools create "$CPU_POOL_NAME" \
        --project="$PROJECT_ID" \
        --cluster="$CLUSTER_NAME" \
        --zone="$ZONE" \
        --node-locations="$ZONE" \
        --machine-type=n2-standard-8 \
        --num-nodes=1
fi

echo "=== [7/7] Configuring kubectl context & deploying workloads ==="
gcloud container clusters get-credentials "$CLUSTER_NAME" --zone="$ZONE" --project="$PROJECT_ID"

# Deploy TPU Serving Pod
kubectl apply -f jina_v5e_deployment.yaml

# Deploy CPU Benchmark Runner Pod
kubectl apply -f cpu_benchmark_runner.yaml

echo "=== Setup complete! Verify pods with: kubectl get pods -o wide ==="
