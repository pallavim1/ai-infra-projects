#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ==========================================
# WARNING: CIDR RANGE REVIEW
# ==========================================
echo -e "${YELLOW}=====================================================================${NC}"
echo -e "${YELLOW}WARNING: PLEASE REVIEW AND CHANGE THE CIDR RANGES IN THIS SCRIPT${NC}"
echo -e "${YELLOW}TO AVOID CONFLICTS WITH EXISTING NETWORKS IN YOUR PROJECT!${NC}"
echo -e "${YELLOW}=====================================================================${NC}"

# ==========================================
# 1. Environment Variables (Template)
# ==========================================
# Project and Location
export PROJECT_NAME="${PROJECT_NAME:-}"
if [ -z "$PROJECT_NAME" ]; then
    echo -e "${RED}ERROR: PROJECT_NAME is not set! Please set it in your environment or edit this script.${NC}"
    exit 1
fi
export REGION="${REGION:-us-east5}"
export ZONE="${ZONE:-us-east5-a}"

# Network Names
export NETWORK_NAME="${NETWORK_NAME:-$USER-h100-vpc-1}"
export SUBNETWORK_NAME_1="${SUBNETWORK_NAME_1:-$USER-subnet-1-us-east5}"
export SUBNETWORK_NAME_2="${SUBNETWORK_NAME_2:-$USER-additional-test-subnet-1-us-east5}"

# Cluster and Pool Names
export CLUSTER_NAME="${CLUSTER_NAME:-$USER-gke-h100-cluster}"
export GPU_POOL_NAME="${GPU_POOL_NAME:-$USER-2x-h100-2nic-pool}"
export COMPUTE_POOL_NAME="${COMPUTE_POOL_NAME:-benchmark-client-pool-h100}"

# CIDR Ranges (CRITICAL: Review these!)
export SUBNET_1_RANGE="${SUBNET_1_RANGE:-10.4.0.0/20}"
export SUBNET_1_PODS_RANGE="${SUBNET_1_PODS_RANGE:-10.5.0.0/20}"
export SUBNET_1_SERVICES_RANGE="${SUBNET_1_SERVICES_RANGE:-10.5.16.0/20}"

export SUBNET_2_RANGE="${SUBNET_2_RANGE:-10.4.16.0/20}"
export SUBNET_2_PODS_RANGE="${SUBNET_2_PODS_RANGE:-10.24.16.0/20}"
export SUBNET_2_SERVICES_RANGE="${SUBNET_2_SERVICES_RANGE:-10.24.32.0/20}"

export MASTER_IPV4_CIDR="${MASTER_IPV4_CIDR:-172.16.0.48/28}"

export FW_ALLOW_INTERNAL_SOURCE="${FW_ALLOW_INTERNAL_SOURCE:-10.0.0.0/8}"

# Machine Types and Scaling
export GPU_MACHINE_TYPE="${GPU_MACHINE_TYPE:-a3-megagpu-8g}"
export COMPUTE_MACHINE_TYPE="${COMPUTE_MACHINE_TYPE:-n2-standard-64}"
export GPU_POOL_MIN_NODES="${GPU_POOL_MIN_NODES:-2}"
export GPU_POOL_MAX_NODES="${GPU_POOL_MAX_NODES:-2}"
export GPU_DISK_SIZE="${GPU_DISK_SIZE:-300}"
export COMPUTE_DISK_SIZE="${COMPUTE_DISK_SIZE:-300}"

# Reservation Configuration
export RESERVATION_AFFINITY="${RESERVATION_AFFINITY:-specific}"
export RESERVATION_NAME="${RESERVATION_NAME:-pm-h100s}"

# Paths
export CLOUDSDK_PYTHON="${CLOUDSDK_PYTHON:-/usr/bin/python3}"
export GCLOUD_PATH="${GCLOUD_PATH:-$HOME/google-cloud-sdk/bin/gcloud}"

# Set default project
echo -e "${GREEN}[Step 1] Setting default project to ${PROJECT_NAME}...${NC}"
${GCLOUD_PATH} config set project ${PROJECT_NAME}

# ==========================================
# 2. Network Infrastructure
# ==========================================
# Create the custom VPC network if it doesn't already exist.
echo -e "${GREEN}[Step 2] Checking VPC Network...${NC}"
if ! ${GCLOUD_PATH} compute networks describe ${NETWORK_NAME} --project ${PROJECT_NAME} >/dev/null 2>&1; then
    echo -e "${YELLOW}Creating VPC Network ${NETWORK_NAME}...${NC}"
    ${GCLOUD_PATH} compute networks create ${NETWORK_NAME} \
         --subnet-mode=custom \
         --mtu=8896
else
    echo -e "${GREEN}VPC ${NETWORK_NAME} already exists, skipping creation.${NC}"
fi

# Create the two subnets with secondary ranges for pods and services.
echo -e "${GREEN}[Step 3] Checking Subnets...${NC}"
if ! ${GCLOUD_PATH} compute networks subnets describe ${SUBNETWORK_NAME_1} --region=${REGION} --project ${PROJECT_NAME} >/dev/null 2>&1; then
    echo -e "${YELLOW}Creating Subnet 1: ${SUBNETWORK_NAME_1}...${NC}"
    ${GCLOUD_PATH} compute networks subnets create ${SUBNETWORK_NAME_1} \
         --network=${NETWORK_NAME} \
         --range=${SUBNET_1_RANGE} \
         --region=${REGION} \
         --secondary-range=$USER-pods=${SUBNET_1_PODS_RANGE},$USER-services=${SUBNET_1_SERVICES_RANGE}
else
    echo -e "${GREEN}Subnet ${SUBNETWORK_NAME_1} already exists, skipping creation.${NC}"
fi

if ! ${GCLOUD_PATH} compute networks subnets describe ${SUBNETWORK_NAME_2} --region=${REGION} --project ${PROJECT_NAME} >/dev/null 2>&1; then
    echo -e "${YELLOW}Creating Subnet 2: ${SUBNETWORK_NAME_2}...${NC}"
    ${GCLOUD_PATH} compute networks subnets create ${SUBNETWORK_NAME_2} \
         --network=${NETWORK_NAME} \
         --range=${SUBNET_2_RANGE} \
         --region=${REGION} \
         --secondary-range=$USER-additional-test-pods=${SUBNET_2_PODS_RANGE},$USER-additional-test-services=${SUBNET_2_SERVICES_RANGE}
else
    echo -e "${GREEN}Subnet ${SUBNETWORK_NAME_2} already exists, skipping creation.${NC}"
fi

# ==========================================
# 3. Firewall Rules
# ==========================================
# Create firewall rules to allow internal communication, SSH, and master-to-node traffic.
echo -e "${GREEN}[Step 4] Checking Firewall Rules...${NC}"
check_and_create_fw() {
    local rule_name=$1
    shift
    if ! ${GCLOUD_PATH} compute firewall-rules describe $rule_name --project ${PROJECT_NAME} >/dev/null 2>&1; then
        echo -e "${YELLOW}Creating Firewall Rule $rule_name...${NC}"
        ${GCLOUD_PATH} compute firewall-rules create $rule_name "$@"
    else
        echo -e "${GREEN}Firewall Rule $rule_name already exists, skipping.${NC}"
    fi
}

check_and_create_fw $USER-h100-vpc-1-fw-rule --network ${NETWORK_NAME} --allow tcp:22,tcp:3389,icmp
check_and_create_fw $USER-h100-allow-internal --network ${NETWORK_NAME} --allow tcp,udp,icmp --direction INGRESS --priority 1000 --source-ranges ${FW_ALLOW_INTERNAL_SOURCE}
check_and_create_fw $USER-h100-master-to-nodes --network ${NETWORK_NAME} --allow tcp:10250,tcp:443 --direction INGRESS --priority 1000 --source-ranges ${MASTER_IPV4_CIDR}
check_and_create_fw $USER-h100-allow-all-external --network ${NETWORK_NAME} --allow tcp,udp,icmp --direction INGRESS --priority 1100 --source-ranges 0.0.0.0/0

# ==========================================
# 4. Cloud Router & NAT
# ==========================================
# Create a Cloud Router and NAT to allow private nodes to access the internet.
echo -e "${GREEN}[Step 5] Checking Router and NAT...${NC}"
if ! ${GCLOUD_PATH} compute routers describe $USER-router-h100 --region=${REGION} --project ${PROJECT_NAME} >/dev/null 2>&1; then
    echo -e "${YELLOW}Creating Router $USER-router-h100...${NC}"
    ${GCLOUD_PATH} compute routers create $USER-router-h100 \
         --network=${NETWORK_NAME} \
         --region=${REGION}
else
    echo -e "${GREEN}Router $USER-router-h100 already exists, skipping.${NC}"
fi

if ${GCLOUD_PATH} compute routers describe $USER-router-h100 --region=${REGION} --project ${PROJECT_NAME} | grep -q "name: $USER-nat-h100"; then
    echo -e "${GREEN}NAT $USER-nat-h100 already exists on router, skipping.${NC}"
else
    echo -e "${YELLOW}Creating NAT $USER-nat-h100...${NC}"
    ${GCLOUD_PATH} compute routers nats create $USER-nat-h100 \
         --router=$USER-router-h100 \
         --region=${REGION} \
         --auto-allocate-nat-external-ips \
         --nat-all-subnet-ip-ranges
fi

# ==========================================
# 5. Artifact Registry
# ==========================================
# Create an Artifact Registry repository for storing Docker images.
echo -e "${GREEN}[Step 6] Checking Artifact Registry...${NC}"
if ! ${GCLOUD_PATH} artifacts repositories describe $USER-docker-h100-repo --location=${REGION} --project ${PROJECT_NAME} >/dev/null 2>&1; then
    echo -e "${YELLOW}Creating Artifact Registry $USER-docker-h100-repo...${NC}"
    ${GCLOUD_PATH} artifacts repositories create $USER-docker-h100-repo \
         --repository-format=docker \
         --location=${REGION}
else
    echo -e "${GREEN}Artifact Registry $USER-docker-h100-repo already exists, skipping.${NC}"
fi

# ==========================================
# 6. GKE Cluster & Node Pools
# ==========================================
# Create the GKE cluster with private nodes, multi-networking, and Workload Identity enabled.
echo -e "${GREEN}[Step 7] Creating GKE Cluster ${CLUSTER_NAME}...${NC}"
${GCLOUD_PATH} beta container clusters create ${CLUSTER_NAME} \
     --zone=${ZONE} \
     --network=${NETWORK_NAME} \
     --subnetwork=${SUBNETWORK_NAME_1} \
     --cluster-secondary-range-name=$USER-pods \
     --services-secondary-range-name=$USER-services \
     --enable-ip-alias \
     --enable-private-nodes \
     --master-ipv4-cidr=${MASTER_IPV4_CIDR} \
     --no-enable-private-endpoint \
     --enable-multi-networking \
     --datapath-provider=advanced \
     --workload-pool=${PROJECT_NAME}.svc.id.goog \
     --enable-gcfs \
     --addons=GcsFuseCsiDriver \
     --enable-shielded-nodes \
     --enable-dns-access \
     --num-nodes=1

# Create the GPU node pool using specified reservation H100 instances
echo -e "${GREEN}[Step 8] Creating H100 GPU Node Pool ${GPU_POOL_NAME} from reservation ${RESERVATION_NAME}...${NC}"
${GCLOUD_PATH} beta container node-pools create ${GPU_POOL_NAME} \
     --cluster=${CLUSTER_NAME} \
     --zone=${ZONE} \
     --machine-type=${GPU_MACHINE_TYPE} \
     --num-nodes=${GPU_POOL_MIN_NODES} \
     --reservation-affinity=${RESERVATION_AFFINITY} \
     --reservation=${RESERVATION_NAME} \
     --accelerator=type=nvidia-h100-mega-80gb,count=8,gpu-driver-version=LATEST \
     --ephemeral-storage-local-ssd=count=16 \
     --enable-image-streaming \
     --workload-metadata=GKE_METADATA \
     --disk-size=${GPU_DISK_SIZE} \
     --scopes=https://www.googleapis.com/auth/cloud-platform \
     --additional-node-network=network=${NETWORK_NAME},subnetwork=${SUBNETWORK_NAME_2} \
     --additional-pod-network=subnetwork=${SUBNETWORK_NAME_2},pod-ipv4-range=$USER-additional-test-pods,max-pods-per-node=32

# Create a separate node pool for the benchmark client
echo -e "${GREEN}[Step 9] Creating Benchmark Client Node Pool ${COMPUTE_POOL_NAME}...${NC}"
${GCLOUD_PATH} container node-pools create ${COMPUTE_POOL_NAME} \
     --cluster=${CLUSTER_NAME} \
     --zone=${ZONE} \
     --machine-type=${COMPUTE_MACHINE_TYPE} \
     --num-nodes=1 \
     --disk-size=${COMPUTE_DISK_SIZE} \
     --workload-metadata=GKE_METADATA \
     --scopes=https://www.googleapis.com/auth/cloud-platform \
     --node-labels=workload=benchmark-client

# ==========================================
# 7. Kubernetes Authentication & Deployment
# ==========================================
# Fetch the credentials for the newly created cluster
echo -e "${GREEN}[Step 10] Fetching GKE Credentials...${NC}"
${GCLOUD_PATH} container clusters get-credentials ${CLUSTER_NAME} --zone ${ZONE} --project ${PROJECT_NAME} --dns-endpoint

echo -e "${GREEN}H100 Infrastructure Setup Complete!${NC}"
