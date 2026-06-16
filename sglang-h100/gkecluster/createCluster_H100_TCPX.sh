#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Colors for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ==========================================
# 1. Environment Variables (Overridable)
# ==========================================
export PROJECT_NAME="${PROJECT_NAME:-northam-ce-mlai-tpu}"
if [ -z "$PROJECT_NAME" ]; then
    echo -e "${RED}ERROR: PROJECT_NAME is not set! Please set it in your environment.${NC}"
    exit 1
fi

export REGION="${REGION:-us-east5}"
export ZONE="${ZONE:-us-east5-a}"

# VPC Network to Reuse (MUST already exist)
export NETWORK_NAME="${NETWORK_NAME:-a3-mega-h100-cluster-gpunet-0}"

# New Cluster Details
export CLUSTER_NAME="${CLUSTER_NAME:-$USER-gke-h100-tcpx-cluster}"
export GPU_POOL_NAME="${GPU_POOL_NAME:-$USER-h100-tcpx-pool}"
export COMPUTE_POOL_NAME="${COMPUTE_POOL_NAME:-benchmark-client-pool-tcpx}"

# New Isolated CIDR Ranges (Adjust if needed to avoid conflicts)
export NEW_MAIN_SUBNET_NAME="${NEW_MAIN_SUBNET_NAME:-$USER-tcpx-main-subnet}"
export NEW_MAIN_SUBNET_RANGE="${NEW_MAIN_SUBNET_RANGE:-10.80.0.0/20}"
export NEW_PODS_RANGE="${NEW_PODS_RANGE:-10.81.0.0/20}"
export NEW_SERVICES_RANGE="${NEW_SERVICES_RANGE:-10.81.16.0/20}"
export MASTER_IPV4_CIDR="${MASTER_IPV4_CIDR:-172.16.1.48/28}"

# Machine Types and Scaling
export GPU_MACHINE_TYPE="${GPU_MACHINE_TYPE:-a3-megagpu-8g}"
export COMPUTE_MACHINE_TYPE="${COMPUTE_MACHINE_TYPE:-n2-standard-64}"
export GPU_POOL_MIN_NODES="${GPU_POOL_MIN_NODES:-2}"
export GPU_POOL_MAX_NODES="${GPU_POOL_MAX_NODES:-2}"
export GPU_DISK_SIZE="${GPU_DISK_SIZE:-300}"
export COMPUTE_DISK_SIZE="${COMPUTE_DISK_SIZE:-300}"

# Reservation Configuration (Optional, leave blank if not using)
export RESERVATION_AFFINITY="${RESERVATION_AFFINITY:-specific}"
export RESERVATION_NAME="${RESERVATION_NAME:-pm-h100s}"

# GCloud SDK Path
export GCLOUD_PATH="${GCLOUD_PATH:-gcloud}"

# ==========================================
# 2. Project Setup
# ==========================================
echo -e "${GREEN}[Step 1] Setting default project to ${PROJECT_NAME}...${NC}"
${GCLOUD_PATH} config set project ${PROJECT_NAME}

# Check if network exists
echo -e "${GREEN}[Step 2] Verifying existing VPC Network ${NETWORK_NAME}...${NC}"
if ! ${GCLOUD_PATH} compute networks describe ${NETWORK_NAME} --project ${PROJECT_NAME} >/dev/null 2>&1; then
    echo -e "${RED}ERROR: VPC Network ${NETWORK_NAME} does not exist. Please check NETWORK_NAME or create the network first.${NC}"
    exit 1
fi

# ==========================================
# 3. Create Subnets
# ==========================================
echo -e "${GREEN}[Step 3] Creating new main subnet for GKE cluster...${NC}"
if ! ${GCLOUD_PATH} compute networks subnets describe ${NEW_MAIN_SUBNET_NAME} --region=${REGION} --project ${PROJECT_NAME} >/dev/null 2>&1; then
    ${GCLOUD_PATH} compute networks subnets create ${NEW_MAIN_SUBNET_NAME} \
         --network=${NETWORK_NAME} \
         --range=${NEW_MAIN_SUBNET_RANGE} \
         --region=${REGION} \
         --secondary-range=$USER-tcpx-pods=${NEW_PODS_RANGE},$USER-tcpx-services=${NEW_SERVICES_RANGE}
else
    echo -e "${YELLOW}Subnet ${NEW_MAIN_SUBNET_NAME} already exists, skipping.${NC}"
fi

echo -e "${GREEN}[Step 4] Creating 8 subnets inside ${NETWORK_NAME} VPC for TCPX interfaces...${NC}"
for i in {1..8}
do
  SUBNET_NAME="$USER-tcpx-subnet-$i"
  # Carve ranges out of 10.82.X.0 block
  SUBNET_RANGE="10.82.$(( (i-1) * 16 )).0/20"
  
  if ! ${GCLOUD_PATH} compute networks subnets describe ${SUBNET_NAME} --region=${REGION} --project ${PROJECT_NAME} >/dev/null 2>&1; then
      echo -e "${YELLOW}Creating Subnet: ${SUBNET_NAME} with range ${SUBNET_RANGE}...${NC}"
      ${GCLOUD_PATH} compute networks subnets create ${SUBNET_NAME} \
           --network=${NETWORK_NAME} \
           --range=${SUBNET_RANGE} \
           --region=${REGION}
  else
      echo -e "${YELLOW}Subnet ${SUBNET_NAME} already exists, skipping.${NC}"
  fi
done

# ==========================================
# 4. Firewall Rules
# ==========================================
echo -e "${GREEN}[Step 5] Checking Firewall Rules for new cluster...${NC}"
check_and_create_fw() {
    local rule_name=$1
    shift
    if ! ${GCLOUD_PATH} compute firewall-rules describe $rule_name --project ${PROJECT_NAME} >/dev/null 2>&1; then
        echo -e "${YELLOW}Creating Firewall Rule $rule_name...${NC}"
        ${GCLOUD_PATH} compute firewall-rules create $rule_name "$@"
    else
        echo -e "${YELLOW}Firewall Rule $rule_name already exists, skipping.${NC}"
    fi
}

check_and_create_fw $USER-tcpx-allow-internal --network ${NETWORK_NAME} --allow tcp,udp,icmp --direction INGRESS --priority 1000 --source-ranges 10.80.0.0/12
check_and_create_fw $USER-tcpx-master-to-nodes --network ${NETWORK_NAME} --allow tcp:10250,tcp:443 --direction INGRESS --priority 1000 --source-ranges ${MASTER_IPV4_CIDR}

# ==========================================
# 5. GKE Cluster Creation
# ==========================================
echo -e "${GREEN}[Step 6] Creating GKE Cluster ${CLUSTER_NAME}...${NC}"
if ! ${GCLOUD_PATH} container clusters describe ${CLUSTER_NAME} --zone=${ZONE} --project ${PROJECT_NAME} >/dev/null 2>&1; then
    ${GCLOUD_PATH} beta container clusters create ${CLUSTER_NAME} \
         --zone=${ZONE} \
         --network=${NETWORK_NAME} \
         --subnetwork=${NEW_MAIN_SUBNET_NAME} \
         --cluster-secondary-range-name=$USER-tcpx-pods \
         --services-secondary-range-name=$USER-tcpx-services \
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
else
    echo -e "${YELLOW}Cluster ${CLUSTER_NAME} already exists, skipping.${NC}"
fi

# ==========================================
# 6. GPU Node Pool with Multi-NIC (8 Subnets)
# ==========================================
echo -e "${GREEN}[Step 7] Creating H100 Node Pool ${GPU_POOL_NAME} (GPUDirect Multi-NIC)...${NC}"
if ! ${GCLOUD_PATH} container node-pools describe ${GPU_POOL_NAME} --cluster=${CLUSTER_NAME} --zone=${ZONE} --project ${PROJECT_NAME} >/dev/null 2>&1; then
    
    # Construct reservation arguments if configured
    RESERVATION_ARGS=""
    if [ -n "$RESERVATION_NAME" ]; then
        RESERVATION_ARGS="--reservation-affinity=${RESERVATION_AFFINITY} --reservation=${RESERVATION_NAME}"
    fi

    ${GCLOUD_PATH} beta container node-pools create ${GPU_POOL_NAME} \
         --cluster=${CLUSTER_NAME} \
         --zone=${ZONE} \
         --machine-type=${GPU_MACHINE_TYPE} \
         --num-nodes=${GPU_POOL_MIN_NODES} \
         ${RESERVATION_ARGS} \
         --accelerator=type=nvidia-h100-mega-80gb,count=8,gpu-driver-version=LATEST \
         --ephemeral-storage-local-ssd=count=16 \
         --enable-image-streaming \
         --workload-metadata=GKE_METADATA \
         --disk-size=${GPU_DISK_SIZE} \
         --scopes=https://www.googleapis.com/auth/cloud-platform \
         --additional-node-network=network=${NETWORK_NAME},subnetwork=$USER-tcpx-subnet-1 \
         --additional-node-network=network=${NETWORK_NAME},subnetwork=$USER-tcpx-subnet-2 \
         --additional-node-network=network=${NETWORK_NAME},subnetwork=$USER-tcpx-subnet-3 \
         --additional-node-network=network=${NETWORK_NAME},subnetwork=$USER-tcpx-subnet-4 \
         --additional-node-network=network=${NETWORK_NAME},subnetwork=$USER-tcpx-subnet-5 \
         --additional-node-network=network=${NETWORK_NAME},subnetwork=$USER-tcpx-subnet-6 \
         --additional-node-network=network=${NETWORK_NAME},subnetwork=$USER-tcpx-subnet-7 \
         --additional-node-network=network=${NETWORK_NAME},subnetwork=$USER-tcpx-subnet-8
else
    echo -e "${YELLOW}Node Pool ${GPU_POOL_NAME} already exists, skipping.${NC}"
fi

# ==========================================
# 7. Benchmark Client Node Pool
# ==========================================
echo -e "${GREEN}[Step 8] Creating Benchmark Client Node Pool ${COMPUTE_POOL_NAME}...${NC}"
if ! ${GCLOUD_PATH} container node-pools describe ${COMPUTE_POOL_NAME} --cluster=${CLUSTER_NAME} --zone=${ZONE} --project ${PROJECT_NAME} >/dev/null 2>&1; then
    ${GCLOUD_PATH} container node-pools create ${COMPUTE_POOL_NAME} \
         --cluster=${CLUSTER_NAME} \
         --zone=${ZONE} \
         --machine-type=${COMPUTE_MACHINE_TYPE} \
         --num-nodes=1 \
         --disk-size=${COMPUTE_DISK_SIZE} \
         --workload-metadata=GKE_METADATA \
         --scopes=https://www.googleapis.com/auth/cloud-platform \
         --node-labels=workload=benchmark-client
else
    echo -e "${YELLOW}Node Pool ${COMPUTE_POOL_NAME} already exists, skipping.${NC}"
fi

# ==========================================
# 8. Fetch GKE Credentials
# ==========================================
echo -e "${GREEN}[Step 9] Fetching credentials for ${CLUSTER_NAME}...${NC}"
${GCLOUD_PATH} container clusters get-credentials ${CLUSTER_NAME} --zone ${ZONE} --project ${PROJECT_NAME} --dns-endpoint

echo -e "${GREEN}GPUDirect TCPX Cluster Infrastructure Setup Complete!${NC}"
