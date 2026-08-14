#!/bin/bash
set -e

# ==============================================================================
# GKE 5-Node Cluster & Node Pool Provisioning for 1M Context Kimi-K2.6
# Target Reservation: pm-crwd-poc (us-east5-a)
# ==============================================================================

export PROJECT_NAME="northam-ce-mlai-tpu"
export REGION="us-east5"
export ZONE="us-east5-a"

export NETWORK_NAME="pm-g4-vpc-useast5"
export SUBNETWORK_NAME_1="pm-subnet-1-useast5"
export SUBNETWORK_NAME_2="pm-subnet-2-useast5"

export CLUSTER_NAME="pm-g4-1m-cluster"
export GPU_POOL_NAME="g4-384-pool-pm-crwd"
export RESERVATION_NAME="pm-crwd-poc"

export GPU_MACHINE_TYPE="g4-standard-384"
export NUM_GPU_NODES=5

echo "======================================================================"
echo "Step 1: Setting GCP Project to ${PROJECT_NAME}..."
echo "======================================================================"
gcloud config set project ${PROJECT_NAME}

echo "======================================================================"
echo "Step 2: Ensuring VPC Network (${NETWORK_NAME}) in ${REGION}..."
echo "======================================================================"
if ! gcloud compute networks describe ${NETWORK_NAME} --project ${PROJECT_NAME} >/dev/null 2>&1; then
    gcloud compute networks create ${NETWORK_NAME} --subnet-mode=custom --mtu=8896
fi

if ! gcloud compute networks subnets describe ${SUBNETWORK_NAME_1} --region=${REGION} --project ${PROJECT_NAME} >/dev/null 2>&1; then
    gcloud compute networks subnets create ${SUBNETWORK_NAME_1} \
         --network=${NETWORK_NAME} \
         --range=10.100.0.0/20 \
         --region=${REGION} \
         --secondary-range=pm-pods=10.101.0.0/20,pm-services=10.101.16.0/20
fi

if ! gcloud compute networks subnets describe ${SUBNETWORK_NAME_2} --region=${REGION} --project ${PROJECT_NAME} >/dev/null 2>&1; then
    gcloud compute networks subnets create ${SUBNETWORK_NAME_2} \
         --network=${NETWORK_NAME} \
         --range=10.100.16.0/20 \
         --region=${REGION} \
         --secondary-range=pm-2nic-pods=10.120.16.0/20,pm-2nic-services=10.120.32.0/20
fi

echo "======================================================================"
echo "Step 3: Creating Proxy-Only Subnet for Gateway API (gke-l7-rilb)..."
echo "======================================================================"
if ! gcloud compute networks subnets describe pm-proxy-subnet --region=${REGION} --project ${PROJECT_NAME} >/dev/null 2>&1; then
    gcloud compute networks subnets create pm-proxy-subnet \
         --purpose=REGIONAL_MANAGED_PROXY \
         --role=ACTIVE \
         --region=${REGION} \
         --network=${NETWORK_NAME} \
         --range=172.16.1.0/24
fi

echo "======================================================================"
echo "Step 4: Creating GKE Cluster ${CLUSTER_NAME} with Gateway API & GCS Fuse..."
echo "======================================================================"
if ! gcloud container clusters describe ${CLUSTER_NAME} --zone=${ZONE} --project=${PROJECT_NAME} >/dev/null 2>&1; then
    gcloud beta container clusters create ${CLUSTER_NAME} \
         --zone=${ZONE} \
         --network=${NETWORK_NAME} \
         --subnetwork=${SUBNETWORK_NAME_1} \
         --cluster-secondary-range-name=pm-pods \
         --services-secondary-range-name=pm-services \
         --enable-ip-alias \
         --enable-private-nodes \
         --master-ipv4-cidr=172.16.0.16/28 \
         --no-enable-private-endpoint \
         --gateway-api=standard \
         --enable-multi-networking \
         --datapath-provider=advanced \
         --workload-pool=${PROJECT_NAME}.svc.id.goog \
         --addons=GcsFuseCsiDriver \
         --num-nodes=1
fi

echo "======================================================================"
echo "Step 5: Creating GPU Node Pool Consuming Reservation '${RESERVATION_NAME}'..."
echo "======================================================================"
if ! gcloud container node-pools describe ${GPU_POOL_NAME} --cluster=${CLUSTER_NAME} --zone=${ZONE} --project=${PROJECT_NAME} >/dev/null 2>&1; then
    gcloud beta container node-pools create ${GPU_POOL_NAME} \
         --cluster=${CLUSTER_NAME} \
         --zone=${ZONE} \
         --machine-type=${GPU_MACHINE_TYPE} \
         --num-nodes=${NUM_GPU_NODES} \
         --reservation-affinity=specific \
         --reservation=${RESERVATION_NAME} \
         --accelerator=type=nvidia-rtx-pro-6000,count=8,gpu-driver-version=LATEST \
         --enable-image-streaming \
         --workload-metadata=GKE_METADATA \
         --scopes=https://www.googleapis.com/auth/cloud-platform \
         --additional-node-network=network=${NETWORK_NAME},subnetwork=${SUBNETWORK_NAME_2} \
         --additional-pod-network=subnetwork=${SUBNETWORK_NAME_2},pod-ipv4-range=pm-2nic-pods,max-pods-per-node=32
fi

echo "======================================================================"
echo "Step 6: Fetching Credentials for Cluster ${CLUSTER_NAME}..."
echo "======================================================================"
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone=${ZONE} --project=${PROJECT_NAME}

echo "======================================================================"
echo "Cluster Provisioning Complete! Target Reservation: ${RESERVATION_NAME}"
echo "======================================================================"
