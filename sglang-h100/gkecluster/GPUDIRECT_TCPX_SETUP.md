# GPUDirect TCPX Setup Guide for GKE H100 Cluster

This guide details the steps to set up a new GKE H100 (A3 Mega) cluster with **GPUDirect TCPX** enabled using the existing VPC network `a3-mega-h100-cluster-gpunet-0` and GKE multi-networking.

---

## 1. Architecture Overview
GPUDirect TCPX uses GKE Multi-Networking to attach 9 network interfaces (NICs) to each H100 VM:
* **NIC 0 (Primary)**: Standard cluster control plane, pod traffic, internet, and Hugging Face downloads.
* **NICs 1 - 8 (Secondary)**: 8 dedicated GPU network interfaces, each bound to a separate subnetwork within the `a3-mega-h100-cluster-gpunet-0` VPC network.

---

## 2. Infrastructure Setup Steps

### Step 2.1: Define Environment Variables
Before starting, set up these variables in your shell:
```bash
export PROJECT_NAME="northam-ce-mlai-tpu"
export REGION="us-east5"
export ZONE="us-east5-a"
export NETWORK_NAME="a3-mega-h100-cluster-gpunet-0"
export CLUSTER_NAME="h100-tcpx-gke-cluster"
export GPU_POOL_NAME="h100-tcpx-pool"
export DISK_SIZE="300"
```

### Step 2.2: Create the Subnetworks
Create 1 main subnet for GKE cluster orchestration, and 8 dedicated subnets inside the `a3-mega-h100-cluster-gpunet-0` VPC for the GPU interfaces.

```bash
# 1. Main subnet
gcloud compute networks subnets create "tcpx-main-subnet" \
     --network="${NETWORK_NAME}" \
     --range="10.80.0.0/20" \
     --region="${REGION}" \
     --secondary-range="tcpx-pods=10.81.0.0/20,tcpx-services=10.81.16.0/20" \
     --project="${PROJECT_NAME}"

# 2. 8 TCPX Subnets
for i in {1..8}
do
  gcloud compute networks subnets create "tcpx-subnet-$i" \
    --network="${NETWORK_NAME}" \
    --region="${REGION}" \
    --range="10.82.$(( (i-1) * 16 )).0/20" \
    --project="${PROJECT_NAME}"
done
```

### Step 2.3: Create GKE Cluster with Multi-Networking
Initialize the GKE cluster, ensuring that `--enable-multi-networking` and `--datapath-provider=advanced` (Datapath V2/Cilium) are set:

```bash
gcloud beta container clusters create "${CLUSTER_NAME}" \
     --zone="${ZONE}" \
     --network="${NETWORK_NAME}" \
     --subnetwork="tcpx-main-subnet" \
     --cluster-secondary-range-name="tcpx-pods" \
     --services-secondary-range-name="tcpx-services" \
     --enable-ip-alias \
     --enable-private-nodes \
     --master-ipv4-cidr="172.16.1.48/28" \
     --no-enable-private-endpoint \
     --enable-multi-networking \
     --datapath-provider=advanced \
     --workload-pool="${PROJECT_NAME}.svc.id.goog" \
     --enable-gcfs \
     --addons=GcsFuseCsiDriver \
     --enable-shielded-nodes \
     --enable-dns-access \
     --num-nodes=1 \
     --project="${PROJECT_NAME}"
```

### Step 2.4: Create the H100 Node Pool with 8 Network Attachments
Attach the node pool to the 8 TCPX subnetworks.

```bash
gcloud beta container node-pools create "${GPU_POOL_NAME}" \
     --cluster="${CLUSTER_NAME}" \
     --zone="${ZONE}" \
     --machine-type="a3-megagpu-8g" \
     --num-nodes=2 \
     --accelerator=type=nvidia-h100-mega-80gb,count=8,gpu-driver-version=LATEST \
     --ephemeral-storage-local-ssd=count=16 \
     --enable-image-streaming \
     --workload-metadata=GKE_METADATA \
     --disk-size="${DISK_SIZE}" \
     --scopes=https://www.googleapis.com/auth/cloud-platform \
     --project="${PROJECT_NAME}" \
     --additional-node-network=network="${NETWORK_NAME}",subnetwork=tcpx-subnet-1 \
     --additional-node-network=network="${NETWORK_NAME}",subnetwork=tcpx-subnet-2 \
     --additional-node-network=network="${NETWORK_NAME}",subnetwork=tcpx-subnet-3 \
     --additional-node-network=network="${NETWORK_NAME}",subnetwork=tcpx-subnet-4 \
     --additional-node-network=network="${NETWORK_NAME}",subnetwork=tcpx-subnet-5 \
     --additional-node-network=network="${NETWORK_NAME}",subnetwork=tcpx-subnet-6 \
     --additional-node-network=network="${NETWORK_NAME}",subnetwork=tcpx-subnet-7 \
     --additional-node-network=network="${NETWORK_NAME}",subnetwork=tcpx-subnet-8
```

---

## 3. TCPX Plugin Installation on GKE
Once the cluster is up, deploy the GPUDirect TCPX driver and setup daemonset to install the NCCL TCPX plugin (`libnccl-net.so`) to the nodes.

### Step 3.1: Connect to Cluster
```bash
gcloud container clusters get-credentials "${CLUSTER_NAME}" --zone "${ZONE}" --project "${PROJECT_NAME}" --dns-endpoint
```

### Step 3.2: Deploy GKE TCPXO DaemonSet
Apply the official Google TCPXO installer manifest for A3 Mega nodes:
```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/gpudirect-tcpxo/nccl-tcpxo-installer.yaml
```
*(This daemonset installs the TCPXO libraries to `/home/kubernetes/bin/tcpx` on the host).*

---

## 4. Serving Configuration changes for SGLang

To enable GPUDirect TCPX inside the SGLang container:

### Step 4.1: Volume Mounts
Update the Pod spec to mount the host's `/home/kubernetes/bin/tcpx` directory:
```yaml
      containers:
      - name: sglang-container
        volumeMounts:
        - mountPath: /usr/local/tcpx
          name: tcpx-library
...
      volumes:
      - name: tcpx-library
        hostPath:
          path: /home/kubernetes/bin/tcpx/
```

### Step 4.2: Environment Variables
Add the following TCPX configuration environment variables to enable the NCCL plugin:
```yaml
        env:
        # Enable IB/RDMA interfaces (TCPX runs as an IB provider)
        - name: NCCL_IB_DISABLE
          value: "0"
        # Point to the custom TCPX libraries
        - name: LD_LIBRARY_PATH
          value: "/usr/local/tcpx/lib64:/usr/local/nvidia/lib64"
        # Bind NCCL network to host interfaces 1-8 (GPUDirect NICs)
        - name: NCCL_SOCKET_IFNAME
          value: "eth0"
        # TCPX Tuning settings
        - name: NCCL_GPUDIRECTTCPX_FORCE_RX_FILE
          value: "1"
        - name: NCCL_GPUDIRECTTCPX_PROGRAM_FLOW_STEERING_WAIT_MICROS
          value: "1000000"
        - name: NCCL_NET_GDR_LEVEL
          value: "5"
```
*(With these variables, SGLang will route all distributed tensor activations (via TP/PP) using GPUDirect TCPX over the 8 physical GPU interfaces).*
