# H200 GKE Cluster & SGLang Benchmarking Setup

This document records the design decisions, network details, and step-by-step procedures to provision a 2-node H200 GKE cluster with GPUDirect TCPX enabled in zone `europe-west4-a`, and run SGLang inference benchmarks.

---

## 1. Plan Overview

### Phase 1: Free Up the H200 Zonal Reservation (Downtime required for Nemo)
* **Goal**: Free up the `pm-h200-testing` reservation in `europe-west4-a`.
* **Action**: Delete the node pool `gpu-pool-1` in GKE cluster `nemo-rl-cluster`.
* **Verification**: Verify that the reservation shows 0 consumed resources (`IN_USE_COUNT = 0`).

### Phase 2: Subnetworks Provisioning
* **Goal**: Create GKE orchestration and TCPX RDMA subnetworks in `europe-west4` under the shared VPC `a3-mega-h100-cluster-gpunet-0`.
* **Action**:
  - Main orchestration subnet: `h200-tcpx-main-subnet` (CIDR: `10.88.0.0/20`, Pods: `10.89.0.0/20`, Services: `10.89.16.0/20`).
  - 8 TCPX Subnets: `h200-tcpx-subnet-1` to `h200-tcpx-subnet-8` (CIDR ranges starting at `10.90.0.0/20` up to `10.90.112.0/20` incremented by `16` in the third octet).

### Phase 3: Provision H200 GKE Cluster with TCPX
* **Goal**: Launch a GKE cluster `h200-tcpx-gke-cluster` and node pool `h200-tcpx-pool` configured with GPUDirect TCPX (8 network attachments).
* **Action**:
  - GKE Cluster: Private nodes, `--enable-multi-networking`, and Cilium DPv2.
  - Node Pool: `a3-megagpu-8g` (H200s), 2 nodes, using the zonal reservation `pm-h200-testing`, and attached to the 8 secondary subnets.

### Phase 4: SGLang Serving & Benchmarking Configuration
* **Goal**: Deploy SGLang serving and run performance benchmarks.
* **Action**:
  - Deploy `nccl-tcpxo-installer` and configure TCPX.
  - Create SGLang Kubernetes deployment manifest for serving (e.g. Kimi-K2.6).
  - Create benchmarking client job.

---

## 2. Phase 1 Execution: Deleting Existing Node Pool

Before we can allocate the H200 GPUs to our new cluster, we must delete the existing node pool that is occupying the reservation:
* **Cluster**: `nemo-rl-cluster` (Location: `europe-west4`)
* **Node Pool**: `gpu-pool-1`
* **Reservation**: `pm-h200-testing`

## 2. Infrastructure Setup & Verification Logs

### Phase 1: Freeing up the Reservation `pm-h200-testing`
The node pool `gpu-pool-1` of `nemo-rl-cluster` was successfully deleted:
```bash
gcloud container node-pools delete gpu-pool-1 \
  --cluster nemo-rl-cluster \
  --location europe-west4 \
  --project northam-ce-mlai-tpu \
  --quiet
```
We verified the reservation was released:
```bash
gcloud compute reservations list --project northam-ce-mlai-tpu | grep pm-h200-testing
# Output: pm-h200-testing    0    2    europe-west4-a    LOCAL
```

### Phase 2: Orchestration Subnet Provisioning
We created a main subnetwork inside `a3-mega-h100-cluster-gpunet-0` network for standard control-plane and pod traffic:
```bash
gcloud compute networks subnets create "h200-tcpx-main-subnet" \
     --network="a3-mega-h100-cluster-gpunet-0" \
     --range="10.88.0.0/20" \
     --region="europe-west4" \
     --secondary-range="h200-tcpx-pods=10.89.0.0/20,h200-tcpx-services=10.89.16.0/20" \
     --project="northam-ce-mlai-tpu"
```

### Phase 3: Provisioning H200 GKE Cluster and GPU Node Pools
#### GCE/GKE Network Constraints for H200 (A3 Ultra):
1. **Network Profile Requirement**: The Mellanox high-speed interfaces (`MRDMA` type NICs) used on H200 nodes require a VPC network associated with a **GCE RDMA/RoCE Network Profile** (e.g. `nemo-rdma-net` which carries `europe-west4-a-vpc-roce`).
2. **Interface Sequence Requirement**: GCE requires that the first additional network interface (`nic1`) must be a standard `GVNIC` interface (without a network profile). The remaining 8 interfaces (`nic2` to `nic9`) must be `MRDMA` interfaces connected to the RoCE network.

#### Provisioning Commands:
1. **Create the GKE Cluster**:
   ```bash
   gcloud beta container clusters create "h200-tcpx-gke-cluster" \
        --zone="europe-west4-a" \
        --network="a3-mega-h100-cluster-gpunet-0" \
        --subnetwork="h200-tcpx-main-subnet" \
        --cluster-secondary-range-name="h200-tcpx-pods" \
        --services-secondary-range-name="h200-tcpx-services" \
        --enable-ip-alias \
        --enable-private-nodes \
        --master-ipv4-cidr="172.16.2.48/28" \
        --no-enable-private-endpoint \
        --enable-multi-networking \
        --datapath-provider=advanced \
        --workload-pool="northam-ce-mlai-tpu.svc.id.goog" \
        --enable-gcfs \
        --addons=GcsFuseCsiDriver \
        --enable-shielded-nodes \
        --enable-dns-access \
        --num-nodes=1 \
        --project="northam-ce-mlai-tpu"
   ```

2. **Create the H200 Node Pool**:
   Attached `nic1` to the standard `nemo-gvnic-net` (using GVNIC) and `nic2` through `nic9` to `nemo-rdma-net` (using MRDMA):
   ```bash
   gcloud beta container node-pools create "h200-tcpx-pool" \
        --cluster="h200-tcpx-gke-cluster" \
        --zone="europe-west4-a" \
        --machine-type="a3-ultragpu-8g" \
        --num-nodes=2 \
        --accelerator=type=nvidia-h200-141gb,count=8,gpu-driver-version=LATEST \
        --ephemeral-storage-local-ssd=count=32 \
        --enable-image-streaming \
        --workload-metadata=GKE_METADATA \
        --disk-size=300 \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --reservation-affinity=specific \
        --reservation="pm-h200-testing" \
        --project="northam-ce-mlai-tpu" \
        --additional-node-network=network="nemo-gvnic-net",subnetwork="nemo-gvnic-sub" \
        --additional-node-network=network="nemo-rdma-net",subnetwork="nemo-rdma-sub-0" \
        --additional-node-network=network="nemo-rdma-net",subnetwork="nemo-rdma-sub-1" \
        --additional-node-network=network="nemo-rdma-net",subnetwork="nemo-rdma-sub-2" \
        --additional-node-network=network="nemo-rdma-net",subnetwork="nemo-rdma-sub-3" \
        --additional-node-network=network="nemo-rdma-net",subnetwork="nemo-rdma-sub-4" \
        --additional-node-network=network="nemo-rdma-net",subnetwork="nemo-rdma-sub-5" \
        --additional-node-network=network="nemo-rdma-net",subnetwork="nemo-rdma-sub-6" \
        --additional-node-network=network="nemo-rdma-net",subnetwork="nemo-rdma-sub-7"
   ```

3. **Create the Benchmark Client Node Pool**:
   ```bash
   gcloud container node-pools create "benchmark-client-pool-h200" \
        --cluster="h200-tcpx-gke-cluster" \
        --zone="europe-west4-a" \
        --machine-type="n2-standard-64" \
        --num-nodes=1 \
        --disk-size=300 \
        --workload-metadata=GKE_METADATA \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --node-labels=workload=benchmark-client \
        --project="northam-ce-mlai-tpu"
   ```

### Phase 4: Internet Access (Cloud NAT Setup)
To allow our private GKE nodes to fetch images and connect to Hugging Face, we provisioned a Cloud Router and Cloud NAT gateway:
```bash
# 1. Create Cloud Router
gcloud compute routers create "h200-tcpx-router" \
     --network="a3-mega-h100-cluster-gpunet-0" \
     --region="europe-west4" \
     --project="northam-ce-mlai-tpu"

# 2. Create Cloud NAT Gateway
gcloud compute routers nats create "h200-tcpx-nat" \
     --router="h200-tcpx-router" \
     --region="europe-west4" \
     --auto-allocate-nat-external-ips \
     --nat-all-subnet-ip-ranges \
     --project="northam-ce-mlai-tpu"
```

---

## 3. GPUDirect RDMA (RoCE) Setup on GKE

Instead of GPUDirect TCPXO (which is optimized for A3 Mega/H100), H200s (A3 Ultra) use native GPUDirect RDMA/RoCE using the Google GIB network driver.

### Step 3.1: Connect to Cluster
```bash
gcloud container clusters get-credentials h200-tcpx-gke-cluster --zone europe-west4-a --project northam-ce-mlai-tpu
```

### Step 3.2: Deploy GKE GPUDirect RDMA Installer DaemonSet
Apply the official Google RDMA installer manifest:
```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/gpudirect-rdma/nccl-rdma-installer.yaml
```
*(This daemonset installs the optimized RDMA and NCCL/gIB libraries to `/home/kubernetes/bin/gib` and `/home/kubernetes/bin/nvidia` on the host).*

---

## 4. Run SGLang Inference Benchmark

### Step 4.1: Deploy SGLang Serving (StatefulSet)
1. Ensure the Hugging Face credentials token secret exists:
   ```bash
   kubectl create secret generic hf-secret \
     --from-literal=HF_TOKEN=$HF_TOKEN
   ```
2. Deploy the SGLang Serving StatefulSet:
   ```bash
   kubectl apply -f Inference/sglang-h200/models/KimiK2.6/sglang-kimi-26-2node-h200.yaml
   ```
3. Monitor the serving pods until they are healthy and ready:
   ```bash
   kubectl get pods -w
   ```

### Step 4.2: Run the Benchmark Client Job
1. Deploy the benchmark client Pod:
   ```bash
   kubectl apply -f Inference/sglang-h200/models/KimiK2.6/benchmark-kimik26-h200.yaml
   ```
2. Follow the client execution logs to see the concurrency test progress:
   ```bash
   kubectl logs -f sglang-kimik26-benchmark
   ```
3. Extract the results when finished:
   ```bash
   kubectl cp sglang-kimik26-benchmark:/workspace/results_kimik26.json ./results_kimik26.json
   ```


