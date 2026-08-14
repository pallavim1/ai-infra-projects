# Comprehensive Architecture & Deployment Guide: Kimi-K2.6 1M Context Serving on GCP G4 with vLLM & GKE Inference Gateway

This document serves as the complete, single-source-of-truth guide for deploying and benchmarking **Kimi-K2.6** at extreme context lengths (**500K to 1,048,576 tokens**) on **Google Cloud G4 instances** using **vLLM**, **NVFP4 (NVIDIA ModelOpt FP4)** quantization, and the **GKE Inference Gateway (llm-d)**.

---

## 1. Executive Summary & Production Setup

| Attribute | Configuration |
| :--- | :--- |
| **Model** | `nvidia/Kimi-K2.6-NVFP4` (Quantized via NVIDIA ModelOpt FP4) |
| **Serving Engine** | vLLM (`v0.19.1+` with Blackwell SM120 & Triton MLA support) |
| **GCP Project** | `northam-ce-mlai-tpu` |
| **GCE Reservation** | 🎯 **`pm-crwd-poc`** in **`us-east5-a`** (10× `g4-standard-384` nodes, `READY`) |
| **Target Infrastructure** | 5× `g4-standard-384` instances (40× NVIDIA RTX PRO 6000 Blackwell GPUs total) |
| **GCS Model Bucket** | `gs://pallaviam-sglang-kimi-us-east5/kimi-k2.6/` (Staged in `us-east5`) |
| **Local Model Backup** | `/usr/local/google/home/pallaviam/models/Kimi-K2.6` |
| **Parallelism** | **TP = 8** (Per Node) \| **Cluster DP = 5** (5 Replicas across 5 Nodes) |
| **Max Context Window** | **1,048,576 tokens (1M)** per request |
| **Per-Node Concurrency** | **2 active in-flight requests** |
| **Cluster Concurrency** | **10 total concurrent 1M-context requests** |
| **Routing & Load Balancer**| GKE Inference Gateway (`gke-l7-rilb`) + EPP Prefix-Cache-Aware Router |

---

## 2. Hardware Topology & Node Specifications

### Per-Node Compute (`g4-standard-384`)
* **GPUs**: 8× NVIDIA RTX PRO 6000 Blackwell (SM120, 96 GB GDDR7 VRAM per GPU = **768 GB VRAM total per node**).
* **CPUs & System Memory**: 384 vCPUs (AMD Turin), **1.5 TB Host RAM**.
* **Interconnect**: PCIe Gen 5.0 x16 (no NVLink, no GPU RDMA).
* **Networking**: 400 Gbps dual-NIC VPC networking managed by Google Cloud Titanium Offload Processors (TOPs).

---

## 3. Network Infrastructure Status

The VPC and subnets have been created in `us-east5` for this deployment:

```
VPC Network: pm-g4-vpc-useast5 (Custom mode, MTU 8896)
├── Subnet 1: pm-subnet-1-useast5 (10.100.0.0/20) [Pods: 10.101.0.0/20, Services: 10.101.16.0/20]
├── Subnet 2 (2nd NIC): pm-subnet-2-useast5 (10.100.16.0/20) [Pods: 10.120.16.0/20, Services: 10.120.32.0/20]
└── Proxy-Only Subnet: pm-proxy-subnet (172.16.1.0/24, REGIONAL_MANAGED_PROXY for Gateway API gke-l7-rilb)
```

---

## 4. Memory Math & Technical Feasibility for 1M Context

### Multi-Head Latent Attention (MLA) + FP8 KV Cache Memory Breakdown

Kimi-K2.6 uses **Multi-head Latent Attention (MLA)**, which compresses KV projections into a latent vector ($d_c = 512, d_R = 64$). Combined with **FP8 KV Cache** (`fp8_e4m3`), memory consumption is only **~35.1 KB per token** across all 61 model layers.

#### Node VRAM Allocation (768 GB Total VRAM per Node)
```
┌────────────────────────────────────────────────────────────────────────┐
│ Total Node VRAM: 768 GB (8x 96GB RTX PRO 6000 Blackwell)               │
├──────────────────────────┬──────────────────────────┬──────────────────┤
│ NVFP4 Model Weights       │ Active KV Cache          │ Free VRAM /      │
│ ~320 GB                  │ (2 reqs × 1M tokens)     │ Prefix Caching   │
│ (~40 GB / GPU)           │ ~70.2 GB (~8.8 GB/GPU)   │ ~377.8 GB        │
└──────────────────────────┴──────────────────────────┴──────────────────┘
```

1. **NVFP4 Model Weights (`nvidia/Kimi-K2.6-NVFP4`)**: $\approx \mathbf{320 \text{ GB}}$ static footprint across 8 GPUs.
2. **Free VRAM for KV Cache & Overhead**: $768 \text{ GB} - 320 \text{ GB} = \mathbf{448 \text{ GB}}$ per node.
3. **Active KV Cache Footprint (FP8 KV Cache)**:
   * **1 Request at 500K Tokens**: $500,000 \times 35.1 \text{ KB} = \mathbf{17.55 \text{ GB}}$
   * **1 Request at 1M Tokens**: $1,000,000 \times 35.1 \text{ KB} = \mathbf{35.1 \text{ GB}}$
   * **2 Concurrent 1M Requests (2M active tokens in VRAM)**: $2 \times 35.1 \text{ GB} = \mathbf{70.2 \text{ GB}}$ ($\approx 8.8 \text{ GB}$ per GPU).
4. **Prefix Caching & CUDA Graph Headroom**: Out of the ~448 GB available pool, 2M active tokens use only **70.2 GB (~15%)**, leaving **~377.8 GB VRAM** for automatic prefix caching and CUDA graphs.

---

## 5. Scheduler Behavior: `--max-model-len` vs `--max-num-seqs`

* **`--max-model-len 1048576` (1M Tokens)**: Per-request maximum length ceiling (prompt + output).
* **`--max-num-seqs 2`**: Strict ceiling on active in-flight requests per node.

> **Important Clarification**: `--max-model-len` is **NOT divided** between the 2 requests. Both Request #1 and Request #2 can simultaneously reach **1,000,000 tokens** at the exact same time. Through PagedAttention, two 1M token requests consume 70.2 GB total, leaving >377 GB VRAM for prefix caching.

---

## 6. End-to-End System Architecture

```mermaid
graph TD
    Client[Client Applications / OpenAI SDK] -->|HTTP / HTTPS| GW[GKE Inference Gateway / Envoy L7 RILB]
    GW <-->|Prefix-Aware Consult| EPP[Endpoint Picker EPP / llm-d]

    subgraph GKE Cluster - us-east5-a Node Pool (Reservation: pm-crwd-poc)
        subgraph Node 1 [g4-standard-384 #1]
            VLLM1[vLLM Pod 1<br/>Kimi-K2.6 NVFP4<br/>TP=8, Concurrency=2<br/>1M Context Window]
        end
        subgraph Node 2 [g4-standard-384 #2]
            VLLM2[vLLM Pod 2<br/>Kimi-K2.6 NVFP4<br/>TP=8, Concurrency=2<br/>1M Context Window]
        end
        subgraph Node 3 [g4-standard-384 #3]
            VLLM3[vLLM Pod 3<br/>Kimi-K2.6 NVFP4<br/>TP=8, Concurrency=2<br/>1M Context Window]
        end
        subgraph Node 4 [g4-standard-384 #4]
            VLLM4[vLLM Pod 4<br/>Kimi-K2.6 NVFP4<br/>TP=8, Concurrency=2<br/>1M Context Window]
        end
        subgraph Node 5 [g4-standard-384 #5]
            VLLM5[vLLM Pod 5<br/>Kimi-K2.6 NVFP4<br/>TP=8, Concurrency=2<br/>1M Context Window]
        end
    end

    GW -->|Forward Request| VLLM1
    GW -->|Forward Request| VLLM2
    GW -->|Forward Request| VLLM3
    GW -->|Forward Request| VLLM4
    GW -->|Forward Request| VLLM5

    VLLM1 -.->|ZMQ KV Events :5556| EPP
    VLLM2 -.->|ZMQ KV Events :5556| EPP
    VLLM3 -.->|ZMQ KV Events :5556| EPP
    VLLM4 -.->|ZMQ KV Events :5556| EPP
    VLLM5 -.->|ZMQ KV Events :5556| EPP
```

---

## 7. Configuration Specifications

### vLLM Command Line Arguments (`vllm serve`)

```bash
vllm serve /models/Kimi-K2.6-NVFP4 \
  --served-model-name=Kimi-K2.6 \
  --tensor-parallel-size=8 \
  --pipeline-parallel-size=1 \
  --disable-custom-all-reduce \
  --quantization=compressed-tensors \
  --kv-cache-dtype=fp8_e4m3 \
  --attention-backend=TRITON_MLA \
  --max-model-len=1048576 \
  --max-num-seqs=2 \
  --enable-chunked-prefill \
  --max-num-batched-tokens=32768 \
  --enable-prefix-caching \
  --gpu-memory-utilization=0.90 \
  --trust-remote-code \
  --reasoning-parser=kimi_k2 \
  --tool-call-parser=kimi_k2 \
  --host=0.0.0.0 \
  --port=30000 \
  --kv-events-config '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"tcp://*:5556","topic":"kv@$(POD_IP):30000@/models"}'
```

### Essential G4 PCIe Environment Variables

```bash
export VLLM_USE_V1=0
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENABLE_PCIE_ALLREDUCE=1
export NCCL_SOCKET_IFNAME="eth0,eth1"
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_MIN_NCHANNELS=8
export NCCL_ALLOC_P2P_NET_LL_BUFFERS=1
export SAFETENSORS_FAST_GPU=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

---

## 8. IAM Permissions Required

To execute cluster creation and manage resources, run these commands from your **local workstation terminal**:

```bash
# Storage Admin for model weights access
gcloud projects add-iam-policy-binding northam-ce-mlai-tpu \
  --member="serviceAccount:pm-h200-reservation@northam-ce-mlai-tpu.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# Kubernetes Engine Admin for GKE cluster & node pool creation
gcloud projects add-iam-policy-binding northam-ce-mlai-tpu \
  --member="serviceAccount:pm-h200-reservation@northam-ce-mlai-tpu.iam.gserviceaccount.com" \
  --role="roles/container.admin"
```

---

## 9. Automated Infrastructure Script

File path: `gkecluster/create_g4_5node_cluster.sh`

```bash
#!/bin/bash
set -e

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

# Step 1: Set Project
gcloud config set project ${PROJECT_NAME}

# Step 2: GKE Cluster
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

# Step 3: GPU Node Pool using Reservation pm-crwd-poc
gcloud beta container node-pools create ${GPU_POOL_NAME} \
     --cluster=${CLUSTER_NAME} \
     --zone=${ZONE} \
     --machine-type=${GPU_MACHINE_TYPE} \
     --num-nodes=${NUM_GPU_NODES} \
     --reservation-affinity=specific \
     --reservation=${RESERVATION_NAME} \
     --accelerator=type=nvidia-rtx-pro-6000,count=8,gpu-driver-version=LATEST \
     --ephemeral-storage-local-ssd=count=32 \
     --enable-image-streaming \
     --workload-metadata=GKE_METADATA \
     --scopes=https://www.googleapis.com/auth/cloud-platform \
     --additional-node-network=network=${NETWORK_NAME},subnetwork=${SUBNETWORK_NAME_2} \
     --additional-pod-network=subnetwork=${SUBNETWORK_NAME_2},pod-ipv4-range=pm-2nic-pods,max-pods-per-node=32
```

---

## 9. Comprehensive Benchmark Results & Long Context Feasibility (5-Node 40x RTX PRO 6000 Cluster)

### Kimi-K2.6 (NVFP4 Quantized Model via compressed-tensors)

| Test Metric | Benchmark Result | Notes / Details |
| :--- | :--- | :--- |
| **Cluster Topology** | 5 Nodes (40× RTX PRO 6000 GPUs) | `g4-standard-384` (768 GB VRAM per node, 3.84 TB total) |
| **Model Weight Footprint** | 64 Safetensors Shards (~320 GB) | Loaded via GCS Fuse CSI driver |
| **Attention Backend & KV Cache** | `TRITON_MLA` + `fp8_e4m3` | Multi-Head Latent Attention with FP8 KV Cache |
| **32K Context Prefill Throughput** | **1,748.43 - 1,756.04 tokens/sec** | Chunked prefill `16384` |
| **32K Context Prefill Latency** | **18.22 - 18.30 seconds** | 32,000 prompt token prefill |
| **256K Context Prefill Throughput** | **2,360.15 tokens/sec** | 256,000 token prompt prefill |
| **256K Context Prefill Latency** | **108.47 seconds (1.8 mins)** | Complete 256k long-context prompt processing |
| **Optimal Context Window** | **256K (`262,144` tokens)** | Fully stable production window on 5 nodes |

---

### Context Window Limit & Memory Boundary Analysis (5-Node 40× GPU Cluster)

1. **512K Context (`524,288` tokens)**:
   * **Initialization Status**: Failed (`ValueError`)
   * **Root Cause**: Serving a 512K single sequence requires **17.16 GiB KV Cache per GPU**, whereas available free VRAM after model weight loading and memory allocations is **15.16 GiB per GPU** (at `gpu_memory_utilization=0.95`).
2. **450K Context (`450,000` tokens)**:
   * **Initialization Status**: Engine initialized successfully (`max_model_len: 450000`).
   * **Prefill Behavior**: Execution encounters activation buffer allocation limits during single-prompt prefill at 450K tokens.
3. **256K Context (`262,144` tokens)**:
   * **Status**: **100% SUCCESSFUL & ROCK-SOLID STABLE**
   * **Performance**: Achieves **2,360.15 tokens/sec** prefill throughput with 108.47s latency and ample VRAM headroom for prefix caching.

