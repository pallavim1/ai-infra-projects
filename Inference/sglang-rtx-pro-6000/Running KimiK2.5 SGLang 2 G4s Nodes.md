# Running KimiK2.5 SGLang on 2 G4 (RTX 6000 Ada) Nodes

This document explains how to use the existing `createCluster_template.sh` script to provision the GKE cluster inside the existing `reflection-g4-vpc-1` network using custom, non-overlapping subnetworks in region `us-south1`, and then run the KimiK2.5 SGLang benchmark.

---

## 1. Set Environment Variables

To reuse the existing `reflection-g4-vpc-1` network and configure non-conflicting subnets in `us-south1`, run the following export commands in your terminal:

```bash
# Project, Region and Cluster Details
export PROJECT_NAME="northam-ce-mlai-tpu"
export REGION="us-south1"
export ZONE="us-south1-a"
export CLUSTER_NAME="pm-g4-sglang-cluster"

# Override network parameters to use the existing reflection VPC network
export NETWORK_NAME="reflection-g4-vpc-1"

# Override subnet names to reflect the region and your username
export SUBNETWORK_NAME_1="$USER-subnet-1-us-south1"
export SUBNETWORK_NAME_2="$USER-additional-test-subnet-1-us-south1"

# CRITICAL: Set unique, non-overlapping IP CIDR ranges for the new subnets in us-south1
export SUBNET_1_RANGE="10.2.0.0/20"
export SUBNET_1_PODS_RANGE="10.3.0.0/20"
export SUBNET_1_SERVICES_RANGE="10.3.16.0/20"

export SUBNET_2_RANGE="10.2.16.0/20"
export SUBNET_2_PODS_RANGE="10.23.16.0/20"
export SUBNET_2_SERVICES_RANGE="10.23.32.0/20"

export MASTER_IPV4_CIDR="172.16.0.32/28"

# Override GPU Pool configuration for a 2-node cluster (with max 3 node autoscaling)
export GPU_POOL_MIN_NODES=2
export GPU_POOL_MAX_NODES=3

# Set path to gcloud binary
export GCLOUD_PATH="/usr/local/google/home/pallaviam/google-cloud-sdk/bin/gcloud"
```

---

## 2. Authenticate with Google Cloud

Ensure your gcloud context is set to the correct project:

```bash
gcloud config set project ${PROJECT_NAME}
```

---

## 3. Run the Cluster Creation Script

Execute the template script. Since you set the environment variables above, the script will automatically:
1. Detect that the VPC `reflection-g4-vpc-1` already exists and skip its creation.
2. Create the new subnetworks (`$USER-subnet-1-us-south1` and `$USER-additional-test-subnet-1-us-south1`) inside the `reflection-g4-vpc-1` VPC.
3. Provision the Cloud Router and NAT in `us-south1`.
4. Deploy the GKE Control Plane and the associated GPU and client node pools.

```bash
chmod +x "gkecluster/createCluster_template.sh"
./gkecluster/createCluster_template.sh
```

---

## 4. Run the KimiK2.5 SGLang 2-Node Benchmark

Once the cluster is created, run the following steps to deploy the model and execute the benchmarking client.

### Step 4.1: Create the Hugging Face Token Secret
Create the secret containing your Hugging Face credentials in the default namespace:

```bash
kubectl create secret generic hf-secret --from-literal=HF_TOKEN="<your_hf_token>"
```

### Step 4.2: Deploy SGLang StatefulSet (2 Nodes)
Apply the StatefulSet configuration to spin up the serving cluster:

```bash
kubectl apply -f "models/KimiK2.5/nvfp4/sglang-kimi-nvfp4-2node_v2.yaml"
```

### Step 4.3: Monitor Pods and Server Startup Logs
Wait for the pods to initialize and check the Serving Engine logs:

```bash
# Watch pod statuses
kubectl get pods -w

# Stream logs from the master SGLang container
kubectl logs -f distributed-sglang-nvfp4-0
```
*Wait until you see `Uvicorn running on http://0.0.0.0:8000` in the logs of the master container before proceeding.*

### Step 4.4: Run the Benchmark Client
Deploy the benchmark client pod to run the load test against the SGLang cluster:

```bash
kubectl apply -f "benchmarking_scripts/benchmark-kimik25-nvfp4.yaml"
```

### Step 4.5: Check Benchmark Progress and Results
Stream the client logs to watch the benchmark execute and display the final performance statistics:

```bash
kubectl logs -f sglang-kimik25-nvfp4-benchmark
```

---

## 5. Benchmark Serving Results

Below are the serving benchmark results from the 2-node G4 (RTX 6000 Ada) run on June 11:

```yaml
============ Serving Benchmark Result ============
Backend:                                 sglang-oai
Traffic request rate:                    inf       
Max request concurrency:                 512       
Successful requests:                     1536      
Benchmark duration (s):                  2107.86   
Total input tokens:                      784969    
Total input text tokens:                 784969    
Total generated tokens:                  6434886   
Total generated tokens (retokenized):    6431531   
Request throughput (req/s):              0.73      
Input token throughput (tok/s):          372.40    
Output token throughput (tok/s):         3052.80   
Peak output token throughput (tok/s):    4534.00   
Peak concurrent requests:                516       
Total token throughput (tok/s):          3425.20   
Concurrency:                             413.30    
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   567174.83 
Median E2E Latency (ms):                 567820.00 
P90 E2E Latency (ms):                    992922.62 
P99 E2E Latency (ms):                    1126947.66
---------------Time to First Token----------------
Mean TTFT (ms):                          5804.45   
Median TTFT (ms):                        302.59    
P99 TTFT (ms):                           29602.10  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          144.14    
Median TPOT (ms):                        135.59    
P99 TPOT (ms):                           152.10    
---------------Inter-Token Latency----------------
Mean ITL (ms):                           134.16    
Median ITL (ms):                         127.80    
P95 ITL (ms):                            194.82    
P99 ITL (ms):                            291.42    
Max ITL (ms):                            27666.49  
==================================================
```

