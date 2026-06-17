# GKE H200 2-Node C++ NCCL Test Guide

This document describes the step-by-step procedure to execute the industry-standard C++ `nccl-tests` benchmark across the 2-node H200 cluster. This test isolates the physical network transport layer (Multi-NIC GPUDirect RDMA with Mellanox TCPX) from the PyTorch/SGLang serving stack.

---

## 1. Prerequisites (Releasing GPU Memory)
To test large payload sizes (up to 4GB), you must release the GPU memory held by the SGLang inference engine. SGLang pre-allocates 88% of the VRAM (127GB of 141GB per H200 GPU), which will cause a CUDA Out-of-Memory (OOM) error if you run the benchmark concurrent with the engine.

### Put Containers in Sleep Mode
To clear VRAM without losing the GKE pod configuration (Multi-NIC attachments, GCS Fuse mounts, and tolerations), override the container's entrypoint script to run a shell `sleep`.

1. Open [sglang-kimi-26-2node-h200.yaml](file:///usr/local/google/home/pallaviam/AI%20Infra%20Projects/ai-infra-projects/Inference/sglang-h200/models/KimiK2.6/sglang-kimi-26-2node-h200.yaml) in your editor.
2. Locate the container `args` block and change it to `sleep 86400`:
   ```yaml
           command:
           - /bin/bash
           - -c
           args:
           - |
             sleep 86400
   ```
3. Apply the manifest changes to trigger a rolling restart of the pods:
   ```bash
   cd "/usr/local/google/home/pallaviam/AI Infra Projects/ai-infra-projects"
   kubectl apply -f Inference/sglang-h200/models/KimiK2.6/sglang-kimi-26-2node-h200.yaml
   ```
4. Verify both pods (`distributed-sglang-0` and `distributed-sglang-1`) are in the `Running` state:
   ```bash
   kubectl get pods
   ```

---

## 2. Compile `nccl-tests` with MPI support
You must clone and compile the C++ benchmark binary on both GKE pods. Since we compile with OpenMPI support, we must explicitly locate the OpenMPI headers.

**On Node 0 (distributed-sglang-0):**
```bash
kubectl exec distributed-sglang-0 -c sglang-container -- bash -c "
  apt update && apt install -y iproute2 git make && \
  git clone https://github.com/NVIDIA/nccl-tests.git /workspace/nccl-tests && \
  make -C /workspace/nccl-tests CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local/gib MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
"
```

**On Node 1 (distributed-sglang-1):**
```bash
kubectl exec distributed-sglang-1 -c sglang-container -- bash -c "
  apt update && apt install -y iproute2 git make && \
  git clone https://github.com/NVIDIA/nccl-tests.git /workspace/nccl-tests && \
  make -C /workspace/nccl-tests CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/local/gib MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
"
```

---

## 3. Configure SSH Coordination
OpenMPI coordinates rank orchestration across pods over SSH. Because GKE containers run without an active SSH daemon by default, you must start the daemon on port 222 and exchange SSH credentials.

**On Node 1:**
```bash
kubectl exec distributed-sglang-1 -c sglang-container -- bash -c "
  mkdir -p /run/sshd && ssh-keygen -A && /usr/sbin/sshd -p 222
"
```

**On Node 2 (Node 0):**
1. Start the daemon locally and generate a local keypair:
   ```bash
   kubectl exec distributed-sglang-0 -c sglang-container -- bash -c "
     mkdir -p /run/sshd && ssh-keygen -A && /usr/sbin/sshd -p 222 && \
     ssh-keygen -t rsa -N '' -f /root/.ssh/id_rsa
   "
   ```
2. Copy Node 0's generated public key:
   ```bash
   PUBKEY=$(kubectl exec distributed-sglang-0 -c sglang-container -- cat /root/.ssh/id_rsa.pub)
   ```
3. Append it to Pod 1's authorized keys list to allow passwordless access:
   ```bash
   kubectl exec distributed-sglang-1 -c sglang-container -- bash -c "
     mkdir -p /root/.ssh && chmod 700 /root/.ssh && \
     echo '$PUBKEY' >> /root/.ssh/authorized_keys && \
     chmod 600 /root/.ssh/authorized_keys
   "
   ```

---

## 4. Run the 16-GPU AllReduce Test (64K to 4GB)
With GPU VRAM fully freed, you can launch the benchmark from Node 0.

Run the test:
```bash
kubectl exec distributed-sglang-0 -c sglang-container -- \
  /usr/local/gib/scripts/run_nccl_tests.sh \
    -t all_reduce \
    -d /workspace/nccl-tests/build \
    -g 8 \
    -b 65536 \
    -e 4294967296 \
    -f 2 \
    -w 5 \
    -n 20 \
    -p 222 \
    10.88.0.3 10.88.0.4
```

### Parameter Explanations:
* `-t all_reduce`: Specifies the all-reduce collective pattern.
* `-d /workspace/nccl-tests/build`: Path to the compiled `all_reduce_perf` binary.
* `-g 8`: Runs across all 8 GPUs on each node.
* `-b 65536`: The starting transfer payload size (64 KB).
* `-e 4294967296`: The final transfer payload size (4 GB).
* `-f 2`: Multiplies the payload size by 2 at each step.
* `-w 5`: Number of warmup iterations per payload.
* `-n 20`: Number of benchmark iterations to average.
* `-p 222`: Uses our custom SSH port for pod-to-pod coordination.
* `10.88.0.3 10.88.0.4`: The pod IP addresses of your cluster nodes (verify via `kubectl get pods -o wide`).

---

## 5. Restore SGLang Inference serving
When the benchmarks have completed and you are ready to restart SGLang:

1. Revert the changes to the manifest:
   ```bash
   git checkout Inference/sglang-h200/models/KimiK2.6/sglang-kimi-26-2node-h200.yaml
   ```
2. Apply the manifest to trigger a rolling restart back to inference mode:
   ```bash
   kubectl apply -f Inference/sglang-h200/models/KimiK2.6/sglang-kimi-26-2node-h200.yaml
   ```
3. Validate that SGLang serving boots successfully:
   ```bash
   kubectl logs distributed-sglang-0 -c sglang-container --tail=100
   ```
