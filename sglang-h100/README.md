# SGLang on GCP H100

This directory contains configurations, cluster deployment scripts, and benchmarks optimized for running SGLang multi-node inference on GCP H100 GPU instances (specifically A3 Mega).

## Directory Structure
- [gkecluster/](file:///usr/local/google/home/pallaviam/AI%20Infra%20Projects/ai-infra-projects/sglang-h100/gkecluster)
  - `createCluster_H100_template.sh`: Automates creation of a GKE cluster with a custom VPC and an H100 GPU node pool.
  - `createCluster_H100_TCPX.sh`: Sets up a GKE cluster with GPUDirect TCPX multi-networking.
  - `GPUDIRECT_TCPX_SETUP.md`: Detailed setup guide for GPUDirect TCPX enablement.
- [models/](file:///usr/local/google/home/pallaviam/AI%20Infra%20Projects/ai-infra-projects/sglang-h100/models)
  - `KimiK2.5/`: Configurations for Kimi-K2.5 on H100 (Native INT4 & NVFP4).
  - `KimiK2.6/`: Configurations for Kimi-K2.6 on H100 (full BF16).

---

## Model Serving Configurations

### Kimi-K2.6 (2-Node H100 Distributed)
- **Model**: `moonshotai/Kimi-K2.6` (served in full BF16 precision)
- **Tensor Parallelism**: 8
- **Pipeline Parallelism**: 2
- **KV Cache**: FP8 (`fp8_e5m2`)
- **Serving Image**: `lmsysorg/sglang:v0.5.10.post1`
- **Node Pool**: `pm-2x-h100-2nic-pool`
- **Optimization Parameters**: `--mem-fraction-static 0.75` (allocated 25% GPU RAM for workspace/activations to prevent CUDA OOM crash under 512 concurrency).

#### Deployment:
```bash
kubectl apply -f models/KimiK2.6/sglang-kimi-26-2nd-h100.yaml
```

### Kimi-K2.5 (2-Node H100 Distributed)
Configurations are available for:
1. **Native INT4**: `models/KimiK2.5/sglang-kimik2.5-2node-h100.yaml`
2. **NVFP4 (Quantized)**: `models/KimiK2.5/nvfp4/sglang-kimi-nvfp4-2node-h100.yaml`

---

## Infrastructure Setup
To provision the H100 GKE cluster, navigate to `gkecluster/` and follow the instructions in [GPUDIRECT_TCPX_SETUP.md](file:///usr/local/google/home/pallaviam/AI%20Infra%20Projects/ai-infra-projects/sglang-h100/gkecluster/GPUDIRECT_TCPX_SETUP.md).
