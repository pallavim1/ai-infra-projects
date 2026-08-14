# Ruthless Retrying Reservations Script

A robust bash script designed to continuously retry creating or upsizing Google Compute Engine reservations (especially useful for highly GPU-constrained environments).

## Features

- **Continuous Retrying**: Keeps retrying reservation creation and incremental upsizing until the desired VM count is met.
- **Local Reservations**: Creates zonal reservations restricted to the host project.
- **Shared Reservations**: Shares the reservation with multiple consumer projects within the same organization.
- **Vertex AI Sharing**: Allows the reservation capacity to be consumed by Vertex AI custom training or prediction jobs.
- **Automated IAM Policy Bindings**: When sharing across projects, automatically assigns the `roles/compute.sharedReservationUser` role to the Vertex AI service agents in the consumer projects.

---

## Command Format

```bash
./ruthless-retrying-reservations-shared.sh <reservation-name> <project-id> <zone> <vm-type-and-options> <vm-count> [VERTEX_SHARING] [SHAREDPROJECTS]
```

### Arguments:

1. **`reservation-name`**: The name of the reservation to create or upsize.
2. **`project-id`**: The host GCP project ID where the reservation will live.
3. **`zone`**: Zonal location of the reserved resources (e.g., `asia-southeast1-c`).
4. **`vm-type-and-options`**: The machine type and accelerator settings wrapped in quotes.
5. **`vm-count`**: The target number of VMs to reserve.
6. **`VERTEX_SHARING`** (Optional): Pass `vertex` or `vertex-ai` to set `--reservation-sharing-policy=ALLOW_ALL` allowing Vertex AI to consume this reservation. If not using Vertex AI sharing but wanting to configure `SHAREDPROJECTS`, pass an empty string `""` here.
7. **`SHAREDPROJECTS`** (Optional): A comma-separated list of consumer project IDs to share the reservation with (e.g., `proj-a,proj-b`).

---

## Examples

### 1. Create a local L4 GPU reservation (No Sharing)
Create a standard zonal reservation for 10x L4 GPUs on `g2-standard-4` machines:
```bash
./ruthless-retrying-reservations-shared.sh my-l4-res northam-ce-mlai-tpu us-central1-a "g2-standard-4 --accelerator=count=1,type=nvidia-l4" 10
```

### 2. Create a local A100 GPU reservation with Vertex AI enabled
Allow Vertex AI Custom Jobs within the same project to consume the A100 GPU reservation:
```bash
./ruthless-retrying-reservations-shared.sh my-a100-res northam-ce-mlai-tpu us-central1-a "a2-highgpu-1g --accelerator=count=1,type=nvidia-tesla-a100" 5 vertex
```

### 3. Create a shared H100 GPU reservation with multiple consumer projects (No Vertex AI)
Share the reservation for 2x H100 GPU nodes (8 GPUs each) across `consumer-project-1` and `consumer-project-2` (using `""` for the 6th argument):
```bash
./ruthless-retrying-reservations-shared.sh my-h100-res northam-ce-mlai-tpu us-central1-a "a3-highgpu-8g --accelerator=count=8,type=nvidia-h100-80gb" 2 "" consumer-project-1,consumer-project-2
```

### 4. Create a shared L4 GPU reservation with multiple projects and Vertex AI enabled
Share the reservation with `gpu-launchpad-playground` and allow Vertex AI custom workloads to consume it:
```bash
./ruthless-retrying-reservations-shared.sh pm-testingscript-l4-res northam-ce-mlai-tpu asia-southeast1-c "g2-standard-16 --accelerator=count=1,type=nvidia-l4" 1 vertex gpu-launchpad-playground
```
