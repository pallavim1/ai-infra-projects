#!/bin/bash
# Master script to run ruthless retrying reservations for 1x L4 GPU (g2-standard-4)
# prioritized in US regions starting with us-central1.

PROJECT_ID=${1:-"northam-ce-mlai-tpu"}
TARGET_TOTAL_COUNT=${2:-3}
RES_PREFIX=${3:-"pm-l4-res"}
VERTEX_SHARING=${4:-"vertex"}
SHARED_PROJECTS=${5:-""}

VM_TYPE="g2-standard-4"

# Priority zones starting with us-central1, then all US regions, then global fallbacks
ZONES=(
  # us-central1 (Priority 1)
  "us-central1-a"
  "us-central1-b"
  "us-central1-c"
  # Other US zones
  "us-east1-b"
  "us-east1-c"
  "us-east1-d"
  "us-east4-a"
  "us-east4-c"
  "us-east7-b"
  "us-west1-a"
  "us-west1-b"
  "us-west1-c"
  "us-west4-a"
  "us-west4-c"
  # North America & Europe fallbacks
  "northamerica-northeast1-b"
  "northamerica-northeast1-c"
  "northamerica-northeast2-a"
  "northamerica-northeast2-b"
  "europe-west4-a"
  "europe-west4-b"
  "europe-west4-c"
  "europe-west1-b"
  "europe-west1-c"
  "europe-west2-a"
  "europe-west2-b"
  "europe-west3-a"
  "europe-west3-b"
  "europe-west6-b"
  "europe-west6-c"
  # Asia fallbacks
  "asia-southeast1-a"
  "asia-southeast1-b"
  "asia-southeast1-c"
  "asia-east1-a"
  "asia-east1-b"
  "asia-east1-c"
  "asia-northeast1-a"
  "asia-northeast1-b"
  "asia-northeast1-c"
  "asia-south1-a"
  "asia-south1-b"
  "asia-south1-c"
)

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
RETRY_SCRIPT="${SCRIPT_DIR}/ruthless-retrying-reservations-shared.sh"

if [ ! -f "$RETRY_SCRIPT" ]; then
  echo "Error: Retry script not found at ${RETRY_SCRIPT}"
  exit 1
fi

chmod +x "$RETRY_SCRIPT"

echo "========================================================================="
echo "Starting L4 GPU (g2-standard-4) Reservations Loops in Parallel"
echo "Project:            ${PROJECT_ID}"
echo "VM Type:            ${VM_TYPE}"
echo "Target Total Count: ${TARGET_TOTAL_COUNT}"
echo "Prefix:             ${RES_PREFIX}"
echo "========================================================================="

PIDS=()

for ZONE in "${ZONES[@]}"; do
  RES_NAME="${RES_PREFIX}-${ZONE}"
  LOG_FILE="${SCRIPT_DIR}/log_reserve_l4_${ZONE}.txt"
  
  echo "Starting loop for zone ${ZONE} -> ${RES_NAME} (log: ${LOG_FILE})"
  
  # Run the reservation script in the background requesting 1 to 3 per zone
  "${RETRY_SCRIPT}" "${RES_NAME}" "${PROJECT_ID}" "${ZONE}" "${VM_TYPE}" "${TARGET_TOTAL_COUNT}" "${VERTEX_SHARING}" "${SHARED_PROJECTS}" > "${LOG_FILE}" 2>&1 &
  
  PIDS+=($!)
done

echo "All ${#ZONES[@]} reservation loops have been started in background."
echo "PIDs: ${PIDS[*]}"
