#!/bin/bash
# Master script to run ruthless retrying reservations for H200 (a3-ultragpu-8g)
# in all available GCP zones in parallel background processes.

PROJECT_ID=${1:-"northam-ce-mlai-tpu"}
VM_COUNT=${2:-2}
RES_PREFIX=${3:-"pm-h200-res"}
VERTEX_SHARING=${4:-""}
SHARED_PROJECTS=${5:-""}

VM_TYPE="a3-ultragpu-8g"

# List of zones where H200 (A3 Ultra) is available on GCP
ZONES=(
  "asia-south1-b"
  "asia-south2-c"
  "europe-west1-b"
  "europe-west4-a"
  "us-central1-b"
  "us-east4-b"
  "us-south1-b"
  "us-west1-c"
)

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
RETRY_SCRIPT="${SCRIPT_DIR}/ruthless-retrying-reservations-shared.sh"

if [ ! -f "$RETRY_SCRIPT" ]; then
  echo "Error: Retry script not found at ${RETRY_SCRIPT}"
  exit 1
fi

chmod +x "$RETRY_SCRIPT"

echo "========================================================================="
echo "Starting H200 Reservations Loops in Parallel"
echo "Project:       ${PROJECT_ID}"
echo "VM Type:       ${VM_TYPE}"
echo "Count per zone: ${VM_COUNT}"
echo "Prefix:        ${RES_PREFIX}"
echo "========================================================================="

for ZONE in "${ZONES[@]}"; do
  RES_NAME="${RES_PREFIX}-${ZONE}"
  LOG_FILE="${SCRIPT_DIR}/log_reserve_${ZONE}.txt"
  
  echo "Starting loop for zone ${ZONE}..."
  echo "Reservation name: ${RES_NAME}"
  echo "Logs will be written to: ${LOG_FILE}"
  
  # Run the reservation script in the background
  # Pass arguments: <reservation-name> <project-id> <zone> <vm-type-and-options> <vm-count> [VERTEX_SHARING] [SHAREDPROJECTS]
  "${RETRY_SCRIPT}" "${RES_NAME}" "${PROJECT_ID}" "${ZONE}" "${VM_TYPE}" "${VM_COUNT}" "${VERTEX_SHARING}" "${SHARED_PROJECTS}" > "${LOG_FILE}" 2>&1 &
  
  # Store PID
  PID=$!
  echo "Zone ${ZONE} loop started with PID ${PID}."
  echo "------------------------------------------------------------------------"
done

echo "All reservation loops have been started in the background."
echo "Use 'ps aux | grep ruthless-retrying' or inspect log files to monitor progress."
echo "Check reservation status with: gcloud compute reservations list --project=${PROJECT_ID}"
