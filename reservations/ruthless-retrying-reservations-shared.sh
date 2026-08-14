#!/bin/bash
# Ruthless Retrying Reservations Script
# command format: ruthless-retrying-reservations-shared.sh <reservation name> <project-id> <zone> <vm type> <vm count> [VERTEX_SHARING] [SHAREDPROJECTS]
#
# Examples:
#
# Create a local L4 GPU reservation (without sharing with other projects or Vertex AI)
# retrying-reservations-shared.sh my-l4-res my-gcp-project us-central1-a "g2-standard-4 --accelerator=count=1,type=nvidia-l4" 10
#
# Create a local H100 GPU reservation allowed to be consumed by Vertex AI (within the same project)
# retrying-reservations-shared.sh my-h100-res my-gcp-project us-central1-a a3-megagpu-8g 5 vertex
#
# Create a shared H100 GPU reservation shared with project-a and project-b, WITHOUT Vertex AI sharing
# retrying-reservations-shared.sh my-h100-res my-gcp-project us-central1-a a3-megagpu-8g 2 "" project-a,project-b
#
# Create a shared H100 GPU reservation shared with project-a and project-b, WITH Vertex AI sharing
# retrying-reservations-shared.sh my-h100-res my-gcp-project us-central1-a a3-megagpu-8g 10 vertex project-a,project-b


NAME=$1
PROJECT=$2
ZONE=$3
VM_TYPE=$4
VM_COUNT=$5
VERTEX_SHARING=$6
SHAREDPROJECTS=$7
RETRY_RATE=15
INCREMENT=1

# Build sharing flags
SHARE_SETTING_FLAGS=""
BIND_IAM_ROLES=false

if [[ -n "$SHAREDPROJECTS" ]]; then
  SHARE_SETTING_FLAGS="--share-setting=projects --share-with=$SHAREDPROJECTS"
  BIND_IAM_ROLES=true
fi

if [[ "$VERTEX_SHARING" == "vertex" || "$VERTEX_SHARING" == "vertex-ai" || "$VERTEX_SHARING" == "allow-vertex" ]]; then
  SHARE_SETTING_FLAGS="$SHARE_SETTING_FLAGS --reservation-sharing-policy=ALLOW_ALL"
fi

# First check for existing reservations
EXISTING_RESERVATIONS=$(gcloud compute reservations describe $NAME --project=$PROJECT --zone=$ZONE --format 'get(specificReservation.assured_count)' 2> /dev/null)
EXISTING_RES_CODE=$?
# No Existing reservation exists to use create it
if [[ $EXISTING_RES_CODE -ne 0 ]]; then
 echo "No existing reservation creating..."
 EXISTING_RESERVATIONS=0
# Existing reservation found use it
elif [[ $EXISTING_RES_CODE -eq 0 ]]; then
 echo "Found existing reservation..."
fi


while true; do

 # If the existing reservation does not exist then create it until we exhaust retries
 if [[ $EXISTING_RES_CODE -ne 0 ]] && [[ $EXISTING_RESERVATIONS -eq 0 ]]; then
  echo "Attempting to create reservation..."
   for (( retry=1; retry<=5; retry++))
    do
     gcloud compute reservations create $NAME --project=$PROJECT --zone=$ZONE --machine-type=$VM_TYPE --vm-count=1 \
       $SHARE_SETTING_FLAGS;
      if [[ $? -eq 0 ]]; then
       echo "Successfully created reservation"
       EXISTING_RES_CODE=$?
       EXISTING_RESERVATIONS=$(gcloud compute reservations describe $NAME --project=$PROJECT --zone=$ZONE --format 'get(specificReservation.assured_count)' 2> /dev/null)
       
       # Setup Vertex AI IAM bindings if it's a shared reservation
       if [[ "$BIND_IAM_ROLES" = true ]]; then
         echo "Setting up IAM bindings for Vertex AI service agents..."
         IFS=',' read -ra ADDR <<< "$SHAREDPROJECTS"
         for p_id in "${ADDR[@]}"; do
           p_id=$(echo "$p_id" | xargs) # trim whitespace
           if [[ -n "$p_id" && "$p_id" != "$PROJECT" ]]; then
             p_num=$(gcloud projects describe "$p_id" --format="value(projectNumber)" 2>/dev/null)
             if [[ -n "$p_num" ]]; then
               echo "Binding roles/compute.sharedReservationUser to Vertex AI service agent in project $p_id..."
               gcloud compute reservations add-iam-policy-binding "$NAME" \
                 --project="$PROJECT" \
                 --zone="$ZONE" \
                 --member="serviceAccount:service-${p_num}@gcp-sa-aiplatform.iam.gserviceaccount.com" \
                 --role="roles/compute.sharedReservationUser" >/dev/null 2>&1
             fi
           fi
         done
       fi
       
       break
      elif [[ $? -ne 0 ]]; then
       echo "Failed to create reservation retrying... attempt: $retry" 
      fi
    done
 fi
 
 # Check existing res status code if 0 then use the existing named reservation
 # If the amount of existing reservations is less than the requested count attempt to upsize
 if [[ $EXISTING_RES_CODE -eq 0 ]] && [[ $EXISTING_RESERVATIONS -lt $VM_COUNT ]]; then
  echo "Existing reservations count: $EXISTING_RESERVATIONS"
  # Make an initial attempt to upsize the reservation
  echo "Attempting to upsize reservation..."
  gcloud compute reservations update $NAME --zone=$ZONE --project=$PROJECT --vm-count $((EXISTING_RESERVATIONS+INCREMENT));
  UPSIZE_STATUS=$?
  if [[ $UPSIZE_STATUS -eq 0 ]]; then
   echo "Successfully upsized reservation"
   EXISTING_RESERVATIONS=$(gcloud compute reservations describe $NAME --project=$PROJECT --zone=$ZONE --format 'get(specificReservation.assured_count)' 2> /dev/null)
   echo "Reservation count: $EXISTING_RESERVATIONS"
     if [[ $EXISTING_RESERVATIONS -ge $VM_COUNT ]]; then
       echo "Desired reservation count met exiting..."
       exit 0
     fi
  fi
   # Unless we reached desired count with the initial upsize then start attempting to upsize until hit the desired count, fail or exhaust retries
   # If we are repeatedly successful try and grab more in a single pass until we fail
   # Each time we succeed in incrementing the reservation increment and try for more
   RETRY=1
   while [[ $EXISTING_RESERVATIONS -lt $VM_COUNT && $UPSIZE_STATUS -eq 0 ]] || [[ $RETRY -le 5 ]]; do
     echo "Trying to upsize reservation again..."
      gcloud compute reservations update $NAME --zone=$ZONE --project=$PROJECT --vm-count $((EXISTING_RESERVATIONS+INCREMENT));
      UPSIZE_STATUS=$?
       if [[ $UPSIZE_STATUS -eq 0 && $EXISTING_RESERVATIONS -lt $VM_COUNT ]]; then
        EXISTING_RESERVATIONS=$(gcloud compute reservations describe $NAME --project=$PROJECT --zone=$ZONE --format 'get(specificReservation.assured_count)' 2> /dev/null)
        echo "Reservation upsized count: $EXISTING_RESERVATIONS"
         if [[ $EXISTING_RESERVATIONS -ge $VM_COUNT ]]; then
          echo "Desired reservation count met exiting..."
          exit 0
         elif [[ $((EXISTING_RESERVATIONS+INCREMENT)) -lt $VM_COUNT ]]; then
           INCREMENT=$((INCREMENT+1))
         elif [[ $((EXISTING_RESERVATIONS+INCREMENT)) -ge $VM_COUNT ]]; then
           INCREMENT=$((VM_COUNT-EXISTING_RESERVATIONS))
         fi
       elif [[ $UPSIZE_STATUS -ne 0 ]]; then
        echo "Failed to upsize reservation retrying... Attempt: $RETRY"
        INCREMENT=1
        RETRY=$((RETRY+1))
       fi
    done 
 fi
 
 if [[ $EXISTING_RESERVATIONS -ge $VM_COUNT ]]; then
  echo "Desired reservation count met exiting..."
  exit 0
 else 
  echo "Re-Scanning in $RETRY_RATE seconds"
  sleep $RETRY_RATE
 fi
done
