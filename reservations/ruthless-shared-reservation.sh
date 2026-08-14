#!/bin/bash
# Ruthless Retrying Shared Reservations Script
# command format: ruthless-shared-reservation.sh <reservation name> <project-id> <zone> <vm type> <vm count> <comma-separated-consumer-projects>
#
# Examples:
#
# Create a shared reservation and share it with GKE and Vertex AI across projects
# ruthless-shared-reservation.sh my-shared-res my-gcp-project us-central1-a n1-standard-64 10 "consumer-proj-1,consumer-proj-2"
#

if [[ "$#" -ne 6 ]]; then
    echo "Usage: $0 <reservation name> <project-id> <zone> \"<vm type with options>\" <vm count> <comma-separated-consumer-projects>"
    echo "Example: $0 my-shared-res my-gcp-project us-central1-a \"n2-standard-8\" 10 \"project-a,project-b\""
    exit 1
fi

NAME=$1
PROJECT=$2
ZONE=$3
VM_TYPE=$4
VM_COUNT=$5
CONSUMER_PROJECTS=$6
RETRY_RATE=60
INCREMENT=1

# Filter out the owner project ID from the consumer projects list to prevent sharing with oneself error
CLEANED_CONSUMERS=""
IFS=',' read -ra ADDR <<< "$CONSUMER_PROJECTS"
for p_id in "${ADDR[@]}"; do
  p_id=$(echo "$p_id" | xargs)
  if [[ -n "$p_id" && "$p_id" != "$PROJECT" ]]; then
    if [[ -z "$CLEANED_CONSUMERS" ]]; then
      CLEANED_CONSUMERS="$p_id"
    else
      CLEANED_CONSUMERS="$CLEANED_CONSUMERS,$p_id"
    fi
  fi
done

setup_iam_bindings() {
  echo "Setting up IAM bindings for GKE and Vertex AI service agents..."
  # Parse all projects (owner + clean consumers)
  local projects_to_bind="$PROJECT"
  if [[ -n "$CLEANED_CONSUMERS" ]]; then
    projects_to_bind="$PROJECT,$CLEANED_CONSUMERS"
  fi
  IFS=',' read -ra ADDR <<< "$projects_to_bind"
  for p_id in "${ADDR[@]}"; do
    # Trim whitespace
    p_id=$(echo "$p_id" | xargs)
    if [[ -z "$p_id" ]]; then
      continue
    fi
    
    echo "Processing project: $p_id"
    p_num=$(gcloud projects describe "$p_id" --format="value(projectNumber)" 2>/dev/null)
    if [[ -z "$p_num" ]]; then
      echo "Warning: Could not get project number for project: $p_id. Skipping IAM binding for this project."
      continue
    fi
    
    gke_sa="service-${p_num}@container-engine-robot.iam.gserviceaccount.com"
    vertex_sa="service-${p_num}@gcp-sa-aiplatform.iam.gserviceaccount.com"
    
    echo "Binding roles/compute.sharedReservationUser to GKE service agent in project $p_id..."
    gcloud compute reservations add-iam-policy-binding "$NAME" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --member="serviceAccount:$gke_sa" \
      --role="roles/compute.sharedReservationUser" >/dev/null 2>&1
    if [[ $? -eq 0 ]]; then
      echo "Successfully bound role for GKE service agent in project $p_id"
    else
      echo "Warning: Failed to bind role for GKE service agent in project $p_id"
    fi
    
    echo "Binding roles/compute.sharedReservationUser to Vertex AI service agent in project $p_id..."
    gcloud compute reservations add-iam-policy-binding "$NAME" \
      --project="$PROJECT" \
      --zone="$ZONE" \
      --member="serviceAccount:$vertex_sa" \
      --role="roles/compute.sharedReservationUser" >/dev/null 2>&1
    if [[ $? -eq 0 ]]; then
      echo "Successfully bound role for Vertex AI service agent in project $p_id"
    else
      echo "Warning: Failed to bind role for Vertex AI service agent in project $p_id"
    fi
  done
}

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
 # Attempt to ensure sharing and policy settings on existing reservation
 if [[ -n "$CLEANED_CONSUMERS" ]]; then
   gcloud compute reservations update $NAME \
     --zone=$ZONE \
     --project=$PROJECT \
     --add-share-with=$CLEANED_CONSUMERS \
     --reservation-sharing-policy=ALLOW_ALL >/dev/null 2>&1
 else
   gcloud compute reservations update $NAME \
     --zone=$ZONE \
     --project=$PROJECT \
     --reservation-sharing-policy=ALLOW_ALL >/dev/null 2>&1
 fi
 
 # Set up IAM bindings on the existing reservation
 setup_iam_bindings
fi


while true; do

 # If the existing reservation does not exist then create it until we exhaust retries
 if [[ $EXISTING_RES_CODE -ne 0 ]] && [[ $EXISTING_RESERVATIONS -eq 0 ]]; then
  echo "Attempting to create reservation..."
   for (( retry=1; retry<=5; retry++))
    do
     if [[ -n "$CLEANED_CONSUMERS" ]]; then
       echo "Creating shared reservation shared with: $CLEANED_CONSUMERS"
       gcloud compute reservations create $NAME \
         --project=$PROJECT \
         --zone=$ZONE \
         --machine-type=$VM_TYPE \
         --vm-count=1 \
         --share-setting=projects \
         --share-with=$CLEANED_CONSUMERS \
         --reservation-sharing-policy=ALLOW_ALL;
     else
       echo "Creating local reservation for project $PROJECT with ALLOW_ALL sharing policy..."
       gcloud compute reservations create $NAME \
         --project=$PROJECT \
         --zone=$ZONE \
         --machine-type=$VM_TYPE \
         --vm-count=1 \
         --reservation-sharing-policy=ALLOW_ALL;
     fi
      if [[ $? -eq 0 ]]; then
       echo "Successfully created reservation"
       EXISTING_RES_CODE=$?
       EXISTING_RESERVATIONS=$(gcloud compute reservations describe $NAME --project=$PROJECT --zone=$ZONE --format 'get(specificReservation.assured_count)' 2> /dev/null)
       
       # Bind IAM roles for GKE and Vertex AI service agents
       setup_iam_bindings
       
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
