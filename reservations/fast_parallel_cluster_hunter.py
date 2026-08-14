#!/usr/bin/env python3
import subprocess
import time
import json
import os
import sys

PROJECT_ID = "northam-ce-mlai-tpu"
TARGET_COUNT = 3
PREFIX = "pm-l4-cluster"
VM_TYPE = "g2-standard-4"
GCLOUD = "/usr/local/google/home/pallaviam/google-cloud-sdk/bin/gcloud"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Priority ordered list of zones
ZONES = [
  "us-central1-a", "us-central1-b", "us-central1-c",
  "us-east1-b", "us-east1-c", "us-east1-d",
  "us-east4-a", "us-east4-c",
  "us-east7-b",
  "us-west1-a", "us-west1-b", "us-west1-c",
  "us-west4-a", "us-west4-c",
  "northamerica-northeast1-b", "northamerica-northeast1-c",
  "northamerica-northeast2-a", "northamerica-northeast2-b",
  "europe-west4-a", "europe-west4-b", "europe-west4-c",
  "europe-west1-b", "europe-west1-c",
  "europe-west2-a", "europe-west2-b",
  "europe-west3-a", "europe-west3-b",
  "europe-west6-b", "europe-west6-c",
  "asia-southeast1-a", "asia-southeast1-b", "asia-southeast1-c",
  "asia-east1-a", "asia-east1-b", "asia-east1-c",
  "asia-northeast1-a", "asia-northeast1-b", "asia-northeast1-c",
  "asia-south1-a", "asia-south1-b", "asia-south1-c"
]

def launch_parallel_loops():
    retry_script = f"{SCRIPT_DIR}/ruthless-retrying-reservations-shared.sh"
    os.chmod(retry_script, 0o755)
    for zone in ZONES:
        res_name = f"{PREFIX}-{zone}"
        log_file = f"{SCRIPT_DIR}/log_cluster_{zone}.txt"
        # Run ruthless-retrying-reservations-shared.sh with target count 3
        cmd = [retry_script, res_name, PROJECT_ID, zone, VM_TYPE, str(TARGET_COUNT), "vertex"]
        with open(log_file, "w") as out:
            subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
    print(f"Launched parallel retrying reservation loops across {len(ZONES)} zones targeting {TARGET_COUNT} VMs per reservation.")

def monitor():
    while True:
        try:
            res = subprocess.run(
                [GCLOUD, "compute", "reservations", "list", f"--project={PROJECT_ID}", "--format=json"],
                capture_output=True, text=True, check=True
            )
            data = json.loads(res.stdout)
            for item in data:
                name = item.get("name", "")
                if name.startswith(PREFIX):
                    zone = item.get("zone", "").split("/")[-1]
                    spec = item.get("specificReservation", {})
                    count = int(spec.get("count", 0))
                    assured = int(spec.get("assuredCount", count))
                    status = item.get("status", "")
                    if count >= TARGET_COUNT:
                        print(f"\n=======================================================")
                        print(f"SUCCESS: Reservation {name} in {zone} reached {count} VMs (Status: {status})!")
                        print(f"=======================================================")
                        # Stop background loops
                        subprocess.run(["pkill", "-f", "ruthless-retrying-reservations-shared.sh"])
                        return
                    elif count > 1:
                        print(f"[{time.strftime("%H:%M:%S")}] Found partial multi-GPU reservation: {name} in {zone} with {count} VMs (Status: {status}). Still aiming for {TARGET_COUNT}...")
        except Exception as e:
            print(f"Monitor error: {e}")
        time.sleep(10)

if __name__ == "__main__":
    launch_parallel_loops()
    monitor()
