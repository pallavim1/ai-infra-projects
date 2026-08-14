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

ZONES = [
  "us-central1-a",
  "us-central1-b",
  "us-central1-c",
  "us-east1-b",
  "us-east1-c",
  "us-east1-d",
  "us-east4-a",
  "us-east4-c",
  "us-east7-b",
  "us-west1-a",
  "us-west1-b",
  "us-west1-c",
  "us-west4-a",
  "us-west4-c",
  "northamerica-northeast1-b",
  "northamerica-northeast1-c",
  "northamerica-northeast2-a",
  "northamerica-northeast2-b",
  "europe-west4-a",
  "europe-west4-b",
  "europe-west4-c",
  "europe-west1-b",
  "europe-west1-c",
  "europe-west2-a",
  "europe-west2-b",
  "europe-west3-a",
  "europe-west3-b",
  "europe-west6-b",
  "europe-west6-c",
  "asia-southeast1-a",
  "asia-southeast1-b",
  "asia-southeast1-c",
  "asia-east1-a",
  "asia-east1-b",
  "asia-east1-c",
  "asia-northeast1-a",
  "asia-northeast1-b",
  "asia-northeast1-c",
  "asia-south1-a",
  "asia-south1-b",
  "asia-south1-c"
]

def check_existing_reservations():
    try:
        res = subprocess.run(
            [GCLOUD, "compute", "reservations", "list", f"--project={PROJECT_ID}", "--format=json"],
            capture_output=True, text=True, check=True
        )
        data = json.loads(res.stdout)
        found = {}
        for item in data:
            name = item.get("name", "")
            if name.startswith(PREFIX):
                zone = item.get("zone", "").split("/")[-1]
                spec = item.get("specificReservation", {})
                count = int(spec.get("count", 0))
                status = item.get("status", "")
                found[zone] = {"name": name, "count": count, "status": status}
        return found
    except Exception as e:
        print(f"Error checking reservations: {e}")
        return {}

def try_create_or_upsize(zone, target_count):
    res_name = f"{PREFIX}-{zone}"
    
    # Check if exists
    desc = subprocess.run(
        [GCLOUD, "compute", "reservations", "describe", res_name, f"--zone={zone}", f"--project={PROJECT_ID}", "--format=json"],
        capture_output=True, text=True
    )
    
    if desc.returncode == 0:
        # Exists, check count
        try:
            item = json.loads(desc.stdout)
            curr_count = int(item.get("specificReservation", {}).get("count", 0))
            if curr_count >= target_count:
                return curr_count
            # Attempt to update count directly to target_count
            for new_c in range(target_count, curr_count, -1):
                up = subprocess.run(
                    [GCLOUD, "compute", "reservations", "update", res_name, f"--zone={zone}", f"--project={PROJECT_ID}", f"--vm-count={new_c}"],
                    capture_output=True, text=True
                )
                if up.returncode == 0:
                    print(f"[{zone}] Successfully upsized {res_name} to {new_c} VMs!")
                    return new_c
        except Exception:
            pass
    else:
        # Does not exist, attempt to create directly with target_count (e.g. 3, or down to 2)
        for c in range(target_count, 1, -1):
            create = subprocess.run(
                [GCLOUD, "compute", "reservations", "create", res_name, f"--zone={zone}", f"--project={PROJECT_ID}", f"--machine-type={VM_TYPE}", f"--vm-count={c}", "--reservation-sharing-policy=ALLOW_ALL"],
                capture_output=True, text=True
            )
            if create.returncode == 0:
                print(f"[{zone}] SUCCESS: Created reservation {res_name} with {c} VMs!")
                return c
    return 0

def main():
    print(f"=== Multi-VM Reservation Hunter Started (Target: >= {TARGET_COUNT} VMs in a single reservation) ===")
    
    while True:
        existing = check_existing_reservations()
        for zone, info in existing.items():
            if info["count"] >= TARGET_COUNT:
                print(f"\n=======================================================")
                print(f"TARGET SECURED: {info["name"]} in {zone} has {info["count"]} VMs!")
                print(f"=======================================================")
                return

        for zone in ZONES:
            cnt = try_create_or_upsize(zone, TARGET_COUNT)
            if cnt >= TARGET_COUNT:
                print(f"\n=======================================================")
                print(f"TARGET SECURED: {PREFIX}-{zone} in {zone} has {cnt} VMs!")
                print(f"=======================================================")
                return
            elif cnt > 1:
                print(f"[{zone}] Currently holding {cnt} VMs. Will continue attempting to upsize to {TARGET_COUNT}...")
                
        print(f"[{time.strftime("%H:%M:%S")}] Completed scan across all zones. Re-scanning in 10 seconds...")
        time.sleep(10)

if __name__ == "__main__":
    main()
