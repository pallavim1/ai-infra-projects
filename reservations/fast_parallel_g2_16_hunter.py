#!/usr/bin/env python3
import subprocess
import time
import json
import os
import sys

PROJECT_ID = "northam-ce-mlai-tpu"
TARGET_COUNT = 3
PREFIX = "pm-g2-16-res"
VM_TYPE = "g2-standard-16"
GCLOUD = "/usr/local/google/home/pallaviam/google-cloud-sdk/bin/gcloud"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Priority ordered list of zones (US first, then Europe/Asia fallbacks)
ZONES = [
  # us-central1 (Priority 1)
  "us-central1-a", "us-central1-b", "us-central1-c",
  # Other US zones
  "us-east1-b", "us-east1-c", "us-east1-d",
  "us-east4-a", "us-east4-c",
  "us-east7-b",
  "us-west1-a", "us-west1-b", "us-west1-c",
  "us-west4-a", "us-west4-c",
  "us-south1-a", "us-south1-b",
  # North America & Europe fallbacks
  "northamerica-northeast1-b", "northamerica-northeast1-c",
  "northamerica-northeast2-a", "northamerica-northeast2-b",
  "europe-west4-a", "europe-west4-b", "europe-west4-c",
  "europe-west1-b", "europe-west1-c",
  "europe-west2-a", "europe-west2-b",
  "europe-west3-a", "europe-west3-b",
  "europe-west6-b", "europe-west6-c",
  # Asia fallbacks
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
        log_file = f"{SCRIPT_DIR}/log_g2_16_{zone}.txt"
        # Run ruthless-retrying-reservations-shared.sh with target count 3
        cmd = [retry_script, res_name, PROJECT_ID, zone, VM_TYPE, str(TARGET_COUNT), "vertex"]
        with open(log_file, "w") as out:
            subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
    print(f"Launched parallel retrying reservation loops for {VM_TYPE} across {len(ZONES)} zones targeting {TARGET_COUNT} VMs per reservation.")

def cleanup_other_probes(winning_zone):
    print(f"\nCleaning up partial probe reservations except winner in {winning_zone}...")
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
                if zone != winning_zone:
                    print(f"Deleting probe reservation {name} in {zone}...")
                    subprocess.Popen([GCLOUD, "compute", "reservations", "delete", name, f"--zone={zone}", f"--project={PROJECT_ID}", "--quiet"])
    except Exception as e:
        print(f"Cleanup error: {e}")

def monitor():
    start_time = time.time()
    while True:
        try:
            res = subprocess.run(
                [GCLOUD, "compute", "reservations", "list", f"--project={PROJECT_ID}", "--format=json"],
                capture_output=True, text=True, check=True
            )
            data = json.loads(res.stdout)
            
            candidates = []
            for item in data:
                name = item.get("name", "")
                if name.startswith(PREFIX):
                    zone = item.get("zone", "").split("/")[-1]
                    spec = item.get("specificReservation", {})
                    count = int(spec.get("count", 0))
                    assured = int(spec.get("assuredCount", count))
                    status = item.get("status", "")
                    candidates.append((name, zone, count, assured, status))
                    
            # Check for winners (count >= TARGET_COUNT)
            winners = [c for c in candidates if c[2] >= TARGET_COUNT and c[4] == "READY"]
            if winners:
                w_name, w_zone, w_count, w_assured, w_status = winners[0]
                print(f"\n=======================================================")
                print(f"SUCCESS: Reservation {w_name} in {w_zone} secured {w_count} VMs (Status: {w_status})!")
                print(f"=======================================================")
                # Stop background loops
                subprocess.run(["pkill", "-f", "ruthless-retrying-reservations-shared.sh"])
                cleanup_other_probes(w_zone)
                return w_name, w_zone, w_count
            elif candidates:
                for c in candidates:
                    print(f"[{time.strftime("%H:%M:%S")}] {c[0]} in {c[1]}: {c[2]} VMs ({c[4]}). Target: {TARGET_COUNT}")
        except Exception as e:
            print(f"Monitor error: {e}")
        time.sleep(10)

if __name__ == "__main__":
    launch_parallel_loops()
    monitor()
