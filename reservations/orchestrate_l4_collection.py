#!/usr/bin/env python3
import subprocess
import time
import json
import os
import sys

PROJECT_ID = sys.argv[1] if len(sys.argv) > 1 else "northam-ce-mlai-tpu"
TARGET_COUNT = int(sys.argv[2]) if len(sys.argv) > 2 else 3
PREFIX = sys.argv[3] if len(sys.argv) > 3 else "pm-l4-res"

GCLOUD = "/usr/local/google/home/pallaviam/google-cloud-sdk/bin/gcloud"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def list_l4_reservations():
    try:
        res = subprocess.run(
            [GCLOUD, "compute", "reservations", "list", f"--project={PROJECT_ID}", "--format=json"],
            capture_output=True, text=True, check=True
        )
        data = json.loads(res.stdout)
        l4_res = []
        for item in data:
            name = item.get("name", "")
            if name.startswith(PREFIX):
                spec = item.get("specificReservation", {})
                count = int(spec.get("count", 0))
                assured = int(spec.get("assuredCount", count))
                zone = item.get("zone", "").split("/")[-1]
                machine = spec.get("instanceProperties", {}).get("machineType", "")
                status = item.get("status", "")
                l4_res.append({
                    "name": name,
                    "zone": zone,
                    "count": count,
                    "assured": assured,
                    "machine": machine,
                    "status": status
                })
        return l4_res
    except Exception as e:
        print(f"Error checking reservations: {e}")
        return []

def main():
    print(f"=== Starting L4 GPU Reservation Orchestrator for {TARGET_COUNT} VMs (Project: {PROJECT_ID}) ===")
    
    # 1. Launch the background multi-zone reservation script
    cmd = ["bash", f"{SCRIPT_DIR}/reserve-l4-all-regions.sh", PROJECT_ID, str(TARGET_COUNT), PREFIX, "vertex"]
    print("Launching reservation loops across all candidate zones...")
    proc = subprocess.Popen(cmd, cwd=SCRIPT_DIR)
    
    start_time = time.time()
    
    while True:
        reservations = list_l4_reservations()
        total_acquired = sum(r["count"] for r in reservations)
        
        print(f"[{time.strftime("%H:%M:%S")}] Total L4 VMs Acquired: {total_acquired}/{TARGET_COUNT}")
        for r in reservations:
            print(f"  -> {r["name"]} in {r["zone"]}: {r["count"]} VMs ({r["status"]})")
            
        if total_acquired >= TARGET_COUNT:
            print(f"\nSUCCESS: Successfully collected {total_acquired} L4 VMs across zones!")
            # Stop remaining background loops
            subprocess.run(["pkill", "-f", "ruthless-retrying-reservations-shared.sh"])
            break
            
        # Check if 15 minutes elapsed
        if time.time() - start_time > 900:
            print("Reached 15 min scan window. Current status:")
            break
            
        time.sleep(15)

if __name__ == "__main__":
    main()
