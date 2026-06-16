import os
import sys
import json
import urllib.request
import urllib.error

# Config
HF_REPO = "moonshotai/Kimi-K2.6"
HF_TOKEN = os.getenv("HF_TOKEN", "")
LOCAL_DIR = "/usr/local/google/home/pallaviam/models/Kimi-K2.6"

IGNORE_EXTENSIONS = ['.msgpack', '.h5', '.ot']

def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    # 1. Fetch file list from Hugging Face API
    api_url = f"https://huggingface.co/api/models/{HF_REPO}"
    print(f"Fetching file list from {api_url}...")
    
    req = urllib.request.Request(api_url)
    req.add_header("Authorization", f"Bearer {HF_TOKEN}")
    
    try:
        with urllib.request.urlopen(req) as response:
            repo_info = json.loads(response.read().decode())
    except Exception as e:
        print(f"Failed to fetch repo info: {e}")
        sys.exit(1)
        
    siblings = repo_info.get("siblings", [])
    files_to_download = []
    
    for s in siblings:
        rpath = s.get("rfilename")
        if not rpath:
            continue
        
        # Filter out ignored extensions
        _, ext = os.path.splitext(rpath)
        if ext in IGNORE_EXTENSIONS:
            continue
            
        files_to_download.append(rpath)
        
    print(f"Found {len(files_to_download)} files to download (excluding ignored extensions).")
    
    # 2. Download files
    for idx, rpath in enumerate(files_to_download, 1):
        local_path = os.path.join(LOCAL_DIR, rpath)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        download_url = f"https://huggingface.co/{HF_REPO}/resolve/main/{rpath}"
        
        # Check if file already exists with same size
        # (Very simple verification, check if file exists)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            print(f"[{idx}/{len(files_to_download)}] Skipping {rpath} (already exists).")
            continue
            
        print(f"[{idx}/{len(files_to_download)}] Downloading {rpath}...")
        
        file_req = urllib.request.Request(download_url)
        file_req.add_header("Authorization", f"Bearer {HF_TOKEN}")
        
        try:
            with urllib.request.urlopen(file_req) as response:
                with open(local_path, "wb") as f:
                    chunk_size = 16 * 1024 * 1024  # 16MB chunk
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
            print(f"Successfully downloaded {rpath}.")
        except Exception as e:
            print(f"Error downloading {rpath}: {e}")
            # Remove partial file if download failed
            if os.path.exists(local_path):
                os.remove(local_path)
            # Stop execution so user can see what failed
            sys.exit(1)

    print("All downloads complete!")

if __name__ == "__main__":
    main()
