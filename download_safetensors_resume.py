import os
import sys
import shutil
import requests
from huggingface_hub import get_token, hf_hub_url

def main():
    repo_id = "google/gemma-3-4b-it"
    filename = "model-00001-of-00002.safetensors"
    revision = "093f9f388b31de276ce2de164bdc2081324b9767"
    
    # Paths
    cache_dir = r"C:\Users\riti3\.cache\huggingface\hub\models--google--gemma-3-4b-it"
    snapshot_dir = os.path.join(cache_dir, "snapshots", revision)
    target_path = os.path.join(snapshot_dir, filename)
    incomplete_source = os.path.join(cache_dir, "blobs", "eb5fd5e97ddd07b56778733e9653c07312529cb00980a318fc3e1c4e3b5a8f1f.045245af.incomplete")
    
    # 1. Check if we need to copy/move the incomplete file
    if not os.path.exists(target_path):
        if os.path.exists(incomplete_source):
            print(f"Moving incomplete file from {incomplete_source} to {target_path}...")
            os.makedirs(snapshot_dir, exist_ok=True)
            shutil.copy2(incomplete_source, target_path)
        else:
            print("No incomplete file found, will download from scratch.")
    
    # Get current file size
    current_size = os.path.getsize(target_path) if os.path.exists(target_path) else 0
    print(f"Current file size: {current_size} bytes")
    
    # 2. Get download URL and token
    try:
        token = get_token()
    except Exception:
        token = None
        
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        print("Using cached Hugging Face token.")
    else:
        print("No cached Hugging Face token found. Attempting download without authentication.")
        
    url = hf_hub_url(repo_id, filename, revision=revision)
    print(f"Resolving download URL: {url}")
    
    # 3. Get total file size from HEAD
    r_head = requests.head(url, headers=headers, allow_redirects=True, timeout=30)
    if r_head.status_code != 200:
        print(f"Failed to fetch HEAD, status code: {r_head.status_code}")
        print(r_head.headers)
        sys.exit(1)
        
    total_size = int(r_head.headers.get("content-length", 0))
    print(f"Total remote file size: {total_size} bytes")
    
    if current_size >= total_size and total_size > 0:
        print("File is already complete!")
        sys.exit(0)
        
    # 4. Prepare resume headers
    download_url = r_head.url
    if current_size > 0:
        headers["Range"] = f"bytes={current_size}-"
        print(f"Resuming download from byte {current_size}...")
    
    # 5. Start streaming download
    print(f"Downloading to {target_path}...")
    r_get = requests.get(download_url, headers=headers, stream=True, timeout=60)
    
    if r_get.status_code not in (200, 206):
        print(f"Failed to start download, status code: {r_get.status_code}")
        sys.exit(1)
        
    mode = "ab" if current_size > 0 else "wb"
    downloaded = current_size
    last_printed = current_size
    
    with open(target_path, mode) as f:
        for chunk in r_get.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                # Print progress every 10 MB
                if downloaded - last_printed >= 10 * 1024 * 1024 or downloaded == total_size:
                    progress_pct = (downloaded / total_size) * 100 if total_size > 0 else 0
                    print(f"Downloaded {downloaded} / {total_size} bytes ({progress_pct:.2f}%)", flush=True)
                    last_printed = downloaded
                    
    print("Download completed successfully!")
    print(f"Final file size: {os.path.getsize(target_path)} bytes")

if __name__ == "__main__":
    main()
