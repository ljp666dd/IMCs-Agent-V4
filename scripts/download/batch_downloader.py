import json
import os
import requests
import time

QUEUE_FILE = "data/literature/raw/download_queue.json"
OUTPUT_DIR = "data/literature/papers/fresh_downloads"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def batch_download():
    print("⬇️ Starting Batch Downloader (Campus Mode)...")
    
    if not os.path.exists(QUEUE_FILE):
        print(f"❌ Queue file not found: {QUEUE_FILE}")
        return

    with open(QUEUE_FILE, "r") as f:
        queue = json.load(f)
        
    print(f"📦 Found {len(queue)} items in queue.")
    
    success_count = 0
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for i, item in enumerate(queue):
        url = item.get("url")
        filename = item.get("filename", f"doc_{i}.pdf")
        
        # If it's a DOI link (e.g. from paywall), usually we need a resolver or browser automation.
        # But specifically for campus IPs, many publishers redirect DOI -> PDF if you are lucky, 
        # or at least we can try the URL found in the metadata.
        # For this script we assume the URL is direct-ish or redirectable.
        
        print(f"\n[{i+1}/{len(queue)}] {item.get('title')[:50]}...")
        print(f"  Target: {url}")
        
        try:
            r = requests.get(url, headers=headers, allow_redirects=True, timeout=30)
            if r.status_code == 200 and "application/pdf" in r.headers.get("Content-Type", ""):
                save_path = os.path.join(OUTPUT_DIR, filename)
                with open(save_path, "wb") as f:
                    f.write(r.content)
                print(f"  ✅ Saved to {filename}")
                success_count += 1
            else:
                print(f"  ⚠️ Failed or Not PDF (Status: {r.status_code}, Type: {r.headers.get('Content-Type')})")
                # Backup: Should try to open in browser?
                # User can manual download if script fails.
        except Exception as e:
            print(f"  ❌ Error: {e}")
            
        time.sleep(1) # Be polite

    print(f"\n🎉 Batch Complete. Downloaded {success_count}/{len(queue)}.")
    print(f"📂 Check folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    batch_download()
