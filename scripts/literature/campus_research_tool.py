import requests
import json
import os
import time

# --- Configuration ---
# Search Keywords (English)
QUERIES = [
    "ordered alloy electrocatalysis",
    "intermetallic compound electrocatalysis",
    "ordered intermetallic electrocatalyst"
]
LIMIT_PER_QUERY = 20 # Number of papers to fetch per keyword

# Paths
BASE_DIR = "data/literature/campus_research"
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
LOG_FILE = os.path.join(BASE_DIR, "search_log.json")

os.makedirs(PDF_DIR, exist_ok=True)

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"

def search_semantic_scholar(query, limit, retries=3):
    print(f"🔍 Searching API for: '{query}'...")
    params = {
        "query": query,
        "limit": limit,
        "fields": "paperId,title,abstract,year,openAccessPdf,externalIds,url,venue"
    }
    
    for attempt in range(retries):
        try:
            r = requests.get(SEMANTIC_SCHOLAR_API, params=params, timeout=15)
            if r.status_code == 200:
                print(f"  ✅ Success on attempt {attempt+1}")
                return r.json().get("data", [])
            elif r.status_code == 429:
                wait_time = (attempt + 1) * 5 + 5 # 10s, 15s, 20s
                print(f"  ⚠️ Rate Limit (429). Waiting {wait_time}s before retry ({attempt+1}/{retries})...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ API Error {r.status_code}")
                return []
        except Exception as e:
            print(f"  ❌ Connection Error: {e}")
            return []
            
    print("  ❌ Failed after retries.")
    return []

def download_paper(paper):
    paper_id = paper.get("paperId")
    title = paper.get("title", "Untitled")
    
    # Clean filename
    safe_title = "".join([c if c.isalnum() else "_" for c in title])[:50]
    filename = f"{safe_title}_{paper_id}.pdf"
    save_path = os.path.join(PDF_DIR, filename)
    
    if os.path.exists(save_path):
        print(f"  ⏩ Exists: {filename}")
        return True

    # 1. Try Open Access Link first
    url = None
    if paper.get("openAccessPdf"):
        url = paper["openAccessPdf"].get("url")
    
    # 2. If no OA link, try DOI resolution (Best Effort for Campus Net)
    # Note: This is tricky with plain requests, but we'll try standard DOI.org matching
    if not url and paper.get("externalIds") and paper["externalIds"].get("DOI"):
        # Just logging the DOI link, usually requests can't bypass landing page
        # But sometimes publisher redirects to PDF directly if authorized
        doi = paper["externalIds"]["DOI"]
        # url = f"https://doi.org/{doi}" 
        # We won't try to download typical DOI links blindly as they return HTML landing pages.
        pass

    if url:
        print(f"  ⬇️  Downloading: {title[:40]}...")
        try:
            # Fake User-Agent to look like browser
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            
            if r.status_code == 200 and "application/pdf" in r.headers.get("Content-Type", "").lower():
                with open(save_path, "wb") as f:
                    f.write(r.content)
                print(f"    ✅ Saved!")
                return True
            else:
                print(f"    ⚠️  Failed (Not PDF/Protected). Status: {r.status_code}")
        except Exception as e:
            print(f"    ❌ Download Error: {e}")
            
    return False

def main():
    print("🎓 Campus Research Tool Initiated")
    print(f"📂 Output Directory: {os.path.abspath(PDF_DIR)}")
    print("------------------------------------------------")
    
    all_papers = {}
    
    # 1. Search Phase
    for q in QUERIES:
        results = search_semantic_scholar(q, LIMIT_PER_QUERY)
        print(f"  Found {len(results)} results.")
        for p in results:
            if p["paperId"] not in all_papers:
                all_papers[p["paperId"]] = p
        
        print("  RESTING: Waiting 5 seconds before next query to respect API limits...")
        time.sleep(5) # Increase delay to avoid 429

    print(f"\n📦 Total Unique Papers Found: {len(all_papers)}")
    print("🚀 Starting Batch Download (Campus Network Mode)...")
    
    # 2. Download Phase
    success_count = 0
    download_list = [] # For record keeping
    
    for pid, paper in all_papers.items():
        if download_paper(paper):
            success_count += 1
            paper["local_path"] = os.path.join(PDF_DIR, f"{pid}.pdf")
        else:
            paper["download_status"] = "failed_or_paywalled"
            
        download_list.append(paper)
        
    # Save Log
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(download_list, f, indent=2, ensure_ascii=False)
        
    print("\n------------------------------------------------")
    print(f"🎉 Session Complete.")
    print(f"📥 Downloaded: {success_count}/{len(all_papers)}")
    print(f"📊 Metadata Log: {LOG_FILE}")
    print("👉 For failed downloads, check the log and download manually via browser.")

if __name__ == "__main__":
    main()
