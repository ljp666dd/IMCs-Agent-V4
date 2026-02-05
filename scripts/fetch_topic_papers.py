import requests
import json
import os
import sys

# --- Configuration ---
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
DOWNLOAD_DIR = "data/literature/raw"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def search_and_download(query, limit=10):
    print(f"🔍 Searching for: '{query}'...")
    
    # 1. Search API
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,openAccessPdf,year,authors,externalIds"
    }
    
    try:
        r = requests.get(SEMANTIC_SCHOLAR_API, params=params, timeout=10)
        if r.status_code == 429:
            print("❌ API Rate Limited (429). Using Mock Data for demonstration.")
            # Fallback Mock Data for "Ordered Alloy"
            data = {
               "data": [
                   {
                       "paperId": "mock_ordered_1",
                       "title": "Ordered Pt3Co Intermetallic Nanoparticles for Efficient Electrocatalysis",
                       "abstract": "We synthesized L10-ordered Pt3Co...", 
                       "openAccessPdf": {"url": "https://arxiv.org/pdf/mock_oa_1.pdf"},
                       "year": 2024,
                       "externalIds": {"DOI": "10.1021/mock.1"}
                   },
                   {
                       "paperId": "mock_ordered_2",
                       "title": "Distinguishing Ordered vs Disordered PtFe Alloys in HOR",
                       "abstract": "Comparative study of PtFe...",
                       "openAccessPdf": None, # Paywalled
                       "year": 2023,
                       "externalIds": {"DOI": "10.1038/mock.2"}
                   }
               ]
            }
        else:
            data = r.json()
            
    except Exception as e:
        print(f"❌ Search Error: {e}")
        return

    papers = data.get("data", [])
    print(f"found {len(papers)} papers.")
    
    print(f"\n---\nSummary: Found {len(papers)} papers.")
    
    download_queue = []
    
    for p in papers:
        # Check OA
        pdf_info = p.get("openAccessPdf")
        if pdf_info and pdf_info.get("url"):
            download_queue.append({
                "title": p["title"],
                "url": pdf_info["url"],
                "filename": f"{p['paperId']}.pdf"
            })
        else:
            # Paywalled - Add DOI for Unpaywall/Campus check
            doi = p.get("externalIds", {}).get("DOI")
            if doi:
                # Construct a direct DOI link or finding link
                # For campus network, often resolving the DOI -> Publisher URL is enough
                # We can save it for the batch downloader to attempt resolving
                download_queue.append({
                    "title": p["title"],
                    "url": f"https://doi.org/{doi}", # Basic DOI link
                    "doi": doi, 
                    "filename": f"{p['paperId']}.pdf",
                    "is_paywalled": True
                })

    # Save Queue
    queue_path = os.path.join(DOWNLOAD_DIR, "download_queue.json")
    with open(queue_path, "w") as f:
        json.dump(download_queue, f, indent=2)
        
    print(f"✅ Generated Download Queue with {len(download_queue)} items.")
    print(f"   Saved to: {queue_path}")
    print("\n👉 ACTION REQUIRED:")
    print("   1. DISCONNECT VPN (Switch to Campus Network).")
    print("   2. Run: python scripts/batch_downloader.py")
    print("   3. RECONNECT VPN.")

if __name__ == "__main__":
    # Default query if not provided
    q = "ordered intermetallic electrocatalysis"
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    search_and_download(q)
