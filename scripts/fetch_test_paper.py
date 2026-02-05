import requests
import json
import os

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
CACHE_DIR = "data/literature/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def search_and_download(query):
    print(f"Searching for: {query}")
    params = {
        "query": query,
        "limit": 5,
        "fields": "title,abstract,openAccessPdf,year"
    }
    r = requests.get(SEMANTIC_SCHOLAR_API, params=params)
    data = r.json()
    
    if "data" not in data:
        print("No results found.")
        return None

    for paper in data["data"]:
        print(f"Checking: {paper.get('title')}")
        pdf_info = paper.get("openAccessPdf")
        if pdf_info and pdf_info.get("url"):
            url = pdf_info["url"]
            print(f"Found PDF URL: {url}")
            try:
                # Try downloading
                headers = {'User-Agent': 'Mozilla/5.0'}
                pdf_r = requests.get(url, headers=headers, timeout=10)
                if pdf_r.status_code == 200:
                    filename = f"test_hor_paper_{paper['paperId']}.pdf"
                    path = os.path.join(CACHE_DIR, filename)
                    with open(path, "wb") as f:
                        f.write(pdf_r.content)
                    print(f"✅ Downloaded to {path}")
                    return path
            except Exception as e:
                print(f"Download failed: {e}")
    
    print("❌ No downloadable PDF found in top results.")
    return None

if __name__ == "__main__":
    search_and_download("PtFe hydrogen oxidation reaction electrocatalyst")
