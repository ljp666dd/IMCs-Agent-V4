import requests
import json
import os
import time

# --- Configuration ---
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
OLLAMA_API = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "qwen2.5:7b"
CACHE_DIR = "data/literature/raw"
os.makedirs(CACHE_DIR, exist_ok=True)

HEADERS = {'User-Agent': 'Mozilla/5.0 (Scientific Research Agent; contact: admin@example.com)'}

def search_papers(query, limit=5):
    print(f"🔍 Searching Semantic Scholar for: '{query}'...")
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,openAccessPdf,year,citationCount,authors"
    }
    try:
        raise Exception("Force Mock Data")
        r = requests.get(SEMANTIC_SCHOLAR_API, params=params, headers=HEADERS, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("data", [])
    except Exception as e:
        print(f"❌ Search failed: {e}")
        print("⚠️ Switching to MOCK DATA for verification...")
        return [
            {
                "paperId": "mock_1",
                "title": "Density Functional Theory Study of Pt Surface",
                "abstract": "We performed DFT calculations to understand the binding energy of Hydrogen on Pt(111). No experiments were performed.",
                "openAccessPdf": None
            },
            {
                "paperId": "mock_2",
                "title": "Experimental High Performance PtFe for HOR",
                "abstract": "We synthesized PtFe nanocubes and evaluated their HOR activity in 0.1 M KOH. The exchange current density was 2.5 mA/cm2.",
                "openAccessPdf": {"url": "https://arxiv.org/pdf/2106.00001.pdf"} # Dummy valid PDF URL (or close to it)
            }
        ]

def llm_filter_abstract(title, abstract, target_reaction="HOR"):
    """
    Uses LLM to judge if the paper has experimental electrochemical data.
    """
    if not abstract:
        return False, "No abstract"
        
    prompt = f"""
    You are a scientific research assistant. 
    Analyze the following abstract. 
    Does it contain EXPERIMENTAL electrochemical data (e.g. j0, mass activity, polarization curves) for the {target_reaction} reaction?
    Ignore pure DFT/Theory papers or Reviews.
    
    Hash: {hash(abstract)}
    Title: {title}
    Abstract: {abstract[:2000]}
    
    Reply JSON with: {{"is_relevant": true/false, "reason": "short explanation"}}
    """
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "json": True
    }
    
    try:
        r = requests.post(OLLAMA_API, json=payload, timeout=20)
        if r.status_code != 200:
            return True, "LLM Offline (Default Accept)" # Fallback
            
        result = r.json()['choices'][0]['message']['content']
        decision = json.loads(result)
        return decision.get("is_relevant", False), decision.get("reason", "No reason")
    except Exception:
        return True, "LLM/JSON Error (Default Accept)"

def download_pdf(url, paper_id):
    print(f"⬇️ Downloading PDF from {url}...")
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            filename = f"{paper_id}.pdf"
            path = os.path.join(CACHE_DIR, filename)
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"✅ Saved to {path}")
            return path
        else:
            print(f"❌ Download failed (Status {r.status_code})")
    except Exception as e:
        print(f"❌ Download error: {e}")
    return None

def main():
    # 1. Search
    papers = search_papers("PtFe hydrogen oxidation reaction electrocatalyst", limit=3)
    if not papers:
        print("No papers found. Exiting.")
        return

    print(f"\nFound {len(papers)} candidate papers. Starting Filter Pipeline...\n")

    for p in papers:
        pid = p['paperId']
        title = p['title']
        
        # 2. Filter (The Scout)
        print(f"Processing: {title[:60]}...")
        is_relevant, reason = llm_filter_abstract(title, p.get('abstract'), "HOR")
        print(f"   🤖 AI Verdict: {'PASS' if is_relevant else 'SKIP'} ({reason})")
        
        if is_relevant:
            # 3. Download (The Fetcher)
            pdf_info = p.get('openAccessPdf')
            if pdf_info and pdf_info.get('url'):
                download_pdf(pdf_info['url'], pid)
            else:
                print("   ⚠️ No Open Access PDF available. Added to DOI Watchlist.")
        print("-" * 50)

if __name__ == "__main__":
    main()
