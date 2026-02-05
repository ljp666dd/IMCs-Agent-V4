import requests
import json
import datetime

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper"
GOLDEN_PAPERS = {
    "10.1038/s41586-020-2160-9": "Review of HOR (Example)",
    "fake_doi_for_test": "Seminal PtFe Paper"
}

def check_citations(doi):
    print(f"📡 Checking citations for: {doi}...")
    # Mocking response to avoid API Rate Limits during dev
    # In production: url = f"{SEMANTIC_SCHOLAR_API}/{doi}/citations?fields=title,year,url&limit=100"
    
    # Simulating data: found a new 2025 paper
    current_year = datetime.datetime.now().year
    mock_citations = [
        {"citingPaper": {"paperId": "abc", "title": "Old Paper", "year": 2020}},
        {"citingPaper": {"paperId": "xyz", "title": "Advanced High Entropy Alloy for HOR", "year": 2025}}
    ]
    
    new_papers = []
    for item in mock_citations:
        paper = item["citingPaper"]
        if paper["year"] and paper["year"] >= (current_year - 1): # Last 2 years
            new_papers.append(paper)
            
    return new_papers

def main():
    print("Starting Citation Watch Service (The Scout)...")
    
    alerts = []
    
    for doi, name in GOLDEN_PAPERS.items():
        new_hits = check_citations(doi)
        if new_hits:
            print(f"  🔔 ALERT: Found {len(new_hits)} new papers citing '{name}'")
            for p in new_hits:
                print(f"     - [New] {p['title']} ({p['year']})")
                alerts.append(p)
        else:
            print(f"  ✅ No new citations for '{name}'")

    print("\nTotal Alerts:", len(alerts))
    # In real pipeline, these would be added to the Download Queue

if __name__ == "__main__":
    main()
