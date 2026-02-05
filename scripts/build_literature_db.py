import os
import json
import glob
import sys
# Fix import path
sys.path.append(os.getcwd())
try:
    from scripts.pipeline_full_extract import process_pdf_full
except ImportError:
    # Fallback if running from scripts dir
    sys.path.append(os.path.dirname(os.getcwd()))
    from scripts.pipeline_full_extract import process_pdf_full

DATA_DIRS = [
    "data/literature/raw",
    "data/literature/cache",
    "data/literature/papers" # New root
]

OUTPUT_DB_DIR = "data/literature/databases"
os.makedirs(OUTPUT_DB_DIR, exist_ok=True)

REACTION_KEYWORDS = {
    "hor": ["hydrogen oxidation", "hor", "hydrogen electrooxidation"],
    "orr": ["oxygen reduction", "orr"],
    "her": ["hydrogen evolution", "her"],
    "nitraterr": ["nitrate reduction", "nitraterr", "ammonia synthesis"]
}

def classify_pdf(text_snippet):
    """
    Simple keyword classifier. Returns 'hor', 'orr', etc. or 'unknown'.
    """
    text = text_snippet.lower()
    scores = {r: 0 for r in REACTION_KEYWORDS}
    
    for reaction, keywords in REACTION_KEYWORDS.items():
        for k in keywords:
            if k in text:
                scores[reaction] += 1
    
    # Return reaction with max hits if > 0
    best_match = max(scores, key=scores.get)
    if scores[best_match] > 0:
        return best_match
    return "unknown"

def classify_pdf_with_hint(pdf_path, text_snippet):
    # 1. Folder Hint (High Confidence)
    parent_dir = os.path.basename(os.path.dirname(pdf_path)).lower()
    for reaction in REACTION_KEYWORDS:
        if reaction in parent_dir:
            return reaction
            
    # 2. Text Content (Fallback)
    return classify_pdf(text_snippet)

def build_db():
    print("🏭 Starting Literature Data Factory...")
    
    all_pdfs = []
    for d in DATA_DIRS:
        # Recursive glob for subdirectories
        pdfs = glob.glob(os.path.join(d, "**/*.pdf"), recursive=True)
        print(f"  Found {len(pdfs)} PDFs in {d} (Recursive)")
        all_pdfs.extend(pdfs)
        
    print(f"  Total Assets to Process: {len(all_pdfs)}")
    
    databases = {r: [] for r in REACTION_KEYWORDS}
    databases["unknown"] = []
    
    for i, pdf_path in enumerate(all_pdfs):
        print(f"\n[Processing {i+1}/{len(all_pdfs)}] {os.path.basename(pdf_path)}...")
        
        try:
            # 1. Run Extraction
            result = process_pdf_full(pdf_path)
            
            # 2. Classify (Smart Router)
            context_str = json.dumps(result.get("text_data", {}))
            reaction_type = classify_pdf_with_hint(pdf_path, context_str)
            
            print(f"  🏷️ Classified as: {reaction_type.upper()}")
            
            entry = {
                "id": os.path.basename(pdf_path),
                "reaction": reaction_type,
                "data": result["merged_metrics"],
                "raw_extraction": result
            }
            databases[reaction_type].append(entry)
            
        except Exception as e:
            print(f"  ❌ Failed to process {pdf_path}: {e}")

    # 4. Save Databases
    print("\n💾 Saving Databases...")
    for r, items in databases.items():
        if items:
            out_path = os.path.join(OUTPUT_DB_DIR, f"{r}_db.json")
            with open(out_path, "w") as f:
                json.dump(items, f, indent=2)
            print(f"  ✅ Saved {r}_db.json ({len(items)} entries)")

if __name__ == "__main__":
    build_db()
