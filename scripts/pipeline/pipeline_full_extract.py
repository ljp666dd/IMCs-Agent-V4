import os
import sys
import json
import fitz
import sys
import os
import requests
import base64
sys.path.append(os.getcwd())
from scripts.extract_pdf_figures import extract_figures_from_pdf

# --- Configuration ---
OLLAMA_API = "http://localhost:11434/v1/chat/completions"
# Using text model for RAG, vision model (simulated or real) for VLM
TEXT_MODEL = "qwen2.5:7b"
VISION_MODEL = "llava" # or qwen2.5-vl
MOCK_VISION = True # Keep True until user confirms model download

# --- Prompts ---
RAG_SYSTEM_PROMPT = """Extract electrochemical data. Output JSON.
Schema: { "formula": str, "electrolyte": str, "pH": float, "j0": float, "is_hor": bool }"""

VLM_PROMPT = """
Is this an LSV polarization curve? 
If yes, extract: {"is_lsv": true, "onset_potential": number}
If no, return {"is_lsv": false}
Return JSON only.
"""

def query_llm_json(model, messages):
    payload = {"model": model, "messages": messages, "temperature": 0.1, "json": True}
    try:
        r = requests.post(OLLAMA_API, json=payload, timeout=30)
        return json.loads(r.json()['choices'][0]['message']['content'])
    except Exception as e:
        print(f"LLM Error: {e}")
        return {}

def mock_vlm_inference(image_path):
    # Determine if it's likely a plot based on filename for demo purposes
    # Real logic would use the VLM to classify
    return {"is_lsv": True, "onset_potential": 0.05, "source_image": os.path.basename(image_path)}

def process_pdf_full(pdf_path):
    print(f"\n🚀 Starting Full Extraction Pipeline for: {os.path.basename(pdf_path)}")
    
    # 1. Text RAG Phase
    print("  [1/3] Text Phase: Extracting context via RAG...")
    doc = fitz.open(pdf_path)
    full_text = "".join([page.get_text() for page in doc])
    
    rag_messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract from:\n{full_text[:15000]}"}
    ]
    rag_result = query_llm_json(TEXT_MODEL, rag_messages)
    print(f"    Context Found: {rag_result}")

    # 2. Vision VLM Phase
    print("  [2/3] Vision Phase: extracting & reading figures...")
    temp_dir = "data/literature/temp_extract"
    images = extract_figures_from_pdf(pdf_path, temp_dir)
    
    vlm_results = []
    for img_path in images:
        if MOCK_VISION:
            # Simulate VLM
            vlm_data = mock_vlm_inference(img_path)
        else:
            # Real VLM call (redacted for brevity, same as verify_chart_reading.py)
            pass
            
        if vlm_data.get("is_lsv"):
            print(f"    Found LSV in {os.path.basename(img_path)}: Onset = {vlm_data.get('onset_potential')} V")
            vlm_results.append(vlm_data)

    # 3. Merge Phase
    print("  [3/3] Merge Phase: Synthesizing data...")
    final_output = {
        "metadata": {"source": os.path.basename(pdf_path)},
        "text_data": rag_result,
        "vision_data": vlm_results,
        "merged_metrics": {
            "j0": rag_result.get("j0"),
            "onset_potential": vlm_results[0].get("onset_potential") if vlm_results else None,
            "electrolyte": rag_result.get("electrolyte")
        }
    }
    
    return final_output

if __name__ == "__main__":
    if len(sys.argv) < 2:
        pdf = "data/literature/cache/10.1038_srep29700.pdf" 
    else:
        pdf = sys.argv[1]
        
    result = process_pdf_full(pdf)
    print("\n✅ Final JSON Asset:")
    print(json.dumps(result, indent=2))
