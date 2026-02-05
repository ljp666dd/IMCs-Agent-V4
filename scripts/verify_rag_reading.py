import os
import sys
import json
import fitz  # PyMuPDF
import requests

# Configuration: Local LLM (Ollama)
OLLAMA_API = "http://localhost:11434/v1/chat/completions"
# Using Qwen2.5-7B (assuming it's pulled as 'qwen2.5:7b' or similar, strict prompt following)
# If user has different model name, this might need adjustment. I'll default to 'qwen2.5:7b-instruct' or just 'qwen2.5'
MODEL_NAME = "qwen2.5:7b" 

SYSTEM_PROMPT = """You are an expert scientific researcher. Extract structured data from the provided text.
Output JSON only. No markdown formatting. No preamble.
Schema:
{
  "material_formula": "string",
  "conditions": {
    "electrolyte": "string (concentration + chemical)",
    "pH": "number",
    "reference_electrode": "string (e.g. RHE)",
    "loading": "string"
  },
  "metrics": [
    {
      "name": "string (e.g. j0, mass_activity)",
      "value": "number",
      "unit": "string",
      "notes": "string"
    }
  ]
}
If data is missing, use null.
"""

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def query_llm(context):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract experimental HOR data from this text:\n\n{context[:12000]}"} # truncated context window
        ],
        "temperature": 0.1,
        "json": True # Ollama supports JSON mode? Or just ask nicely.
    }
    
    try:
        response = requests.post(OLLAMA_API, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error querying LLM: {e}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_rag_reading.py <pdf_path>")
        return

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print(f"Reading {pdf_path}...")
    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(text)} characters.")
    
    print("Querying Local LLM (Ollama)...")
    result = query_llm(text)
    
    print("\n--- LLM Output ---")
    print(result)
    print("------------------")
    
    # Validation
    try:
        data = json.loads(result)
        if "metrics" in data and len(data["metrics"]) > 0:
            print("✅ JSON Parsing Successful. Found metrics.")
        else:
            print("⚠️ JSON Parsed but no metrics found.")
    except:
        print("❌ Failed to parse JSON response.")

if __name__ == "__main__":
    main()
