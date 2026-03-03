import requests
import json

OLLAMA_API = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "qwen2.5:7b" 

SYSTEM_PROMPT = """You are an expert scientific researcher. Extract structured data from the provided text.
Output JSON only. No markdown.
Schema:
{
  "material_formula": "string",
  "conditions": {
    "electrolyte": "string",
    "pH": "number (estimate from electrolyte)",
    "loading": "string"
  },
  "metrics": [
    {
      "name": "string",
      "value": "number",
      "unit": "string",
      "notes": "string"
    }
  ]
}
"""

TEST_TEXT = """
The electrocatalytic hydrogen oxidation reaction (HOR) activity of the prepared PtFe/C nanocubes was evaluated in 0.1 M KOH solution using a rotating disk electrode (RDE). 
The Pt loading on the electrode was maintained at 0.02 mg_Pt/cm2.
The polarization curves were recorded at a scan rate of 5 mV/s at room temperature. 
The kinetic current density (jk) at 0.05 V vs. RHE was calculated to be 2.5 mA/cm2, which is significantly higher than that of commercial Pt/C (1.1 mA/cm2). 
The specific exchange current density (j0) was determined to be 0.85 mA/cm2 based on the Butler-Volmer equation fitting. 
The electrochemical surface area (ECSA) was measured to be 65 m2/g using CO stripping voltammetry.
"""

def query_llm(context):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract experimental HOR data from this text:\n\n{context}"} 
        ],
        "temperature": 0.1,
        "json": True
    }
    
    try:
        response = requests.post(OLLAMA_API, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

def main():
    print("Testing RAG Engine with Synthetic HOR Text...")
    print(f"Input Text:\n{TEST_TEXT.strip()}")
    print("-" * 20)
    
    result = query_llm(TEST_TEXT)
    
    print("LLM Output:")
    print(result)
    
    # Validation
    if "0.85" in result and "KOH" in result:
        print("\n✅ SUCCESS: Extracted j0 (0.85) and Electrolyte (KOH)")
    else:
        print("\n❌ FAILURE: Failed to extract key data")

if __name__ == "__main__":
    main()
