import json
import re
import requests

# --- Configuration ---
OLLAMA_API = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "qwen2.5:7b"

# Sample "Incomplete" Extration (Simulator)
# Scenario: RAG extracted j0 but missed Loading because it was in the Methods section, not Results.
INITIAL_DATA = {
    "material_formula": "PtFe/C",
    "metrics": [{"name": "j0", "value": 0.85, "unit": "mA/cm2"}],
    "conditions": {
        "electrolyte": "0.1 M KOH",
        "loading": None # MISSING!
    },
    "text_context": "The activity was measured... j0 was 0.85... (Result Section)" 
}

# The full text (Simulator) contains the missing info
FULL_TEXT_SIMULATION = """
Experimental Section:
Catalyst inks were prepared by dispersing 5 mg of catalyst in 1 mL ethanol.
The suspension was pipetted onto the glassy carbon electrode.
The final Pt loading was carefully controlled at 0.02 mg_Pt/cm2 for all measurements.
...
Results:
The activity was measured in 0.1 M KOH. j0 was 0.85 mA/cm2.
"""

REQUIRED_FIELDS = ["electrolyte", "loading"]

KEYWORD_MAP = {
    "loading": ["loading", "mg", "cm-2", "cm2", "dispersion"],
    "electrolyte": ["solution", "M ", "KOH", "HClO4", "electrolyte", "pH"],
    "reference_electrode": ["RHE", "SCE", "Ag/AgCl", "vs."]
}

def query_llm_correction(field, context):
    prompt = f"""
    Context: "{context}"
    Task: Extract the value for '{field}' from the context above.
    Format: JSON only. {{ "found": true, "value": "...", "unit": "..." }}
    """
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "json": True # Ollama mode
    }
    
    try:
        r = requests.post(OLLAMA_API, json=payload, timeout=20)
        raw_content = r.json()['choices'][0]['message']['content']
        # print(f"DEBUG LLM: {raw_content}") # Uncomment for debug
        return json.loads(raw_content)
    except Exception as e:
        print(f"LLM Error: {e} | Raw: {r.text if 'r' in locals() else 'None'}")
        return {"found": False}

def find_context_for_field(text, field):
    """
    Simple keyword scanner to find relevant sentences.
    """
    keywords = KEYWORD_MAP.get(field, [])
    hits = []
    sentences = text.split('.')
    for s in sentences:
        if any(k in s for k in keywords):
            hits.append(s.strip())
    
    # Return top 3 hits
    return ". ".join(hits[:3])

def active_loop(data, full_text):
    print("🔄 Starting Active Reading Loop...")
    
    updated_conditions = data["conditions"].copy()
    
    for field in REQUIRED_FIELDS:
        current_value = updated_conditions.get(field)
        
        if current_value is None:
            print(f"  ⚠️ Missing Field: {field}. Initiating Search...")
            
            # 1. Hunt
            context_snippet = find_context_for_field(full_text, field)
            if not context_snippet:
                print(f"    ❌ No relevant keywords found in text for {field}.")
                continue
                
            print(f"    🔎 Context Found: \"{context_snippet[:60]}...\"")
            
            # 2. Refine (LLM Query)
            correction = query_llm_correction(field, context_snippet)
            
            if correction.get("found"):
                val = correction.get("value")
                unit = correction.get("unit")
                print(f"    ✅ FIXED: {field} = {val} {unit}")
                updated_conditions[field] = f"{val} {unit}" if unit else val
            else:
                 print(f"    ❌ LLM could not extract {field} from context.")
        else:
            print(f"  ✅ {field} is present ({current_value})")

    data["conditions"] = updated_conditions
    return data

def main():
    print("--- Before Loop ---")
    print(json.dumps(INITIAL_DATA["conditions"], indent=2))
    
    final_data = active_loop(INITIAL_DATA, FULL_TEXT_SIMULATION)
    
    print("\n--- After Loop ---")
    print(json.dumps(final_data["conditions"], indent=2))

    if final_data["conditions"]["loading"]:
        print("\n✅ Verification Success: Missing 'loading' was recovered.")

if __name__ == "__main__":
    main()
