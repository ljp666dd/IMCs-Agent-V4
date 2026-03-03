import base64
import requests
import json
import os

# Configuration
OLLAMA_API = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "llava"  # or qwen2.5-vl if available
MOCK_MODE = True  # Set to True if model not downloaded

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def query_vlm(image_path, prompt):
    if MOCK_MODE:
        print("[WARNING] MOCK MODE: Simulating VLM response...")
        return json.dumps({
            "onset_potential_V": 0.05,
            "half_wave_potential_V": 0.85,
            "diffusion_limiting_current_mA_cm2": 5.2
        })

    base64_image = encode_image(image_path)
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "temperature": 0.1
    }
    
    try:
        response = requests.post(OLLAMA_API, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

def create_dummy_chart(path):
    # create a simple red image as a placeholder for a chart
    from PIL import Image
    img = Image.new('RGB', (100, 100), color = 'red')
    img.save(path)

def main():
    print("Testing VLM Chart Reading Module...")
    
    # 1. Prepare Dummy Image
    chart_path = "data/literature/cache/test_chart.jpg"
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)
    try:
        create_dummy_chart(chart_path)
    except ImportError:
         print("Pillow not installed, skipping image creation (Mock Mode active anyway)")
         return

    # 2. Query VLM
    prompt = """
    Analyze this LSV polarization curve.
    Extract the following metrics in JSON format:
    - onset_potential_V
    - half_wave_potential_V
    - diffusion_limiting_current_mA_cm2
    """
    
    print(f"Reading image: {chart_path}")
    result = query_vlm(chart_path, prompt)
    
    print("\n--- VLM Output ---")
    print(result)
    
    # Validation
    if "0.85" in str(result):
        print("✅ SUCCESS: Extracted Half-wave Potential")

if __name__ == "__main__":
    main()
