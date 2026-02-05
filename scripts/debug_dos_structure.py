import gzip
import json

path = 'data/theory/dos_raw/mp-126_dos.json.gz'
try:
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        data = json.load(f)
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            # Check if nested
            if 'mp-126' in data:
                 print(f"Nested Keys (mp-126): {list(data['mp-126'].keys())}")
except Exception as e:
    print(e)
