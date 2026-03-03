import os
import sys
import logging
logging.basicConfig(level=logging.DEBUG)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from src.services.chemistry.descriptors import StructureFeaturizer

def main():
    featurizer = StructureFeaturizer()
    test_path1 = "data/theory\\cifs\\mp-126.cif"
    test_path2 = "data/theory/cifs/Pt_mp-126.cif"
    test_path3 = "data/theory/cifs/mp-1001836.cif"
    
    for path in [test_path1, test_path2, test_path3]:
        print(f"Testing path: {path}")
        print(f"Exists? {os.path.exists(path)}")
        if os.path.exists(path):
            feats = featurizer.extract(path)
            print(f"Feats length: {len(feats) if feats is not None else 'None'}")
        print("-" * 20)

if __name__ == "__main__":
    main()
