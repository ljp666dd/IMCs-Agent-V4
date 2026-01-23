import os
import json
import re
from pymatgen.core import Composition

class ExperimentalAgent:
    def __init__(self, db_path="data/literature/experimental_db.json"):
        self.db_path = db_path
        self.data = []
        self.load_database()

    def load_database(self):
        """Load the structured experimental database."""
        if os.path.exists(self.db_path):
            with open(self.db_path, "r", encoding='utf-8') as f:
                self.data = json.load(f)
            # Normalize keys on load
            for entry in self.data:
                try:
                    entry['normalized_formula'] = Composition(entry['formula']).reduced_formula
                except:
                    entry['normalized_formula'] = entry['formula']
        else:
            print(f"Warning: Literature DB not found at {self.db_path}")

    def query_literature(self, query):
        """
        Query the database for a material.
        Returns a list of matching entries (Formula, Overpotential, Reference).
        """
        results = []
        
        # 1. Normalize Query
        try:
            norm_query = Composition(query).reduced_formula
        except:
            norm_query = query.strip()
            
        # 2. Search
        # Direct Match
        for entry in self.data:
            # Check normalized formula match
            if entry.get('normalized_formula') == norm_query:
                results.append(entry)
                continue
            
            # Fuzzy match (e.g. searching for "PtFe" finds "PtFeCo") - maybe too broad?
            # Let's stick to exact or "contains" logic if query is a subset
            # For 5-element alloys, users usually want specific composition.
            
            if norm_query in entry.get('normalized_formula', ''):
                 results.append(entry)

        return results

    def extract_from_text(self, text):
        """
        Simulate extraction from raw text using Regex.
        Finds patterns like 'Pt3Co ... 50 mV'.
        (Prototype for future PDF reading)
        """
        # 1. Find Formulas (Simple Regex)
        formula_pattern = r'\b[A-Z][a-z]?\d*[A-Z][a-z]?\d*[A-Z]?[a-z]?\d*\b'
        # 2. Find Values (e.g. 10-100 mV)
        value_pattern = r'(\d{2,3})\s?mV'
        
        comp_matches = re.findall(formula_pattern, text)
        val_matches = re.findall(value_pattern, text)
        
        return {
            "potential_formulas": list(set(comp_matches)),
            "potential_values": list(set(val_matches))
        }

# Initial Mock Data for Testing
if __name__ == "__main__":
    agent = ExperimentalAgent()
    print(agent.query_literature("PtFeCoNiCu"))
