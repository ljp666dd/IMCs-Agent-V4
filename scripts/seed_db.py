import sys
import os
import sqlite3

# Add src path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from services.db.database import DatabaseService

# Platinum (Pt) CIF Content
PT_CIF = """data_Pt
_audit_creation_date              2024-01-24
_chemical_name_common             Platinum
_chemical_formula_sum             Pt
_cell_length_a                    3.92420
_cell_length_b                    3.92420
_cell_length_c                    3.92420
_cell_angle_alpha                 90.00000
_cell_angle_beta                  90.00000
_cell_angle_gamma                 90.00000
_space_group_name_H-M_alt         'F m -3 m'
_space_group_IT_number            225
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Pt1 Pt 0.00000 0.00000 0.00000
"""

def seed_database():
    print("Seeding database with Platinum structure...")
    
    # Ensure data directory exists
    os.makedirs("data/theory/cifs", exist_ok=True)
    
    # Write CIF file
    cif_path = os.path.abspath("data/theory/cifs/Pt_mp-126.cif")
    with open(cif_path, "w") as f:
        f.write(PT_CIF)
        
    # Insert into DB
    db = DatabaseService()
    row_id = db.save_material(
        material_id="mp-126",
        formula="Pt",
        energy=-6.05,
        cif_path=cif_path
    )
    
    print(f"Inserted Pt (mp-126) at Row ID: {row_id}")
    print(f"CIF Path: {cif_path}")

if __name__ == "__main__":
    seed_database()
