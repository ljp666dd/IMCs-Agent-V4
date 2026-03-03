import os
import json
import random

def generate_mock_ml_data():
    """Generates mock material data for testing the ML pipeline."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    theory_dir = os.path.join(base_dir, "data", "theory")
    os.makedirs(theory_dir, exist_ok=True)
    
    # Required output files
    fe_full_path = os.path.join(theory_dir, "formation_energy_full.json")
    fe_ext_path = os.path.join(theory_dir, "formation_energy_extended.json")
    dos_full_path = os.path.join(theory_dir, "dos_descriptors_full.json")
    dos_ext_path = os.path.join(theory_dir, "dos_data_extended.json")
    
    print("Generating mock ML data...")
    
    mock_dataset = []
    
    # Generate 100 realistic-looking mock materials (mostly Pt alloys for HOR context)
    elements = ["Pt", "Pd", "Ni", "Co", "Fe", "Cu", "Ru", "Rh", "Ir"]
    
    for i in range(1, 101):
        # Pick 1-3 random elements
        num_els = random.randint(1, 3)
        chosen = random.sample(elements, num_els)
        formula = "".join(chosen)
        
        # 42 structural features (as defined in ML Agent / Featurizer)
        record = {
            "material_id": f"mock-{1000+i}",
            "formula": formula,
            "formation_energy": random.uniform(-4.0, 1.0),  # Target 1
            "energy_above_hull": random.uniform(0.0, 0.5),
            
            # The 42 ML Features (Simulated)
            "n_atoms": float(random.randint(1, 10)),
            "n_elements": float(num_els),
            "volume_per_atom": random.uniform(10.0, 20.0),
            "density": random.uniform(5.0, 22.0),
            "packing_fraction": random.uniform(0.5, 0.74),
            
            "avg_Z": random.uniform(20.0, 80.0),
            "std_Z": random.uniform(0.0, 20.0),
            "max_Z": random.uniform(20.0, 80.0),
            "min_Z": random.uniform(10.0, 40.0),
            "range_Z": random.uniform(0.0, 60.0),
            
            "avg_mass": random.uniform(50.0, 200.0),
            "std_mass": random.uniform(0.0, 50.0),
            "max_mass": random.uniform(50.0, 250.0),
            "min_mass": random.uniform(10.0, 100.0),
            
            "avg_electronegativity": random.uniform(1.5, 2.5),
            "std_electronegativity": random.uniform(0.0, 0.5),
            "max_electronegativity": random.uniform(1.8, 2.6),
            "min_electronegativity": random.uniform(1.0, 2.2),
            "range_electronegativity": random.uniform(0.0, 1.0),
            
            "avg_radius": random.uniform(1.2, 1.6),
            "std_radius": random.uniform(0.0, 0.2),
            "max_radius": random.uniform(1.3, 1.8),
            "min_radius": random.uniform(0.5, 1.3),
            "radius_ratio": random.uniform(1.0, 1.5),
            
            "composition_entropy": random.uniform(0.0, 1.5),
            "composition_variance": random.uniform(0.0, 0.5),
            "max_composition": random.uniform(0.3, 1.0),
            "min_composition": random.uniform(0.1, 0.5),
            "n_elements_comp": float(num_els),
            
            "lattice_a": random.uniform(3.0, 10.0),
            "lattice_b": random.uniform(3.0, 10.0),
            "lattice_c": random.uniform(3.0, 15.0),
            "alpha": random.uniform(60.0, 120.0),
            "beta": random.uniform(60.0, 120.0),
            "gamma": random.uniform(60.0, 120.0),
            "c_over_a": random.uniform(0.8, 3.0),
            
            "avg_lattice": random.uniform(3.0, 12.0),
            "lattice_distortion": random.uniform(0.0, 0.1),
            
            "mixing_enthalpy_proxy": random.uniform(-1.0, 0.5),
            "avg_valence_electrons": random.uniform(4.0, 10.0),
            "std_valence_electrons": random.uniform(0.0, 3.0),
            "volume": random.uniform(10.0, 300.0),
            
            # The 11 DOS Targets / Features
            "d_band_center": random.uniform(-4.0, 0.0),     # Target 2
            "d_band_width": random.uniform(1.0, 5.0),
            "d_band_filling": random.uniform(0.0, 1.0),
            "DOS_EF": random.uniform(0.1, 5.0),
            "DOS_window": random.uniform(5.0, 15.0),
            "unoccupied_d_states": random.uniform(0.0, 5.0),
            "epsilon_d_minus_EF": random.uniform(-4.0, 1.0),
            "valence_DOS_slope": random.uniform(-1.0, 1.0),
            "num_DOS_peaks": float(random.randint(1, 5)),
            "first_peak_position": random.uniform(-6.0, -1.0),
            "total_states": random.uniform(10.0, 100.0)
        }
        mock_dataset.append(record)

    # Save to all expected target files so training works immediately
    for p in [fe_full_path, fe_ext_path, dos_full_path, dos_ext_path]:
        with open(p, 'w') as f:
            json.dump(mock_dataset, f, indent=2)
            print(f"[{p}] -> Written {len(mock_dataset)} mock records.")

    print("\nMock ML Data Generation Complete.")
    print("You can now test the ML Training tab in the UI without real MP downloads.")

if __name__ == "__main__":
    generate_mock_ml_data()
