import argparse
import json
import math
import os
import random
import sqlite3
import sys
from typing import List, Dict, Any

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.agents.core.theory_agent import TheoryDataConfig


def _load_allowed_formulas(db_path: str, allowed: set) -> List[str]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT formula FROM materials WHERE formula IS NOT NULL")
    formulas = [row[0] for row in cur.fetchall()]
    conn.close()
    # Keep only single-element formulas in allowed set
    return [f for f in formulas if f in allowed]


def _make_curve(potentials, j0, tafel_b, j_lim):
    # Tafel-like kinetic current
    # j_k = j0 * 10^(eta / b)
    eta = potentials
    j_k = j0 * (10 ** (eta / tafel_b))
    # Koutecky-Levich mix
    j = 1.0 / (1.0 / j_k + 1.0 / j_lim)
    # add small noise
    noise = (0.02 * j_lim) * (2 * (random.random() - 0.5))
    return j + noise


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic LSV curves for RDE analysis.")
    parser.add_argument("--db", dest="db_path", default="data/imcs.db", help="SQLite DB path")
    parser.add_argument("--output-dir", default="data/experimental/rde_lsv", help="Output directory")
    parser.add_argument("--materials", default="", help="Comma-separated formulas (e.g., Pt,Pd,Ni)")
    parser.add_argument("--count", type=int, default=5, help="Number of materials if auto-selected")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    cfg = TheoryDataConfig()
    allowed = set(cfg.elements)

    if args.materials.strip():
        materials = [m.strip() for m in args.materials.split(",") if m.strip()]
    else:
        materials = _load_allowed_formulas(args.db_path, allowed)[: args.count]

    if not materials:
        raise SystemExit("No materials found. Provide --materials or ensure DB has allowed formulas.")

    os.makedirs(args.output_dir, exist_ok=True)

    rpm_list = [2500, 1600, 900, 400]
    # potential range (V vs RHE)
    potentials = [round(x, 4) for x in [i * 0.005 for i in range(0, 121)]]

    # Fixed experimental conditions
    conditions = {
        "electrolyte": "0.1M KOH",
        "method": "RDE",
        "pH": 13,
        "temperature_K": 298,
        "precious_metal_loading_fraction": 0.20,
        "ink": {
            "catalyst_mg": 5,
            "water_uL": 475,
            "ethanol_uL": 475,
            "nafion_uL": 50,
            "drop_uL": 10,
        },
        "loading_mg_cm2": 0.25,
    }

    manifest = {"conditions": conditions, "materials": materials, "files": []}

    for formula in materials:
        # material-specific kinetic params
        j0 = random.uniform(0.01, 0.08)  # mA/cm2
        tafel_b = random.uniform(0.04, 0.08)  # V/dec
        jlim_1600 = random.uniform(4.0, 10.0)  # mA/cm2

        for rpm in rpm_list:
            j_lim = jlim_1600 * math.sqrt(rpm / 1600.0)
            currents = [_make_curve(p, j0, tafel_b, j_lim) for p in potentials]

            filename = f"{formula}_LSV_{rpm}rpm.csv"
            path = os.path.join(args.output_dir, filename)

            # Include metadata columns for traceability
            rows = []
            for v, j in zip(potentials, currents):
                rows.append({
                    "Potential_V": v,
                    "Current_mAcm2": round(j, 6),
                    "RPM": rpm,
                    "pH": conditions["pH"],
                    "Temperature_K": conditions["temperature_K"],
                    "Electrolyte": conditions["electrolyte"],
                })

            import pandas as pd
            pd.DataFrame(rows).to_csv(path, index=False)
            manifest["files"].append(path)

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(manifest['files'])} files -> {args.output_dir}")


if __name__ == "__main__":
    main()
