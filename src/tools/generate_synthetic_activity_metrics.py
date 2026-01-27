import argparse
import json
import os
import random
import sqlite3
import sys
from typing import List, Dict, Any

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.agents.core.theory_agent import TheoryDataConfig


def _load_materials(db_path: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT material_id, formula FROM materials WHERE material_id IS NOT NULL AND formula IS NOT NULL")
    rows = cur.fetchall()
    conn.close()
    return [{"material_id": r[0], "formula": r[1]} for r in rows]


def _allowed_formula(formula: str, allowed: set) -> bool:
    if not formula:
        return False
    try:
        from pymatgen.core import Composition
        elements = {el.symbol for el in Composition(formula).elements}
        return bool(elements) and elements.issubset(allowed)
    except Exception:
        return False


def _make_metric_records(material_id: str, formula: str) -> List[Dict[str, Any]]:
    metrics = []
    metrics.append({
        "metric_name": "exchange_current_density",
        "metric_value": round(10 ** random.uniform(-5.0, -2.0), 8),
        "unit": "A/cm2",
    })
    metrics.append({
        "metric_name": "overpotential_10mAcm2",
        "metric_value": round(random.uniform(20.0, 300.0), 2),
        "unit": "mV",
    })
    metrics.append({
        "metric_name": "tafel_slope",
        "metric_value": round(random.uniform(30.0, 150.0), 2),
        "unit": "mV/dec",
    })
    conditions = {
        "pH": random.choice([0, 1, 7, 13]),
        "temperature_K": random.choice([298, 303, 313]),
        "electrolyte": random.choice(["H2SO4", "KOH", "HClO4"]),
        "method": random.choice(["RDE", "CV", "LSV"]),
    }
    records = []
    for m in metrics:
        records.append({
            "material_id": material_id,
            "metric_name": m["metric_name"],
            "metric_value": m["metric_value"],
            "unit": m["unit"],
            "conditions": json.dumps(conditions),
            "source": "synthetic",
            "source_id": f"synthetic:{material_id}:{m['metric_name']}",
            "metadata": json.dumps({"formula": formula, "note": "synthetic_for_testing"}),
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic activity metrics CSV.")
    parser.add_argument("--db", dest="db_path", default="data/imcs.db", help="SQLite DB path")
    parser.add_argument("--rows", type=int, default=30, help="Number of materials to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        default="data/experimental/synthetic_activity_metrics.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    cfg = TheoryDataConfig()
    allowed = set(cfg.elements)

    materials = _load_materials(args.db_path)
    filtered = [m for m in materials if _allowed_formula(m.get("formula"), allowed)]

    if not filtered:
        raise SystemExit("No materials match TheoryDataConfig.elements. Download theory data first.")

    sample_size = min(args.rows, len(filtered))
    sampled = random.sample(filtered, sample_size)

    records: List[Dict[str, Any]] = []
    for mat in sampled:
        records.extend(_make_metric_records(mat["material_id"], mat["formula"]))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    import pandas as pd
    pd.DataFrame(records).to_csv(args.output, index=False)
    print(f"Generated {len(records)} rows -> {args.output}")


if __name__ == "__main__":
    main()
