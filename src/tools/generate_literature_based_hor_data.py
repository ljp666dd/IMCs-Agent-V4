import argparse
import csv
import json
import math
import os
import random
import sys
from typing import Dict, Any, List, Optional

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.services.db.database import DatabaseService
from src.agents.core.theory_agent import TheoryDataConfig


DEFAULT_CONDITIONS = {
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


def _allowed_formula(formula: str, allowed: set) -> bool:
    if not formula:
        return False
    try:
        from pymatgen.core import Composition
        elements = {el.symbol for el in Composition(formula).elements}
        return elements.issubset(allowed)
    except Exception:
        return False


def _parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _sample_range(min_val: Optional[float], max_val: Optional[float]) -> Optional[float]:
    if min_val is None and max_val is None:
        return None
    if max_val is None:
        return min_val
    if min_val is None:
        return max_val
    if max_val < min_val:
        min_val, max_val = max_val, min_val
    if min_val == max_val:
        return min_val
    return random.uniform(min_val, max_val)


def _tafel_from_activity(j0: float) -> float:
    if j0 is None or j0 <= 0:
        return random.uniform(80, 120)
    if j0 >= 1.0:
        return random.uniform(30, 70)
    if j0 >= 0.1:
        return random.uniform(50, 90)
    return random.uniform(80, 120)


def _overpotential_from_tafel(j0: float, tafel_mV: float, j_target: float = 10.0) -> Optional[float]:
    if j0 is None or j0 <= 0 or tafel_mV is None:
        return None
    b = tafel_mV / 1000.0
    return float(b * math.log10(j_target / j0))


def _koutecky_levich_current(potential: float, j0: float, tafel_mV: float, j_lim: float) -> float:
    if j0 is None or j0 <= 0:
        j0 = 0.05
    if tafel_mV is None or tafel_mV <= 0:
        tafel_mV = 80
    b = tafel_mV / 1000.0
    jk = j0 * (10 ** (potential / b))
    return 1.0 / (1.0 / jk + 1.0 / j_lim)


def main():
    parser = argparse.ArgumentParser(description="Generate literature-based HOR synthetic data (metrics + LSV curves).")
    parser.add_argument("--seed", default="data/experimental/literature_hor_seed.csv", help="Seed CSV")
    parser.add_argument("--output-dir", default="data/experimental/literature_rde_lsv", help="LSV output directory")
    parser.add_argument("--metrics-csv", default="data/experimental/literature_activity_metrics.csv", help="Output metrics CSV")
    parser.add_argument("--seeded", action="store_true", help="Use fixed seed values (no random sampling)")
    parser.add_argument("--seed-value", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed_value)

    if not os.path.exists(args.seed):
        raise SystemExit(f"Seed file not found: {args.seed}")

    db = DatabaseService()
    allowed = set(TheoryDataConfig().elements)

    os.makedirs(args.output_dir, exist_ok=True)

    metrics_rows: List[Dict[str, Any]] = []
    manifest = {
        "conditions_default": DEFAULT_CONDITIONS,
        "seed_file": args.seed,
        "files": [],
    }

    with open(args.seed, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("material_label") or row.get("formula")
            formula = row.get("formula") or label
            if not _allowed_formula(formula, allowed):
                continue

            material_id = f"lit:{label}"
            db.save_material(material_id=material_id, formula=formula, energy=None, cif_path=None)

            j0_min = _parse_float(row.get("j0_specific_min_mA_cm2"))
            j0_max = _parse_float(row.get("j0_specific_max_mA_cm2"))
            j0_mass_min = _parse_float(row.get("j0_mass_min_A_g"))
            j0_mass_max = _parse_float(row.get("j0_mass_max_A_g"))
            spec_50 = _parse_float(row.get("specific_activity_50mV_mA_cm2"))
            mass_50 = _parse_float(row.get("mass_activity_50mV_A_mg"))

            j0_specific = _sample_range(j0_min, j0_max) if not args.seeded else (j0_min or j0_max)
            j0_mass = _sample_range(j0_mass_min, j0_mass_max) if not args.seeded else (j0_mass_min or j0_mass_max)

            conditions = DEFAULT_CONDITIONS.copy()
            if row.get("conditions_json"):
                try:
                    conditions.update(json.loads(row["conditions_json"]))
                except Exception:
                    pass

            # Derive spec_50 from mass_50 if missing
            if spec_50 is None and mass_50 is not None:
                spec_50 = mass_50 * conditions["loading_mg_cm2"] * conditions["precious_metal_loading_fraction"] * 1000.0

            # Derive j0 from spec_50 if missing
            if j0_specific is None and spec_50 is not None:
                tafel_guess = 70.0
                b = tafel_guess / 1000.0
                j0_specific = spec_50 / (10 ** (0.05 / b))

            tafel_mV = _tafel_from_activity(j0_specific)
            overpotential = _overpotential_from_tafel(j0_specific, tafel_mV, j_target=10.0)

            source_id = row.get("source_doi") or "literature"
            source_note = row.get("source_note") or ""
            source_title = row.get("source_title") or ""
            source_url = row.get("source_url") or ""
            source_year = row.get("source_year") or ""

            def add_metric(name, value, unit, source, meta_extra=None):
                if value is None:
                    return
                meta = {
                    "material_label": label,
                    "formula": formula,
                    "source_note": source_note,
                    "source_title": source_title,
                    "source_url": source_url,
                    "source_year": source_year,
                }
                if meta_extra:
                    meta.update(meta_extra)
                metrics_rows.append({
                    "material_id": material_id,
                    "metric_name": name,
                    "metric_value": value,
                    "unit": unit,
                    "conditions": json.dumps(conditions, ensure_ascii=False),
                    "source": source,
                    "source_id": source_id,
                    "metadata": json.dumps(meta, ensure_ascii=False),
                })

            add_metric("exchange_current_density", j0_specific, "mA/cm2", "literature")
            add_metric("exchange_current_density_mass", j0_mass, "A/g", "literature")
            add_metric("Jk_ref", spec_50, "mA/cm2", "literature", {"reference_potential_V": 0.05})
            add_metric("mass_activity", mass_50, "A/mg", "literature", {"reference_potential_V": 0.05})
            add_metric("tafel_slope", tafel_mV, "mV/dec", "synthetic_from_literature")
            add_metric("overpotential_10mA", overpotential, "V", "synthetic_from_literature")

            rpm_list = [2500, 1600, 900, 400]
            potentials = [round(i * 0.005, 4) for i in range(0, 121)]
            j_lim_1600 = max(4.0, min(20.0, (spec_50 or 5.0) * 1.2))

            for rpm in rpm_list:
                j_lim = j_lim_1600 * math.sqrt(rpm / 1600.0)
                currents = []
                for v in potentials:
                    j = _koutecky_levich_current(v, j0_specific, tafel_mV, j_lim)
                    noise = (0.02 * j_lim) * (2 * (random.random() - 0.5))
                    currents.append(j + noise)

                filename = f"{label}_LSV_{rpm}rpm.csv"
                path = os.path.join(args.output_dir, filename)
                manifest["files"].append(path)

                rows_out = []
                for v, j in zip(potentials, currents):
                    rows_out.append({
                        "Potential_V": v,
                        "Current_mAcm2": round(j, 6),
                        "RPM": rpm,
                        "pH": conditions.get("pH", 13),
                        "Temperature_K": conditions.get("temperature_K", 298),
                        "Electrolyte": conditions.get("electrolyte", "0.1M KOH"),
                    })
                import pandas as pd
                pd.DataFrame(rows_out).to_csv(path, index=False)

    os.makedirs(os.path.dirname(args.metrics_csv), exist_ok=True)
    with open(args.metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["material_id", "metric_name", "metric_value", "unit", "conditions", "source", "source_id", "metadata"],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Wrote metrics CSV: {args.metrics_csv}")
    print(f"Generated {len(manifest['files'])} LSV files -> {args.output_dir}")


if __name__ == "__main__":
    main()
