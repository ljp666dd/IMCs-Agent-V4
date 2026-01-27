import argparse
import json
import os
import sys
from collections import defaultdict

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.agents.core.experiment_agent import ExperimentDataAgent


def _group_by_formula(files):
    groups = defaultdict(list)
    for path in files:
        name = os.path.basename(path)
        formula = name.split("_")[0].split("-")[0]
        if formula:
            groups[formula].append(path)
    return groups


def main():
    parser = argparse.ArgumentParser(description="Analyze RDE LSV series (multi-RPM) and store metrics.")
    parser.add_argument("--dir", dest="data_dir", default="data/experimental/rde_lsv", help="Directory with LSV files")
    parser.add_argument("--reference-potential", type=float, default=0.2, help="Reference potential for MA/Jk")
    parser.add_argument("--loading", type=float, default=0.25, help="Total catalyst loading (mg/cm2)")
    parser.add_argument("--precious-fraction", type=float, default=0.20, help="Precious metal fraction")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise SystemExit(f"Directory not found: {args.data_dir}")

    # Load optional manifest conditions
    manifest_path = os.path.join(args.data_dir, "manifest.json")
    conditions = None
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        conditions = manifest.get("conditions")

    files = [
        os.path.join(args.data_dir, f)
        for f in os.listdir(args.data_dir)
        if f.lower().endswith(".csv")
    ]
    groups = _group_by_formula(files)
    agent = ExperimentDataAgent()

    for formula, paths in groups.items():
        if len(paths) < 2:
            continue
        result = agent.analyze_rde_series(
            paths,
            sample_id=formula,
            reference_potential=args.reference_potential,
            loading_mg_cm2=args.loading,
            precious_fraction=args.precious_fraction,
            conditions=conditions,
        )
        print(f"[OK] {formula}: J0={result.exchange_current_density} MA={result.mass_activity}")


if __name__ == "__main__":
    main()
