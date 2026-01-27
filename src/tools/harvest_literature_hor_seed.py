import argparse
import csv
import os
import subprocess
import sys
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.agents.core.literature_agent import LiteratureAgent


SEED_FIELDS = [
    "material_label",
    "formula",
    "specific_activity_50mV_mA_cm2",
    "mass_activity_50mV_A_mg",
    "j0_specific_min_mA_cm2",
    "j0_specific_max_mA_cm2",
    "j0_mass_min_A_g",
    "j0_mass_max_A_g",
    "source_doi",
    "source_note",
    "source_title",
    "source_url",
    "source_year",
    "source_abstract",
    "conditions_json",
]


def _read_existing(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_seed(path: str, rows: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SEED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _dedupe(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    unique = []
    for row in rows:
        key = (row.get("material_label"), row.get("source_doi"), row.get("source_note"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="Harvest HOR literature and build seed CSV.")
    parser.add_argument("--query", required=True, help="Search query (e.g., HOR ordered alloy catalyst)")
    parser.add_argument("--limit", type=int, default=15, help="Max papers")
    parser.add_argument("--max-pdfs", type=int, default=5, help="Max open-access PDFs to download")
    parser.add_argument("--min-elements", type=int, default=2, help="Minimum unique elements in formula")
    parser.add_argument("--seed-out", default="data/experimental/literature_hor_seed.csv", help="Seed CSV output")
    parser.add_argument("--persist", action="store_true", help="Persist literature evidence into DB")
    parser.add_argument("--generate", action="store_true", help="Generate metrics + LSV after harvesting")
    parser.add_argument("--metrics-csv", default="data/experimental/literature_activity_metrics.csv", help="Metrics CSV output")
    parser.add_argument("--lsv-dir", default="data/experimental/literature_rde_lsv", help="LSV output dir")
    parser.add_argument("--seeded", action="store_true", help="Use fixed seed values in generation")
    parser.add_argument("--import-metrics", action="store_true", help="Import generated metrics into DB")
    parser.add_argument("--analyze-lsv", action="store_true", help="Analyze generated LSV curves into DB")
    parser.add_argument("--run-all", action="store_true", help="Run generate + import + analyze in one pipeline")
    args = parser.parse_args()

    agent = LiteratureAgent()
    rows = agent.harvest_hor_seed(
        query=args.query,
        limit=args.limit,
        max_pdfs=args.max_pdfs,
        min_elements=args.min_elements,
        persist=args.persist,
    )

    existing = _read_existing(args.seed_out)
    merged = _dedupe(existing + rows)
    _write_seed(args.seed_out, merged)
    print(f"Harvested {len(rows)} rows, seed size: {len(merged)} -> {args.seed_out}")

    if args.run_all:
        args.generate = True
        args.import_metrics = True
        args.analyze_lsv = True

    if args.generate:
        script = os.path.join(BASE_DIR, "src", "tools", "generate_literature_based_hor_data.py")
        cmd = [sys.executable, script, "--seed", args.seed_out, "--metrics-csv", args.metrics_csv, "--output-dir", args.lsv_dir]
        if args.seeded:
            cmd.append("--seeded")
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            return result.returncode

    if args.import_metrics:
        script = os.path.join(BASE_DIR, "src", "tools", "import_activity_metrics.py")
        cmd = [sys.executable, script, args.metrics_csv]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            return result.returncode

    if args.analyze_lsv:
        script = os.path.join(BASE_DIR, "src", "tools", "analyze_rde_series.py")
        cmd = [sys.executable, script, "--dir", args.lsv_dir]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
