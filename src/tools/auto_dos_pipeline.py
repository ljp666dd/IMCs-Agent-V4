import argparse
import json
import os
import sqlite3
import subprocess
import sys
from typing import Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.agents.core.theory_agent import TheoryDataConfig


def _allowed_formula(formula: str, allowed: set) -> bool:
    if not formula:
        return False
    try:
        from pymatgen.core import Composition
        elements = {el.symbol for el in Composition(formula).elements}
        return elements.issubset(allowed)
    except Exception:
        return False


def _coverage(db_path: str) -> Dict[str, float]:
    cfg = TheoryDataConfig()
    allowed = set(cfg.elements)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT material_id, formula, dos_data FROM materials")
    rows = cur.fetchall()
    conn.close()

    total = 0
    with_dos = 0
    with_curve = 0
    with_plot = 0

    for mid, formula, dos_data in rows:
        if not _allowed_formula(formula, allowed):
            continue
        total += 1
        if dos_data:
            with_dos += 1
            try:
                meta = json.loads(dos_data) if isinstance(dos_data, str) else (dos_data or {})
            except Exception:
                meta = {}
            if isinstance(meta, dict):
                if meta.get("dos_curve_path"):
                    with_curve += 1
                if meta.get("dos_plot_path"):
                    with_plot += 1

    def ratio(x):
        return round(x / total, 4) if total else 0.0

    return {
        "total_materials": total,
        "dos_descriptor_count": with_dos,
        "dos_descriptor_coverage": ratio(with_dos),
        "dos_curve_count": with_curve,
        "dos_curve_coverage": ratio(with_curve),
        "dos_plot_count": with_plot,
        "dos_plot_coverage": ratio(with_plot),
    }


def main():
    parser = argparse.ArgumentParser(description="Automate DOS fill + curve render pipeline.")
    parser.add_argument("--db", default="data/imcs.db")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--energy-min", type=float, default=-15.0)
    parser.add_argument("--energy-max", type=float, default=10.0)
    parser.add_argument("--pdos", default="data/theory/orbital_pdos.json")
    parser.add_argument("--plots-dir", default="data/theory/orbital_dos_plots")
    parser.add_argument("--curves-dir", default="data/theory/orbital_dos_curves")
    parser.add_argument("--report", default="docs/dos_coverage_report.json")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--merge-pdos", action="store_true")
    parser.add_argument("--update-db", action="store_true")
    args = parser.parse_args()

    before = _coverage(args.db)
    print("Coverage before:", before)

    if not args.skip_download:
        script = os.path.join(BASE_DIR, "src", "tools", "fill_missing_dos_from_mp.py")
        cmd = [
            sys.executable, script,
            "--limit", str(args.limit),
            "--batch-size", str(args.batch_size),
            "--energy-min", str(args.energy_min),
            "--energy-max", str(args.energy_max),
        ]
        if args.merge_pdos:
            cmd.append("--merge-pdos")
        if args.pdos:
            cmd.extend(["--output-pdos", args.pdos])
        print("Running:", " ".join(cmd))
        ret = subprocess.run(cmd, check=False)
        if ret.returncode != 0:
            print("DOS download failed.")
            return ret.returncode

    if not args.skip_render:
        script = os.path.join(BASE_DIR, "src", "tools", "render_orbital_dos_outputs.py")
        cmd = [
            sys.executable, script,
            "--pdos", args.pdos,
            "--plots-dir", args.plots_dir,
            "--curves-dir", args.curves_dir,
        ]
        if args.update_db:
            cmd.append("--update-db")
        print("Running:", " ".join(cmd))
        ret = subprocess.run(cmd, check=False)
        if ret.returncode != 0:
            print("DOS render failed.")
            return ret.returncode

    after = _coverage(args.db)
    print("Coverage after:", after)

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump({"before": before, "after": after}, f, indent=2)
    print("Report written:", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
