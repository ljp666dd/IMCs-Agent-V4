import argparse
import csv
import json
import os
import sys
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.services.db.database import DatabaseService
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


def _get_formula(db: DatabaseService, material_id: str) -> str:
    with db._get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT formula FROM materials WHERE material_id = ?", (material_id,))
        row = cur.fetchone()
        return row[0] if row else None


def _merge_dos_meta(db: DatabaseService, material_id: str, updates: Dict[str, Any]) -> None:
    with db._get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT dos_data FROM materials WHERE material_id = ?", (material_id,))
        row = cur.fetchone()
        base = {}
        if row and row[0]:
            try:
                base = json.loads(row[0]) if isinstance(row[0], str) else (row[0] or {})
            except Exception:
                base = {}
        if not isinstance(base, dict):
            base = {}
        base.update(updates)
        cur.execute("UPDATE materials SET dos_data = ? WHERE material_id = ?", (json.dumps(base), material_id))
        conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Render orbital DOS curves and plots from PDOS JSON.")
    parser.add_argument("--pdos", default="data/theory/orbital_pdos.json", help="PDOS JSON file")
    parser.add_argument("--plots-dir", default="data/theory/orbital_dos_plots")
    parser.add_argument("--curves-dir", default="data/theory/orbital_dos_curves")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--update-db", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.pdos):
        raise SystemExit(f"PDOS file not found: {args.pdos}")

    with open(args.pdos, "r") as f:
        data = json.load(f)

    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.curves_dir, exist_ok=True)

    cfg = TheoryDataConfig()
    allowed = set(cfg.elements)
    db = DatabaseService()

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"matplotlib required: {e}")

    count = 0
    for material_id, payload in data.items():
        if args.limit and count >= args.limit:
            break
        if not isinstance(payload, dict):
            continue
        formula = _get_formula(db, material_id)
        if not _allowed_formula(formula, allowed):
            continue

        energies = payload.get("energies") or []
        s_dos = payload.get("s_dos") or []
        p_dos = payload.get("p_dos") or []
        d_dos = payload.get("d_dos") or []
        total = payload.get("total_dos") or []
        if not energies or not total:
            continue

        # Save curve CSV
        csv_path = os.path.join(args.curves_dir, f"{material_id}_dos_curve.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["energy", "s_dos", "p_dos", "d_dos", "total_dos"])
            for i in range(len(energies)):
                writer.writerow([
                    energies[i],
                    s_dos[i] if i < len(s_dos) else "",
                    p_dos[i] if i < len(p_dos) else "",
                    d_dos[i] if i < len(d_dos) else "",
                    total[i] if i < len(total) else "",
                ])

        # Plot
        plt.figure(figsize=(5.5, 3.8))
        plt.plot(energies, total, label="total", color="black", linewidth=1.2)
        if s_dos:
            plt.plot(energies, s_dos, label="s", linewidth=0.8)
        if p_dos:
            plt.plot(energies, p_dos, label="p", linewidth=0.8)
        if d_dos:
            plt.plot(energies, d_dos, label="d", linewidth=0.8)
        plt.axvline(0.0, color="#888", linestyle="--", linewidth=0.8)
        plt.xlabel("E - Ef (eV)")
        plt.ylabel("DOS")
        plt.title(f"{material_id} ({formula})")
        plt.legend(fontsize=7)
        plt.tight_layout()

        plot_path = os.path.join(args.plots_dir, f"{material_id}_dos.png")
        plt.savefig(plot_path, dpi=160)
        plt.close()

        if args.update_db:
            _merge_dos_meta(db, material_id, {
                "dos_curve_path": csv_path,
                "dos_plot_path": plot_path,
            })

        count += 1

    print(f"Rendered DOS outputs: {count}")


if __name__ == "__main__":
    main()
