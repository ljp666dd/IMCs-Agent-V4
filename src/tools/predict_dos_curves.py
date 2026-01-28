import argparse
import json
import os
import sys
from typing import Dict

import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.services.db.database import DatabaseService
from src.services.chemistry.descriptors import StructureFeaturizer
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


def _merge_dos_meta(db: DatabaseService, material_id: str, updates: Dict) -> None:
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
    parser = argparse.ArgumentParser(description="Predict DOS curves from structure features using trained model.")
    parser.add_argument("--model", default="data/ml_agent/dos_curve_model.pkl")
    parser.add_argument("--out-curves", default="data/theory/dos_curve_predictions")
    parser.add_argument("--out-plots", default="data/theory/dos_curve_pred_plots")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only-missing", action="store_true", help="Only predict for materials missing dos_data")
    parser.add_argument("--update-db", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"Model not found: {args.model}")

    payload = joblib.load(args.model)
    model = payload["model"]
    pca = payload["pca"]
    energy = payload["energy_grid"]
    channel = payload.get("channel", "total")

    db = DatabaseService()
    featurizer = StructureFeaturizer()
    cfg = TheoryDataConfig()
    allowed = set(cfg.elements)

    os.makedirs(args.out_curves, exist_ok=True)
    os.makedirs(args.out_plots, exist_ok=True)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise SystemExit(f"matplotlib required for plots: {e}")

    with db._get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT material_id, formula, cif_path, dos_data FROM materials")
        rows = cur.fetchall()

    count = 0
    for mid, formula, cif_path, dos_data in rows:
        if args.limit and count >= args.limit:
            break
        if not mid or not cif_path or not os.path.exists(cif_path):
            continue
        if not _allowed_formula(formula, allowed):
            continue
        if args.only_missing and dos_data:
            continue

        feats = featurizer.extract(cif_path)
        if feats is None:
            continue
        z = model.predict(np.array([feats]))
        curve = pca.inverse_transform(z)[0]

        csv_path = os.path.join(args.out_curves, f"{mid}_{channel}_pred.csv")
        with open(csv_path, "w") as f:
            f.write("energy,dos\n")
            for e, d in zip(energy, curve):
                f.write(f"{e},{d}\n")

        plot_path = None
        if args.plot:
            plt.figure(figsize=(5.5, 3.8))
            plt.plot(energy, curve, label=f"{channel}", color="black", linewidth=1.2)
            plt.axvline(0.0, color="#888", linestyle="--", linewidth=0.8)
            plt.xlabel("E - Ef (eV)")
            plt.ylabel("DOS")
            plt.title(f"{mid} ({formula})")
            plt.legend(fontsize=7)
            plt.tight_layout()
            plot_path = os.path.join(args.out_plots, f"{mid}_{channel}_pred.png")
            plt.savefig(plot_path, dpi=160)
            plt.close()

        if args.update_db:
            updates = {
                "dos_curve_pred_path": csv_path,
                "dos_curve_pred_channel": channel,
            }
            if plot_path:
                updates["dos_plot_pred_path"] = plot_path
            _merge_dos_meta(db, mid, updates)

        count += 1

    print(f"Predicted DOS curves: {count}")


if __name__ == "__main__":
    main()
