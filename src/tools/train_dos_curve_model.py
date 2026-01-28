import argparse
import json
import os
import sys
from typing import Dict, Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
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


def _load_pdos(pdos_path: str, channel: str) -> Dict[str, Dict[str, Any]]:
    with open(pdos_path, 'r') as f:
        data = json.load(f)
    out = {}
    for mid, payload in data.items():
        if not isinstance(payload, dict):
            continue
        if "energies" not in payload:
            continue
        key = channel
        if channel == "total":
            key = "total_dos"
        if key not in payload:
            continue
        out[mid] = {
            "energies": payload.get("energies"),
            "dos": payload.get(key),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Train model to predict orbital DOS curves from structure features.")
    parser.add_argument("--pdos", default="data/theory/orbital_pdos.json")
    parser.add_argument("--channel", default="total", choices=["total", "s", "p", "d"], help="DOS channel")
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--model-out", default="data/ml_agent/dos_curve_model.pkl")
    parser.add_argument("--report", default="data/ml_agent/dos_curve_report.json")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    if not os.path.exists(args.pdos):
        raise SystemExit(f"PDOS file not found: {args.pdos}")

    cfg = TheoryDataConfig()
    allowed = set(cfg.elements)
    db = DatabaseService()
    featurizer = StructureFeaturizer()

    pdos = _load_pdos(args.pdos, args.channel)
    if not pdos:
        raise SystemExit("No PDOS entries found for requested channel.")

    # Build lookup for CIF paths
    with db._get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT material_id, formula, cif_path FROM materials")
        db_rows = {row[0]: {"formula": row[1], "cif_path": row[2]} for row in cur.fetchall() if row[0]}

    # Pick base energy grid
    base_mid = next(iter(pdos.keys()))
    base_e = np.array(pdos[base_mid]["energies"], dtype=float)

    X_list = []
    Y_list = []
    ids = []

    for mid, item in pdos.items():
        info = db_rows.get(mid)
        if not info:
            continue
        if not _allowed_formula(info.get("formula"), allowed):
            continue
        cif_path = info.get("cif_path")
        if not cif_path or not os.path.exists(cif_path):
            continue
        dos = np.array(item.get("dos") or [], dtype=float)
        if dos.size == 0:
            continue
        energies = np.array(item.get("energies"), dtype=float)
        if energies.shape != base_e.shape:
            # interpolate to base grid
            dos = np.interp(base_e, energies, dos)
        feats = featurizer.extract(cif_path)
        if feats is None:
            continue
        X_list.append(feats)
        Y_list.append(dos)
        ids.append(mid)

    if not X_list:
        raise SystemExit("No samples available for DOS curve training.")

    X = np.array(X_list)
    Y = np.array(Y_list)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_size, random_state=42)

    pca = PCA(n_components=min(args.n_components, Y_train.shape[0], Y_train.shape[1]))
    Y_train_z = pca.fit_transform(Y_train)
    Y_test_z = pca.transform(Y_test)

    model = MultiOutputRegressor(Ridge(alpha=1.0))
    model.fit(X_train, Y_train_z)

    Y_pred_z = model.predict(X_test)
    Y_pred = pca.inverse_transform(Y_pred_z)

    # Metrics
    comp_r2 = r2_score(Y_test_z, Y_pred_z, multioutput='variance_weighted')
    curve_r2 = r2_score(Y_test.flatten(), Y_pred.flatten())

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump({
        "model": model,
        "pca": pca,
        "energy_grid": base_e,
        "channel": args.channel,
        "feature_names": featurizer.feature_names,
    }, args.model_out)

    report = {
        "samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_points": int(Y.shape[1]),
        "n_components": int(pca.n_components_),
        "component_r2": float(comp_r2),
        "curve_r2": float(curve_r2),
        "channel": args.channel,
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved model:", args.model_out)
    print("Report:", report)


if __name__ == "__main__":
    main()
