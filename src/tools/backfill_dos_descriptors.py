import argparse
import json
import os
import sys
from typing import Dict, Any, Iterable, Tuple

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.services.db.database import DatabaseService
from src.services.theory.physics import PhysicsCalc
from src.agents.core.theory_agent import TheoryDataConfig


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _allowed_formula(formula: str, allowed: set) -> bool:
    if not formula:
        return False
    try:
        from pymatgen.core import Composition
        elements = {el.symbol for el in Composition(formula).elements}
        return bool(elements) and elements.issubset(allowed)
    except Exception:
        return False


def _iter_descriptors(data: Any) -> Iterable[Tuple[str, Dict[str, Any], str]]:
    """
    Yield (material_id, descriptors, formula_guess).
    Accepts list of dicts or dict of material_id -> dos_data/descriptor dict.
    """
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            material_id = item.get("material_id")
            if not material_id:
                continue
            formula = item.get("formula")
            yield material_id, item, formula
    elif isinstance(data, dict):
        for material_id, payload in data.items():
            if not material_id or not isinstance(payload, dict):
                continue
            formula = payload.get("formula")
            yield material_id, payload, formula


def _build_descriptors_from_pdos(pdos_data: Dict[str, Any]) -> Dict[str, Any]:
    calc = PhysicsCalc()
    return calc.extract_dos_descriptors(pdos_data) or {}


def _lookup_formula(db: DatabaseService, material_id: str) -> str:
    try:
        with db._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT formula FROM materials WHERE material_id = ?", (material_id,))
            row = cur.fetchone()
            if row:
                return row[0]
    except Exception:
        return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Backfill DOS descriptors into IMCs DB.")
    parser.add_argument("--db", dest="db_path", default="data/imcs.db", help="SQLite DB path")
    parser.add_argument(
        "--descriptors",
        default="data/theory/dos_descriptors_full.json",
        help="Path to DOS descriptors JSON (list or dict)",
    )
    parser.add_argument(
        "--pdos",
        default="data/theory/orbital_pdos.json",
        help="Path to orbital PDOS JSON (used if descriptors missing)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max records to process (0 = all)")
    args = parser.parse_args()

    cfg = TheoryDataConfig()
    allowed = set(cfg.elements)
    db = DatabaseService(db_path=args.db_path)

    processed = 0
    updated = 0
    skipped = 0

    if os.path.exists(args.descriptors):
        data = _load_json(args.descriptors)
        for material_id, payload, formula in _iter_descriptors(data):
            if args.limit and processed >= args.limit:
                break
            processed += 1
            if not formula:
                formula = _lookup_formula(db, material_id)
            if not _allowed_formula(formula, allowed):
                skipped += 1
                continue
            # Remove non-descriptor fields if present
            descriptors = {
                k: v for k, v in payload.items()
                if k not in {"material_id", "formula", "formation_energy", "dos_source"}
            }
            if not descriptors:
                skipped += 1
                continue
            db.update_material_dos(material_id, descriptors)
            updated += 1
    elif os.path.exists(args.pdos):
        data = _load_json(args.pdos)
        for material_id, payload, formula in _iter_descriptors(data):
            if args.limit and processed >= args.limit:
                break
            processed += 1
            if not formula:
                formula = _lookup_formula(db, material_id)
            if not _allowed_formula(formula, allowed):
                skipped += 1
                continue
            descriptors = _build_descriptors_from_pdos(payload)
            if not descriptors:
                skipped += 1
                continue
            db.update_material_dos(material_id, descriptors)
            updated += 1
    else:
        raise SystemExit("No DOS source JSON found. Provide --descriptors or --pdos.")

    print(f"Processed: {processed} | Updated: {updated} | Skipped: {skipped}")


if __name__ == "__main__":
    main()
