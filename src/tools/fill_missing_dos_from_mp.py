import argparse
import os
import sys
from typing import List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.agents.core.theory_agent import TheoryDataAgent, TheoryDataConfig
from src.services.db.database import DatabaseService


def _allowed_formula(formula: str, allowed: set) -> bool:
    if not formula:
        return False
    try:
        from pymatgen.core import Composition
        elements = {el.symbol for el in Composition(formula).elements}
        return elements.issubset(allowed)
    except Exception:
        return False


def _get_missing_ids(db: DatabaseService, allowed: set, limit: int = 0) -> List[str]:
    missing = []
    with db._get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT material_id, formula, dos_data, cif_path FROM materials")
        for mid, formula, dos_data, cif_path in cur.fetchall():
            if not mid or not cif_path:
                continue
            if dos_data:
                continue
            if not _allowed_formula(formula, allowed):
                continue
            missing.append(mid)
            if limit and len(missing) >= limit:
                break
    return missing


def main():
    parser = argparse.ArgumentParser(description="Fill missing DOS descriptors via Materials Project.")
    parser.add_argument("--limit", type=int, default=0, help="Max number of materials to fetch (0=all)")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for MP DOS calls")
    parser.add_argument("--energy-min", type=float, default=-15.0)
    parser.add_argument("--energy-max", type=float, default=10.0)
    parser.add_argument("--merge-pdos", action="store_true", help="Merge PDOS results into orbital_pdos.json")
    parser.add_argument("--output-pdos", default=None, help="Override PDOS output path")
    parser.add_argument("--dry-run", action="store_true", help="Only print missing IDs")
    args = parser.parse_args()

    cfg = TheoryDataConfig()
    allowed = set(cfg.elements)
    db = DatabaseService()
    missing = _get_missing_ids(db, allowed, limit=args.limit)
    print(f"Missing DOS records: {len(missing)}")
    if args.dry_run:
        print("Sample IDs:", missing[:10])
        return

    if not missing:
        return

    agent = TheoryDataAgent(cfg)
    total_updated = 0
    for i in range(0, len(missing), args.batch_size):
        batch = missing[i:i + args.batch_size]
        updated = agent.download_orbital_dos(
            material_ids=batch,
            energy_range=(args.energy_min, args.energy_max),
            output_path=args.output_pdos,
            merge_existing=args.merge_pdos,
        )
        total_updated += updated
        print(f"Batch {i//args.batch_size + 1}: updated {updated}")

    print(f"Total updated: {total_updated}")


if __name__ == "__main__":
    main()
