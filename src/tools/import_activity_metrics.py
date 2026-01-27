import argparse
import json
import os
from typing import Any, Dict, Optional

import pandas as pd

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(BASE_DIR)

from src.services.db.database import DatabaseService
from src.agents.core.theory_agent import TheoryDataConfig


def _lookup_formula(db: DatabaseService, material_id: str) -> Optional[str]:
    if not material_id:
        return None
    try:
        with db._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT formula FROM materials WHERE material_id = ?", (material_id,))
            row = cursor.fetchone()
            return row[0] if row else None
    except Exception:
        return None


def _allowed_formula(formula: Optional[str], allowed: set) -> bool:
    if not formula:
        return False
    try:
        from pymatgen.core import Composition
        elements = {el.symbol for el in Composition(formula).elements}
        return elements.issubset(allowed)
    except Exception:
        return False


def _to_none(value: Any) -> Optional[Any]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _parse_json(value: Any) -> Optional[Dict[str, Any]]:
    value = _to_none(value)
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return None


def _to_float(value: Any) -> Optional[float]:
    value = _to_none(value)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Import activity metrics into IMCs DB.")
    parser.add_argument("csv_path", help="Path to CSV file")
    parser.add_argument("--db", dest="db_path", default="data/imcs.db", help="SQLite DB path")
    parser.add_argument("--source", dest="source", default=None, help="Override source for all rows")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        raise SystemExit(f"CSV not found: {args.csv_path}")

    df = pd.read_csv(args.csv_path)

    required = {"metric_name", "metric_value"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    db = DatabaseService(db_path=args.db_path)
    allowed = set(TheoryDataConfig().elements)
    total = 0
    inserted = 0
    skipped = 0

    for _, row in df.iterrows():
        total += 1
        metric_name = _to_none(row.get("metric_name"))
        metric_value = _to_float(row.get("metric_value"))
        if metric_name is None or metric_value is None:
            skipped += 1
            continue

        material_id = _to_none(row.get("material_id"))
        if material_id:
            formula = _lookup_formula(db, material_id)
            if not _allowed_formula(formula, allowed):
                skipped += 1
                continue
        unit = _to_none(row.get("unit"))
        conditions = _parse_json(row.get("conditions"))
        metadata = _parse_json(row.get("metadata"))
        source = args.source or _to_none(row.get("source")) or "literature"
        source_id = _to_none(row.get("source_id"))

        db.save_activity_metric(
            material_id=material_id,
            metric_name=str(metric_name),
            metric_value=metric_value,
            unit=unit,
            conditions=conditions,
            source=source,
            source_id=source_id,
            metadata=metadata,
        )
        inserted += 1

    print(f"Imported {inserted}/{total} rows. Skipped {skipped}.")


if __name__ == "__main__":
    main()
