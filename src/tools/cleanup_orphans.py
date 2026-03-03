import argparse
import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Tuple

from src.config.config import config
from src.services.db.database import DatabaseService


def _backup_sqlite(db_path: str, backup_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(backup_path)) or ".", exist_ok=True)
    src = sqlite3.connect(db_path)
    try:
        try:
            src.execute("PRAGMA wal_checkpoint(FULL);")
        except Exception:
            pass
        dst = sqlite3.connect(backup_path)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def _select_orphan_rows(conn: sqlite3.Connection, table: str) -> List[Tuple[int, str, str]]:
    """
    Return list of (id, material_id, metadata_json) for rows whose material_id
    is non-null but missing in materials table.
    """
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT t.id, t.material_id, t.metadata
        FROM {table} t
        LEFT JOIN materials m ON t.material_id = m.material_id
        WHERE t.material_id IS NOT NULL AND m.material_id IS NULL
        """
    )
    out: List[Tuple[int, str, str]] = []
    for row in cur.fetchall():
        rid = int(row[0])
        mid = (str(row[1]) if row[1] is not None else "").strip()
        meta = row[2]
        out.append((rid, mid, meta))
    return out


def _update_orphan_to_null(
    conn: sqlite3.Connection,
    table: str,
    rows: List[Tuple[int, str, str]],
) -> int:
    """
    For adsorption_energies/activity_metrics: set material_id=NULL and store the
    orphan material_id into metadata (best-effort).
    """
    cur = conn.cursor()
    updated = 0
    for rid, mid, meta_json in rows:
        meta: Dict[str, Any] = {}
        if meta_json:
            try:
                meta = json.loads(meta_json) if isinstance(meta_json, str) else dict(meta_json)
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}
        meta.setdefault("orphan_material_id", mid)
        meta.setdefault("orphan_reason", "material_missing")
        new_meta_json = json.dumps(meta, ensure_ascii=False)
        cur.execute(
            f"UPDATE {table} SET material_id = NULL, metadata = ? WHERE id = ?",
            (new_meta_json, int(rid)),
        )
        updated += int(cur.rowcount or 0)
    return updated


def _delete_orphan_evidence(conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        DELETE FROM evidence
        WHERE NOT EXISTS (
            SELECT 1 FROM materials m WHERE m.material_id = evidence.material_id
        )
        """
    )
    return int(cur.rowcount or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="IMCs: cleanup legacy orphan rows (dry-run by default).")
    parser.add_argument("--db", default=config.DB_PATH, help="SQLite DB path (default: data/imcs.db)")
    parser.add_argument("--apply", action="store_true", help="Apply changes to DB (otherwise dry-run)")
    parser.add_argument("--no-backup", action="store_true", help="Skip sqlite backup even when --apply is set")
    parser.add_argument("--sample", type=int, default=10, help="Show up to N orphan material_ids per table")
    args = parser.parse_args()

    db_path = args.db
    db = DatabaseService(db_path=db_path)
    before = db.get_data_integrity_stats()
    print("Integrity before:")
    print(json.dumps(before, ensure_ascii=False, indent=2))

    with db._get_conn() as conn:
        conn.row_factory = sqlite3.Row

        # Gather orphan rows (non-destructive)
        adsorption_orphans = _select_orphan_rows(conn, "adsorption_energies")
        activity_orphans = _select_orphan_rows(conn, "activity_metrics")

        ev_cur = conn.cursor()
        ev_cur.execute(
            """
            SELECT e.material_id, COUNT(*) AS cnt
            FROM evidence e
            LEFT JOIN materials m ON e.material_id = m.material_id
            WHERE m.material_id IS NULL
            GROUP BY e.material_id
            ORDER BY cnt DESC
            """
        )
        evidence_orphan_ids = [r[0] for r in ev_cur.fetchall()]

        def _print_samples(title: str, ids: List[str]) -> None:
            sample_n = max(0, int(args.sample or 0))
            if not sample_n:
                return
            uniq: List[str] = []
            seen = set()
            for item in ids:
                s = (str(item) if item is not None else "").strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                uniq.append(s)
                if len(uniq) >= sample_n:
                    break
            if uniq:
                print(f"{title} sample material_ids ({len(uniq)} shown): {uniq}")

        _print_samples("Evidence orphan", evidence_orphan_ids)
        _print_samples("Adsorption orphan", [mid for _, mid, _ in adsorption_orphans])
        _print_samples("Activity orphan", [mid for _, mid, _ in activity_orphans])

    if not args.apply:
        print("Dry-run only. Re-run with --apply to modify the DB.")
        return

    if not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{db_path}.bak_{stamp}"
        print(f"Creating backup: {backup_path}")
        _backup_sqlite(db_path, backup_path)

    with db._get_conn() as conn:
        conn.execute("BEGIN;")
        try:
            deleted_evidence = _delete_orphan_evidence(conn)
            adsorption_orphans = _select_orphan_rows(conn, "adsorption_energies")
            activity_orphans = _select_orphan_rows(conn, "activity_metrics")
            updated_adsorption = _update_orphan_to_null(conn, "adsorption_energies", adsorption_orphans)
            updated_activity = _update_orphan_to_null(conn, "activity_metrics", activity_orphans)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    after = db.get_data_integrity_stats()
    print("Changes applied:")
    print(json.dumps(
        {
            "deleted_orphan_evidence_rows": deleted_evidence,
            "updated_adsorption_orphan_rows": updated_adsorption,
            "updated_activity_orphan_rows": updated_activity,
        },
        ensure_ascii=False,
        indent=2,
    ))
    print("Integrity after:")
    print(json.dumps(after, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
