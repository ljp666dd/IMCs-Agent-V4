import argparse
import sqlite3

# Evidence coverage stats by source type

def main(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # total materials
    cur.execute("SELECT COUNT(*) FROM materials")
    total = cur.fetchone()[0] or 0

    # evidence by type
    cur.execute("SELECT source_type, COUNT(DISTINCT material_id) FROM evidence GROUP BY source_type")
    rows = cur.fetchall()

    print(f"Total materials: {total}")
    for stype, cnt in rows:
        ratio = (cnt / total) if total else 0
        print(f"{stype}: {cnt} materials ({ratio:.2%})")

    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='data/imcs.db', help='Path to SQLite DB')
    args = parser.parse_args()
    main(args.db)
