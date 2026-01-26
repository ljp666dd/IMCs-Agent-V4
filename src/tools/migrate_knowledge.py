import argparse
import json
import os
import sqlite3
import sys
from typing import Dict, Optional

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.config.config import config


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_entity(conn: sqlite3.Connection, entity_type: str, name: str,
                  canonical_id: Optional[str] = None,
                  metadata: Optional[Dict] = None) -> Optional[int]:
    if not name:
        return None
    meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO knowledge_entities (entity_type, name, canonical_id, metadata)
        VALUES (?, ?, ?, ?)
        """,
        (entity_type, name, canonical_id, meta_json),
    )
    if canonical_id or metadata:
        updates = []
        params = []
        if canonical_id:
            updates.append("canonical_id = ?")
            params.append(canonical_id)
        if metadata:
            updates.append("metadata = ?")
            params.append(meta_json)
        if updates:
            params.extend([entity_type, name])
            cur.execute(
                f"UPDATE knowledge_entities SET {', '.join(updates)} WHERE entity_type = ? AND name = ?",
                params,
            )
    cur.execute(
        """
        SELECT id FROM knowledge_entities WHERE entity_type = ? AND name = ?
        """,
        (entity_type, name),
    )
    row = cur.fetchone()
    return row["id"] if row else None


def create_source(conn: sqlite3.Connection, source_type: str, source_id: Optional[str],
                  title: Optional[str] = None, url: Optional[str] = None,
                  year: Optional[int] = None, metadata: Optional[Dict] = None) -> Optional[int]:
    meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
    cur = conn.cursor()
    if source_id:
        cur.execute(
            """
            SELECT id FROM knowledge_sources WHERE source_type = ? AND source_id = ?
            """,
            (source_type, source_id),
        )
        row = cur.fetchone()
        if row:
            return row["id"]
    if title:
        cur.execute(
            """
            SELECT id FROM knowledge_sources WHERE source_type = ? AND title = ?
            """,
            (source_type, title),
        )
        row = cur.fetchone()
        if row:
            return row["id"]

    cur.execute(
        """
        INSERT INTO knowledge_sources (source_type, source_id, title, url, year, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (source_type, source_id, title, url, year, meta_json),
    )
    return cur.lastrowid


def _source_entity_type(source_type: str) -> str:
    st = (source_type or "").lower()
    if st in ("literature", "paper", "publication"):
        return "paper"
    if st in ("experiment", "experimental"):
        return "experiment"
    if st in ("ml_prediction", "ml", "model"):
        return "model"
    if st in ("theory", "adsorption_energy", "dataset"):
        return "dataset"
    return "source"


def create_relation(conn: sqlite3.Connection, subject_id: int, predicate: str, object_id: int,
                    confidence: Optional[float] = None,
                    evidence_source_id: Optional[int] = None,
                    metadata: Optional[Dict] = None) -> Optional[int]:
    meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, confidence, evidence_source_id, metadata FROM knowledge_relations
        WHERE subject_id = ? AND predicate = ? AND object_id = ?
        """,
        (subject_id, predicate, object_id),
    )
    row = cur.fetchone()
    if row:
        rel_id = row["id"]
        updates = []
        params = []
        if confidence is not None and row["confidence"] is None:
            updates.append("confidence = ?")
            params.append(confidence)
        if evidence_source_id is not None and row["evidence_source_id"] is None:
            updates.append("evidence_source_id = ?")
            params.append(evidence_source_id)
        if metadata is not None and row["metadata"] is None:
            updates.append("metadata = ?")
            params.append(meta_json)
        if updates:
            params.append(rel_id)
            cur.execute(
                f"UPDATE knowledge_relations SET {', '.join(updates)} WHERE id = ?",
                params,
            )
        return rel_id
    cur.execute(
        """
        INSERT INTO knowledge_relations
        (subject_id, predicate, object_id, confidence, evidence_source_id, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (subject_id, predicate, object_id, confidence, evidence_source_id, meta_json),
    )
    return cur.lastrowid


def add_relation_evidence(conn: sqlite3.Connection, relation_id: int, source_id: int,
                          score: Optional[float] = None, metadata: Optional[Dict] = None) -> None:
    meta_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO knowledge_relation_evidence (relation_id, source_id, score, metadata)
        VALUES (?, ?, ?, ?)
        """,
        (relation_id, source_id, score, meta_json),
    )


def migrate_materials_and_evidence(conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    cur.execute("SELECT material_id, formula, formation_energy FROM materials")
    mats = cur.fetchall()

    migrated = 0
    for m in mats:
        mat_id = m["material_id"]
        formula = m["formula"]
        formation_energy = m["formation_energy"]
        mat_name = mat_id or formula
        mat_entity_id = ensure_entity(
            conn,
            "material",
            name=mat_name,
            canonical_id=mat_id,
            metadata={"formula": formula, "formation_energy": formation_energy},
        )
        if not mat_entity_id:
            continue

        # Evidence -> Knowledge sources + relations
        cur.execute(
            "SELECT source_type, source_id, score, metadata FROM evidence WHERE material_id = ?",
            (mat_id,),
        )
        evidences = cur.fetchall()
        for ev in evidences:
            ev_meta = _safe_json(ev["metadata"])
            title = None
            url = None
            year = None
            if ev_meta:
                title = ev_meta.get("title")
                url = ev_meta.get("url")
                year = ev_meta.get("year")
            source_id = create_source(
                conn,
                ev["source_type"],
                ev["source_id"],
                title=title,
                url=url,
                year=year,
                metadata=ev_meta,
            )
            source_entity_type = _source_entity_type(ev["source_type"])
            source_name = ev["source_id"] or title or f"{ev['source_type']}_source"
            source_entity_id = ensure_entity(
                conn,
                source_entity_type,
                name=source_name,
                canonical_id=ev["source_id"],
                metadata={"source_type": ev["source_type"], "title": title, "url": url, "year": year},
            )
            rel_id = create_relation(
                conn,
                subject_id=mat_entity_id,
                predicate="supported_by",
                object_id=source_entity_id or mat_entity_id,
                confidence=ev["score"],
                evidence_source_id=source_id,
                metadata={"source_type": ev["source_type"]},
            )
            if rel_id and source_id:
                add_relation_evidence(conn, rel_id, source_id, score=ev["score"])
        migrated += 1
    return migrated


def _slug(text: str) -> str:
    if not text:
        return ""
    import re
    return re.sub(r"[^A-Za-z0-9\\-\\+]+", "_", text)


def migrate_adsorption(conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT material_id, surface_composition, facet, adsorbate, reaction_energy, activation_energy
        FROM adsorption_energies
        """
    )
    rows = cur.fetchall()
    count = 0
    for row in rows:
        mat_id = row["material_id"]
        surface = row["surface_composition"] or ""
        facet = row["facet"] or ""
        ads = row["adsorbate"] or ""

        if mat_id:
            mat_entity_id = ensure_entity(conn, "material", name=mat_id, canonical_id=mat_id)
        else:
            mat_entity_id = ensure_entity(conn, "surface", name=surface or "surface", canonical_id=None)

        ads_tag = _slug(ads) or "ads"
        facet_tag = _slug(facet) or "facet"
        surface_tag = _slug(surface) or "surface"
        prop_name = f"adsorption_{ads_tag}_{facet_tag}_{surface_tag}"
        prop_entity_id = ensure_entity(
            conn,
            "property",
            name=prop_name,
            metadata={
                "adsorbate": ads,
                "facet": facet,
                "surface": surface,
                "reaction_energy": row["reaction_energy"],
                "activation_energy": row["activation_energy"],
            },
        )
        if mat_entity_id and prop_entity_id:
            create_relation(
                conn,
                subject_id=mat_entity_id,
                predicate="has_property",
                object_id=prop_entity_id,
                confidence=None,
                metadata={"type": "adsorption"},
            )
            count += 1
    return count


def _safe_json(meta: Optional[str]) -> Optional[Dict]:
    if not meta:
        return None
    try:
        return json.loads(meta)
    except Exception:
        return {"raw": meta}


def migrate(db_path: str):
    conn = _connect(db_path)
    try:
        total_mat = migrate_materials_and_evidence(conn)
        total_ads = migrate_adsorption(conn)
        conn.commit()
        print(f"Migrated materials: {total_mat}")
        print(f"Migrated adsorption records: {total_ads}")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=config.DB_PATH, help="Path to SQLite DB")
    args = parser.parse_args()
    migrate(args.db)


if __name__ == "__main__":
    main()
