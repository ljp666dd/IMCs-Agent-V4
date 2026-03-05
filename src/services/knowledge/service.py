import json
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from src.config.config import config


class KnowledgeService:
    """Knowledge Core service (entities, relations, evidence)."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.DB_PATH

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _to_meta(self, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        if metadata is None:
            return None
        try:
            return json.dumps(metadata, ensure_ascii=False)
        except Exception:
            return json.dumps(str(metadata))

    def ensure_entity(
        self,
        entity_type: str,
        name: str,
        canonical_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        meta_json = self._to_meta(metadata)
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT OR IGNORE INTO knowledge_entities (entity_type, name, canonical_id, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (entity_type, name, canonical_id, meta_json),
            )

            updates = []
            params: List[Any] = []
            if canonical_id is not None:
                updates.append("canonical_id = ?")
                params.append(canonical_id)
            if metadata is not None:
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
                SELECT id, entity_type, name, canonical_id, metadata, created_at
                FROM knowledge_entities
                WHERE entity_type = ? AND name = ?
                """,
                (entity_type, name),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def get_entity(self, entity_id: int) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, entity_type, name, canonical_id, metadata, created_at
                FROM knowledge_entities WHERE id = ?
                """,
                (entity_id,),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def get_entity_by_name(self, entity_type: str, name: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, entity_type, name, canonical_id, metadata, created_at
                FROM knowledge_entities
                WHERE entity_type = ? AND name = ?
                """,
                (entity_type, name),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def get_entity_by_canonical(self, entity_type: str, canonical_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, entity_type, name, canonical_id, metadata, created_at
                FROM knowledge_entities
                WHERE entity_type = ? AND canonical_id = ?
                """,
                (entity_type, canonical_id),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def create_source(
        self,
        source_type: str,
        source_id: Optional[str] = None,
        title: Optional[str] = None,
        url: Optional[str] = None,
        year: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        meta_json = self._to_meta(metadata)
        with self._get_conn() as conn:
            cur = conn.cursor()
            if source_id:
                cur.execute(
                    """
                    SELECT id, source_type, source_id, title, url, year, metadata, created_at
                    FROM knowledge_sources
                    WHERE source_type = ? AND source_id = ?
                    """,
                    (source_type, source_id),
                )
                row = cur.fetchone()
                if row:
                    return dict(row)
            if title:
                cur.execute(
                    """
                    SELECT id, source_type, source_id, title, url, year, metadata, created_at
                    FROM knowledge_sources
                    WHERE source_type = ? AND title = ?
                    """,
                    (source_type, title),
                )
                row = cur.fetchone()
                if row:
                    return dict(row)

            cur.execute(
                """
                INSERT INTO knowledge_sources (source_type, source_id, title, url, year, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source_type, source_id, title, url, year, meta_json),
            )
            source_id_pk = cur.lastrowid
            cur.execute(
                """
                SELECT id, source_type, source_id, title, url, year, metadata, created_at
                FROM knowledge_sources WHERE id = ?
                """,
                (source_id_pk,),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def create_relation(
        self,
        subject_id: int,
        predicate: str,
        object_id: int,
        confidence: Optional[float] = None,
        evidence_source_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        meta_json = self._to_meta(metadata)
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id FROM knowledge_relations
                WHERE subject_id = ? AND predicate = ? AND object_id = ?
                """,
                (subject_id, predicate, object_id),
            )
            row = cur.fetchone()
            if row:
                rel_id = row["id"]
                updates = []
                params: List[Any] = []
                if confidence is not None:
                    updates.append("confidence = ?")
                    params.append(confidence)
                if evidence_source_id is not None:
                    updates.append("evidence_source_id = ?")
                    params.append(evidence_source_id)
                if metadata is not None:
                    updates.append("metadata = ?")
                    params.append(meta_json)
                if updates:
                    params.append(rel_id)
                    cur.execute(
                        f"UPDATE knowledge_relations SET {', '.join(updates)} WHERE id = ?",
                        params,
                    )
            else:
                cur.execute(
                    """
                    INSERT INTO knowledge_relations
                    (subject_id, predicate, object_id, confidence, evidence_source_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (subject_id, predicate, object_id, confidence, evidence_source_id, meta_json),
                )
                rel_id = cur.lastrowid
            cur.execute(
                """
                SELECT id, subject_id, predicate, object_id, confidence, evidence_source_id, metadata, created_at
                FROM knowledge_relations WHERE id = ?
                """,
                (rel_id,),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def add_relation_evidence(
        self,
        relation_id: int,
        source_id: int,
        score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        meta_json = self._to_meta(metadata)
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO knowledge_relation_evidence (relation_id, source_id, score, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (relation_id, source_id, score, meta_json),
            )
            evid_id = cur.lastrowid
            cur.execute(
                """
                SELECT id, relation_id, source_id, score, metadata, created_at
                FROM knowledge_relation_evidence WHERE id = ?
                """,
                (evid_id,),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def list_relations(
        self,
        entity_id: int,
        direction: str = "both",
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        direction = direction.lower()
        with self._get_conn() as conn:
            cur = conn.cursor()
            if direction == "out":
                cur.execute(
                    """
                    SELECT * FROM knowledge_relations
                    WHERE subject_id = ? ORDER BY created_at DESC LIMIT ?
                    """,
                    (entity_id, limit),
                )
            elif direction == "in":
                cur.execute(
                    """
                    SELECT * FROM knowledge_relations
                    WHERE object_id = ? ORDER BY created_at DESC LIMIT ?
                    """,
                    (entity_id, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM knowledge_relations
                    WHERE subject_id = ? OR object_id = ?
                    ORDER BY created_at DESC LIMIT ?
                    """,
                    (entity_id, entity_id, limit),
                )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def _get_entities_by_ids(self, ids: List[int]) -> Dict[int, Dict[str, Any]]:
        if not ids:
            return {}
        placeholders = ",".join(["?"] * len(ids))
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                SELECT id, entity_type, name, canonical_id, metadata, created_at
                FROM knowledge_entities WHERE id IN ({placeholders})
                """,
                ids,
            )
            rows = cur.fetchall()
        return {r["id"]: dict(r) for r in rows}

    def trace_entity(self, entity_id: int, depth: int = 1, limit: int = 200) -> Dict[str, Any]:
        """Return a small subgraph (nodes + edges) around an entity."""
        edges = []
        visited_edges = set()
        frontier = {entity_id}
        visited_nodes = {entity_id}

        for _ in range(max(1, depth)):
            next_frontier = set()
            for nid in frontier:
                rels = self.list_relations(nid, direction="both", limit=limit)
                for rel in rels:
                    key = (rel["subject_id"], rel["predicate"], rel["object_id"])
                    if key in visited_edges:
                        continue
                    visited_edges.add(key)
                    edges.append(rel)
                    next_frontier.add(rel["subject_id"])
                    next_frontier.add(rel["object_id"])
                    visited_nodes.add(rel["subject_id"])
                    visited_nodes.add(rel["object_id"])
            frontier = next_frontier

        nodes = self._get_entities_by_ids(list(visited_nodes))
        return {
            "center_id": entity_id,
            "nodes": list(nodes.values()),
            "edges": edges,
        }

    def link_entities(
        self,
        subject: Tuple[str, str],
        predicate: str,
        obj: Tuple[str, str],
        confidence: Optional[float] = None,
        evidence_source_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Convenience helper for (type,name) -> (type,name) relation."""
        subj = self.ensure_entity(subject[0], subject[1])
        obj_ent = self.ensure_entity(obj[0], obj[1])
        if not subj or not obj_ent:
            return None
        return self.create_relation(
            subject_id=subj["id"],
            predicate=predicate,
            object_id=obj_ent["id"],
            confidence=confidence,
            evidence_source_id=evidence_source_id,
            metadata=metadata,
        )

    def _source_entity_type(self, source_type: str) -> str:
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

    def upsert_material_evidence(
        self,
        material_id: str,
        source_type: str,
        source_id: Optional[str],
        score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not material_id:
            return None
        mat_entity = self.ensure_entity("material", name=material_id, canonical_id=material_id)
        if not mat_entity:
            return None

        title = None
        url = None
        year = None
        if metadata:
            title = metadata.get("title")
            url = metadata.get("url")
            year = metadata.get("year")

        source = self.create_source(
            source_type=source_type,
            source_id=source_id,
            title=title,
            url=url,
            year=year,
            metadata=metadata,
        )
        source_entity = self.ensure_entity(
            self._source_entity_type(source_type),
            name=source_id or title or f"{source_type}_source",
            canonical_id=source_id,
            metadata={"source_type": source_type, "title": title, "url": url, "year": year},
        )
        if not source_entity:
            return None

        return self.create_relation(
            subject_id=mat_entity["id"],
            predicate="supported_by",
            object_id=source_entity["id"],
            confidence=score,
            evidence_source_id=source["id"] if source else None,
            metadata={"source_type": source_type},
        )

    def score_material(self, material_id: str) -> Dict[str, Any]:
        """Compute a simple knowledge quality score for a material."""
        ent = self.get_entity_by_canonical("material", material_id) or self.get_entity_by_name("material", material_id)
        if not ent:
            return {"material_id": material_id, "score": 0.0, "counts": {}}

        rels = self.list_relations(ent["id"], direction="out", limit=500)
        source_ids = [r.get("evidence_source_id") for r in rels if r.get("predicate") == "supported_by"]
        source_ids = [sid for sid in source_ids if sid]
        counts: Dict[str, int] = {}
        weights = {
            "literature": 1.2,
            "experiment": 1.5,
            "ml_prediction": 1.0,
            "theory": 0.8,
            "dataset": 0.9,
            "adsorption_energy": 1.1,
            "activity_metric": 1.3
        }

        if source_ids:
            placeholders = ",".join(["?"] * len(source_ids))
            with self._get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    f"SELECT id, source_type FROM knowledge_sources WHERE id IN ({placeholders})",
                    source_ids,
                )
                rows = cur.fetchall()
            for row in rows:
                stype = (row["source_type"] or "unknown").lower()
                counts[stype] = counts.get(stype, 0) + 1

        score = 0.0
        for stype, cnt in counts.items():
            score += cnt * weights.get(stype, 0.5)

        return {
            "material_id": material_id,
            "score": round(score, 3),
            "counts": counts
        }

    def generate_structure_preview(self, material_id: str) -> Dict[str, Any]:
        """
        Generate a 3D HTML preview for a material's crystal structure (V5.7 Preview).
        Uses 3Dmol.js for browser-side rendering.
        """
        from src.services.db.database import DatabaseService
        db = DatabaseService(self.db_path)
        
        material = db.get_material_by_id(material_id)
        if not material or not material.get("cif_path"):
            return {"error": "Material or CIF path not found"}
            
        cif_content = db._read_cif_content(material["cif_path"])
        if not cif_content:
            return {"error": "Could not read CIF content"}
            
        # V6: Enhanced py3Dmol style HTML template for responsive dark mode UI
        html_template = f"""
        <div style="height: 100%; width: 100%; min-height: 400px; position: relative;" class='viewer_3Dmoljs' 
             data-selectable='true' data-category='crystal' data-type='cif' data-backgroundcolor='#0e1117'
             data-style='{{"stick": {{"radius": 0.2}}, "sphere": {{"scale": 0.3}}}}'>
            <textarea style="display:none;">{cif_content}</textarea>
        </div>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        """
        
        return {
            "success": True,
            "material_id": material_id,
            "html_snippet": html_template,
            "viewer_type": "3Dmol.js"
        }
