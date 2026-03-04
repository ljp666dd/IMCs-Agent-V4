from typing import List, Dict, Any, Optional
import sqlite3
import json
import os
from src.core.logger import get_logger, log_exception

logger = get_logger(__name__)

class MaterialRepoMixin:
    # ========== Materials ==========
    
    @log_exception(logger)
    def save_material(self, material_id: str, formula: str, 
                      energy: float = None, cif_path: str = None) -> int:
        """Save theoretical material data."""
        query = """
        INSERT INTO materials (material_id, formula, formation_energy, cif_path)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(material_id) DO UPDATE SET
            formula = COALESCE(excluded.formula, materials.formula),
            formation_energy = COALESCE(excluded.formation_energy, materials.formation_energy),
            cif_path = COALESCE(excluded.cif_path, materials.cif_path)
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (material_id, formula, energy, cif_path))
            return cursor.lastrowid

    def _normalize_material_id(self, material_id: Optional[str]) -> Optional[str]:
        mid = (str(material_id) if material_id is not None else "").strip()
        return mid or None

    def _guess_stub_formula(self, material_id: str, formula: Optional[str] = None) -> str:
        if formula is not None:
            val = str(formula).strip()
            if val:
                return val

        mid = (str(material_id) if material_id is not None else "").strip()
        if not mid:
            return "UNKNOWN"
        if mid.startswith("fake:"):
            rest = mid.split("fake:", 1)[1].strip()
            return rest or "UNKNOWN"
        if mid.startswith("mp-"):
            return "UNKNOWN"
        for ch in (":", "/", "\\", " ", "-"):
            if ch in mid:
                return "UNKNOWN"
        return mid

    def ensure_material_stub(self, material_id: str, formula: Optional[str] = None) -> Optional[str]:
        """
        Ensure a materials row exists for the given material_id without overwriting
        any existing formula/fields.

        This is used to prevent FK errors when external systems (robot/importers)
        report metrics for material_ids that are not yet present in the local DB.
        """
        mid = self._normalize_material_id(material_id)
        if not mid:
            return None
        stub_formula = self._guess_stub_formula(mid, formula=formula)
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO materials (material_id, formula, formation_energy, cif_path)
                VALUES (?, ?, NULL, NULL)
                ON CONFLICT(material_id) DO NOTHING
                """,
                (mid, stub_formula),
            )
        return mid
            
    def list_materials(self, limit: int = 100, allowed_elements: Optional[List[str]] = None, require_cif: bool = False) -> List[Dict]:
        """List stored materials."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = "SELECT * FROM materials"
            if require_cif:
                query += " WHERE cif_path IS NOT NULL"
            query += " ORDER BY created_at DESC LIMIT ?"
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            records = [dict(row) for row in rows]
            return self._filter_material_records(records, allowed_elements)

    def list_materials_since(self, created_at: str, limit: int = 100, allowed_elements: Optional[List[str]] = None) -> List[Dict]:
        """List materials created since a timestamp."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM materials WHERE created_at >= ? ORDER BY created_at DESC LIMIT ?",
                (created_at, limit),
            )
            rows = cursor.fetchall()
            records = [dict(row) for row in rows]
            return self._filter_material_records(records, allowed_elements)

    def list_materials_by_ids(
        self,
        material_ids: List[str],
        allowed_elements: Optional[List[str]] = None,
    ) -> List[Dict]:
        """List material records by material_ids (preserve input order)."""
        if not material_ids:
            return []

        ordered: List[str] = []
        seen = set()
        for mid in material_ids:
            mid = (str(mid) if mid is not None else "").strip()
            if not mid or mid in seen:
                continue
            seen.add(mid)
            ordered.append(mid)

        if not ordered:
            return []

        placeholders = ",".join(["?"] * len(ordered))
        query = f"SELECT * FROM materials WHERE material_id IN ({placeholders})"
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, tuple(ordered))
            rows = cursor.fetchall()
            records = [dict(row) for row in rows]
            records = self._filter_material_records(records, allowed_elements)
            by_id = {r.get("material_id"): r for r in records if r.get("material_id")}
            return [by_id[mid] for mid in ordered if mid in by_id]
            
    def get_material_by_id(self, material_id: str) -> Optional[Dict]:
        """Get material details including CIF content."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM materials WHERE material_id = ?", (material_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            data = dict(row)
            # Read CIF content if path exists
            data["cif_content"] = self._read_cif_content(data.get("cif_path"))
                
            return data

    def get_material_with_evidence(self, material_id: str, include_cif: bool = False) -> Optional[Dict]:
        """Get material details with evidence (optionally include CIF content)."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM materials WHERE material_id = ?", (material_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            if include_cif:
                data["cif_content"] = self._read_cif_content(data.get("cif_path"))
            
            # Auto-parse DOS data
            if data.get("dos_data") and isinstance(data["dos_data"], str):
                try:
                    data["dos_data"] = json.loads(data["dos_data"])
                except Exception:
                    pass

            data["evidence"] = self.get_evidence_for_material(material_id)
            # 附加吸附能记录(若存在)
            data["adsorption_energies"] = self.list_adsorption_energies(material_id)
            data["activity_metrics"] = self.list_activity_metrics(material_id)
            return data

    # ========== Evidence & Coverage ==========

    def get_evidence_stats(self, allowed_elements: Optional[List[str]] = None) -> Dict[str, Any]:
        """Return system-wide evidence coverage stats."""
        stats: Dict[str, Any] = {}
        allowed_ids = self._allowed_material_ids(allowed_elements)
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if allowed_ids is None:
                cursor.execute("SELECT COUNT(*) AS total FROM materials")
                stats["total_materials"] = cursor.fetchone()["total"] or 0

                cursor.execute("SELECT COUNT(*) AS total FROM materials WHERE formation_energy IS NOT NULL")
                stats["formation_energy_count"] = cursor.fetchone()["total"] or 0

                cursor.execute("SELECT COUNT(*) AS total FROM materials WHERE dos_data IS NOT NULL")
                stats["dos_count"] = cursor.fetchone()["total"] or 0

                cursor.execute("SELECT source_type, COUNT(DISTINCT material_id) AS cnt FROM evidence GROUP BY source_type")
                stats["evidence_by_source"] = {row["source_type"]: row["cnt"] for row in cursor.fetchall()}

                cursor.execute("SELECT COUNT(*) AS total FROM adsorption_energies")
                stats["adsorption_rows"] = cursor.fetchone()["total"] or 0
                cursor.execute("SELECT COUNT(DISTINCT material_id) AS total FROM adsorption_energies WHERE material_id IS NOT NULL")
                stats["adsorption_materials"] = cursor.fetchone()["total"] or 0

                cursor.execute("SELECT COUNT(*) AS total FROM activity_metrics")
                stats["activity_rows"] = cursor.fetchone()["total"] or 0
                cursor.execute("SELECT COUNT(DISTINCT material_id) AS total FROM activity_metrics WHERE material_id IS NOT NULL")
                stats["activity_materials"] = cursor.fetchone()["total"] or 0
            else:
                cursor.execute("SELECT material_id, formation_energy, dos_data FROM materials")
                total = 0
                fe_count = 0
                dos_count = 0
                for row in cursor.fetchall():
                    mid = row["material_id"]
                    if mid not in allowed_ids:
                        continue
                    total += 1
                    if row["formation_energy"] is not None:
                        fe_count += 1
                    if row["dos_data"] is not None:
                        dos_count += 1
                stats["total_materials"] = total
                stats["formation_energy_count"] = fe_count
                stats["dos_count"] = dos_count

                cursor.execute("SELECT material_id, source_type FROM evidence")
                ev_map: Dict[str, set] = {}
                for row in cursor.fetchall():
                    mid = row["material_id"]
                    if mid not in allowed_ids:
                        continue
                    stype = row["source_type"] or "unknown"
                    ev_map.setdefault(stype, set()).add(mid)
                stats["evidence_by_source"] = {k: len(v) for k, v in ev_map.items()}

                cursor.execute("SELECT material_id FROM adsorption_energies WHERE material_id IS NOT NULL")
                ads_rows = 0
                ads_ids = set()
                for (mid,) in cursor.fetchall():
                    if mid in allowed_ids:
                        ads_rows += 1
                        ads_ids.add(mid)
                stats["adsorption_rows"] = ads_rows
                stats["adsorption_materials"] = len(ads_ids)

                cursor.execute("SELECT material_id FROM activity_metrics WHERE material_id IS NOT NULL")
                act_rows = 0
                act_ids = set()
                for (mid,) in cursor.fetchall():
                    if mid in allowed_ids:
                        act_rows += 1
                        act_ids.add(mid)
                stats["activity_rows"] = act_rows
                stats["activity_materials"] = len(act_ids)

            cursor.execute("SELECT COUNT(*) AS total FROM models")
            stats["model_count"] = cursor.fetchone()["total"] or 0

        return stats

    def get_data_integrity_stats(self) -> Dict[str, Any]:
        """
        Lightweight data integrity / orphan-row checks.

        Rationale:
        - In SQLite, foreign key enforcement is connection-scoped and must be
          explicitly enabled. We also want visibility into any legacy orphan
          rows that were inserted before enforcement was turned on.
        """
        stats: Dict[str, Any] = {}
        with self._get_conn() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys;")
                row = cursor.fetchone()
                stats["foreign_keys_enabled"] = bool(row and int(row[0]) == 1)
            except Exception:
                stats["foreign_keys_enabled"] = None

            # Evidence (material_id is NOT NULL)
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM evidence e
                LEFT JOIN materials m ON e.material_id = m.material_id
                WHERE m.material_id IS NULL
                """
            )
            stats["evidence_orphan_rows"] = int(cursor.fetchone()[0] or 0)
            cursor.execute(
                """
                SELECT COUNT(DISTINCT e.material_id)
                FROM evidence e
                LEFT JOIN materials m ON e.material_id = m.material_id
                WHERE m.material_id IS NULL
                """
            )
            stats["evidence_orphan_material_ids"] = int(cursor.fetchone()[0] or 0)

            # Activity metrics (material_id is nullable)
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM activity_metrics a
                LEFT JOIN materials m ON a.material_id = m.material_id
                WHERE a.material_id IS NOT NULL AND m.material_id IS NULL
                """
            )
            stats["activity_metric_orphan_rows"] = int(cursor.fetchone()[0] or 0)
            cursor.execute(
                """
                SELECT COUNT(DISTINCT a.material_id)
                FROM activity_metrics a
                LEFT JOIN materials m ON a.material_id = m.material_id
                WHERE a.material_id IS NOT NULL AND m.material_id IS NULL
                """
            )
            stats["activity_metric_orphan_material_ids"] = int(cursor.fetchone()[0] or 0)

            # Adsorption energies (material_id is nullable)
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM adsorption_energies ae
                LEFT JOIN materials m ON ae.material_id = m.material_id
                WHERE ae.material_id IS NOT NULL AND m.material_id IS NULL
                """
            )
            stats["adsorption_orphan_rows"] = int(cursor.fetchone()[0] or 0)
            cursor.execute(
                """
                SELECT COUNT(DISTINCT ae.material_id)
                FROM adsorption_energies ae
                LEFT JOIN materials m ON ae.material_id = m.material_id
                WHERE ae.material_id IS NOT NULL AND m.material_id IS NULL
                """
            )
            stats["adsorption_orphan_material_ids"] = int(cursor.fetchone()[0] or 0)

        return stats

    def get_material_by_formula(self, formula: str) -> Optional[Dict]:
        """Get material by formula (Exact Match)."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM materials WHERE formula = ?", (formula,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # ========== M3: Evidence Chain ==========

    def save_evidence(self, material_id: str, source_type: str, source_id: str, score: float = 1.0, metadata: Dict = None) -> int:
        """
        Save evidence linking a material to a source (Lit, Exp, ML).
        """
        query = """
        INSERT INTO evidence (material_id, source_type, source_id, score, metadata)
        VALUES (?, ?, ?, ?, ?)
        """
        meta_json = json.dumps(metadata) if metadata else None
        
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (material_id, source_type, source_id, score, meta_json))
            evid_id = cursor.lastrowid

        # Best-effort sync into Knowledge Core
        try:
            from src.services.knowledge import KnowledgeService
            ks = KnowledgeService(self.db_path)
            ks.upsert_material_evidence(
                material_id=material_id,
                source_type=source_type,
                source_id=source_id,
                score=score,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Knowledge sync failed: {e}")

        return evid_id

    def get_evidence_for_material(self, material_id: str) -> List[Dict]:
        """Get all evidence linked to a material."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM evidence WHERE material_id = ? ORDER BY created_at DESC", (material_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_evidence_counts(self, material_ids: List[str]) -> Dict[str, Dict[str, int]]:
        """Get evidence counts by source_type for each material_id."""
        if not material_ids:
            return {}
        placeholders = ",".join(["?"] * len(material_ids))
        counts: Dict[str, Dict[str, int]] = {mid: {} for mid in material_ids}
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT material_id, source_type, COUNT(*) AS cnt
                FROM evidence
                WHERE material_id IN ({placeholders})
                GROUP BY material_id, source_type
                """,
                material_ids,
            )
            for row in cursor.fetchall():
                mid = row["material_id"]
                stype = row["source_type"] or "unknown"
                counts.setdefault(mid, {})[stype] = row["cnt"] or 0
        return counts

    def get_material_feature_flags(self, material_ids: List[str]) -> Dict[str, Dict[str, bool]]:
        """Return material-level feature flags for evidence gap analysis."""
        if not material_ids:
            return {}
        placeholders = ",".join(["?"] * len(material_ids))
        flags: Dict[str, Dict[str, bool]] = {
            mid: {"formation_energy": False, "dos_data": False} for mid in material_ids
        }
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT material_id, formation_energy, dos_data
                FROM materials
                WHERE material_id IN ({placeholders})
                """,
                material_ids,
            )
            for row in cursor.fetchall():
                mid = row["material_id"]
                if not mid:
                    continue
                flags[mid] = {
                    "formation_energy": row["formation_energy"] is not None,
                    "dos_data": row["dos_data"] is not None,
                }
        return flags

    # ========== Dataset Snapshots (Reproducibility) ==========

    def create_dataset_snapshot(self, plan_id: Optional[str], name: str,
                                description: str = "", metadata: Dict = None) -> Optional[int]:
        """Create a dataset snapshot entry."""
        meta_json = json.dumps(metadata) if metadata else None
        query = """
        INSERT INTO dataset_snapshots (plan_id, name, description, metadata)
        VALUES (?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (plan_id, name, description, meta_json))
            return cursor.lastrowid

    def add_snapshot_item(self, snapshot_id: int, item_type: str, item_id: str,
                          metadata: Dict = None) -> Optional[int]:
        """Add an item to a dataset snapshot."""
        meta_json = json.dumps(metadata) if metadata else None
        query = """
        INSERT INTO dataset_snapshot_items (snapshot_id, item_type, item_id, metadata)
        VALUES (?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (snapshot_id, item_type, item_id, meta_json))
            return cursor.lastrowid

    def list_snapshot_items(self, snapshot_id: int) -> List[Dict]:
        """List items of a dataset snapshot."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM dataset_snapshot_items WHERE snapshot_id = ? ORDER BY created_at DESC",
                (snapshot_id,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_snapshot(self, snapshot_id: int) -> Optional[Dict]:
        """Get snapshot metadata by id."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM dataset_snapshots WHERE id = ?", (snapshot_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_snapshot_by_plan(self, plan_id: str) -> Optional[Dict]:
        """Get latest snapshot for a plan."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM dataset_snapshots WHERE plan_id = ? ORDER BY created_at DESC LIMIT 1",
                (plan_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    # ========== Experiments ==========
    
    @log_exception(logger)
    def save_experiment(self, name: str, exp_type: str, raw_path: str, results: Dict, material_id: str = None) -> int:
        """Save experiment results."""
        query = """
        INSERT INTO experiments (name, type, raw_data_path, results, material_id)
        VALUES (?, ?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (name, exp_type, raw_path, json.dumps(results), material_id))
            return cursor.lastrowid
            
    def fetch_training_set(self, target_col: str = "formation_energy",
                           allowed_elements: Optional[List[str]] = None) -> List[Dict]:
        """Fetch data for training."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Check if target_col exists in table
            cursor.execute("PRAGMA table_info(materials)")
            columns = [row["name"] for row in cursor.fetchall()]
            
            if target_col in columns:
                query = f"SELECT material_id, formula, cif_path, dos_data, {target_col} FROM materials WHERE {target_col} IS NOT NULL"
                cursor.execute(query)
                rows = cursor.fetchall()
            else:
                # Fallback: Fetch dos_data and extract in Python
                query = "SELECT material_id, formula, cif_path, dos_data FROM materials WHERE dos_data IS NOT NULL"
                cursor.execute(query)
                rows = cursor.fetchall()
                
            records = []
            for row in rows:
                rec = dict(row)
                if target_col not in rec:
                     # Try extract from dos_data
                     try:
                         dos = json.loads(rec.get("dos_data", "{}"))
                         val = dos.get(target_col)
                         if val is not None:
                             rec[target_col] = val
                         else:
                             continue # Skip if target not found
                     except Exception:
                         continue
                records.append(rec)
                
            return self._filter_material_records(records, allowed_elements)

    def fetch_activity_training_set(self, metric_name: str,
                                    allowed_elements: Optional[List[str]] = None) -> List[Dict]:
        """Fetch activity metrics joined with materials for ML training."""
        if not metric_name:
            return []
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT m.material_id, m.formula, m.cif_path, m.dos_data,
                       a.metric_value, a.created_at
                FROM activity_metrics a
                JOIN materials m ON a.material_id = m.material_id
                WHERE a.metric_name = ? AND a.metric_value IS NOT NULL
                ORDER BY a.created_at DESC
                """,
                (metric_name,),
            )
            rows = cursor.fetchall()
            if not rows:
                return []
            # Keep only latest per material_id
            seen = set()
            records = []
            for row in rows:
                mid = row["material_id"]
                if not mid or mid in seen:
                    continue
                rec = dict(row)
                rec["target"] = rec.pop("metric_value")
                records.append(rec)
                seen.add(mid)
            return self._filter_material_records(records, allowed_elements)

    # ========== Adsorption Energies (Catalysis-Hub) ==========

    def save_adsorption_energy(self, material_id: Optional[str], surface_composition: str,
                               facet: str, adsorbate: str,
                               reaction_energy: Optional[float],
                               activation_energy: Optional[float],
                               source: str = "Catalysis-Hub",
                               metadata: Dict = None) -> int:
        """Save adsorption energy record (proxy for activity)."""
        material_id = self._normalize_material_id(material_id)
        if material_id:
            # Ensure FK target exists (do not override existing formula).
            try:
                self.ensure_material_stub(material_id)
            except Exception:
                pass
        query = """
        INSERT INTO adsorption_energies
        (material_id, surface_composition, facet, adsorbate, reaction_energy, activation_energy, source, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        meta_json = json.dumps(metadata) if metadata else None
        record_id = None
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (
                material_id,
                surface_composition,
                facet,
                adsorbate,
                reaction_energy,
                activation_energy,
                source,
                meta_json
            ))
            record_id = cursor.lastrowid
        # Link as evidence when material_id is known (new connection to avoid locks)
        if material_id:
            try:
                self.save_evidence(
                    material_id=material_id,
                    source_type="adsorption_energy",
                    source_id=str(record_id),
                    score=0.8,
                    metadata={
                        "surface_composition": surface_composition,
                        "facet": facet,
                        "adsorbate": adsorbate,
                        "reaction_energy": reaction_energy,
                        "activation_energy": activation_energy,
                        "source": source,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to link adsorption evidence: {e}")

        return record_id

    def list_adsorption_energies(self, material_id: str) -> List[Dict]:
        """List adsorption energy records for a material."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM adsorption_energies WHERE material_id = ? ORDER BY created_at DESC",
                (material_id,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]


    # ========== Activity Metrics (HOR/HER Indicators) ==========

    def save_activity_metric(self, material_id: Optional[str], metric_name: str,
                             metric_value: Optional[float], unit: Optional[str] = None,
                             conditions: Dict = None, source: str = "experiment",
                             source_id: Optional[str] = None, metadata: Dict = None) -> int:
        """Save activity metric record and link as evidence when possible."""
        material_id = self._normalize_material_id(material_id)
        if material_id:
            # Ensure FK target exists (do not override existing formula).
            try:
                self.ensure_material_stub(material_id, formula=(metadata or {}).get("formula") if isinstance(metadata, dict) else None)
            except Exception:
                # If we can't ensure, fall back to NULL to avoid rejecting the metric row.
                material_id = None
        query = """
        INSERT INTO activity_metrics
        (material_id, metric_name, metric_value, unit, conditions, source, source_id, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        cond_json = json.dumps(conditions) if conditions else None
        meta_json = json.dumps(metadata) if metadata else None
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (material_id, metric_name, metric_value, unit, cond_json, source, source_id, meta_json)
            )
            record_id = cursor.lastrowid

        if material_id:
            try:
                self.save_evidence(
                    material_id=material_id,
                    source_type="activity_metric",
                    source_id=str(record_id),
                    score=0.9,
                    metadata={
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "unit": unit,
                        "conditions": conditions,
                        "source": source,
                        "source_id": source_id,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to link activity metric evidence: {e}")

        return record_id

    def list_activity_metrics(self, material_id: str) -> List[Dict]:
        """List activity metrics for a material."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM activity_metrics WHERE material_id = ? ORDER BY created_at DESC",
                (material_id,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    def update_material_dos(self, material_id: str, dos_data: Dict) -> None:
        """Update DOS descriptors for a material (JSON)."""
        if dos_data is None:
            return
        dos_json = json.dumps(dos_data)
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE materials SET dos_data = ? WHERE material_id = ?",
                (dos_json, material_id)
            )

