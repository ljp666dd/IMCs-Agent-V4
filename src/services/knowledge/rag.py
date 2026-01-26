import json
import sqlite3
from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config.config import config


class KnowledgeRAG:
    """Minimal GraphRAG: graph-filtered retrieval over knowledge_sources."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.DB_PATH
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        self._source_ids: List[int] = []
        self._source_rows: List[Dict[str, Any]] = []
        self._source_count: int = 0

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_sources(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, source_type, source_id, title, url, year, metadata
                FROM knowledge_sources
                ORDER BY created_at DESC
                """
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def _build_text(self, row: Dict[str, Any]) -> str:
        parts = []
        title = row.get("title") or ""
        if title:
            parts.append(title)
        meta = row.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {"raw": meta}
        if isinstance(meta, dict):
            for key in ("abstract", "keywords", "summary", "notes", "content"):
                val = meta.get(key)
                if not val:
                    continue
                if isinstance(val, list):
                    parts.extend([str(v) for v in val if v])
                else:
                    parts.append(str(val))
        return " ".join([p for p in parts if p]).strip()

    def refresh(self) -> None:
        rows = self._load_sources()
        corpus = [self._build_text(r) for r in rows]
        has_text = any(text.strip() for text in corpus)
        if not rows or not has_text:
            # Avoid empty-vocabulary errors
            self._vectorizer = None
            self._matrix = None
        else:
            self._vectorizer = TfidfVectorizer(stop_words="english")
            self._matrix = self._vectorizer.fit_transform(corpus)
        self._source_ids = [r.get("id") for r in rows]
        self._source_rows = rows
        self._source_count = len(rows)

    def _ensure_index(self) -> None:
        with self._get_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM knowledge_sources")
            count = cur.fetchone()[0] or 0
        if self._vectorizer is None or self._matrix is None or count != self._source_count:
            self.refresh()

    def query(
        self,
        query_text: str,
        candidate_source_ids: Optional[List[int]] = None,
        top_k: int = 5,
        source_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not query_text:
            return []
        self._ensure_index()
        if not self._source_rows:
            return []
        if self._vectorizer is None or self._matrix is None:
            return []

        query_vec = self._vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, self._matrix).flatten()

        candidates = []
        for idx, row in enumerate(self._source_rows):
            sid = row.get("id")
            if candidate_source_ids and sid not in candidate_source_ids:
                continue
            if source_type and (row.get("source_type") or "").lower() != source_type.lower():
                continue
            candidates.append((idx, row, float(scores[idx])))

        candidates.sort(key=lambda x: x[2], reverse=True)
        results = []
        for idx, row, score in candidates[: max(1, top_k)]:
            results.append({
                "id": row.get("id"),
                "source_type": row.get("source_type"),
                "source_id": row.get("source_id"),
                "title": row.get("title"),
                "url": row.get("url"),
                "year": row.get("year"),
                "score": round(score, 4),
            })
        return results
