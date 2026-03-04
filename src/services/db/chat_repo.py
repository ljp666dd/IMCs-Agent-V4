from typing import List, Dict, Any, Optional
import sqlite3
import json
import os
from src.core.logger import get_logger, log_exception

logger = get_logger(__name__)

class ChatRepoMixin:
    # ========== Users (v4.0) ==========

    def create_user(self, username: str, password_hash: str) -> Optional[int]:
        """Create a new user."""
        query = "INSERT INTO users (username, password_hash) VALUES (?, ?)"
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (username, password_hash))
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None # Duplicate username

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user by username."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # ========== Chat Sessions & Messages (v4.0) ==========

    def create_chat_session(self, title: str, user_id: Optional[int] = None) -> int:
        """Create a chat session and return its id."""
        query = "INSERT INTO chat_sessions (user_id, title) VALUES (?, ?)"
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (user_id, title))
            return cursor.lastrowid

    def list_chat_sessions(self, limit: int = 50, user_id: Optional[int] = None) -> List[Dict]:
        """List chat sessions ordered by last update."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if user_id is None:
                cursor.execute(
                    "SELECT * FROM chat_sessions ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                )
            else:
                cursor.execute(
                    "SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                    (user_id, limit),
                )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_chat_session(self, session_id: int) -> Optional[Dict]:
        """Get a chat session by id."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_chat_session_title(self, session_id: int, title: str):
        """Update chat session title."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (title, session_id),
            )

    def touch_chat_session(self, session_id: int):
        """Touch chat session updated_at."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )

    def add_chat_message(self, session_id: int, role: str, content: str, artifacts: Dict = None) -> int:
        """Add a chat message to a session."""
        artifacts_json = json.dumps(artifacts) if artifacts else None
        query = """
        INSERT INTO chat_messages (session_id, role, content, artifacts)
        VALUES (?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (session_id, role, content, artifacts_json))
            msg_id = cursor.lastrowid
            cursor.execute(
                "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )
            return msg_id

    def list_chat_messages(self, session_id: int) -> List[Dict]:
        """List chat messages for a session."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            )
            rows = cursor.fetchall()
            messages = []
            for row in rows:
                data = dict(row)
                if data.get("artifacts"):
                    try:
                        data["artifacts"] = json.loads(data["artifacts"])
                    except Exception:
                        pass
                messages.append(data)
            return messages

    def delete_chat_session(self, session_id: int):
        """Delete a chat session and its messages."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))

