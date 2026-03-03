from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.core.logger import get_logger
from src.services.task.types import TaskStep

logger = get_logger(__name__)


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def classify_failure(exc: BaseException) -> Tuple[str, str]:
    """Classify an exception into a coarse failure category.

    Returns:
        (category, reason)
    """
    try:
        name = type(exc).__name__
    except Exception:
        name = "Exception"
    msg = str(exc or "")
    lower = msg.lower()

    if isinstance(exc, PermissionError) or "winerror 5" in lower or "access is denied" in lower or "permission denied" in lower:
        return "permission", f"{name}: {msg}"
    if isinstance(exc, FileNotFoundError) or "no such file" in lower or "not found" in lower or "404" in lower:
        return "not_found", f"{name}: {msg}"
    if isinstance(exc, TimeoutError) or "timeout" in lower or "timed out" in lower or "deadline exceeded" in lower:
        return "timeout", f"{name}: {msg}"
    if "429" in lower or "rate limit" in lower or "too many requests" in lower:
        return "rate_limit", f"{name}: {msg}"
    if "401" in lower or "403" in lower or "unauthorized" in lower or "forbidden" in lower or "api key" in lower or "apikey" in lower or "invalid key" in lower:
        return "auth", f"{name}: {msg}"
    if (
        "connection" in lower
        or "network" in lower
        or "ssl" in lower
        or "dns" in lower
        or "name resolution" in lower
        or "failed to establish a new connection" in lower
    ):
        return "network", f"{name}: {msg}"
    if (
        "no data" in lower
        or "empty" in lower
        or "insufficient" in lower
        or "not enough" in lower
        or "0 rows" in lower
        or "no records" in lower
    ):
        return "missing_data", f"{name}: {msg}"

    return "unknown", f"{name}: {msg}"


@dataclass(frozen=True)
class FailureDecision:
    action: str  # skip | replan | fail
    category: str
    note: str = ""
    reason: str = ""
    max_replans: int = 0
    spec: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "action": self.action,
            "category": self.category,
        }
        if self.note:
            payload["note"] = self.note
        if self.reason:
            payload["reason"] = self.reason
        if self.max_replans:
            payload["max_replans"] = self.max_replans
        if self.spec:
            payload["spec"] = self.spec
        return payload


class FailurePolicyEngine:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(_repo_root(), "configs", "failure_policies.json")
        self.policies = self._load()

    def _load(self) -> Dict[str, Any]:
        try:
            if not os.path.exists(self.config_path):
                return {}
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning(f"Failed to load failure policies: {e}")
            return {}

    def decide(self, step: TaskStep, exc: BaseException) -> FailureDecision:
        category, reason = classify_failure(exc)
        key = f"{getattr(step, 'agent', '')}.{getattr(step, 'action', '')}"

        policies = (self.policies.get("policies") or {}) if isinstance(self.policies, dict) else {}
        default = (self.policies.get("default") or {}) if isinstance(self.policies, dict) else {}

        policy = policies.get(key) if isinstance(policies.get(key), dict) else {}

        def _pick(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            cats = obj.get("categories") or {}
            if isinstance(cats, dict) and isinstance(cats.get(category), dict):
                return cats.get(category)
            fallback = obj.get("fallback")
            if isinstance(fallback, dict):
                return fallback
            return None

        picked = _pick(policy) or _pick(default) or {"action": "fail", "note": "No failure policy matched."}
        action = picked.get("action") if isinstance(picked, dict) else None
        if action not in ("skip", "replan", "fail"):
            action = "fail"
        note = picked.get("note") if isinstance(picked, dict) else ""
        max_replans = 0
        try:
            max_replans = int(picked.get("max_replans") or 0) if isinstance(picked, dict) else 0
        except Exception:
            max_replans = 0
        spec = picked.get("spec") if isinstance(picked, dict) and isinstance(picked.get("spec"), dict) else None
        return FailureDecision(action=action, category=category, note=str(note or ""), reason=str(reason or ""), max_replans=max_replans, spec=spec)

