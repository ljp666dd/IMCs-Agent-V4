"""
IMCs Token Usage Tracker (V5 - Phase B)

Monitors and audits LLM API consumption:
1. Per-call token counting (input/output)
2. Cost estimation by model tier
3. Tiered model routing (simple tasks → lightweight, reasoning → flagship)
4. Usage report generation
"""

import time
import json
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from src.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TokenUsageRecord:
    """Single LLM API call record."""
    timestamp: float
    model: str
    task_type: str  # "report", "audit", "parse", "vision", etc.
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    success: bool = True


# Pricing per 1M tokens (approximate, as of 2026)
MODEL_PRICING = {
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    "default": {"input": 0.50, "output": 2.00},
}

# Task → recommended model tier
TASK_MODEL_MAP = {
    "parse": "lightweight",       # Simple extraction
    "vision": "lightweight",      # Image analysis
    "report": "flagship",         # Expert report generation
    "audit": "flagship",          # Consistency auditing
    "debate": "flagship",         # Conflict resolution
    "general": "lightweight",
}


class TokenTracker:
    """
    Singleton Token Usage Tracker.
    Thread-safe for concurrent agent execution.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.records: List[TokenUsageRecord] = []
        self._records_lock = threading.Lock()
        self.session_start = time.time()
        logger.info("TokenTracker initialized.")

    def log_usage(
        self,
        model: str,
        task_type: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        success: bool = True
    ):
        """Log a single LLM API call."""
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        record = TokenUsageRecord(
            timestamp=time.time(),
            model=model,
            task_type=task_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
            success=success
        )

        with self._records_lock:
            self.records.append(record)

        logger.debug(
            f"Token usage: {model}/{task_type} "
            f"in={input_tokens} out={output_tokens} "
            f"cost=${cost:.6f} latency={latency_ms:.0f}ms"
        )

    def get_usage_report(self) -> Dict[str, Any]:
        """Generate a comprehensive usage report."""
        with self._records_lock:
            records = list(self.records)

        if not records:
            return {"total_calls": 0, "total_cost_usd": 0, "message": "No API calls recorded."}

        by_model: Dict[str, Dict] = defaultdict(lambda: {
            "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0
        })
        by_task: Dict[str, Dict] = defaultdict(lambda: {
            "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0
        })

        total_cost = 0.0
        total_input = 0
        total_output = 0
        total_latency = 0.0

        for r in records:
            by_model[r.model]["calls"] += 1
            by_model[r.model]["input_tokens"] += r.input_tokens
            by_model[r.model]["output_tokens"] += r.output_tokens
            by_model[r.model]["cost"] += r.estimated_cost_usd

            by_task[r.task_type]["calls"] += 1
            by_task[r.task_type]["input_tokens"] += r.input_tokens
            by_task[r.task_type]["output_tokens"] += r.output_tokens
            by_task[r.task_type]["cost"] += r.estimated_cost_usd

            total_cost += r.estimated_cost_usd
            total_input += r.input_tokens
            total_output += r.output_tokens
            total_latency += r.latency_ms

        elapsed = time.time() - self.session_start

        return {
            "total_calls": len(records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost_usd": round(total_cost, 6),
            "avg_latency_ms": round(total_latency / len(records), 1),
            "session_duration_s": round(elapsed, 1),
            "by_model": dict(by_model),
            "by_task": dict(by_task),
            "cost_optimization_tip": self._generate_optimization_tip(by_task)
        }

    def recommend_model(self, task_type: str) -> str:
        """
        Recommend a model tier for a given task type (tiered routing).
        Returns 'lightweight' or 'flagship'.
        """
        return TASK_MODEL_MAP.get(task_type, "lightweight")

    def reset(self):
        """Reset all records (for testing)."""
        with self._records_lock:
            self.records.clear()
        self.session_start = time.time()

    def _generate_optimization_tip(self, by_task: Dict) -> str:
        """Generate cost optimization suggestions."""
        tips = []
        for task, data in by_task.items():
            recommended_tier = TASK_MODEL_MAP.get(task, "lightweight")
            if recommended_tier == "lightweight" and data["cost"] > 0.01:
                tips.append(
                    f"任务 '{task}' 建议使用轻量模型（当前花费 ${data['cost']:.4f}）"
                )
        return "；".join(tips) if tips else "当前调度策略合理。"


def get_token_tracker() -> TokenTracker:
    """Get singleton TokenTracker instance."""
    return TokenTracker()
