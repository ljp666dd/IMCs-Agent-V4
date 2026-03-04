"""
IMCs Replan Engine — 任务重规划引擎

从 executor.py 拆分，负责：
1. 加载/格式化重规划策略模板
2. 构建失败回退方案
3. 应用重规划步骤并重连依赖
4. 追加动态步骤（MetaController 建议）
5. 解析 gap 依赖占位符
"""

import json
import os
from typing import Dict, Any, Optional, List

from src.core.logger import get_logger
from src.services.task.types import TaskPlan, TaskStep

logger = get_logger(__name__)


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_replan_strategies() -> Dict[str, Any]:
    """Load replan/fallback strategies from configs/replan_strategies.json."""
    try:
        path = os.path.join(_repo_root(), "configs", "replan_strategies.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def format_params(params: Any, template_vars: Dict[str, str]) -> Any:
    if isinstance(params, dict):
        return {k: format_params(v, template_vars) for k, v in params.items()}
    if isinstance(params, list):
        return [format_params(v, template_vars) for v in params]
    if isinstance(params, str):
        out = params
        for k, v in (template_vars or {}).items():
            out = out.replace("{" + str(k) + "}", str(v or ""))
        return out
    return params


def simplify_query(query: str) -> str:
    """Best-effort query simplification for fallback searches."""
    if not query:
        return ""
    import re
    tokens = re.findall(r"[A-Za-z0-9\\-\\+]+", query)
    if not tokens:
        return query.strip()
    return " ".join(tokens[:8]).strip()


def replan_from_strategy(
    replan_strategies: Dict[str, Any],
    step: TaskStep,
    active_plan: Optional[TaskPlan],
) -> Optional[Dict[str, Any]]:
    if not replan_strategies:
        return None
    group = replan_strategies.get("default") or {}
    key = f"{step.agent}.{step.action}"
    spec = group.get(key)
    if not isinstance(spec, dict):
        return None
    try:
        query = None
        if isinstance(step.params, dict):
            query = step.params.get("query")
        if not query and active_plan:
            query = active_plan.description
        template_vars = {
            "query": query or "",
            "query_simplified": simplify_query(query or ""),
        }
        spec = json.loads(json.dumps(spec))
        spec["steps"] = [
            {
                **item,
                "params": format_params(item.get("params") or {}, template_vars),
            }
            for item in (spec.get("steps") or [])
        ]
        return spec
    except Exception:
        return spec


def build_replan_spec(
    step: TaskStep,
    replan_strategies: Dict[str, Any],
    active_plan: Optional[TaskPlan],
) -> Optional[Dict[str, Any]]:
    """Return a minimal fallback plan when a step fails."""
    agent = step.agent
    action = step.action
    strategy_spec = replan_from_strategy(replan_strategies, step, active_plan)
    if strategy_spec:
        return strategy_spec

    if agent == "literature" and action == "search":
        query = (step.params or {}).get("query", "")
        simplified = simplify_query(query)
        if simplified and simplified.lower() != query.lower():
            fallback_query = simplified
        elif query:
            fallback_query = f"{query} catalyst"
        else:
            fallback_query = "HOR catalyst"
        return {
            "note": "fallback: simplified literature query",
            "steps": [
                {
                    "agent": "literature",
                    "action": "search",
                    "params": {"query": fallback_query, "limit": 5},
                    "deps": []
                }
            ]
        }

    if agent == "theory" and action == "download":
        data_types = (step.params or {}).get("data_types") or []
        fallback_types = ["cif"]
        if "formation_energy" in data_types:
            fallback_types = ["cif", "formation_energy"]
        return {
            "note": "fallback: reduced theory download scope",
            "steps": [
                {
                    "agent": "theory",
                    "action": "download",
                    "params": {"data_types": fallback_types, "limit": 20},
                    "deps": []
                }
            ]
        }

    if agent == "ml" and action in ("train", "train_all"):
        return {
            "note": "fallback: refresh theory data then retrain",
            "steps": [
                {
                    "agent": "theory",
                    "action": "download",
                    "params": {"data_types": ["cif", "formation_energy"], "limit": 50},
                    "deps": []
                },
                {
                    "agent": "ml",
                    "action": action,
                    "params": step.params or {},
                    "deps": ["$prev"]
                }
            ]
        }

    return None


def next_step_id(plan: TaskPlan) -> str:
    """Allocate a unique step_id within a plan."""
    max_idx = 0
    for step in plan.steps:
        sid = getattr(step, "step_id", "")
        if isinstance(sid, str) and sid.startswith("step_"):
            try:
                max_idx = max(max_idx, int(sid.split("_", 1)[1]))
            except Exception:
                continue
    return f"step_{max_idx + 1}"


def apply_replan(
    plan: TaskPlan,
    failed_step: TaskStep,
    spec: Dict[str, Any],
    pending: Dict[str, TaskStep],
    db,
) -> List[str]:
    """Insert replan steps and rewire dependencies. Returns new step_ids."""
    new_step_ids: List[str] = []
    prev_id: Optional[str] = None

    for item in spec.get("steps", []):
        step_id = next_step_id(plan)
        deps = []
        for dep in item.get("deps", []):
            if dep == "$prev":
                if prev_id:
                    deps.append(prev_id)
            else:
                deps.append(dep)

        step = TaskStep(
            step_id=step_id,
            agent=item.get("agent", ""),
            action=item.get("action", ""),
            params=item.get("params") or {},
            dependencies=deps,
            max_retries=item.get("max_retries", 0),
            max_replans=0
        )
        plan.steps.append(step)
        pending[step.step_id] = step
        new_step_ids.append(step.step_id)
        prev_id = step.step_id

        # Persist the new pending step
        db.log_plan_step(
            plan_id=plan.task_id,
            step_id=step.step_id,
            agent=step.agent,
            action=step.action,
            status="pending",
            dependencies=step.dependencies,
            params=step.params
        )

    if not new_step_ids:
        return new_step_ids

    anchor_id = new_step_ids[-1]
    for step in pending.values():
        deps = step.dependencies or []
        if failed_step.step_id in deps:
            step.dependencies = [anchor_id if d == failed_step.step_id else d for d in deps]
    return new_step_ids


def append_dynamic_steps(
    plan: TaskPlan,
    specs: List[Dict[str, Any]],
    pending: Dict[str, TaskStep],
    db,
) -> List[str]:
    """Append meta-controller suggested steps to the plan."""
    new_step_ids: List[str] = []
    step_id_map = {}

    for step in plan.steps:
        if step.agent:
            step_id_map[step.agent] = step.step_id

    for item in specs:
        step_id = next_step_id(plan)
        deps = []
        for dep in item.get("deps", []):
            if isinstance(dep, str) and dep.startswith("$"):
                key = dep[1:]
                if key in step_id_map:
                    deps.append(step_id_map[key])
            else:
                deps.append(dep)

        step = TaskStep(
            step_id=step_id,
            agent=item.get("agent", ""),
            action=item.get("action", ""),
            params=item.get("params") or {},
            dependencies=deps,
            max_retries=item.get("max_retries", 0),
            max_replans=0
        )
        plan.steps.append(step)
        pending[step.step_id] = step
        new_step_ids.append(step.step_id)

        db.log_plan_step(
            plan_id=plan.task_id,
            step_id=step.step_id,
            agent=step.agent,
            action=step.action,
            status="pending",
            dependencies=step.dependencies,
            params=step.params
        )

        step_id_map[step.agent] = step.step_id

    return new_step_ids


def resolve_gap_deps(plan: TaskPlan, deps: List[Any]) -> List[str]:
    """Resolve '$agent' placeholders in gap-step dependencies."""
    if not deps:
        return []
    step_id_map: Dict[str, str] = {}
    for step in plan.steps:
        if step.agent and step.step_id:
            step_id_map[step.agent] = step.step_id
    resolved: List[str] = []
    for dep in deps:
        if isinstance(dep, str) and dep.startswith("$"):
            key = dep[1:]
            if key in step_id_map:
                resolved.append(step_id_map[key])
        else:
            resolved.append(dep)
    return resolved
