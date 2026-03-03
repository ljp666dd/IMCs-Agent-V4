import argparse
import json
import os
from datetime import datetime
from typing import List

from src.services.task.types import TaskPlan, TaskStep, TaskType
from src.services.task.executor_factory import new_plan_executor


def build_steps(query: str, theory_limit: int, include_literature: bool, include_ml: bool) -> List[TaskStep]:
    steps: List[TaskStep] = []

    def add_step(agent: str, action: str, params: dict, deps: List[str]) -> TaskStep:
        step_id = f"step_{len(steps) + 1}"
        step = TaskStep(step_id=step_id, agent=agent, action=action, params=params, dependencies=deps)
        steps.append(step)
        return step

    lit_extract = None
    if include_literature:
        lit_search = add_step(
            "literature",
            "search",
            {"query": query, "limit": 5},
            []
        )
        lit_extract = add_step(
            "literature",
            "extract_knowledge",
            {"topic": query},
            [lit_search.step_id]
        )

    theory_step = add_step(
        "theory",
        "download",
        {"data_types": ["cif", "formation_energy"], "limit": theory_limit},
        []
    )

    ml_step = None
    if include_ml:
        ml_step = add_step(
            "ml",
            "train",
            {"include_deep_learning": False},
            [theory_step.step_id]
        )

    deps = []
    if lit_extract:
        deps.append(lit_extract.step_id)
    if ml_step:
        deps.append(ml_step.step_id)

    add_step("task_manager", "recommend", {}, deps)
    add_step("task_manager", "knowledge_pack", {}, deps)

    return steps


def main():
    parser = argparse.ArgumentParser(description="IMCs quick E2E verification runner")
    parser.add_argument("--query", default="HOR ordered alloy", help="Query text")
    parser.add_argument("--theory-limit", type=int, default=20, help="Theory download limit")
    parser.add_argument("--no-literature", action="store_true", help="Skip literature steps")
    parser.add_argument("--no-ml", action="store_true", help="Skip ML training")
    parser.add_argument("--report-dir", default=os.path.join("data", "tasks"), help="Report output directory")
    args = parser.parse_args()

    executor = new_plan_executor()
    executor.max_adaptive_rounds = 0

    task_id = f"e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    plan = TaskPlan(
        task_id=task_id,
        task_type=TaskType.CATALYST_DISCOVERY,
        description=f"E2E verify: {args.query}",
        steps=build_steps(
            query=args.query,
            theory_limit=args.theory_limit,
            include_literature=not args.no_literature,
            include_ml=not args.no_ml,
        ),
    )

    results = executor.execute_plan(plan)

    os.makedirs(args.report_dir, exist_ok=True)
    report_path = os.path.join(args.report_dir, f"e2e_report_{task_id}.json")
    summary = {
        "task_id": task_id,
        "status": plan.status,
        "has_evidence_gap": bool(plan.results.get("evidence_gap")),
        "ranking_current": bool(plan.results.get("ranking_current")),
        "ranking_metric": plan.results.get("ranking_metric"),
        "knowledge_pack": os.path.join("data", "tasks", f"knowledge_{task_id}.json"),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("E2E done:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
