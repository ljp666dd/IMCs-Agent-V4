import json
import os
from datetime import datetime

from src.services.db.database import DatabaseService
from src.services.task.eval_suite import EvalSuiteConfig, build_eval_report


def _cleanup_db_files(db_path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(db_path + suffix)
        except Exception:
            pass


def _cleanup_dir(path: str) -> None:
    try:
        for name in os.listdir(path):
            try:
                os.remove(os.path.join(path, name))
            except Exception:
                pass
        os.rmdir(path)
    except Exception:
        pass


def test_eval_suite_builds_report_with_topk_and_rag_metrics():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    db_path = os.path.join("data", f"test_eval_suite_{stamp}.db")
    knowledge_dir = os.path.join("data", "analysis", f"eval_suite_packs_{stamp}")
    os.makedirs(knowledge_dir, exist_ok=True)

    db = DatabaseService(db_path=db_path)
    try:
        plan_ok = f"eval_suite_ok_{stamp}"
        plan_bad = f"eval_suite_bad_{stamp}"

        db.create_plan(plan_ok, user_id=None, task_type="catalyst_discovery", description="ok")
        db.create_plan(plan_bad, user_id=None, task_type="catalyst_discovery", description="bad")
        db.update_plan_status(plan_ok, "completed")
        db.update_plan_status(plan_bad, "failed")

        db.log_plan_step(plan_ok, "step_1", agent="task_manager", action="recommend", status="completed", result={"x": 1})
        db.log_plan_step(plan_bad, "step_1", agent="task_manager", action="recommend", status="failed", result=None, error="boom")

        pack_ok = {
            "ranking_current": [{"material_id": "mp-1"}, {"material_id": "mp-2"}],
            "candidate_material_ids": ["mp-1", "mp-2", "mp-3"],
            "knowledge_rag": [
                {
                    "material_id": "mp-1",
                    "results": [
                        {
                            "id": 123,
                            "source_type": "literature",
                            "source_id": "doi:10.1/xxx",
                            "title": "paper",
                            "url": "https://example.com",
                            "year": 2024,
                            "score": 0.9,
                        }
                    ],
                }
            ],
            "evidence_stats_before_gap": {
                "total_materials": 10,
                "formation_energy_count": 2,
                "evidence_by_source": {"literature": 1},
            },
            "evidence_stats_after_gap": {
                "total_materials": 10,
                "formation_energy_count": 3,
                "evidence_by_source": {"literature": 2},
            },
            "evaluation_metrics": {"version": "1.0", "plan_status": "completed"},
        }
        with open(os.path.join(knowledge_dir, f"knowledge_{plan_ok}.json"), "w", encoding="utf-8") as f:
            json.dump(pack_ok, f, ensure_ascii=False, indent=2)

        cfg = EvalSuiteConfig(
            db_path=db_path,
            knowledge_dir=knowledge_dir,
            recent=10,
            ks=(1, 2),
            ground_truth_ids={"mp-1", "mp-x"},
        )
        report = build_eval_report(cfg)

        assert report.get("summary", {}).get("plan_count") == 2
        assert report.get("summary", {}).get("status_counts", {}).get("completed") == 1
        assert report.get("summary", {}).get("status_counts", {}).get("failed") == 1
        assert report.get("summary", {}).get("success_rate") == 0.5
        assert report.get("summary", {}).get("terminal_success_rate") == 0.5
        assert report.get("summary", {}).get("blocked_ratio") == 0.0
        assert report.get("summary", {}).get("knowledge_pack_present_count") == 1
        assert report.get("summary", {}).get("knowledge_pack_missing_count") == 1
        integrity = report.get("summary", {}).get("data_integrity_stats") or {}
        assert integrity.get("foreign_keys_enabled") in (True, 1)
        assert integrity.get("evidence_orphan_rows") == 0
        assert integrity.get("activity_metric_orphan_rows") == 0
        assert integrity.get("adsorption_orphan_rows") == 0

        plans = {p.get("plan_id"): p for p in report.get("plans") or []}
        ok = plans.get(plan_ok) or {}
        assert ok.get("candidate_count") == 2  # ranking_current overrides candidate_material_ids
        assert ok.get("knowledge_pack_present") is True
        assert (plans.get(plan_bad) or {}).get("knowledge_pack_present") is False

        recall = ok.get("topk_recall") or {}
        assert recall.get("1") == 0.5
        assert recall.get("2") == 0.5

        rag = ok.get("rag_metrics") or {}
        assert rag.get("rag_results_count") == 1
        assert rag.get("rag_unique_sources") == 1
        assert rag.get("rag_avg_score") == 0.9
        assert rag.get("rag_reference_rate") == 1.0

        ev_delta = ok.get("evidence_delta") or {}
        assert ev_delta.get("formation_energy_count") == 1
        assert (ev_delta.get("evidence_by_source") or {}).get("literature") == 1
    finally:
        _cleanup_db_files(db_path)
        _cleanup_dir(knowledge_dir)
