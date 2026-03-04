"""
IMCs Step Dispatcher — 步骤分发器

从 executor.py 拆分，负责：
将 TaskStep 分发到对应的 Agent 方法调用
"""

import math
from typing import Dict, Any

from src.core.logger import get_logger

logger = get_logger(__name__)


def dispatch_step(step, agents: Dict[str, Any], active_plan=None, db=None) -> Any:
    """Dispatch single step (TaskStep object) to the appropriate agent."""
    agent_name = step.agent
    action = step.action
    params = step.params

    if agent_name not in agents and agent_name != "task_manager":
        return {"error": f"Unknown agent: {agent_name}"}

    target_agent = agents.get(agent_name)

    if agent_name == "literature":
        if action == "search":
            return target_agent.search_all_sources(params.get("query", ""), params.get("limit"))
        elif action == "extract_knowledge":
            return target_agent.extract_knowledge(params.get("topic", ""))
        elif action == "harvest_hor_seed":
            return target_agent.harvest_hor_seed(
                query=params.get("query", ""),
                limit=params.get("limit", 10),
                max_pdfs=params.get("max_pdfs", 5),
                min_elements=params.get("min_elements", 2),
                persist=bool(params.get("persist", False)),
            )
        elif action == "ingest_local_library":
            return target_agent.ingest_local_library(
                min_elements=params.get("min_elements", 2)
            )

    elif agent_name == "theory":
        if action == "load_data":
            return target_agent.get_status()
        elif action == "download":
            data_types = params.get("data_types") or []
            limit = params.get("limit", 50)
            if not data_types or "cif" in data_types:
                target_agent.download_structures(limit=limit)
            if "formation_energy" in data_types:
                target_agent.download_formation_energy()
            if "dos" in data_types:
                mats = target_agent.list_stored_materials(limit=20)
                mat_ids = [
                    m.get("material_id")
                    for m in mats
                    if m.get("material_id") and str(m.get("material_id")).startswith("mp-")
                ]
                if mat_ids:
                    target_agent.download_orbital_dos(material_ids=mat_ids)
            if "adsorption" in data_types:
                try:
                    target_agent.download_adsorption_energies(adsorbates=["H*", "OH*"], limit=limit)
                except TypeError:
                    target_agent.download_adsorption_energies()
            return {"message": "Downloaded theory data to DB", "data_types": data_types}

    elif agent_name == "experiment":
        if action == "process":
            data_dir = params.get("data_dir") if isinstance(params, dict) else None
            reference_potential = params.get("reference_potential", 0.2) if isinstance(params, dict) else 0.2
            loading_mg_cm2 = params.get("loading_mg_cm2", 0.25) if isinstance(params, dict) else 0.25
            precious_fraction = params.get("precious_fraction", 0.20) if isinstance(params, dict) else 0.20
            return target_agent.process_rde_directory(
                data_dir=data_dir or "data/experimental/rde_lsv",
                reference_potential=reference_potential,
                loading_mg_cm2=loading_mg_cm2,
                precious_fraction=precious_fraction,
            )

    elif agent_name == "ml":
        if action == "seed_predictions":
            preds: Dict[str, float] = {}
            if isinstance(params, dict):
                raw = params.get("predictions") or {}
                if isinstance(raw, dict):
                    for mid, score in raw.items():
                        mid_str = (str(mid) if mid is not None else "").strip()
                        if not mid_str:
                            continue
                        try:
                            val = float(score)
                        except Exception:
                            continue
                        if not math.isfinite(val):
                            continue
                        preds[mid_str] = val
            return {"models": [], "predictions": preds}

        # AUTO-LOAD DATA from DB before training
        target_col = params.get("target_col") if isinstance(params, dict) else None
        if isinstance(target_col, str) and target_col.startswith("activity_metric:"):
            metric = target_col.split(":", 1)[1] or "exchange_current_density"
            target_agent.load_activity_metrics_from_db(metric)
        else:
            target_agent.load_from_db(target_col=target_col or "formation_energy")

        if action == "train":
            results = target_agent.train_traditional_models()
            models = [
                {
                    "name": r.name,
                    "r2_test": r.r2_test,
                    "rmse_test": r.rmse_test
                } for r in results
            ]
            pred_map = target_agent.predict_best()
            return {"models": models, "predictions": pred_map}
        elif action == "train_all":
            r1 = target_agent.train_traditional_models()
            top_models = target_agent.get_top_models(k=3)
            top_summary = [f"{m.name} (R2={m.r2_test:.3f})" for m in top_models]
            pred_map = target_agent.predict_best()
            return {"trained": len(r1), "top_3": top_summary, "predictions": pred_map}
        elif action == "shap_analysis":
            return {"message": "SHAP analysis ready"}
        elif action == "predict":
            return {"message": "Prediction ready"}

    elif agent_name == "task_manager":
        if action == "recommend":
            return _build_recommendation(agents)
        elif action == "knowledge_pack":
            return _build_knowledge_pack(active_plan, agents, db)
        elif action == "summarize":
            return {"summary": "Task completed."}

    return {"status": "executed", "action": action}


def _build_recommendation(agents: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesize recommendation report from agent states."""
    lit_agent = agents.get("literature")
    ml_agent = agents.get("ml")
    theory_agent = agents.get("theory")

    report = []
    report.append("### 🔬 Research Synthesis")

    if lit_agent and hasattr(lit_agent, 'papers') and lit_agent.papers:
        titles = [p.title for p in lit_agent.papers[:3]]
        report.append(f"**📚 Literature**: Analyzed {len(lit_agent.papers)} papers. Key sources:\n" + "\n".join([f"- *{t}*" for t in titles]))

    if theory_agent:
        status = theory_agent.get_status()
        cif_count = status.get("cif_files", 0)
        report.append(f"**💎 Theory Data**: Integrated {cif_count} crystal structures from Materials Project.")

    if ml_agent:
        top = ml_agent.get_top_models(k=3)
        if top:
            best = top[0]
            report.append(f"**🤖 AI Models**: Trained {len(ml_agent.results)} candidates. Best Performer:\n" +
                          f"- **{best.name}** (R²={best.r2_test:.3f})")
            if best.r2_test > 0.7:
                report.append(f"> ✅ High confidence model detected. Ready for prediction tasks.")
            else:
                report.append(f"> ⚠️ Model performance needs improvement (R²<0.7). Suggest adding more data.")

    report.append("\n### 💡 Final Recommendation")
    report.append("1. **Experiment**: Proceed with synthesis of materials identified in the downloaded CIF set.")
    report.append("2. **Analysis**: Use the trained High-Performance Model to screen new candidates.")
    report.append("3. **Next Step**: Upload experimental characterization data to `/experiments` for validation.")

    return {"recommendation": "\n\n".join(report)}


def _build_knowledge_pack(active_plan, agents, db) -> Dict[str, Any]:
    """Build and persist the knowledge pack."""
    import json
    import os
    try:
        plan = active_plan
        try:
            from src.agents.core.theory_agent import TheoryDataConfig
            allowed = TheoryDataConfig().elements
        except Exception:
            allowed = None
        pack = {
            "task_id": plan.task_id if plan else None,
            "query": plan.description if plan else None,
            "task_type": plan.task_type.value if plan else None,
            "evidence_stats": db.get_evidence_stats(allowed_elements=allowed),
            "knowledge_rag": (plan.results.get("knowledge_rag") if plan else []),
            "reasoning_report": (plan.results.get("reasoning_report") if plan else []),
            "evidence_gap": (plan.results.get("evidence_gap") if plan else None),
            "candidate_material_ids": (plan.results.get("candidate_material_ids") if plan else []),
            "ranking_before_gap": (plan.results.get("ranking_before_gap") if plan else []),
            "ranking_after_gap": (plan.results.get("ranking_after_gap") if plan else []),
            "ranking_current": (plan.results.get("ranking_current") if plan else []),
            "ranking_metric": (plan.results.get("ranking_metric") if plan else None),
            "evaluation_metrics": (plan.results.get("evaluation_metrics") if plan else None),
        }
        out_dir = os.path.join("data", "tasks")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"knowledge_{plan.task_id if plan else 'unknown'}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pack, f, ensure_ascii=False, indent=2)
        return {"knowledge_pack": pack, "path": out_path}
    except Exception as e:
        return {"error": f"knowledge_pack failed: {e}"}
