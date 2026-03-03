# -*- coding: utf-8 -*-
"""
IMCs System Functional Verification Script
===========================================
Comprehensive pre-remediation baseline test.
Tests every agent, the orchestrator, fusion engine, and LLM service.
Outputs a structured verification report.
"""
import os, sys, time, json, traceback, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

import logging
logging.basicConfig(level=logging.WARNING)  # suppress noise

# ============================================================
# Helpers
# ============================================================
PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
results = []

def record(module, test_name, status, detail=""):
    results.append((module, test_name, status, detail))
    tag = {"[PASS]": "OK", "[FAIL]": "XX", "[WARN]": "!!"}[status]
    print(f"  {tag}  {module} / {test_name}: {detail[:120]}")

# ============================================================
# 1. Database Layer
# ============================================================
print("\n=== 1. Database Layer ===")
try:
    from src.services.db.database import DatabaseService
    db = DatabaseService()
    record("DB", "init", PASS, f"db_path={db.db_path}")

    all_mats = db.list_materials(limit=5000)
    record("DB", "list_materials(5000)", PASS, f"returned {len(all_mats)} rows")

    mats_cif = db.list_materials(limit=5000, require_cif=True)
    record("DB", "list_materials(require_cif=True)", PASS, f"returned {len(mats_cif)} rows with CIF")

    if mats_cif:
        sample = mats_cif[0]
        detail = db.get_material_by_id(sample["material_id"])
        has_cif = bool(detail and detail.get("cif_path"))
        record("DB", "get_material_by_id", PASS if has_cif else WARN,
               f"id={sample['material_id']} has_cif={has_cif}")
    else:
        record("DB", "get_material_by_id", WARN, "no CIF materials to test")

except Exception as e:
    record("DB", "init", FAIL, str(e))

# ============================================================
# 2. Structure Featurizer
# ============================================================
print("\n=== 2. Structure Featurizer ===")
try:
    from src.services.chemistry.descriptors import StructureFeaturizer
    feat = StructureFeaturizer()
    record("Featurizer", "init", PASS)

    if mats_cif:
        cif_path = mats_cif[0].get("cif_path", "")
        if os.path.exists(cif_path):
            vec = feat.extract(cif_path)
            record("Featurizer", "extract", PASS if vec is not None else FAIL,
                   f"path={os.path.basename(cif_path)} dim={len(vec) if vec is not None else 0}")
        else:
            record("Featurizer", "extract", WARN, f"CIF path does not exist: {cif_path}")
    else:
        record("Featurizer", "extract", WARN, "no CIF material available")
except Exception as e:
    record("Featurizer", "init", FAIL, str(e))

# ============================================================
# 3. HOR Predictor
# ============================================================
print("\n=== 3. HOR Predictor ===")
try:
    from src.ml.hor_predictor import get_predictor
    pred = get_predictor()
    available = pred.is_available()
    record("HOR_Predictor", "init", PASS if available else WARN,
           f"is_available={available}")
except Exception as e:
    record("HOR_Predictor", "init", FAIL, str(e))

# ============================================================
# 4. Theory Agent (Protocol)
# ============================================================
print("\n=== 4. Theory Agent ===")
try:
    from src.agents.core.theory_agent import TheoryDataAgent, TheoryDataConfig
    from src.agents.protocol_impl import TheoryAgentProtocolMixin
    class TA(TheoryAgentProtocolMixin, TheoryDataAgent): pass
    ta = TA(TheoryDataConfig())
    record("TheoryAgent", "init", PASS)

    cap = ta.assess_capability("Recommend HOR alloy catalysts")
    record("TheoryAgent", "assess_capability", PASS,
           f"can={cap.can_contribute} conf={cap.confidence:.2f}")

    from src.agents.protocol import QueryContext
    ctx = QueryContext(user_query="Recommend HOR alloy catalysts", task_type="catalyst_discovery")
    contrib = ta.contribute(ctx)
    record("TheoryAgent", "contribute", PASS if contrib.success else FAIL,
           f"candidates={len(contrib.candidates)} reasoning={contrib.reasoning[:80]}")
except Exception as e:
    record("TheoryAgent", "init/test", FAIL, str(e))

# ============================================================
# 5. ML Agent (Protocol)
# ============================================================
print("\n=== 5. ML Agent ===")
try:
    from src.agents.core.ml_agent import MLAgent
    from src.agents.protocol_impl import MLAgentProtocolMixin
    class MA(MLAgentProtocolMixin, MLAgent): pass
    ma = MA()
    record("MLAgent", "init", PASS)

    cap = ma.assess_capability("Recommend HOR alloy catalysts")
    record("MLAgent", "assess_capability", PASS,
           f"can={cap.can_contribute} conf={cap.confidence:.2f}")

    ctx = QueryContext(user_query="Recommend HOR alloy catalysts", task_type="catalyst_discovery")
    t0 = time.time()
    contrib = ma.contribute(ctx)
    elapsed = time.time() - t0
    n_cand = len(contrib.candidates)
    
    # Check for uncertainty values
    has_uq = any(c.get("uncertainty") is not None for c in contrib.candidates[:5])
    record("MLAgent", "contribute", PASS if contrib.success and n_cand > 0 else FAIL,
           f"candidates={n_cand} has_UQ={has_uq} time={elapsed:.1f}s")

    if n_cand > 0:
        top = contrib.candidates[0]
        record("MLAgent", "top_candidate", PASS,
               f"formula={top.get('formula')} score={top.get('predicted_activity','?'):.4f} unc={top.get('uncertainty','?')}")
except Exception as e:
    record("MLAgent", "init/test", FAIL, traceback.format_exc()[-200:])

# ============================================================
# 6. Experiment Agent (Protocol)
# ============================================================
print("\n=== 6. Experiment Agent ===")
try:
    from src.agents.core.experiment_agent import ExperimentDataAgent
    from src.agents.protocol_impl import ExperimentAgentProtocolMixin
    class EA(ExperimentAgentProtocolMixin, ExperimentDataAgent): pass
    ea = EA()
    record("ExperimentAgent", "init", PASS)

    cap = ea.assess_capability("Recommend HOR alloy catalysts with high activity")
    record("ExperimentAgent", "assess_capability", PASS,
           f"can={cap.can_contribute} conf={cap.confidence:.2f} reason={cap.reason[:60]}")

    ctx = QueryContext(user_query="test", task_type="catalyst_discovery")
    contrib = ea.contribute(ctx)
    n_metrics = len(contrib.metrics) if contrib.metrics else 0
    n_cand = len(contrib.candidates)
    record("ExperimentAgent", "contribute", 
           PASS if contrib.success and (n_metrics > 0 or n_cand > 0) else WARN,
           f"success={contrib.success} candidates={n_cand} metrics={n_metrics} reasoning={contrib.reasoning[:60]}")
except Exception as e:
    record("ExperimentAgent", "init/test", FAIL, str(e)[:200])

# ============================================================
# 7. Literature Agent (Protocol)
# ============================================================
print("\n=== 7. Literature Agent ===")
try:
    from src.agents.core.literature_agent import LiteratureAgent
    from src.agents.protocol_impl import LiteratureAgentProtocolMixin
    class LA(LiteratureAgentProtocolMixin, LiteratureAgent): pass
    la = LA()
    record("LitAgent", "init", PASS)

    cap = la.assess_capability("Recommend HOR alloy catalysts")
    record("LitAgent", "assess_capability", PASS,
           f"can={cap.can_contribute} conf={cap.confidence:.2f}")

    ctx = QueryContext(user_query="Recommend HOR alloy catalysts", task_type="catalyst_discovery")
    t0 = time.time()
    contrib = la.contribute(ctx)
    elapsed = time.time() - t0
    n_insights = len(contrib.insights) if contrib.insights else 0
    n_cand = len(contrib.candidates)
    record("LitAgent", "contribute",
           PASS if n_insights > 0 or n_cand > 0 else WARN,
           f"insights={n_insights} candidates={n_cand} time={elapsed:.1f}s reasoning={contrib.reasoning[:60]}")
except Exception as e:
    record("LitAgent", "init/test", FAIL, str(e)[:200])

# ============================================================
# 8. FusionEngine
# ============================================================
print("\n=== 8. Fusion Engine ===")
try:
    from src.agents.orchestrator import FusionEngine
    from src.agents.protocol import AgentContribution, ContributionType
    fe = FusionEngine()
    
    # Simulate multi-agent contributions
    mock_contribs = {
        "theory": AgentContribution("theory", ContributionType.CANDIDATES, True,
            candidates=[{"material_id": "mp-126", "formula": "Pt", "formation_energy": 0.0}],
            confidence=0.8),
        "ml": AgentContribution("ml", ContributionType.PREDICTIONS, True,
            candidates=[
                {"material_id": "mp-126", "formula": "Pt", "predicted_activity": 0.95},
                {"material_id": "mp-999", "formula": "Ni3Al", "predicted_activity": 0.7}
            ],
            predictions={"mp-126": 0.95, "mp-999": 0.7},
            confidence=0.9),
    }
    
    fused = fe.synthesize(mock_contribs)
    record("FusionEngine", "synthesize", PASS if len(fused) > 0 else FAIL,
           f"output={len(fused)} candidates, top_score={fused[0].get('final_score',0):.3f}" if fused else "empty")
    
    # Check multi-source bonus
    mp126 = [c for c in fused if c["material_id"] == "mp-126"]
    if mp126:
        sources = mp126[0].get("sources", [])
        record("FusionEngine", "multi_source_bonus", PASS if len(sources) >= 2 else WARN,
               f"mp-126 sources={sources}")
except Exception as e:
    record("FusionEngine", "test", FAIL, str(e)[:200])

# ============================================================
# 9. AdvancedFusionEngine
# ============================================================
print("\n=== 9. Advanced Fusion Engine ===")
try:
    from src.agents.fusion import AdvancedFusionEngine, create_fusion_report
    afe = AdvancedFusionEngine()
    
    sorted_cands, explanations = afe.synthesize(mock_contribs)
    record("AdvFusion", "synthesize", PASS if len(sorted_cands) > 0 else FAIL,
           f"candidates={len(sorted_cands)} explanations={len(explanations)}")
    
    if explanations:
        exp0 = explanations[0]
        record("AdvFusion", "explanations", PASS,
               f"rank={exp0.rank} score={exp0.final_score:.3f} reasons={[r.value for r in exp0.reasons]}")
        
        report = create_fusion_report(explanations, top_n=3)
        record("AdvFusion", "report_gen", PASS if len(report) > 50 else WARN,
               f"report_length={len(report)} chars")
except Exception as e:
    record("AdvFusion", "test", FAIL, str(e)[:200])

# ============================================================
# 10. LLM Expert Reasoning Service
# ============================================================
print("\n=== 10. LLM Expert Reasoning ===")
try:
    from src.services.llm.expert_reasoning import ExpertReasoningService
    llm = ExpertReasoningService()
    record("LLM", "init", PASS if llm.available else WARN,
           f"available={llm.available} has_key={bool(llm.api_key)}")
    
    test_cands = [{"formula": "Pt3Ni", "properties": {"predicted_activity": 0.92, "uncertainty": 0.03, "formation_energy": -0.2}}]
    report = llm.generate_report(test_cands, is_active_learning=False)
    is_mock = "MOCK" in report
    record("LLM", "generate_report", WARN if is_mock else PASS,
           f"is_mock={is_mock} length={len(report)} chars")
except Exception as e:
    record("LLM", "test", FAIL, str(e)[:200])

# ============================================================
# 11. Full Orchestrator E2E
# ============================================================
print("\n=== 11. Full Orchestrator E2E ===")
try:
    from src.agents.orchestrator import AgentOrchestrator
    orch = AgentOrchestrator()
    record("Orchestrator", "init", PASS, f"agents={list(orch.agents.keys())}")

    t0 = time.time()
    result = orch.orchestrate("Recommend ordered alloy HOR catalysts containing Pt, Ru, or Ni with extremely high activity potential.", max_iterations=1)
    elapsed = time.time() - t0
    
    record("Orchestrator", "orchestrate", PASS if result.success else FAIL,
           f"success={result.success} candidates={len(result.candidates)} iter={result.iteration} time={elapsed:.1f}s")
    
    # Check which agents contributed
    for name, contrib in result.contributions.items():
        n = len(contrib.candidates) + len(contrib.predictions) + len(contrib.insights)
        status = PASS if contrib.success and n > 0 else (WARN if contrib.success else FAIL)
        record("Orchestrator", f"agent_{name}_contribution", status,
               f"success={contrib.success} items={n}")
    
    # Check Active Learning
    has_al = any("active_learning_reason" in c for c in result.candidates)
    record("Orchestrator", "active_learning_triggered", PASS if not has_al else WARN,
           f"triggered={has_al}")

    # Check iterate_with_feedback interface
    has_iterate = hasattr(orch, 'iterate_with_feedback')
    record("Orchestrator", "iterate_with_feedback_exists", PASS if has_iterate else FAIL)

    # Top 3 candidates
    if result.candidates:
        print("\n  --- Top 3 Candidates ---")
        for i, c in enumerate(result.candidates[:3]):
            f = c.get("formula", c.get("material_id", "?"))
            s = c.get("properties", {}).get("predicted_activity", c.get("final_score", "?"))
            u = c.get("properties", {}).get("uncertainty", "?")
            print(f"    [{i+1}] {f}  score={s}  uncertainty={u}")

except Exception as e:
    record("Orchestrator", "e2e_test", FAIL, traceback.format_exc()[-300:])

# ============================================================
# REPORT
# ============================================================
print("\n" + "=" * 80)
print("       F U N C T I O N A L   V E R I F I C A T I O N   R E P O R T")
print("=" * 80)

pass_count = sum(1 for r in results if r[2] == PASS)
warn_count = sum(1 for r in results if r[2] == WARN)
fail_count = sum(1 for r in results if r[2] == FAIL)
total = len(results)

print(f"\n  Total: {total}  |  PASS: {pass_count}  |  WARN: {warn_count}  |  FAIL: {fail_count}")
print(f"  Pass Rate: {pass_count/total*100:.0f}%  |  Health Score: {(pass_count + 0.5*warn_count)/total*100:.0f}%\n")

if fail_count > 0:
    print("  FAILURES:")
    for module, test, status, detail in results:
        if status == FAIL:
            print(f"    XX  {module}/{test}: {detail[:100]}")

if warn_count > 0:
    print("\n  WARNINGS:")
    for module, test, status, detail in results:
        if status == WARN:
            print(f"    !!  {module}/{test}: {detail[:100]}")

print("\n" + "=" * 80)
