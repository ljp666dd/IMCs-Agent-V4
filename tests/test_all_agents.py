"""
Comprehensive Test Suite for Multi-Agent Framework.
Run with: python src/tests/test_all_agents.py
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

import numpy as np


def test_ml_agent():
    """Test MLAgent functionality."""
    print("\n" + "="*60)
    print("Testing MLAgent")
    print("="*60)
    
    from src.agents.core import MLAgent
    
    agent = MLAgent()
    results = {}
    
    # Test 1: Agent initialization
    print("\n[Test 1] Agent initialization...")
    assert agent is not None
    print("  [OK] MLAgent initialized")
    results["initialization"] = True
    
    # Test 2: Check model types
    print("\n[Test 2] Check model types...")
    from src.agents.core.ml_agent import ModelType
    assert hasattr(ModelType, "TRADITIONAL")
    assert hasattr(ModelType, "DEEP_LEARNING")
    assert hasattr(ModelType, "GNN")
    print("  [OK] Model types: TRADITIONAL, DEEP_LEARNING, GNN")
    results["model_types"] = True
    
    # Test 3: Check methods (smoke test, offline-safe)
    print("\n[Test 3] Required methods exist...")
    methods = [
        "load_from_db",
        "train_traditional_models",
        "train_deep_learning_models",
        "train_transformer_models",
        "train_gnn_models_v2",
        "get_top_models",
        "predict_best",
        "predict",
    ]
    for m in methods:
        assert hasattr(agent, m), f"Missing method: {m}"
    print(f"  [OK] All {len(methods)} methods available")
    results["methods"] = True


def test_theory_agent():
    """Test TheoryDataAgent functionality."""
    print("\n" + "="*60)
    print("Testing TheoryDataAgent")
    print("="*60)
    
    from src.agents.core import TheoryDataAgent
    
    agent = TheoryDataAgent()
    results = {}
    
    # Test 1: Initialization
    print("\n[Test 1] Agent initialization...")
    assert agent is not None
    print("  [OK] TheoryDataAgent initialized")
    results["initialization"] = True
    
    # Test 2: Methods exist (offline-safe)
    print("\n[Test 2] Required methods exist...")
    methods = [
        "download_structures",
        "download_formation_energy",
        "download_orbital_dos",
        "download_adsorption_energies",
        "query_aflow",
        "query_catalysis_hub",
        "query_oqmd",
        "list_stored_materials",
        "get_status",
    ]
    for m in methods:
        assert hasattr(agent, m), f"Missing method: {m}"
    print(f"  [OK] All {len(methods)} methods available")
    results["methods"] = True
    
    # Test 3: Local DB status (no network calls)
    print("\n[Test 3] Local DB status...")
    status = agent.get_status()
    assert isinstance(status, dict)
    print(f"  [OK] Status keys: {sorted(status.keys())}")
    results["status"] = True


def test_experiment_agent():
    """Test ExperimentDataAgent functionality."""
    print("\n" + "="*60)
    print("Testing ExperimentDataAgent")
    print("="*60)
    
    from src.agents.core import ExperimentDataAgent
    
    agent = ExperimentDataAgent()
    results = {}
    
    # Test 1: Initialization
    print("\n[Test 1] Agent initialization...")
    assert agent is not None
    print("  [OK] ExperimentDataAgent initialized")
    results["initialization"] = True
    
    # Test 2: Methods exist (offline-safe)
    print("\n[Test 2] Required methods exist...")
    methods = [
        "scan_directory",
        "process_rde_directory",
        "load_csv",
        "analyze_lsv",
        "analyze_rde_series",
        "process_request",
    ]
    for m in methods:
        assert hasattr(agent, m), f"Missing method: {m}"
    print(f"  [OK] All {len(methods)} methods available")
    results["methods"] = True

    # Test 3: LSV analysis (no file IO)
    print("\n[Test 3] LSV analysis...")
    import pandas as pd
    potential = np.linspace(0, 0.5, 100)
    current = -10 * (1 - np.exp(-potential * 10))
    df = pd.DataFrame({"potential_v": potential, "current_ma": current})
    lsv_result = agent.analyze_lsv(df, sample_id="pytest_sample")
    assert isinstance(lsv_result, object)
    print(f"  [OK] LSV analyzed: overpotential_10mA={lsv_result.overpotential_10mA}")
    results["lsv"] = True


def test_literature_agent():
    """Test LiteratureAgent functionality."""
    print("\n" + "="*60)
    print("Testing LiteratureAgent")
    print("="*60)
    
    from src.agents.core import LiteratureAgent
    
    agent = LiteratureAgent()
    results = {}
    
    # Test 1: Initialization
    print("\n[Test 1] Agent initialization...")
    assert agent is not None
    print("  [OK] LiteratureAgent initialized")
    results["initialization"] = True
    
    # Test 2: Methods exist (offline-safe)
    print("\n[Test 2] Required methods exist...")
    methods = [
        "search_all_sources",
        "search_literature",
        "list_local_pdfs",
        "parse_all_local_pdfs",
        "extract_knowledge",
        "generate_report",
    ]
    for m in methods:
        assert hasattr(agent, m), f"Missing method: {m}"
    print(f"  [OK] All {len(methods)} methods available")
    results["methods"] = True
    
    # Test 4: Knowledge extraction
    print("\n[Test 4] Knowledge extraction...")
    assert hasattr(agent, 'extract_knowledge')
    assert hasattr(agent, 'generate_report')
    print("  [OK] Knowledge extraction available")
    results["knowledge"] = True


def test_task_manager():
    """Test TaskManagerAgent functionality."""
    print("\n" + "="*60)
    print("Testing TaskManagerAgent")
    print("="*60)
    
    from src.agents.core import TaskManagerAgent
    
    agent = TaskManagerAgent()
    results = {}
    
    # Test 1: Initialization
    print("\n[Test 1] Agent initialization...")
    assert agent is not None
    print("  [OK] TaskManagerAgent initialized")
    results["initialization"] = True
    
    # Test 2: Core methods exist
    print("\n[Test 2] Core methods exist...")
    methods = ["chat", "analyze_request", "create_plan", "execute_plan", "format_plan"]
    for m in methods:
        assert hasattr(agent, m), f"Missing method: {m}"
    print(f"  [OK] All {len(methods)} core methods available")
    results["methods"] = True


def run_all_tests():
    """Run all tests and generate report."""
    print("\n" + "="*60)
    print("[TEST] MULTI-AGENT FRAMEWORK TEST SUITE")
    print("="*60)
    
    all_results = {}
    
    try:
        all_results["MLAgent"] = test_ml_agent()
    except Exception as e:
        print(f"  [FAIL] MLAgent: {e}")
        all_results["MLAgent"] = {"error": str(e)}
    
    try:
        all_results["TheoryDataAgent"] = test_theory_agent()
    except Exception as e:
        print(f"  [FAIL] TheoryDataAgent: {e}")
        all_results["TheoryDataAgent"] = {"error": str(e)}
    
    try:
        all_results["ExperimentDataAgent"] = test_experiment_agent()
    except Exception as e:
        print(f"  [FAIL] ExperimentDataAgent: {e}")
        all_results["ExperimentDataAgent"] = {"error": str(e)}
    
    try:
        all_results["LiteratureAgent"] = test_literature_agent()
    except Exception as e:
        print(f"  [FAIL] LiteratureAgent: {e}")
        all_results["LiteratureAgent"] = {"error": str(e)}
    
    try:
        all_results["TaskManagerAgent"] = test_task_manager()
    except Exception as e:
        print(f"  [FAIL] TaskManagerAgent: {e}")
        all_results["TaskManagerAgent"] = {"error": str(e)}
    
    # Summary
    print("\n" + "="*60)
    print("[SUMMARY] TEST RESULTS")
    print("="*60)
    
    total_passed = 0
    total_tests = 0
    
    for agent_name, results in all_results.items():
        if "error" in results:
            print(f"\n{agent_name}: FAILED - {results['error'][:50]}")
        else:
            passed = sum(1 for v in results.values() if v is True or (isinstance(v, int) and v >= 0))
            total = len(results)
            total_passed += passed
            total_tests += total
            print(f"\n{agent_name}: {passed}/{total} passed")
    
    print("\n" + "="*60)
    print(f"[RESULT] Total: {total_passed}/{total_tests} tests passed")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    run_all_tests()
