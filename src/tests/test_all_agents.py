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
    
    # Test 3: Check GNN models
    print("\n[Test 3] GNN model classes...")
    try:
        from src.agents.core.ml_agent import SimpleCGCNN, SimpleSchNet, SimpleMEGNet
        print("  [OK] CGCNN, SchNet, MEGNet classes available")
        results["gnn_classes"] = True
    except ImportError as e:
        print(f"  [WARN] GNN classes: {e}")
        results["gnn_classes"] = False
    
    # Test 4: Check Transformer
    print("\n[Test 4] Transformer model class...")
    try:
        from src.agents.core.ml_agent import TransformerRegressor
        print("  [OK] TransformerRegressor available")
        results["transformer"] = True
    except ImportError:
        print("  [WARN] TransformerRegressor not available")
        results["transformer"] = False
    
    # Test 5: Check methods
    print("\n[Test 5] Required methods exist...")
    methods = ["train_traditional_models", "train_deep_learning_models", 
               "train_gnn_models", "train_transformer_models", "shap_analysis", 
               "get_top_n_models", "predict"]
    for m in methods:
        assert hasattr(agent, m), f"Missing method: {m}"
    print(f"  [OK] All {len(methods)} methods available")
    results["methods"] = True
    
    return results


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
    
    # Test 2: API Key
    print("\n[Test 2] API key configured...")
    assert len(agent.config.api_key) > 0
    print(f"  [OK] API key: {agent.config.api_key[:8]}...")
    results["api_key"] = True
    
    # Test 3: Methods exist
    print("\n[Test 3] Required methods exist...")
    methods = ["download_structures", "download_formation_energy", 
               "download_orbital_dos", "download_band_structure",
               "query_aflow", "query_catalysis_hub", "query_oqmd"]
    for m in methods:
        assert hasattr(agent, m), f"Missing method: {m}"
    print(f"  [OK] All {len(methods)} methods available")
    results["methods"] = True
    
    # Test 4: Load existing data
    print("\n[Test 4] Load existing data...")
    structures = agent.load_structures()
    fe_data = agent.load_formation_energy()
    dos_data = agent.load_orbital_dos()
    print(f"  [OK] CIF: {len(structures)}, FE: {len(fe_data)}, DOS: {len(dos_data)}")
    results["load_data"] = len(structures) + len(fe_data) + len(dos_data)
    
    return results


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
    
    # Test 2: LSV analysis
    print("\n[Test 2] LSV analysis...")
    potential = np.linspace(0, 0.5, 100)
    current = -10 * (1 - np.exp(-potential * 10))
    lsv_result = agent.analyze_lsv(potential, current)
    print(f"  [OK] LSV: overpotential = {lsv_result.overpotential_10mA:.3f} V")
    results["lsv"] = True
    
    # Test 3: CV analysis
    print("\n[Test 3] CV analysis...")
    cv_potential = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0, 50)])
    cv_current = np.sin(cv_potential * np.pi * 2) * 0.1
    cv_result = agent.analyze_cv(cv_potential, cv_current)
    print(f"  [OK] CV: ECSA = {cv_result.ecsa:.3f}")
    results["cv"] = True
    
    # Test 4: RDE & RRDE methods
    print("\n[Test 4] RDE/RRDE methods...")
    assert hasattr(agent, 'analyze_rde')
    assert hasattr(agent, 'analyze_rrde')
    rde_result = agent.analyze_rde(potential, current)
    print(f"  [OK] RDE: j_limit = {rde_result['limiting_current_density']:.3f}")
    results["rde_rrde"] = True
    
    # Test 5: File parsers
    print("\n[Test 5] File parser methods...")
    assert hasattr(agent, 'load_eclab_mpt')
    assert hasattr(agent, 'load_chi_txt')
    print("  [OK] EC-Lab and CHI parsers available")
    results["parsers"] = True
    
    return results


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
    
    # Test 2: Search methods
    print("\n[Test 2] Search methods exist...")
    assert hasattr(agent, 'search_semantic_scholar')
    assert hasattr(agent, 'search_arxiv')
    assert hasattr(agent, 'search_all_sources')
    print("  [OK] All search methods available")
    results["search_methods"] = True
    
    # Test 3: PDF methods
    print("\n[Test 3] PDF methods exist...")
    assert hasattr(agent, 'parse_pdf')
    assert hasattr(agent, 'parse_all_local_pdfs')
    assert hasattr(agent, 'list_local_pdfs')
    print("  [OK] PDF parsing methods available")
    results["pdf_methods"] = True
    
    # Test 4: Knowledge extraction
    print("\n[Test 4] Knowledge extraction...")
    assert hasattr(agent, 'extract_knowledge')
    assert hasattr(agent, 'generate_report')
    print("  [OK] Knowledge extraction available")
    results["knowledge"] = True
    
    return results


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
    
    # Test 2: Sub-agents loaded
    print("\n[Test 2] Sub-agents loaded...")
    assert agent.ml_agent is not None
    assert agent.theory_agent is not None
    assert agent.experiment_agent is not None
    assert agent.literature_agent is not None
    print("  [OK] All 4 sub-agents loaded")
    results["sub_agents"] = True
    
    # Test 3: Core methods
    print("\n[Test 3] Core methods exist...")
    methods = ["chat", "analyze_request", "create_plan", "execute_plan"]
    for m in methods:
        assert hasattr(agent, m), f"Missing method: {m}"
    print(f"  [OK] All {len(methods)} core methods available")
    results["methods"] = True
    
    return results


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
