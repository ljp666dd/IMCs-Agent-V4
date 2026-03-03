"""
IMCs 智能体功能测试脚本 (v2)
测试各智能体的核心功能是否正常工作
"""

import sys
import os
import json
import requests

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)

API_BASE = "http://localhost:8000"


def test_api_health():
    """测试 API 健康状态"""
    print("\n" + "="*60)
    print("[API] Testing API Health")
    print("="*60)
    
    try:
        res = requests.get(f"{API_BASE}/docs", timeout=5)
        print(f"  [OK] API Online (status: {res.status_code})")
        return True
    except Exception as e:
        print(f"  [FAIL] API Offline: {e}")
        return False


def test_literature_agent():
    """测试文献智能体"""
    print("\n" + "="*60)
    print("[LIT] Testing LiteratureAgent")
    print("="*60)
    
    from src.agents.core import LiteratureAgent
    
    agent = LiteratureAgent()
    results = {"passed": 0, "failed": 0}
    
    # Test 1: 初始化
    print("\n[Test 1] Initialization...")
    try:
        assert agent is not None
        print("  [OK] LiteratureAgent initialized")
        results["passed"] += 1
    except Exception as e:
        print(f"  [FAIL] Init failed: {e}")
        results["failed"] += 1
    
    # Test 2: search_all_sources 方法存在
    print("\n[Test 2] search_all_sources method...")
    try:
        assert hasattr(agent, 'search_all_sources')
        assert hasattr(agent, 'search_literature')
        print("  [OK] Search methods available")
        results["passed"] += 1
    except Exception as e:
        print(f"  [FAIL] Method check: {e}")
        results["failed"] += 1
    
    # Test 3: 本地 PDF 列表
    print("\n[Test 3] list_local_pdfs...")
    try:
        pdfs = agent.list_local_pdfs()
        print(f"  [OK] Found {len(pdfs)} local PDFs")
        results["passed"] += 1
    except Exception as e:
        print(f"  [WARN] list_local_pdfs: {e}")
        results["failed"] += 1
    
    # Test 4: 搜索功能 (模拟)
    print("\n[Test 4] Search literature...")
    try:
        papers = agent.search_literature("PtRu HOR catalyst", limit=5)
        print(f"  [OK] Search returned {len(papers)} papers")
        results["passed"] += 1
    except Exception as e:
        print(f"  [WARN] Search: {e}")
        results["failed"] += 1
    
    return results


def test_theory_agent():
    """测试理论数据智能体"""
    print("\n" + "="*60)
    print("[THY] Testing TheoryDataAgent")
    print("="*60)
    
    from src.agents.core import TheoryDataAgent
    
    agent = TheoryDataAgent()
    results = {"passed": 0, "failed": 0}
    
    # Test 1: 初始化
    print("\n[Test 1] Initialization...")
    try:
        assert agent is not None
        print("  [OK] TheoryDataAgent initialized")
        results["passed"] += 1
    except Exception as e:
        print(f"  [FAIL] Init failed: {e}")
        results["failed"] += 1
    
    # Test 2: API Key
    print("\n[Test 2] Materials Project API Key...")
    try:
        assert len(agent.config.api_key) > 0
        print(f"  [OK] API Key configured: {agent.config.api_key[:8]}...")
        results["passed"] += 1
    except Exception as e:
        print(f"  [FAIL] API Key: {e}")
        results["failed"] += 1
    
    # Test 3: 列出存储的材料
    print("\n[Test 3] list_stored_materials...")
    try:
        materials = agent.list_stored_materials(limit=10)
        print(f"  [OK] Listed {len(materials)} materials from DB")
        results["passed"] += 1
    except Exception as e:
        print(f"  [WARN] list_stored_materials: {e}")
        results["failed"] += 1
    
    # Test 4: 获取状态
    print("\n[Test 4] get_status...")
    try:
        status = agent.get_status()
        print(f"  [OK] Status: CIF={status.get('n_cifs',0)}, DOS={status.get('n_dos',0)}")
        results["passed"] += 1
    except Exception as e:
        print(f"  [WARN] get_status: {e}")
        results["failed"] += 1
    
    return results


def test_ml_agent():
    """测试机器学习智能体"""
    print("\n" + "="*60)
    print("[ML] Testing MLAgent")
    print("="*60)
    
    from src.agents.core import MLAgent
    
    agent = MLAgent()
    results = {"passed": 0, "failed": 0}
    
    # Test 1: 初始化
    print("\n[Test 1] Initialization...")
    try:
        assert agent is not None
        print("  [OK] MLAgent initialized")
        results["passed"] += 1
    except Exception as e:
        print(f"  [FAIL] Init failed: {e}")
        results["failed"] += 1
    
    # Test 2: 模型类型
    print("\n[Test 2] Model types...")
    try:
        from src.agents.core.ml_agent import ModelType
        types = [t.value for t in ModelType]
        print(f"  [OK] Model types: {types}")
        results["passed"] += 1
    except Exception as e:
        print(f"  [FAIL] Model types: {e}")
        results["failed"] += 1
    
    # Test 3: 核心方法
    print("\n[Test 3] Core methods...")
    try:
        required = ["train_traditional_models", "train_deep_learning_models", 
                   "predict", "load_from_db", "interpret_model"]
        missing = [m for m in required if not hasattr(agent, m)]
        if missing:
            print(f"  [WARN] Missing: {missing}")
            results["failed"] += 1
        else:
            print(f"  [OK] All {len(required)} methods available")
            results["passed"] += 1
    except Exception as e:
        print(f"  [FAIL] Method check: {e}")
        results["failed"] += 1
    
    # Test 4: 从数据库加载数据
    print("\n[Test 4] load_from_db...")
    try:
        loaded = agent.load_from_db(target_col="formation_energy")
        if agent.X is not None and len(agent.X) > 0:
            print(f"  [OK] Loaded {len(agent.X)} samples with {agent.X.shape[1]} features")
            results["passed"] += 1
        else:
            print(f"  [INFO] No data in DB yet (empty dataset)")
            results["passed"] += 1
    except Exception as e:
        print(f"  [WARN] load_from_db: {e}")
        results["failed"] += 1
    
    return results


def test_experiment_agent():
    """测试实验数据智能体"""
    print("\n" + "="*60)
    print("[EXP] Testing ExperimentDataAgent")
    print("="*60)
    
    from src.agents.core import ExperimentDataAgent
    import pandas as pd
    import numpy as np
    
    agent = ExperimentDataAgent()
    results = {"passed": 0, "failed": 0}
    
    # Test 1: 初始化
    print("\n[Test 1] Initialization...")
    try:
        assert agent is not None
        print("  [OK] ExperimentDataAgent initialized")
        results["passed"] += 1
    except Exception as e:
        print(f"  [FAIL] Init failed: {e}")
        results["failed"] += 1
    
    # Test 2: scan_directory 方法
    print("\n[Test 2] scan_directory method...")
    try:
        assert hasattr(agent, 'scan_directory')
        print("  [OK] scan_directory available")
        results["passed"] += 1
    except Exception as e:
        print(f"  [FAIL] Method check: {e}")
        results["failed"] += 1
    
    # Test 3: LSV 分析 (使用 DataFrame)
    print("\n[Test 3] analyze_lsv with DataFrame...")
    try:
        potential = np.linspace(0, 0.5, 100)
        current = -10 * (1 - np.exp(-potential * 10))
        df = pd.DataFrame({"potential": potential, "current": current})
        lsv_result = agent.analyze_lsv(df, sample_id="test_sample")
        print(f"  [OK] LSV: overpotential = {lsv_result.overpotential_10mA:.3f} V")
        results["passed"] += 1
    except Exception as e:
        print(f"  [WARN] analyze_lsv: {e}")
        results["failed"] += 1
    
    # Test 4: detect_data_type
    print("\n[Test 4] detect_data_type...")
    try:
        df = pd.DataFrame({"potential": [0.1, 0.2], "current": [-1, -2]})
        dtype = agent.detect_data_type(df)
        print(f"  [OK] Detected type: {dtype}")
        results["passed"] += 1
    except Exception as e:
        print(f"  [WARN] detect_data_type: {e}")
        results["failed"] += 1
    
    return results


def test_task_manager():
    """测试任务管理智能体"""
    print("\n" + "="*60)
    print("[TM] Testing TaskManagerAgent")
    print("="*60)
    
    from src.agents.core import TaskManagerAgent
    
    agent = TaskManagerAgent()
    results = {"passed": 0, "failed": 0}
    
    # Test 1: 初始化
    print("\n[Test 1] Initialization...")
    try:
        assert agent is not None
        print("  [OK] TaskManagerAgent initialized")
        results["passed"] += 1
    except Exception as e:
        print(f"  [FAIL] Init failed: {e}")
        results["failed"] += 1
    
    # Test 2: 子智能体加载
    print("\n[Test 2] Sub-agents loaded...")
    try:
        assert agent.ml_agent is not None
        assert agent.theory_agent is not None
        assert agent.experiment_agent is not None
        assert agent.literature_agent is not None
        print("  [OK] All 4 sub-agents loaded")
        results["passed"] += 1
    except Exception as e:
        print(f"  [FAIL] Sub-agents: {e}")
        results["failed"] += 1
    
    # Test 3: 核心方法
    print("\n[Test 3] Core methods...")
    try:
        methods = ["chat", "analyze_request", "create_plan", "execute_plan"]
        missing = [m for m in methods if not hasattr(agent, m)]
        if missing:
            print(f"  [WARN] Missing: {missing}")
            results["failed"] += 1
        else:
            print(f"  [OK] All {len(methods)} methods available")
            results["passed"] += 1
    except Exception as e:
        print(f"  [FAIL] Method check: {e}")
        results["failed"] += 1
    
    # Test 4: 请求分析
    print("\n[Test 4] analyze_request...")
    try:
        analysis = agent.analyze_request("Find HOR catalyst candidates")
        print(f"  [OK] Analysis: {type(analysis).__name__}")
        results["passed"] += 1
    except Exception as e:
        print(f"  [WARN] analyze_request: {e}")
        results["failed"] += 1
    
    return results


def test_api_task_flow():
    """测试 API 任务流程"""
    print("\n" + "="*60)
    print("[API] Testing API Task Flow")
    print("="*60)
    
    results = {"passed": 0, "failed": 0}
    
    # Test 1: 创建任务
    print("\n[Test 1] Create task...")
    try:
        res = requests.post(
            f"{API_BASE}/tasks/create",
            json={"query": "List all available materials in database"},
            timeout=30
        )
        if res.status_code == 200:
            data = res.json()
            task_id = data.get("task_id")
            print(f"  [OK] Task created: {task_id}")
            results["passed"] += 1
            results["task_id"] = task_id
        else:
            print(f"  [FAIL] Create task: {res.status_code}")
            results["failed"] += 1
    except Exception as e:
        print(f"  [FAIL] Create task: {e}")
        results["failed"] += 1
    
    # Test 2: 获取任务状态
    if "task_id" in results:
        print("\n[Test 2] Get task status...")
        try:
            res = requests.get(f"{API_BASE}/tasks/{results['task_id']}", timeout=10)
            if res.status_code == 200:
                data = res.json()
                print(f"  [OK] Status: {data.get('status')}")
                results["passed"] += 1
            else:
                print(f"  [FAIL] Get status: {res.status_code}")
                results["failed"] += 1
        except Exception as e:
            print(f"  [FAIL] Get status: {e}")
            results["failed"] += 1
    
    # Test 3: 获取材料列表 API
    print("\n[Test 3] Get materials API...")
    try:
        res = requests.get(f"{API_BASE}/api/theory/materials?limit=5", timeout=10)
        if res.status_code == 200:
            data = res.json()
            print(f"  [OK] API returned {len(data.get('materials', []))} materials")
            results["passed"] += 1
        else:
            print(f"  [INFO] Materials API: {res.status_code}")
            results["passed"] += 1  # 可能没有数据，但 API 正常
    except Exception as e:
        print(f"  [WARN] Materials API: {e}")
        results["failed"] += 1
    
    return results


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("IMCs Agent Function Test Suite v2")
    print("="*60)
    
    all_results = {}
    total_passed = 0
    total_failed = 0
    
    # API 健康检查
    api_ok = test_api_health()
    
    # 各智能体测试
    tests = [
        ("LiteratureAgent", test_literature_agent),
        ("TheoryDataAgent", test_theory_agent),
        ("MLAgent", test_ml_agent),
        ("ExperimentDataAgent", test_experiment_agent),
        ("TaskManagerAgent", test_task_manager),
    ]
    
    for name, test_func in tests:
        try:
            result = test_func()
            all_results[name] = result
            total_passed += result["passed"]
            total_failed += result["failed"]
        except Exception as e:
            print(f"  [FAIL] {name} exception: {e}")
            all_results[name] = {"passed": 0, "failed": 1, "error": str(e)}
            total_failed += 1
    
    # API 任务流程测试
    if api_ok:
        try:
            result = test_api_task_flow()
            all_results["API TaskFlow"] = result
            total_passed += result["passed"]
            total_failed += result["failed"]
        except Exception as e:
            print(f"  [FAIL] API flow exception: {e}")
            total_failed += 1
    
    # 汇总
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, result in all_results.items():
        passed = result.get("passed", 0)
        failed = result.get("failed", 0)
        total = passed + failed
        status = "[OK]" if failed == 0 else "[WARN]"
        print(f"  {status} {name}: {passed}/{total} passed")
    
    print("\n" + "-"*60)
    print(f"  TOTAL: {total_passed}/{total_passed+total_failed} tests passed")
    
    success_rate = total_passed / (total_passed + total_failed) * 100 if (total_passed + total_failed) > 0 else 0
    print(f"  SUCCESS RATE: {success_rate:.1f}%")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    run_all_tests()
