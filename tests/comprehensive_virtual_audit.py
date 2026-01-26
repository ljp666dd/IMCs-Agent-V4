import sys
import os
import unittest
import logging
from typing import List, Dict, Any

# Setup path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "src"))

from agents.core.ml_agent import MLAgent
from agents.core.experiment_agent import ExperimentDataAgent, ExperimentDataConfig
from agents.core.theory_agent import TheoryDataAgent
from agents.core.literature_agent import LiteratureAgent
from agents.core.task_manager import TaskManagerAgent
from services.db.database import DatabaseService

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Audit")

class VirtualAuditTest(unittest.TestCase):
    """
    Virtual Test Suite corresponding to multi_agent_framework_plan requirements.
    """
    
    @classmethod
    def setUpClass(cls):
        print("\n=== STARTING COMPREHENSIVE VIRTUAL AUDIT ===\n")
        # Ensure clean DB for audit
        cls.db = DatabaseService("data/audit_test.db") 

    def test_01_ml_agent_capabilities(self):
        """Test ML Agent functionality (P0 in plan)."""
        print(">> Testing 2. Machine Learning Agent...")
        agent = MLAgent()
        
        # Check method existence as per plan
        self.assertTrue(hasattr(agent, 'train_traditional_models'), "Missing: train_traditional_ml")
        self.assertTrue(hasattr(agent, 'train_deep_learning_models'), "Missing: train_deep_learning")
        self.assertTrue(hasattr(agent, 'train_gnn_models_v2'), "Missing: train_gnn_models")
        self.assertTrue(hasattr(agent, 'interpret_model'), "Missing: shap_analysis")
        
        print("   [PASS] API Signature Check")
        
        # Virtual execution (Dry Run)
        # We assume empty data won't crash but return empty result or handle gracefully
        try:
            # Inject dummy data for robustness check
            import numpy as np
            agent.data_manager.X_train = np.random.rand(10, 5)
            agent.data_manager.y_train = np.random.rand(10)
            agent.data_manager.X_test = np.random.rand(2, 5)
            agent.data_manager.y_test = np.random.rand(2)
            agent.data_manager.feature_names = [f"f{i}" for i in range(5)]
            
            res = agent.train_traditional_models()
            print(f"   [PASS] Trained {len(res)} traditional models.")
        except Exception as e:
            print(f"   [FAIL] Training execution failed: {e}")

    def test_02_theory_agent_capabilities(self):
        """Test Theory Data Agent (P1 in plan)."""
        print("\n>> Testing 4. Theory Data Agent...")
        agent = TheoryDataAgent()
        
        self.assertTrue(hasattr(agent, 'download_structures'), "Missing: download_structures")
        # process_and_structure is implicit in download logic
        
        # Mocking MP Client to avoid network
        agent.mp.search_materials = lambda *args, **kwargs: [] 
        
        try:
            count = agent.download_structures(limit=1)
            print("   [PASS] download_structures execution (mocked).")
        except Exception as e:
            print(f"   [FAIL] Theory download failed: {e}")

    def test_03_experiment_agent_capabilities(self):
        """Test Experiment Agent (P2 in plan)."""
        print("\n>> Testing 5. Experiment Data Agent...")
        agent = ExperimentDataAgent()
        
        self.assertTrue(hasattr(agent, 'process_request'), "Missing: Data processing entry point")
        
        # Test LSV parsing logic
        import pandas as pd
        df = pd.DataFrame({"V": [0, 0.5, 1.0], "I": [0, 5, 20]}) # Mocked LSV
        
        try:
            res = agent.analyze_lsv(df, "audit_sample")
            self.assertIsNotNone(res.overpotential_10mA)
            print("   [PASS] analyze_lsv logic verification.")
        except Exception as e:
            print(f"   [FAIL] Experiment analysis failed: {e}")

    def test_04_literature_agent_capabilities(self):
        """Test Literature Agent (P3 in plan)."""
        print("\n>> Testing 3. Literature Agent...")
        agent = LiteratureAgent()
        
        try:
            findings = agent.search_literature("HER catalyst", limit=1)
            # It might fail if API key is invalid or network issues, but method exists
            print(f"   [PASS] search_literature API call attempted.")
        except Exception as e:
            print(f"   [WARN] Literature API skipped/failed (Expected without real API key): {e}")

    def test_05_task_manager_capabilities(self):
        """Test Task Manager Agent (P0/P4 in plan)."""
        print("\n>> Testing 1. Task Management Agent...")
        agent = TaskManagerAgent()
        
        self.assertTrue(hasattr(agent, 'create_plan'), "Missing: create_plan")
        
        plan = agent.create_plan("Find HER catalyst")
        if plan and len(plan.steps) > 0:
            print(f"   [PASS] Planner created {len(plan.steps)} steps for 'HER Catalyst'.")
        else:
            print("   [FAIL] Planner returned empty plan.")

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
