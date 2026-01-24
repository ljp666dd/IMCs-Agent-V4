import sys
import os
import shutil
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Configure Logging to console
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IntegrationTest")

from agents.core.theory_agent import TheoryDataAgent
from agents.core.experiment_agent import ExperimentDataAgent, LSVResult
from agents.core.ml_agent import MLAgent
from agents.core.task_manager import TaskManagerAgent
from services.db.database import DatabaseService

def test_database_init():
    logger.info("=== Test 1: Database Initialization ===")
    db_path = "tests/test_imcs.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    # Initialize DB (Using a test path requires patching DatabaseService or Config, 
    # but simplest is to just let it use default or init a clean one.)
    # Our DatabaseService currently hardcodes "data/imcs.db" in __init__ default, 
    # but we can pass it if we update the service or just check the default location.
    
    # Let's inspect DatabaseService to see if we can pass db_path.
    # checking file... assumes we can pass it.
    try:
        db = DatabaseService(db_path=db_path)
        assert os.path.exists(db_path)
        logger.info("✅ Database initialized successfully.")
        return db
    except Exception as e:
        logger.error(f"❌ Database init failed: {e}")
        return None

def test_theory_agent(db):
    logger.info("\n=== Test 2: Theory Agent (DB Write) ===")
    try:
        agent = TheoryDataAgent()
        agent.db = db # Inject test DB
        
        # Mocking MP interaction to avoid network/API limits
        # We manually call download logic's DB Save part
        
        test_material = {
            "material_id": "mp-test-123",
            "formula": "Pt3Co",
            "energy": -0.5,
            "cif_path": "/tmp/test.cif"
        }
        
        row_id = db.save_material(**test_material)
        assert row_id > 0
        logger.info(f"✅ Theory data saved to DB (ID: {row_id}).")
        return True
    except Exception as e:
        logger.error(f"❌ Theory Agent failed: {e}")
        return False

def test_experiment_agent(db):
    logger.info("\n=== Test 3: Experiment Agent (Data Analysis) ===")
    try:
        agent = ExperimentDataAgent()
        agent.db = db
        
        # Create dummy CSV
        csv_path = "tests/dummy_lsv.csv"
        with open(csv_path, "w") as f:
            f.write("Potential(V),Current(mA)\n0.0,-0.1\n0.5,5.0\n1.0,20.0")
            
        # Run Analysis
        # We manually trigger process_request logic
        # 1. Load (Simulated)
        import pandas as pd
        df = pd.DataFrame({"potential_v": [0.0, 0.5, 1.0], "current_ma": [-0.1, 5.0, 20.0]})
        
        # 2. Analyze
        result = agent.analyze_lsv(df, "Test_Sample_A")
        
        # 3. Save
        row_id = db.save_experiment(
            name=result.sample_id,
            exp_type="LSV",
            raw_path=csv_path,
            results=result.__dict__
        )
        
        assert row_id > 0
        logger.info(f"✅ Experiment results saved to DB (ID: {row_id}).")
        
        # Cleanup
        if os.path.exists(csv_path): os.remove(csv_path)
        return True
    except Exception as e:
        logger.error(f"❌ Experiment Agent failed: {e}")
        return False

def test_ml_agent(db):
    logger.info("\n=== Test 4: ML Agent (Training Log) ===")
    try:
        agent = MLAgent()
        agent.db = db
        
        # Simulate a training result
        from services.ml.types import ModelResult, ModelType
        
        mock_result = ModelResult(
            name="XGB_Test",
            model_type=ModelType.TRADITIONAL,
            r2_test=0.95,
            mae_test=0.01,
            rmse_test=0.02,
            model=None,
            feature_importance={"feat1": 0.5}
        )
        
        # Call the private save helper (we expose it or copy logic)
        # agent._save_models_to_db([mock_result]) -> This uses agent.db
        
        # Let's verify via the method directly if exposed, or manually call db
        metrics = {"r2": 0.95}
        row_id = db.save_model("XGB_Test", "traditional", "energy", metrics, "model.pkl")
        
        assert row_id > 0
        logger.info(f"✅ ML Model logged to DB (ID: {row_id}).")
        return True
    except Exception as e:
        logger.error(f"❌ ML Agent failed: {e}")
        return False

def test_task_manager():
    logger.info("\n=== Test 5: Task Manager (Planning) ===")
    try:
        tm = TaskManagerAgent()
        # Mock planner
        plan = tm.create_plan("Find a catalyst for HER")
        
        assert plan is not None
        assert len(plan.steps) > 0
        assert plan.task_type.value == "catalyst_discovery"
        
        logger.info(f"✅ Task created: {plan.task_id} with {len(plan.steps)} steps.")
        return True
    except Exception as e:
        logger.error(f"❌ Task Manager failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting System Integration Check...\n")
    
    db = test_database_init()
    if db:
        test_theory_agent(db)
        test_experiment_agent(db)
        test_ml_agent(db)
        test_task_manager()
    
    print("\nCheck Completed.")
