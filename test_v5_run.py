import os
import sys
import logging
import json
from src.core.logger import get_logger
from src.agents.orchestrator import AgentOrchestrator
from src.services.common.token_tracker import get_token_tracker
from src.services.robot.protocol_generator import get_protocol_generator
from src.agents.core.theory_agent import TheoryDataAgent

# Configure basic logging to terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("SystemTest")

def main():
    logger.info("=== IMCs V5 System Integration Test ===")
    
    # 1. Initialize Tracker
    tracker = get_token_tracker()
    tracker.reset()
    logger.info("Token Tracker initialized.")

    # 2. Test TheoryAgent & Pretrained GNN Bridge (Phase E)
    logger.info("--- Testing TheoryAgent (Pretrained GNN) ---")
    try:
        theory = TheoryDataAgent()
        # Test an arbitrary MP ID if database has it, otherwise skip gracefullly
        mats = theory.list_stored_materials(limit=1)
        if mats:
            test_mid = mats[0].get("material_id")
            logger.info(f"Testing GNN stability prediction for {test_mid}")
            stab = theory.predict_stability(test_mid)
            logger.info(f"GNN Prediction Result: {stab}")
        else:
            logger.info("No materials in local DB for GNN test.")
    except Exception as e:
        logger.error(f"TheoryAgent test failed: {e}")

    # 3. Test Robot Protocol Generator (Phase C)
    logger.info("--- Testing Robot SDK ---")
    try:
        robot = get_protocol_generator()
        protocol = robot.generate_ink_protocol("Pt3Ni", catalyst_mass_mg=5.0)
        logger.info(f"Generated Protocol: {protocol.robot_platform} for {protocol.material_formula}")
        logger.info(f"Estimated Duration: {protocol.estimated_duration_min} min")
        logger.info(f"OT-2 Script Snippet:\\n{protocol.python_script[:150]}...")
    except Exception as e:
        logger.error(f"Robot SDK test failed: {e}")

    # 4. Test Orchestrator End-to-End Task (Phase A, B, D)
    logger.info("--- Testing Orchestrator End-to-End ---")
    orchestrator = AgentOrchestrator()
    
    test_query = "寻找适合碱性介质的高效且低成本基于非贵金属的HOR催化剂，要求具有良好的抗CO毒化能力。"
    logger.info(f"Submitting query: {test_query}")
    
    try:
        result = orchestrator.orchestrate(test_query)
        logger.info(f"Orchestration Success: {result.success}")
        logger.info(f"Candidates Recommended: {len(result.candidates)}")
        logger.info(f"Reasoning Excerpt:\\n{result.reasoning[:300]}...")
        
        # Check active learning
        al_requests = [c for c in result.candidates if "active_learning_reason" in c]
        logger.info(f"Active Learning Candidates (Bayesian AL): {len(al_requests)}")
    except Exception as e:
        logger.error(f"Orchestration failed: {e}", exc_info=True)

    # 5. Token Usage Report
    logger.info("--- Token Usage Audit Report ---")
    report = tracker.get_usage_report()
    logger.info(json.dumps(report, indent=2, ensure_ascii=False))

    logger.info("=== Test Complete ===")

if __name__ == "__main__":
    main()
