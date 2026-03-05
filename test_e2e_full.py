import os
import sys
import logging
import json
import time
from src.core.logger import get_logger

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("Deep_E2E_Test")

# Monkey patch os.environ for test
os.environ["ENVIRONMENT"] = "testing"

def deep_test_pipeline():
    logger.info("=========================================")
    logger.info("   IMCs Deep Autonomous E2E Test Suite   ")
    logger.info("=========================================")
    
    # 1. Literature Crawler (V6)
    logger.info("--- [1] Testing Literature Crawler (V6) ---")
    try:
        from src.services.literature.crawler import ArxivCrawler
        crawler = ArxivCrawler()
        papers = crawler.fetch_latest_papers(max_results=1)
        logger.info(f"Crawler OK. Fetched: {papers[0]['title'] if papers else 'None'}")
    except Exception as e:
        logger.error(f"Crawler Failed: {e}", exc_info=True)
        
    # 2. Knowledge Graph DB (V6)
    logger.info("--- [2] Testing Knowledge Graph Matrix (V6) ---")
    try:
        from src.services.knowledge.graph_db import KnowledgeGraphDB
        gdb = KnowledgeGraphDB()
        gdb.rebuild_graph_from_sqlite()
        export = gdb.export_to_json()
        logger.info(f"Graph DB OK. Graph Data Exported to {export}")
    except Exception as e:
        logger.error(f"Graph DB Failed: {e}", exc_info=True)
        
    # 3. Multi-Objective Market Data (V6)
    logger.info("--- [3] Testing Market Data Penalities (V6) ---")
    try:
        from src.services.theory.market_data import get_market_data
        md = get_market_data()
        penalty = md.get_cost_penalty({"Pt": 1, "Cu": 3})
        logger.info(f"Market Data OK. Cost Penalty (PtCu3): {penalty:.3f}")
    except Exception as e:
        logger.error(f"Market Data Failed: {e}", exc_info=True)

    # 4. Multi-Agent Orchestrator & Bayesian Active Learning (V5/V6 Fusion)
    logger.info("--- [4] Testing Multi-Agent Orchestrator (DAG Fusion + Pareto AL) ---")
    try:
        from src.agents.orchestrator import AgentOrchestrator
        orchestrator = AgentOrchestrator()
        query = "请研发一个适用于碱性介质的极高活性的低成本HOR合金催化剂 (Pt, Ni, Cu, Au等候选)"
        result = orchestrator.orchestrate(query, max_iterations=2)
        logger.info(f"Orchestrator OK. Output {len(result.candidates)} candidates.")
        
        # Test MetaController Feedback
        from src.services.task.meta_controller import MetaController
        meta = MetaController()
        meta.strategy_feedback("TEST_PLAN", "success", evidence_yield=5)
        logger.info("MetaController Feedback OK.")
        
        # Collect Pareto
        pareto_cands = [c for c in result.candidates if "Pareto" in c.get('active_learning_reason', '')]
        logger.info(f"Found {len(pareto_cands)} Pareto Optimal candidates.")
        
    except Exception as e:
        logger.error(f"Orchestrator Failed: {e}", exc_info=True)
        result = None

    # 5. Opentrons SDK generation (V5)
    logger.info("--- [5] Testing Opentrons Physical SDK Generation (V5) ---")
    try:
        if result and result.candidates:
            from src.services.robot.protocol_generator import ProtocolGenerator
            gen = ProtocolGenerator()
            # Use the top candidate
            top_cand = result.candidates[0]
            path = gen.generate_ink_protocol(top_cand.get("material_id", "TEST_MAT_001"), catalyst_mass_mg=5.0)
            logger.info(f"Robot SDK Generator OK. Protocol written to {path.python_script[:50]}...")
        else:
            logger.warning("Robot SDK Generator Skipped (No candidates produced).")
    except Exception as e:
        logger.error(f"Robot SDK Generator Failed: {e}", exc_info=True)

    logger.info("=========================================")
    logger.info("       Deep E2E Test Suite Completed     ")
    logger.info("=========================================")

if __name__ == "__main__":
    deep_test_pipeline()
