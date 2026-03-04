import os
import sys
import logging
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# Set up logging early
from src.core.logger import get_logger
logger = get_logger("End2EndTest")
logging.basicConfig(level=logging.INFO)

from src.agents.orchestrator import AgentOrchestrator

def main():
    logger.info("Initializing Orchestrator for End-to-End Active Learning Test...")
    director = AgentOrchestrator()
    
    query = "Recommend ordered alloy HOR catalysts containing Pt, Ru, or Ni with extremely high activity potential."
    
    logger.info(f"Query: {query}")
    logger.info("Calling director.orchestrate()...")
    
    result = director.orchestrate(query)
    
    print("\n" + "="*80)
    print("  T E S T   R E S U L T S  ")
    print("="*80)
    print(f"Success: {result.success}")
    is_al = "⚠️ [主动发现 - Active Learning] ⚠️" in result.reasoning
    print(f"Active Learning Triggered: {is_al}")
    print("\n" + "-"*40 + " DIRECTOR REPORT / REASONING " + "-"*40)
    print(result.reasoning)
    print("-" * 109)
    print(f"\nCandidates Evaluated/Returned: {len(result.candidates)}")
    print("\nTop Candidates Details:")
    for i, c in enumerate(result.candidates[:5]):
        formula = c.get("formula", str(c.get("material_id")))
        props = c.get("properties", {})
        score = props.get("predicted_activity", "N/A")
        var = props.get("uncertainty", "N/A")
        
        reason = c.get("active_learning_reason", "")
        print(f"  [{i+1}] {formula}")
        print(f"      - Predicted Activity Score (Sabatier logic): {score}")
        print(f"      - Uncertainty (MC-Dropout Variance)      : {var}")
        if reason:
            print(f"      - Active Learning Flag                   : {reason}")
            
if __name__ == "__main__":
    main()
