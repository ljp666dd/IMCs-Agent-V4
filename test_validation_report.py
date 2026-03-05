import os
import sys
import json
import asyncio
from pathlib import Path

# Fix import path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.core.logger import get_logger
from src.agents.core.literature_agent import LiteratureAgent
from src.agents.core.ml_agent import MLAgent
from src.services.literature.crawler import ArxivCrawler
from src.agents.orchestrator import AgentOrchestrator

logger = get_logger("SystemIntegrationTest")

async def run_full_validation():
    print("\n" + "="*50)
    print("IMCs - ALL SYSTEMS INTEGRATION TEST")
    print("="*50)
    
    # 1. Literature Agent Test (Retrieval)
    print("\n[TEST 1] Literature Retrieval (arXiv & Semantic Scholar)")
    try:
        lit_agent = LiteratureAgent()
        res = lit_agent.search_all_sources("PtNi HOR alkaline", limit=2)
        if res and len(res) > 0:
            print(f"SUCCESS: Retrieved {len(res)} papers from all sources.")
            print(f"   Sample Title: {res[0].title[:60]}...")
        else:
            print("WARNING: No papers retrieved.")
    except Exception as e:
        print(f"ERROR in Literature Retrieval: {e}")

    # 2. VLM Parsing Test (Vision)
    print("\n[TEST 2] Multi-Modal Vision Parsing (VLM Image Analysis)")
    try:
        if "GEMINI_API_KEY" not in os.environ:
            print("SKIPPED: GEMINI_API_KEY not set for VLM test.")
        else:
            try:
                from src.services.literature.parser import parse_images_with_vlm
                test_bytes = b"fake_image_bytes"
                try:
                    parse_images_with_vlm([test_bytes])
                    print("ERROR: VLM engine accepted fake bytes incorrectly.")
                except Exception as e:
                    if "Identify" in str(e) or "MIME" in str(e) or "decode" in str(e) or "format" in str(e):
                        print("SUCCESS: Vision engine is active and correctly validated input format.")
                    else:
                        print(f"ERROR: Vision engine crashed unexpectedly: {e}")
            except ImportError:
                print("SKIPPED: VLM parser not available in this module structure.")
    except Exception as e:
        print(f"ERROR in VLM setup: {e}")

    # 3. ML Agent Testing
    print("\n[TEST 3] Machine Learning Agent Training")
    try:
        ml_agent = MLAgent()
        # Mocking a tiny dataset for quick validation
        import pandas as pd
        import numpy as np
        import tempfile
        import os
        df = pd.DataFrame({
            'feature1': np.random.rand(50),
            'feature2': np.random.rand(50),
            'target': np.random.rand(50)
        })
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name
            df.to_csv(temp_path, index=False)
        try:
            ml_agent.load_generic_csv(temp_path, target_col="target", feature_cols=["feature1", "feature2"])
            ml_agent.config.test_size = 0.2
            results = ml_agent.train_traditional_models(auto_select_features=False)
            if results and len(results) > 0:
                print(f"SUCCESS: Trained {len(results)} ML models (XGBoost/RF/etc) on mock data.")
                print(f"   Best Mock Model: {results[0].name} (R2={results[0].r2_test:.2f})")
            else:
                print("ERROR: ML training returned no results.")
        finally:
            os.remove(temp_path)
    except Exception as e:
        print(f"ERROR in ML Training: {e}")

    # 4. Multi-Agent Orchestration & Debate
    print("\n[TEST 4] Multi-Agent Orchestration & Debate")
    try:
        orchestrator = AgentOrchestrator()
        # Run a very short 1-iteration query to test debate logic
        res = orchestrator.orchestrate("Find low cost HOR catalysts without Pt", max_iterations=1)
        if res.candidates and len(res.candidates) > 0:
            print(f"SUCCESS: Agents collaborated and generated {len(res.candidates)} fused candidates.")
            print(f"   Debate Output Length: {len(res.reasoning)} chars")
        else:
            print("WARNING: Orchestration produced no candidates.")
    except Exception as e:
        print(f"ERROR in Orchestration: {e}")

    # 5. Data Analysis & UI Status
    print("\n[TEST 5] Data Sources & Database Health")
    try:
        from src.services.db.database import DatabaseService
        db = DatabaseService()
        stats = db.get_system_stats()
        print(f"SUCCESS: DB Online. Contains {stats.get('total_materials', 0)} materials.")
    except Exception as e:
        print(f"ERROR in DB connection: {e}")
        
    print("\n" + "="*50)
    print("SYSTEM VALIDATION COMPLETE")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(run_full_validation())
