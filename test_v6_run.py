import os
import sys
import logging
import json
from src.core.logger import get_logger

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("V6_SystemTest")

def main():
    logger.info("=== IMCs V6 System Integration Test (Phase I-III) ===")

    # 1. 验证文献抓取器 (Phase I)
    logger.info("--- 1. Testing Arxiv Crawler ---")
    try:
        from src.services.literature.crawler import ArxivCrawler
        crawler = ArxivCrawler()
        papers = crawler.fetch_latest_papers(max_results=2)
        logger.info(f"Crawled {len(papers)} papers. First title: {papers[0]['title'] if papers else 'None'}")
    except Exception as e:
        logger.error(f"Crawler test failed: {e}")

    # 2. 验证图谱构建与导出 (Phase I)
    logger.info("--- 2. Testing Knowledge Graph DB ---")
    try:
        from src.services.knowledge.graph_db import KnowledgeGraphDB
        gdb = KnowledgeGraphDB()
        gdb.rebuild_graph_from_sqlite()
        export_path = gdb.export_to_json()
        logger.info(f"Graph DB built and exported to {export_path}")
    except Exception as e:
        logger.error(f"Graph DB test failed: {e}")

    # 3. 验证市场估价与代价值计算 (Phase II)
    logger.info("--- 3. Testing Market Data & Cost Estimation ---")
    try:
        from src.services.theory.market_data import get_market_data
        md = get_market_data()
        cost_pt3ni = md.estimate_formula_cost({"Pt": 3, "Ni": 1})
        cost_feni = md.estimate_formula_cost({"Fe": 1, "Ni": 1})
        pen_pt3ni = md.get_cost_penalty({"Pt": 3, "Ni": 1})
        pen_feni = md.get_cost_penalty({"Fe": 1, "Ni": 1})
        logger.info(f"Cost Pt3Ni: ${cost_pt3ni:.2f}/kg, Penalty: {pen_pt3ni:.3f}")
        logger.info(f"Cost FeNi: ${cost_feni:.2f}/kg, Penalty: {pen_feni:.3f}")
    except Exception as e:
        logger.error(f"Market Data test failed: {e}")

    # 4. 验证带回调的多目标编排流 (Phase II & III)
    logger.info("--- 4. Testing Multi-Objective Orchestrator with Callbacks ---")
    messages = []
    def prog_callback(data):
        messages.append(f"[{data['stage']}] {data['message']}")

    try:
        from src.agents.orchestrator import AgentOrchestrator
        orchestrator = AgentOrchestrator(on_progress=prog_callback)
        query = "寻可用作HOR的高效非贵金属电催化剂材料，要求成本极低。"
        result = orchestrator.orchestrate(query, max_iterations=2)
        
        logger.info(f"Orchestration Success: {result.success}")
        logger.info(f"Total Candidates Generated: {len(result.candidates)}")
        
        pareto_count = sum(1 for c in result.candidates if "Pareto" in c.get('active_learning_reason', ''))
        logger.info(f"Pareto Optimal Candidates Identified: {pareto_count}")
        logger.info(f"UI Callbacks Received: {len(messages)}")
        logger.info("First 3 Callbacks:\\n" + "\\n".join(messages[:3]))
    except Exception as e:
        logger.error(f"Orchestrator test failed: {e}", exc_info=True)

    logger.info("=== V6 Test Complete ===")


if __name__ == "__main__":
    main()
