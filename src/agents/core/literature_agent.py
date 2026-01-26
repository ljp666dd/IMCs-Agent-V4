"""
Literature Agent (LiteratureAgent)
Refactored (v3.1) to use Service-Oriented Architecture.
"""

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from src.core.logger import get_logger, log_exception
from src.services.literature.types import PaperInfo, KnowledgeExtract
from src.services.literature.parser import DocParser
from src.services.literature.search_engine import SearchEngine
from src.services.literature.analyzer import TopicAnalyzer

logger = get_logger(__name__)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)


@dataclass
class LiteratureConfig:
    """Configuration for Literature Agent."""
    local_library: str = "data/literature/pdfs"
    cache_dir: str = "data/literature/cache"
    semantic_scholar_api: str = "https://api.semanticscholar.org/graph/v1"
    max_results: int = 20


class LiteratureAgent:
    """
    Literature Agent for scientific knowledge extraction.
    Delegates to DocParser, SearchEngine, TopicAnalyzer.
    """
    
    def __init__(self, config: LiteratureConfig = None):
        self.config = config or LiteratureConfig()
        os.makedirs(self.config.local_library, exist_ok=True)
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Services
        self.parser = DocParser()
        self.search_engine = SearchEngine(self.config.semantic_scholar_api)
        self.analyzer = TopicAnalyzer()
        
        self.papers: List[PaperInfo] = []
        logger.info("LiteratureAgent initialized with services.")

    # ========== Local Library ==========
    
    def list_local_pdfs(self) -> List[str]:
        """List all PDF files in local library."""
        pdfs = []
        for root, dirs, files in os.walk(self.config.local_library):
            for f in files:
                if f.lower().endswith('.pdf'):
                    pdfs.append(os.path.join(root, f))
        return pdfs
        
    @log_exception(logger)
    def parse_all_local_pdfs(self) -> List[PaperInfo]:
        """Parse all PDFs in local library."""
        pdf_paths = self.list_local_pdfs()
        logger.info(f"Parsing {len(pdf_paths)} local PDFs...")
        
        parsed_papers = []
        for path in pdf_paths:
            paper = self.parser.parse_pdf(path)
            if paper:
                parsed_papers.append(paper)
        
        self.papers.extend(parsed_papers)
        logger.info(f"Parsed {len(parsed_papers)} papers.")
        return parsed_papers

    # ========== Online Search ==========
    
    @log_exception(logger)
    def search_all_sources(self, query: str, limit: int = None) -> List[PaperInfo]:
        """Search all available sources."""
        limit = limit or self.config.max_results
        results = self.search_engine.search_all(query, limit)
        self.papers.extend(results)
        return results

    # Backward-compatible alias for audit/test scripts
    def search_literature(self, query: str, limit: int = None) -> List[PaperInfo]:
        return self.search_all_sources(query, limit)

    # ========== Analysis ==========
    
    @log_exception(logger)
    def extract_knowledge(self, topic: str, papers: List[PaperInfo] = None) -> KnowledgeExtract:
        """Extract structured knowledge."""
        papers = papers or self.papers
        return self.analyzer.extract_knowledge(topic, papers)
        
    def generate_report(self, topic: str, papers: List[PaperInfo] = None) -> str:
        """Generate a literature report."""
        knowledge = self.extract_knowledge(topic, papers)
        return self.analyzer.generate_report(knowledge)

    # ========== Shortcuts ==========
    
    def search_catalyst(self, catalyst: str, reaction: str = "HER") -> List[PaperInfo]:
        query = f"{catalyst} {reaction} electrocatalyst"
        return self.search_engine.search_semantic_scholar(query, self.config.max_results)
        
    def search_alloy(self, elements: List[str], reaction: str = "HOR") -> List[PaperInfo]:
        alloy_str = "-".join(elements)
        query = f"{alloy_str} alloy {reaction} catalyst"
        return self.search_engine.search_semantic_scholar(query, self.config.max_results)
