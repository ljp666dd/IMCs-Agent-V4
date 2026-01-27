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
from src.services.literature.hor_metrics import extract_hor_metrics, extract_formulas

logger = get_logger(__name__)

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)


@dataclass
class LiteratureConfig:
    """Configuration for Literature Agent."""
    local_library: str = "data/literature"
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

    def _download_pdf(self, pdf_url: str, name_hint: str = "paper") -> Optional[str]:
        if not pdf_url:
            return None
        import re
        import requests
        safe_name = re.sub(r"[^A-Za-z0-9\\-_.]+", "_", name_hint).strip("_") or "paper"
        dest = os.path.join(self.config.cache_dir, f"{safe_name}.pdf")
        if os.path.exists(dest):
            return dest
        try:
            r = requests.get(pdf_url, timeout=60)
            if r.status_code == 200 and r.content:
                with open(dest, "wb") as f:
                    f.write(r.content)
                return dest
        except Exception as e:
            logger.warning(f"PDF download failed: {e}")
        return None
        
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

    # ========== HOR Harvest (Online) ==========

    @log_exception(logger)
    def harvest_hor_seed(self, query: str, limit: int = 10, max_pdfs: int = 5,
                         min_elements: int = 2, persist: bool = False) -> List[Dict[str, str]]:
        """Search online, parse PDFs when available, and return seed rows for HOR metrics."""
        from src.agents.core.theory_agent import TheoryDataConfig
        allowed = set(TheoryDataConfig().elements)
        db = None
        if persist:
            from src.services.db.database import DatabaseService
            db = DatabaseService()

        papers = self.search_all_sources(query, limit)
        seed_rows: List[Dict[str, str]] = []
        pdf_downloads = 0

        for paper in papers:
            text_blob = paper.abstract or ""
            if paper.pdf_url and pdf_downloads < max_pdfs:
                path = self._download_pdf(paper.pdf_url, paper.doi or paper.title or "paper")
                if path:
                    parsed = self.parser.parse_pdf(path)
                    if parsed and parsed.full_text:
                        text_blob = f"{parsed.full_text}\\n\\n{text_blob}"
                    pdf_downloads += 1

            formulas = extract_formulas(text_blob, allowed_elements=allowed, min_elements=min_elements)
            if not formulas:
                continue

            metrics = extract_hor_metrics(text_blob)
            for formula in formulas:
                label = formula
                row = {
                    "material_label": label,
                    "formula": formula,
                    "specific_activity_50mV_mA_cm2": "",
                    "mass_activity_50mV_A_mg": "",
                    "j0_specific_min_mA_cm2": "",
                    "j0_specific_max_mA_cm2": "",
                    "j0_mass_min_A_g": "",
                    "j0_mass_max_A_g": "",
                    "source_doi": paper.doi or "",
                    "source_note": paper.title or "",
                    "source_title": paper.title or "",
                    "source_url": paper.url or "",
                    "source_year": paper.year or "",
                    "source_abstract": (paper.abstract or "")[:2000],
                    "conditions_json": "",
                }
                if metrics.get("specific_activity"):
                    row["specific_activity_50mV_mA_cm2"] = metrics["specific_activity"]["value"]
                if metrics.get("mass_activity"):
                    row["mass_activity_50mV_A_mg"] = metrics["mass_activity"]["value"]
                if metrics.get("exchange_current_density"):
                    val = metrics["exchange_current_density"]["value"]
                    row["j0_specific_min_mA_cm2"] = val
                    row["j0_specific_max_mA_cm2"] = val
                seed_rows.append(row)

                if persist and db:
                    material_id = f"lit:{label}"
                    db.save_material(material_id=material_id, formula=formula, energy=None, cif_path=None)
                    db.save_evidence(
                        material_id=material_id,
                        source_type="literature",
                        source_id=paper.doi or paper.url or paper.title or "paper",
                        score=0.8,
                        metadata={
                            "title": paper.title,
                            "year": paper.year,
                            "abstract": paper.abstract,
                            "doi": paper.doi,
                            "url": paper.url,
                            "authors": paper.authors,
                            "citation_count": paper.citation_count,
                        },
                    )

        return seed_rows

    @log_exception(logger)
    def ingest_local_library(self, min_elements: int = 2) -> Dict[str, int]:
        """Parse local PDFs and index them into knowledge sources (and optional evidence)."""
        from src.services.db.database import DatabaseService
        from src.services.knowledge import KnowledgeService
        from src.agents.core.theory_agent import TheoryDataConfig

        allowed = set(TheoryDataConfig().elements)
        db = DatabaseService()
        ks = KnowledgeService()

        pdf_paths = self.list_local_pdfs()
        indexed = 0
        linked = 0

        for path in pdf_paths:
            paper = self.parser.parse_pdf(path)
            if not paper:
                continue
            meta = {
                "abstract": paper.abstract or "",
                "content": paper.full_text or "",
                "local_path": path,
            }
            source = ks.create_source(
                source_type="literature",
                source_id=os.path.basename(path),
                title=paper.title,
                url=path,
                year=paper.year if hasattr(paper, "year") else None,
                metadata=meta,
            )
            indexed += 1

            formulas = extract_formulas(paper.full_text or paper.abstract, allowed_elements=allowed, min_elements=min_elements)
            for formula in formulas:
                material_id = f"lit:{formula}"
                db.save_material(material_id=material_id, formula=formula, energy=None, cif_path=None)
                db.save_evidence(
                    material_id=material_id,
                    source_type="literature",
                    source_id=source.get("source_id") if source else os.path.basename(path),
                    score=0.6,
                    metadata={"title": paper.title, "url": path},
                )
                linked += 1

        return {"indexed_sources": indexed, "linked_materials": linked}
