"""
Literature Agent (LiteratureAgent)
Refactored (v3.1) to use Service-Oriented Architecture.
"""

import os
import sys
import warnings
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from src.core.logger import get_logger, log_exception
from src.services.literature.types import PaperInfo, KnowledgeExtract
from src.services.literature.parser import DocParser
from src.services.literature.search_engine import SearchEngine
from src.services.literature.analyzer import TopicAnalyzer
from src.services.literature.hor_metrics import extract_hor_metrics, extract_formulas
from src.services.llm.vision_service import get_vision_service

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
        self.vision_service = get_vision_service()
        
        self.papers: List[PaperInfo] = []
        logger.info("LiteratureAgent initialized with services.")

    def _extract_target_elements(self, query: str, allowed: set) -> List[str]:
        if not query:
            return []
        focus_patterns = [
            r"(?:focus on|focus|筛选|聚焦|重点)\s*([A-Z][a-z]?(?:[-/ ,]+[A-Z][a-z]?){1,2})",
            r"([A-Z][a-z]?(?:[-/][A-Z][a-z]?){1,2})",
        ]
        focus_elements = set()
        for pat in focus_patterns:
            for match in re.findall(pat, query):
                focus_elements.update(re.findall(r"[A-Z][a-z]?", match))
        if focus_elements:
            return sorted({el for el in focus_elements if el in allowed})
        tokens = re.findall(r"[A-Z][a-z]?", query)
        elements = [t for t in tokens if t in allowed]
        return sorted(set(elements))

    def _build_hor_query(self, query: str, target_elements: List[str], prefer_metrics: bool = False) -> str:
        base = query or ""
        q = base
        lower = q.lower()
        if "hor" not in lower and "hydrogen oxidation" not in lower:
            q = f"HOR hydrogen oxidation {q}"
            lower = q.lower()
        if "electrocatalyst" not in lower:
            q = f"{q} electrocatalyst"
            lower = q.lower()
        if "alloy" not in lower and "intermetallic" not in lower:
            q = f"{q} alloy"
            lower = q.lower()
        if prefer_metrics and "exchange current density" not in lower and "overpotential" not in lower:
            q = f"{q} exchange current density overpotential j0"
        if target_elements:
            q = f"{q} {' '.join(target_elements[:4])}"
        return q

    def _wants_metrics(self, query: str) -> bool:
        lower = (query or "").lower()
        return any(
            k in lower
            for k in ("j0", "exchange current density", "overpotential", "tafel", "mass activity")
        ) or any(
            k in (query or "")
            for k in ("指标", "过电位", "交换电流", "活性")
        )

    def _paper_matches(
        self,
        paper: PaperInfo,
        target_elements: List[str],
        min_elements: int = 1,
        require_hor: bool = True,
    ) -> bool:
        title = paper.title or ""
        abstract = paper.abstract or ""
        text = f"{title} {abstract}".strip()
        lower = text.lower()
        if require_hor and not (
            re.search(r"\bHOR\b", text, flags=re.IGNORECASE)
            or "hydrogen oxidation" in lower
            or "hydrogen evolution" in lower
            or "electrocatalyst" in lower
            or "alloy catalyst" in lower
        ):
            return False
        if target_elements:
            formulas = extract_formulas(text, allowed_elements=set(target_elements), min_elements=min_elements)
            if formulas:
                return True
            hits = 0
            for el in target_elements:
                if re.search(rf"(?<![A-Za-z0-9]){re.escape(el)}(?![a-z])", text):
                    hits += 1
            return hits >= min_elements
        return True

    def _filter_papers(
        self,
        papers: List[PaperInfo],
        target_elements: List[str],
        min_elements: int = 2,
        require_hor: bool = True,
        allow_fallback: bool = True,
    ) -> List[PaperInfo]:
        if not papers:
            return []
        filtered = [
            p for p in papers
            if self._paper_matches(p, target_elements, min_elements=min_elements, require_hor=require_hor)
        ]
        if filtered:
            return filtered
        if allow_fallback:
            logger.warning("Literature filter removed all results; returning unfiltered list.")
            return papers
        logger.warning("Literature filter removed all results; returning empty list.")
        return []

    def _search_local_pdfs(
        self,
        target_elements: List[str],
        min_elements: int = 2,
        require_hor: bool = True,
        limit: int = 10,
    ) -> List[PaperInfo]:
        pdfs = self.list_local_pdfs()
        if not pdfs:
            return []
        keywords = [
            "hor",
            "hydrogen",
            "oxidation",
            "intermetallic",
            "alloy",
            "electrocatalyst",
            "fuel cell",
        ]
        candidates = []
        for path in pdfs:
            name = os.path.basename(path)
            name_lower = name.lower()
            if any(k in name_lower for k in keywords):
                candidates.append(path)
                continue
            for el in target_elements:
                if el in name:
                    candidates.append(path)
                    break
        if not candidates:
            candidates = pdfs
        max_scan = max(limit * 2, 10)
        matches: List[PaperInfo] = []
        for path in candidates[:max_scan]:
            if len(matches) >= limit:
                break
            paper = self.parser.parse_pdf(path)
            if not paper:
                continue
            if self._paper_matches(paper, target_elements, min_elements=min_elements, require_hor=require_hor):
                matches.append(paper)
        return matches

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
        """Search all available sources with progressive filter relaxation."""
        limit = limit or self.config.max_results
        from src.agents.core.theory_agent import TheoryDataConfig
        allowed = set(TheoryDataConfig().elements)
        target_elements = self._extract_target_elements(query, allowed)
        prefer_metrics = self._wants_metrics(query)
        local_hits = self._search_local_pdfs(
            target_elements,
            min_elements=1,
            require_hor=True,
            limit=min(5, limit),
        )
        if local_hits:
            self.papers.extend(local_hits)
            return local_hits

        # Stage 1: Strict filter (HOR + element match)
        focused_query = self._build_hor_query(query, target_elements, prefer_metrics=prefer_metrics)
        results = self.search_engine.search_all(focused_query, limit)
        results = self._filter_papers(results, target_elements, min_elements=2, require_hor=True, allow_fallback=False)

        # Stage 2: Relaxed element match (min_elements=1)
        if not results:
            fallback_query = f'"hydrogen oxidation" OR HOR electrocatalyst alloy {" ".join(target_elements[:4])}'
            results = self.search_engine.search_all(fallback_query, limit)
            results = self._filter_papers(results, target_elements, min_elements=1, require_hor=False, allow_fallback=False)

        # Stage 3: Generic HOR search, always return whatever we find
        if not results:
            generic_query = "HOR hydrogen oxidation electrocatalyst alloy"
            results = self.search_engine.search_all(generic_query, limit)
            results = self._filter_papers(results, [], min_elements=0, require_hor=False, allow_fallback=True)

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

        target_elements = self._extract_target_elements(query, allowed)
        prefer_metrics = self._wants_metrics(query)
        focused_query = self._build_hor_query(query, target_elements, prefer_metrics=prefer_metrics)
        papers = self.search_all_sources(focused_query, limit)
        if papers and target_elements:
            papers = self._filter_papers(papers, target_elements, min_elements=min_elements, require_hor=True, allow_fallback=False)
        seed_rows: List[Dict[str, str]] = []
        pdf_downloads = 0

        for paper in papers:
            text_blob = paper.abstract or ""
            if getattr(paper, "full_text", None):
                text_blob = f"{paper.full_text}\n\n{text_blob}"
            if paper.pdf_url and pdf_downloads < max_pdfs:
                path = self._download_pdf(paper.pdf_url, paper.doi or paper.title or "paper")
                if path:
                    parsed = self.parser.parse_pdf(path)
                    if parsed and parsed.full_text:
                        text_blob = f"{parsed.full_text}\\n\\n{text_blob}"
                    
                    # V5.2: Multi-modal Vision Analysis
                    vision_seed_data = []
                    if self.vision_service.available:
                        logger.info(f"Triggering vision analysis for {path}")
                        img_paths = self.parser.render_pages_to_images(path, page_indices=[0, 1])
                        for img_p in img_paths:
                            # 1. 提取表格 (原有逻辑)
                            vision_data = self.vision_service.analyze_page(img_p, task="extract_tables")
                            if isinstance(vision_data, list) and vision_data:
                                logger.info(f"Extracted {len(vision_data)} items via vision.")
                                vision_seed_data.extend(vision_data)
                            
                            # 2. V5.5: 数字化曲线 (新增逻辑)
                            curve_data = self.vision_service.digitize_curve(img_p)
                            if curve_data.get("success"):
                                logger.info(f"Digitized curve with {len(curve_data.get('data_points', []))} points.")
                                # 赋予特殊的 insight 类型
                                text_blob = f"--- Digitized Curve Data ---\n{json.dumps(curve_data, ensure_ascii=False)}\n\n{text_blob}"
                                
                            vision_text = json.dumps(vision_data, ensure_ascii=False)
                            text_blob = f"--- Vision Data ---\n{vision_text}\n\n{text_blob}"
                    
                    pdf_downloads += 1

            allowed_for_formula = set(target_elements) if target_elements else allowed
            formulas = extract_formulas(text_blob, allowed_elements=allowed_for_formula, min_elements=min_elements)
            
            # Incorporate vision findings directly if they match element constraints
            for item in vision_seed_data:
                v_formula = item.get("material", "")
                if v_formula and v_formula not in formulas:
                    # Check elements
                    v_parts = re.findall(r"([A-Z][a-z]?)(\d*)", v_formula)
                    v_els = {p[0] for p in v_parts}
                    if not allowed_for_formula or v_els.issubset(allowed_for_formula):
                        formulas.append(v_formula)

            if not formulas and not target_elements:
                formulas = extract_formulas(text_blob, allowed_elements=allowed, min_elements=min_elements)
            
            if not formulas:
                continue

            metrics = extract_hor_metrics(text_blob)
            
            # Merge metrics from vision data if keys match
            for item in vision_seed_data:
                v_metric = item.get("metric", "").lower()
                v_val = item.get("value")
                v_unit = item.get("units", "")
                
                # Simple mapping
                if "exchange current" in v_metric or "j0" in v_metric:
                    metrics["exchange_current_density"] = {"value": v_val, "unit": v_unit}
                elif "mass activity" in v_metric:
                    metrics["mass_activity"] = {"value": v_val, "unit": v_unit}
                elif "overpotential" in v_metric:
                    metrics["overpotential"] = {"value": v_val, "unit": v_unit}
            
            metrics_missing = prefer_metrics and not metrics
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
                    "conditions_json": json.dumps({"metrics_missing": bool(metrics_missing)}) if metrics_missing else "",
                }
                if metrics.get("specific_activity"):
                    row["specific_activity_50mV_mA_cm2"] = metrics["specific_activity"]["value"]
                if metrics.get("mass_activity"):
                    row["mass_activity_50mV_A_mg"] = metrics["mass_activity"]["value"]
                if metrics.get("exchange_current_density"):
                    val = metrics["exchange_current_density"]["value"]
                    row["j0_specific_min_mA_cm2"] = val
                    row["j0_specific_max_mA_cm2"] = val
                if metrics.get("overpotential"):
                    row["conditions_json"] = str(metrics["overpotential"])
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
