import requests
import xml.etree.ElementTree as ET
from typing import List, Optional
from src.core.logger import get_logger, log_exception
from src.services.literature.types import PaperInfo

logger = get_logger(__name__)

class SearchEngine:
    """
    Service for searching scientific databases (Semantic Scholar, arXiv).
    """
    
    def __init__(self, semantic_scholar_api: str = "https://api.semanticscholar.org/graph/v1"):
        self.ss_api = semantic_scholar_api

    @log_exception(logger)
    def search_semantic_scholar(self, query: str, limit: int = 20) -> List[PaperInfo]:
        """Search Semantic Scholar."""
        logger.info(f"Searching Semantic Scholar: {query}")
        papers = []
        
        try:
            url = f"{self.ss_api}/paper/search"
            params = {
                "query": query,
                "limit": limit,
                "fields": "title,authors,year,abstract,citationCount,externalIds,url"
            }
            
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                for item in data.get("data", []):
                    paper = PaperInfo(
                        title=item.get("title", ""),
                        authors=[a.get("name", "") for a in item.get("authors", [])[:5]],
                        year=item.get("year", 0) or 0,
                        abstract=item.get("abstract", "") or "",
                        doi=item.get("externalIds", {}).get("DOI", ""),
                        url=item.get("url", ""),
                        citation_count=item.get("citationCount", 0) or 0
                    )
                    papers.append(paper)
                    
            # Sort by citations
            papers.sort(key=lambda x: x.citation_count, reverse=True)
            logger.info(f"Found {len(papers)} papers on SS.")
            
        except Exception as e:
            logger.error(f"Semantic Scholar Search Failed: {e}")
            
        return papers

    @log_exception(logger)
    def search_arxiv(self, query: str, limit: int = 20) -> List[PaperInfo]:
        """Search arXiv."""
        logger.info(f"Searching arXiv: {query}")
        papers = []
        
        try:
            url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                
                for entry in root.findall("atom:entry", ns):
                    title = entry.find("atom:title", ns).text.strip()
                    summary = entry.find("atom:summary", ns).text.strip()
                    id_url = entry.find("atom:id", ns).text
                    
                    # Authors
                    authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
                    
                    # Date
                    pub = entry.find("atom:published", ns)
                    year = int(pub.text[:4]) if pub is not None else 0
                    
                    doi = id_url.split("abs/")[-1] if "abs/" in id_url else ""
                    
                    papers.append(PaperInfo(
                        title=title,
                        authors=authors[:5],
                        year=year,
                        abstract=summary,
                        doi=doi,
                        url=id_url
                    ))
            
            logger.info(f"Found {len(papers)} papers on arXiv.")
            
        except Exception as e:
            logger.error(f"arXiv Search Failed: {e}")
            
        return papers

    def search_all(self, query: str, limit: int = 20) -> List[PaperInfo]:
        """Search both sources."""
        ss_res = self.search_semantic_scholar(query, limit // 2)
        ar_res = self.search_arxiv(query, limit // 2)
        combined = ss_res + ar_res
        combined.sort(key=lambda x: x.year, reverse=True)
        return combined
