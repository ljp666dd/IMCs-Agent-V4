"""
IMCs Literature Crawler (V6 - Phase I)

Automated literature retrieval from open-access preprint servers (arXiv).
Targets:
- Condensed Matter Physics (cond-mat.mtrl-sci)
- Chemical Physics (physics.chem-ph)
Keywords: electrocatalysis, HER, OER, HOR, ORR, fuel cell, intermetallic
"""

import os
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.core.logger import get_logger

logger = get_logger(__name__)


class ArxivCrawler:
    """
    Crawls latest publications from arXiv via its public API.
    """

    BASE_URL = "http://export.arxiv.org/api/query?"
    
    # Pre-defined query for electrocatalysis IMCs
    DEFAULT_QUERY = 'all:"electrocatalysis" OR all:"hydrogen oxidation" OR all:"intermetallic"'

    def __init__(self, output_dir: str = "data/literature/raw"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"ArxivCrawler initialized. Out dir: {self.output_dir}")

    def fetch_latest_papers(self, query: str = None, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch metadata of latest papers matching the query.
        """
        search_query = query or self.DEFAULT_QUERY
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        url = self.BASE_URL + urllib.parse.urlencode(params)
        logger.info(f"Fetching arXiv data: {url}")
        
        results = []
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                xml_data = response.read()
                
            root = ET.fromstring(xml_data)
            # Namespace for arXiv XML
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns).text
                if title: title = title.replace('\\n', ' ').strip()
                
                summary = entry.find('atom:summary', ns).text
                if summary: summary = summary.replace('\\n', ' ').strip()
                
                published = entry.find('atom:published', ns).text
                entry_id = entry.find('atom:id', ns).text
                
                # Find PDF link
                pdf_url = None
                for link in entry.findall('atom:link', ns):
                    if link.attrib.get('title') == 'pdf':
                        pdf_url = link.attrib.get('href')
                        break
                
                results.append({
                    "id": entry_id.split('/abs/')[-1] if entry_id else "unknown",
                    "title": title,
                    "abstract": summary,
                    "published": published,
                    "pdf_url": pdf_url,
                    "source": "arxiv"
                })
                
            logger.info(f"Found {len(results)} papers.")
            
        except Exception as e:
            logger.error(f"Failed to fetch arXiv papers: {e}")
            
        return results

    def download_paper(self, paper: Dict[str, Any]) -> Optional[str]:
        """
        Download the PDF of a paper. Returns absolute path to local PDF.
        """
        pdf_url = paper.get("pdf_url")
        if not pdf_url:
            logger.warning(f"No PDF URL for {paper.get('title')}")
            return None
            
        paper_id = paper.get("id", "unknown").replace(".", "_")
        # Ensure it has .pdf extension for downloader
        if not pdf_url.endswith('.pdf'):
            pdf_url += '.pdf'
            
        safe_title = "".join(c for c in paper.get("title", paper_id)[:30] if c.isalnum() or c in " -_").strip()
        filename = f"arxiv_{paper_id}_{safe_title.replace(' ', '_')}.pdf"
        filepath = os.path.join(self.output_dir, filename)
        
        if os.path.exists(filepath):
            logger.info(f"Paper already downloaded: {filename}")
            return os.path.abspath(filepath)
            
        try:
            logger.info(f"Downloading: {filename} from {pdf_url}")
            urllib.request.urlretrieve(pdf_url, filepath)
            # Be kind to the server
            time.sleep(1.0)
            return os.path.abspath(filepath)
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            return None

    def execute_daily_crawl(self, max_papers: int = 5) -> List[str]:
        """
        Full crawl workflow: fetch latest -> download PDFs.
        Returns list of newly downloaded PDF paths.
        """
        logger.info("Starting daily autonomous crawl...")
        papers = self.fetch_latest_papers(max_results=max_papers)
        
        downloaded_paths = []
        for p in papers:
            path = self.download_paper(p)
            if path:
                downloaded_paths.append(path)
                
        logger.info(f"Crawl finished. Downloaded {len(downloaded_paths)} new PDFs.")
        return downloaded_paths

if __name__ == "__main__":
    # Test crawler
    logging.basicConfig(level=logging.INFO)
    crawler = ArxivCrawler()
    crawler.execute_daily_crawl(max_papers=2)
