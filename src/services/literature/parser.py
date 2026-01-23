import os
import re
from typing import Optional
from src.core.logger import get_logger, log_exception
from src.services.literature.types import PaperInfo

logger = get_logger(__name__)

# Optional imports
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

class DocParser:
    """
    Service for parsing scientific documents (PDF).
    """

    @log_exception(logger)
    def parse_pdf(self, pdf_path: str) -> Optional[PaperInfo]:
        """
        Parse a local PDF file and extract text and metadata (title/abstract).
        """
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF not found: {pdf_path}")
            return None
            
        text = ""
        title = os.path.basename(pdf_path).replace('.pdf', '').replace('_', ' ')
        
        # 1. Try pdfplumber
        if HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    # Text
                    for page in pdf.pages[:10]:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    
                    # Title heuristic
                    if pdf.pages:
                        first_page = pdf.pages[0].extract_text()
                        if first_page:
                            lines = first_page.split('\n')
                            for line in lines[:5]:
                                if len(line) > 20 and len(line) < 200:
                                    title = line.strip()
                                    break
            except Exception as e:
                logger.debug(f"pdfplumber failed for {pdf_path}: {e}")
        
        # 2. Fallback PyPDF2
        elif HAS_PYPDF2:
            try:
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages[:10]:
                         t = page.extract_text()
                         if t: text += t + "\n"
            except Exception as e:
                 logger.debug(f"PyPDF2 failed for {pdf_path}: {e}")
        else:
            logger.warning("No PDF parser (pdfplumber/PyPDF2) installed.")
            return None
            
        if not text:
            return None
            
        # Extract Abstract
        abstract = ""
        # Regex to look for "Abstract" followed by text until "Introduction" or double newline
        match = re.search(r'abstract[:\s]*(.{100,1000}?)(?=\n\n|introduction|keywords)', 
                          text, re.IGNORECASE | re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            
        return PaperInfo(
            title=title,
            abstract=abstract,
            full_text=text[:50000] # Cap size
        )
