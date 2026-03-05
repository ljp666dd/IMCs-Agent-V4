import os
import re
from typing import Optional, List, Dict, Any
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

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

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

    def render_pages_to_images(self, pdf_path: str, page_indices: List[int] = None, dpi: int = 150) -> List[str]:
        """
        Render specific PDF pages to temporary image files.
        Returns a list of absolute paths to the images.
        """
        if not HAS_FITZ:
            logger.warning("fitz (PyMuPDF) not installed. Cannot render images.")
            return []
        
        if not os.path.exists(pdf_path):
            return []
            
        temp_dir = os.path.join("data", "literature", "temp_images")
        os.makedirs(temp_dir, exist_ok=True)
        
        image_paths = []
        try:
            doc = fitz.open(pdf_path)
            if not page_indices:
                page_indices = [0] # Default to first page
                
            for i in page_indices:
                if i < 0 or i >= len(doc):
                    continue
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=dpi)
                
                base_name = os.path.basename(pdf_path).replace(".pdf", "")
                img_path = os.path.join(temp_dir, f"{base_name}_p{i}.png")
                pix.save(img_path)
                image_paths.append(os.path.abspath(img_path))
                
            doc.close()
        except Exception as e:
            logger.error(f"Failed to render PDF to images: {e}")
            
        return image_paths

    def parse_images_with_vlm(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        V6 Phase I: Utilize VisionService to extract tables and curve insights from rendered pages.
        """
        results = {"tables": [], "curves": []}
        try:
            from src.services.llm.vision_service import get_vision_service
            vision = get_vision_service()
            if not vision.available:
                logger.warning("VisionService not available. Skipping VLM extraction.")
                return results

            for path in image_paths:
                # Extract tables
                tab_res = vision.analyze_page(path, task="extract_tables")
                if isinstance(tab_res, list): # Some JSON responses return a list of tables directly
                    results["tables"].extend(tab_res)
                elif isinstance(tab_res, dict) and "error" not in tab_res:
                    results["tables"].append(tab_res)
                    
                # Analyze curves
                cur_res = vision.analyze_page(path, task="analyze_curves")
                if isinstance(cur_res, dict) and "error" not in cur_res:
                    cur_res["source_image"] = os.path.basename(path)
                    results["curves"].append(cur_res)
                    
        except ImportError:
            logger.warning("Could not import VisionService.")
        except Exception as e:
            logger.error(f"VLM parsing error: {e}")
            
        return results
