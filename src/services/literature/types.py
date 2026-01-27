from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class PaperInfo:
    """Information about a scientific paper."""
    title: str
    authors: List[str] = field(default_factory=list)
    year: int = 0
    abstract: str = ""
    doi: str = ""
    url: str = ""
    pdf_url: str = ""
    is_open_access: bool = False
    citation_count: int = 0
    keywords: List[str] = field(default_factory=list)
    full_text: str = ""

@dataclass
class KnowledgeExtract:
    """Extracted knowledge from literature."""
    topic: str
    key_findings: List[str] = field(default_factory=list)
    synthesis_methods: List[str] = field(default_factory=list)
    characterization: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    mechanism: str = ""
    references: List[str] = field(default_factory=list)
