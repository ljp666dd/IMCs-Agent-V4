import re
from typing import List, Dict, Any
from src.core.logger import get_logger, log_exception
from src.services.literature.types import PaperInfo, KnowledgeExtract

logger = get_logger(__name__)

class TopicAnalyzer:
    """
    Service for extracting structured knowledge from papers.
    """
    
    @log_exception(logger)
    def extract_knowledge(self, topic: str, papers: List[PaperInfo]) -> KnowledgeExtract:
        """Extract key findings, methods, and metrics."""
        knowledge = KnowledgeExtract(topic=topic)
        
        synthesis_keywords = ["synthesis", "preparation", "fabrication", "deposited", 
                              "annealed", "heat treatment", "electrodeposition", "solvothermal"]
        char_keywords = ["XRD", "XPS", "TEM", "SEM", "EXAFS", "XANES", "EDX", "Raman"]
        mech_keywords = ["mechanism", "d-band", "adsorption", "binding", "active site", "volcano"]
        
        seen_methods = set()
        seen_char = set()
        
        for paper in papers:
            abstract = paper.abstract.lower()
            if not abstract: continue
            
            # Findings (First 3 sentences)
            sentences = abstract.split('.')
            for sent in sentences[:3]:
                if len(sent) > 50 and any(kw in sent for kw in ["found", "show", "demonstrate", "reveal", "achieve"]):
                     knowledge.key_findings.append(sent.strip())

            # Synthesis
            for kw in synthesis_keywords:
                if kw in abstract:
                    seen_methods.add(kw.title())
            
            # Characterization
            for kw in char_keywords:
                if kw.lower() in abstract:
                    seen_char.add(kw.upper() if len(kw) <= 5 else kw)
            
            # Performance Metrics (Regex)
            # Find "123 mV" nearby "overpotential"
            if "overpotential" in abstract:
                matches = re.findall(r'(\d+)\s*mv', abstract)
                if matches and "overpotential_mV" not in knowledge.performance_metrics:
                     knowledge.performance_metrics["overpotential_mV"] = matches[0]
            
            # Mechanism
            if not knowledge.mechanism:
                for sent in sentences:
                    if any(kw in sent for kw in mech_keywords):
                        knowledge.mechanism = sent.strip()
                        break
            
            # References
            if paper.authors:
                 knowledge.references.append(f"{paper.authors[0]} et al., {paper.year}")

        knowledge.synthesis_methods = list(seen_methods)
        knowledge.characterization = list(seen_char)
        knowledge.key_findings = knowledge.key_findings[:5]
        knowledge.references = knowledge.references[:10]
        
        return knowledge

    def generate_report(self, knowledge: KnowledgeExtract) -> str:
        """Generate markdown report."""
        report = f"# Literature Review: {knowledge.topic}\n\n"
        report += "## Key Findings\n"
        for f in knowledge.key_findings:
            report += f"- {f}\n"
        
        report += f"\n## Synthesis Methods\n{', '.join(knowledge.synthesis_methods) or 'None'}\n"
        report += f"\n## Characterization\n{', '.join(knowledge.characterization) or 'None'}\n"
        
        report += "\n## Metrics\n"
        for k, v in knowledge.performance_metrics.items():
            report += f"- {k}: {v}\n"
            
        report += f"\n## Mechanism\n{knowledge.mechanism or 'None'}\n"
        
        report += "\n## References\n"
        for ref in knowledge.references:
            report += f"- {ref}\n"
            
        return report
