import requests
from typing import List, Dict, Any, Optional
from src.core.logger import get_logger, log_exception

logger = get_logger(__name__)

class ExternalDBClient:
    """
    Service for querying external material databases (OQMD, AFLOW, Catalysis-Hub).
    """

    @log_exception(logger)
    def query_oqmd(self, elements: List[str], limit: int = 100) -> List[Dict]:
        """Query OQMD for formation energies."""
        logger.info(f"Querying OQMD for {elements}...")
        results = []
        base_url = "http://oqmd.org/oqmdapi/formationenergy"
        
        try:
            # Query first few elements to avoid timeout
            for element in elements[:3]:
                params = {
                    "fields": "name,entry_id,delta_e,stability",
                    "filter": f"element_set={element}",
                    "limit": min(limit // 3, 50),
                    "format": "json"
                }
                
                try:
                    response = requests.get(base_url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        for entry in data.get("data", []):
                            results.append({
                                "source": "OQMD",
                                "composition": entry.get("name", ""),
                                "formation_energy": entry.get("delta_e", None),
                                "stability": entry.get("stability", None)
                            })
                except Exception as e:
                    logger.debug(f"OQMD query for {element} failed: {e}")
                    
            logger.info(f"Retrieved {len(results)} entries from OQMD.")
        except Exception as e:
            logger.error(f"OQMD Service failed: {e}")
            
        return results

    @log_exception(logger)
    def query_aflow(self, elements: List[str], limit: int = 100) -> List[Dict]:
        """Query AFLOW for basic properties."""
        logger.info(f"Querying AFLOW for {elements}...")
        results = []
        base_url = "http://aflowlib.org/API/aflux/"
        
        try:
            for element in elements[:2]:
                query = f"?species({element}),paging(1)"
                try:
                    response = requests.get(base_url + query, timeout=10)
                    if response.status_code == 200:
                        lines = response.text.strip().split('\n')
                        for line in lines[:limit]:
                            if '|' in line:
                                parts = line.split('|')
                                if len(parts) >= 3:
                                    results.append({
                                        "source": "AFLOW",
                                        "auid": parts[0].strip(),
                                        "compound": parts[1].strip() if len(parts) > 1 else "",
                                        "enthalpy_formation": None 
                                    })
                except Exception:
                    continue
                    
            logger.info(f"Retrieved {len(results)} entries from AFLOW.")
        except Exception as e:
            logger.error(f"AFLOW Service failed: {e}")
            
        return results

    @log_exception(logger)
    def query_catalysis_hub(self, reaction: str = "HER", limit: int = 50) -> List[Dict]:
        """Query Catalysis-Hub for adsorption energies."""
        logger.info(f"Querying Catalysis-Hub for {reaction}...")
        results = []
        url = "https://api.catalysis-hub.org/graphql"
        
        query = """
        {
          reactions(first: %d, reactants: "H*") {
            edges {
              node {
                Equation
                reactionEnergy
                activationEnergy
                surfaceComposition
                facet
              }
            }
          }
        }
        """ % limit
        
        try:
            response = requests.post(url, json={"query": query}, timeout=15)
            if response.status_code == 200:
                data = response.json()
                edges = data.get("data", {}).get("reactions", {}).get("edges", [])
                for edge in edges:
                    node = edge.get("node", {})
                    results.append({
                        "source": "Catalysis-Hub",
                        "equation": node.get("Equation", ""),
                        "reaction_energy": node.get("reactionEnergy"),
                        "activation_energy": node.get("activationEnergy"),
                        "surface": node.get("surfaceComposition", ""),
                        "facet": node.get("facet", "")
                    })
            logger.info(f"Retrieved {len(results)} entries from CatHub.")
        except Exception as e:
            logger.error(f"CatHub Service failed: {e}")
            
        return results
