"""
IMCs Knowledge Graph Engine (V6 - Phase I)

Implements an in-memory NetworkX knowledge graph backed by SQLite storage parsing.
Used to find implicit connections (e.g., doping impact, paper sources, shared topologies).
"""

import os
import sqlite3
import json
from typing import Dict, List, Any, Optional
from src.core.logger import get_logger

logger = get_logger(__name__)

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

class KnowledgeGraphDB:
    """
    Lightweight Graph Database wrapper using NetworkX.
    Maps material components, performance metrics, and evidence sources.
    """
    def __init__(self, db_path: str = "data/imcs.db", graph_export_dir: str = "data/knowledge_graph"):
        self.db_path = db_path
        self.export_dir = graph_export_dir
        os.makedirs(self.export_dir, exist_ok=True)
        self.graph = nx.DiGraph() if HAS_NX else None
        
        if not HAS_NX:
            logger.warning("networkx is not installed. Graph capabilities disabled.")
        else:
            logger.info("KnowledgeGraphDB initialized.")

    def rebuild_graph_from_sqlite(self):
        """
        Reads existing Materials and Evidence from SQLite and constructs the NetworkX DiGraph.
        """
        if not self.graph: return
        self.graph.clear()
        
        if not os.path.exists(self.db_path):
            logger.warning(f"Database {self.db_path} not found.")
            return
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                
                # 1. Add material nodes
                cur.execute("SELECT material_id, formula FROM candidates")
                for row in cur.fetchall():
                    mid = row["material_id"]
                    self.graph.add_node(mid, type="material", formula=row["formula"])
                    
                # 2. Add properties / evidence edges
                cur.execute("SELECT material_id, source_type, source_id, score, metadata FROM evidence")
                for row in cur.fetchall():
                    mid = row["material_id"]
                    source_type = row["source_type"]
                    source_id = row["source_id"]
                    score = row["score"]
                    meta_str = row["metadata"]
                    
                    if not self.graph.has_node(mid): continue
                    
                    # Create evidence node if it relates to literature or external graph
                    evidence_node_id = f"evi_{source_type}_{source_id}"
                    if not self.graph.has_node(evidence_node_id):
                        self.graph.add_node(evidence_node_id, type="evidence", source=source_type, ref=source_id)
                        
                    # Add edge describing relationship
                    self.graph.add_edge(mid, evidence_node_id, weight=score or 1.0, relation="supported_by")
                    
                    # Parse metadata for metrics if available
                    if meta_str:
                        try:
                            meta = json.loads(meta_str)
                            if "energy_per_atom_ev" in meta:
                                metric_id = f"metric_stab_{mid}"
                                self.graph.add_node(metric_id, type="metric", name="stability", value=meta["energy_per_atom_ev"])
                                self.graph.add_edge(mid, metric_id, relation="has_property")
                        except Exception:
                            pass
                            
            logger.info(f"Graph built successfully: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")
        except Exception as e:
            logger.error(f"Failed to rebuild graph: {e}")

    def query_similar_materials(self, material_id: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """
        Find materials that share evidence sources or structural properties within N hops.
        """
        if not self.graph or not self.graph.has_node(material_id):
            return []
            
        similar = []
        try:
            # Get shortest path ego graph
            ego = nx.ego_graph(self.graph, material_id, radius=max_hops, undirected=True)
            for n, data in ego.nodes(data=True):
                if n != material_id and data.get("type") == "material":
                    # Calculate graph metric (e.g. shortest path length)
                    dist = nx.shortest_path_length(ego, source=material_id, target=n)
                    similar.append({"material_id": n, "formula": data.get("formula"), "graph_distance": dist})
                    
            # Sort by distance
            similar.sort(key=lambda x: x["graph_distance"])
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            
        return similar

    def export_to_json(self):
        """Export graph to d3.js compatible JSON format for frontend visualization."""
        if not self.graph: return None
        
        try:
            data = nx.node_link_data(self.graph)
            out_path = os.path.join(self.export_dir, "graph_export.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Graph exported to {out_path}")
            return out_path
        except Exception as e:
            logger.error(f"Graph export failed: {e}")
            return None

def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    gdb = KnowledgeGraphDB()
    gdb.rebuild_graph_from_sqlite()
    gdb.export_to_json()

if __name__ == "__main__":
    main()
