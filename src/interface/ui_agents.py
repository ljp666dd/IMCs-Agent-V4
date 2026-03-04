"""
Multi-Agent Catalyst Research System - Web UI
Streamlit-based interface for the multi-agent framework.

Run with: streamlit run src/ui/app.py
"""

import streamlit as st
import os
import sys
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import requests
import time

# Add project root to path (ensure local imports work with `streamlit run`)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.services.db.database import DatabaseService

try:
    from langdetect import detect, LangDetectException
    HAS_LANGDETECT = True
except Exception:
    LangDetectException = Exception
    HAS_LANGDETECT = False

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except Exception:
    HAS_TRANSLATOR = False


# Page config
from src.interface.ui_core import *

# ========== Agent Loading (Lazy) ==========

@st.cache_resource
def get_ml_agent():
    """Load ML Agent (cached, lazy)."""
    try:
        from src.agents.core import MLAgent
        return MLAgent()
    except Exception as e:
        st.error(f"ML Agent loading failed: {e}")
        return None

@st.cache_resource
def get_theory_agent():
    """Load Theory Agent (cached, lazy)."""
    try:
        from src.agents.core import TheoryDataAgent
        return TheoryDataAgent()
    except Exception as e:
        st.error(f"Theory Agent loading failed: {e}")
        return None

@st.cache_resource
def get_experiment_agent():
    """Load Experiment Agent (cached, lazy)."""
    try:
        from src.agents.core import ExperimentDataAgent
        return ExperimentDataAgent()
    except Exception as e:
        st.error(f"Experiment Agent loading failed: {e}")
        return None

@st.cache_resource
def get_literature_agent():
    """Load Literature Agent (cached, lazy)."""
    try:
        from src.agents.core import LiteratureAgent
        return LiteratureAgent()
    except Exception as e:
        st.error(f"Literature Agent loading failed: {e}")
        return None

@st.cache_resource
def get_task_manager():
    """Load Task Manager (cached, lazy)."""
    try:
        from src.agents.core import TaskManagerAgent
        return TaskManagerAgent()
    except Exception as e:
        st.error(f"Task Manager loading failed: {e}")
        return None

def load_agents():
    """Load all agents (for compatibility)."""
    return {
        'task_manager': get_task_manager(),
        'ml': get_ml_agent(),
        'theory': get_theory_agent(),
        'experiment': get_experiment_agent(),
        'literature': get_literature_agent()
    }


# ========== Data Caching ==========

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_json_data(file_path: str):
    """Load JSON file with caching."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data(ttl=3600)
def get_formation_energy_df():
    """Get formation energy as DataFrame (cached)."""
    data = load_json_data(os.path.join(ROOT_DIR, "data", "theory", "formation_energy_full.json"))
    if data:
        return pd.DataFrame(data)
    return None

@st.cache_data(ttl=3600)
def get_dos_descriptors_df():
    """Get DOS descriptors as DataFrame (cached)."""
    data = load_json_data(os.path.join(ROOT_DIR, "data", "theory", "dos_descriptors_full.json"))
    if data:
        return pd.DataFrame(data)
    return None

@st.cache_data(ttl=3600)
def get_model_results():
    """Get model comparison results (cached)."""
    return load_json_data(os.path.join(ROOT_DIR, "data", "ml_agent", "model_comparison.json"))

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_data_stats():
    """Get data statistics (cached)."""
    data_dir = os.path.join(ROOT_DIR, "data", "theory")
    stats = {}
    
    # Count CIF files
    cif_dir = os.path.join(data_dir, "cifs")
    if os.path.exists(cif_dir):
        stats['n_cifs'] = len([f for f in os.listdir(cif_dir) if f.endswith('.cif')])
    else:
        stats['n_cifs'] = 0
    
    # Count DOS records
    dos_file = os.path.join(data_dir, "orbital_pdos.json")
    if os.path.exists(dos_file):
        data = load_json_data(dos_file)
        stats['n_dos'] = len(data) if data else 0
    else:
        stats['n_dos'] = 0
    
    return stats


# ========== Evaluation Helpers ==========

def compute_evidence_coverage(db_path: str) -> Dict[str, Any]:
    """Compute evidence coverage statistics from SQLite DB."""
    if not os.path.exists(db_path):
        return {"error": f"DB not found: {db_path}"}
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM materials")
    total = cur.fetchone()[0] or 0
    cur.execute("SELECT source_type, COUNT(DISTINCT material_id) FROM evidence GROUP BY source_type")
    rows = cur.fetchall()
    conn.close()
    return {"total": total, "rows": rows}


def load_ids_from_df(df: pd.DataFrame) -> List[str]:
    """Extract material ids from a DataFrame."""
    if df is None or df.empty:
        return []
    cols = [c for c in ["material_id", "id", "mid"] if c in df.columns]
    if not cols:
        return []
    col = cols[0]
    return [str(v).strip() for v in df[col].tolist() if str(v).strip()]


def topk_recall(candidates: List[str], ground_truth: set, k: int) -> float:
    if not ground_truth:
        return 0.0
    topk = set(candidates[:k])
    return len(topk & ground_truth) / len(ground_truth)


