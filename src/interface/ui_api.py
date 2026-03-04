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

# ========== API Helpers (Task Graph) ==========

API_BASE_URL = os.getenv("IMCS_API_URL", "http://localhost:8000")

def api_create_task(query: str):
    res = requests.post(f"{API_BASE_URL}/tasks/create", json={"query": query}, timeout=20)
    res.raise_for_status()
    return res.json()

def api_execute_task(task_id: str):
    res = requests.post(f"{API_BASE_URL}/tasks/execute/{task_id}", timeout=20)
    res.raise_for_status()
    return res.json()

def api_get_task_status(task_id: str):
    res = requests.get(f"{API_BASE_URL}/tasks/{task_id}", timeout=20)
    res.raise_for_status()
    return res.json()

def api_confirm_gap_fill(task_id: str, run_step_ids: List[str] = None,
                         params_overrides: Dict[str, Any] = None,
                         mark_complete: bool = False):
    payload = {
        "run_step_ids": run_step_ids,
        "params_overrides": params_overrides or {},
        "mark_complete": mark_complete,
    }
    res = requests.post(f"{API_BASE_URL}/tasks/{task_id}/confirm_gap", json=payload, timeout=20)
    res.raise_for_status()
    return res.json()

@st.cache_data(ttl=60)
def api_list_materials():
    res = requests.get(f"{API_BASE_URL}/theory/materials", timeout=20)
    res.raise_for_status()
    return res.json()

@st.cache_data(ttl=120)
def api_get_materials_batch(material_ids):
    res = requests.post(
        f"{API_BASE_URL}/theory/materials/batch",
        json={"material_ids": material_ids, "include_cif": False},
        timeout=30
    )
    res.raise_for_status()
    return res.json()

@st.cache_data(ttl=120)
def api_get_material_details(material_id: str):
    res = requests.get(f"{API_BASE_URL}/theory/materials/{material_id}", timeout=20)
    res.raise_for_status()
    return res.json()


def api_knowledge_rag(query: str, material_id: str = None, top_k: int = 5, source_type: str = "literature"):
    payload = {
        "query": query,
        "material_id": material_id,
        "top_k": top_k,
        "source_type": source_type
    }
    res = requests.post(f"{API_BASE_URL}/knowledge/rag", json=payload, timeout=30)
    res.raise_for_status()
    return res.json()


def api_knowledge_entity_by_name(entity_type: str, name: str):
    res = requests.get(
        f"{API_BASE_URL}/knowledge/entities/by-name",
        params={"entity_type": entity_type, "name": name},
        timeout=20
    )
    res.raise_for_status()
    return res.json()


def api_knowledge_trace(entity_id: int, depth: int = 1):
    res = requests.get(
        f"{API_BASE_URL}/knowledge/trace/{entity_id}",
        params={"depth": depth, "limit": 200},
        timeout=20
    )
    res.raise_for_status()
    return res.json()


def api_task_report(task_id: str):
    res = requests.get(f"{API_BASE_URL}/tasks/{task_id}/report", timeout=30)
    res.raise_for_status()
    return res.json()


def api_snapshot(snapshot_id: int):
    res = requests.get(f"{API_BASE_URL}/tasks/snapshots/{snapshot_id}", timeout=30)
    res.raise_for_status()
    return res.json()


def api_knowledge_stats():
    res = requests.get(f"{API_BASE_URL}/knowledge/stats", timeout=20)
    res.raise_for_status()
    return res.json()


@st.cache_data(ttl=10)
def api_robot_tasks(limit: int = 50):
    res = requests.get(f"{API_BASE_URL}/robot/tasks", params={"limit": limit}, timeout=20)
    res.raise_for_status()
    return res.json()


@st.cache_data(ttl=10)
def api_robot_task_events(task_id: int, limit: int = 50, status: str = None):
    params = {"limit": limit}
    if status:
        params["status"] = status
    res = requests.get(f"{API_BASE_URL}/robot/task_events/{task_id}", params=params, timeout=20)
    res.raise_for_status()
    return res.json()


def api_robot_iteration_to_taskplan(
    robot_task_id: int,
    event_id: int = None,
    top_n: int = None,
    task_type: str = "catalyst_discovery",
    description: str = None,
):
    payload = {
        "robot_task_id": int(robot_task_id),
        "event_id": int(event_id) if event_id is not None else None,
        "top_n": int(top_n) if top_n is not None else None,
        "task_type": task_type,
        "description": description,
        "auto_execute": True,
    }
    res = requests.post(f"{API_BASE_URL}/robot/iteration_to_taskplan", json=payload, timeout=30)
    res.raise_for_status()
    return res.json()

def load_knowledge_pack(task_id: str):
    if not task_id:
        return None
    path = os.path.join(ROOT_DIR, "data", "tasks", f"knowledge_{task_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def load_strategy_stats():
    path = os.path.join(ROOT_DIR, "data", "strategy", "strategy_stats.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def load_strategy_feedback(task_id: str):
    if not task_id:
        return None
    path = os.path.join(ROOT_DIR, "data", "strategy", "feedback", f"feedback_{task_id}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

