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
# ========== Session State ==========


@st.cache_resource(show_spinner=False)
def get_db_service() -> DatabaseService:
    return DatabaseService(os.path.join(ROOT_DIR, "data", "imcs.db"))


def load_chat_messages(db: DatabaseService, session_id: int) -> None:
    messages = db.list_chat_messages(session_id)
    st.session_state.messages = [
        {"role": m.get("role"), "content": m.get("content")} for m in messages
    ]
    last_task_id = None
    for m in messages:
        artifacts = m.get("artifacts") or {}
        if isinstance(artifacts, dict):
            plan_id = artifacts.get("plan_id")
            if plan_id:
                last_task_id = plan_id
    st.session_state.last_task_id = last_task_id
    st.session_state.last_loaded_session_id = session_id


def detect_language(text: str) -> str:
    if not text:
        return "unknown"
    if HAS_LANGDETECT:
        try:
            lang = detect(text)
            if lang.startswith("zh"):
                return "zh"
            if lang.startswith("en"):
                return "en"
            return lang
        except LangDetectException:
            pass
    # Fallback: simple heuristic
    for ch in text:
        if "一" <= ch <= "鿿":
            return "zh"
    return "en"

def translate_text(text: str, target_lang: str = "zh") -> str:
    if not text:
        return text
    if not HAS_TRANSLATOR:
        return text
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception:
        return text

UI_TRANSLATIONS = {
    "首页": "Home",
    "智能体对话": "Chat",
    "数据分析": "Data Analysis",
    "ML 训练": "ML Training",
    "文献库": "Literature",
    "API 状态": "API Status",
    "设置": "Settings",
    "Evaluation": "Evaluation",
    "语言": "Language",
    "自动翻译 UI 文本": "Auto-translate UI text",
    "科研助手": "Research Assistant",
    "正在创建任务计划...": "Creating task plan...",
    "开始执行": "Start Execution",
    "刷新状态": "Refresh Status",
    "停止自动刷新": "Stop Auto Refresh",
    "下载任务报告": "Download Task Report",
    "导航": "Navigation",
    "功能模块": "Modules",
    "数据统计": "Data Stats",
    "会话管理": "Session Manager",
    "\u641c\u7d22\u4f1a\u8bdd": "Search Sessions",
    "选择会话": "Select Session",
    "新建会话": "New Session",
    "会话标题": "Session Title",
    "保存标题": "Save Title",
    "会话操作": "Session Actions",
    "导出 JSON": "Export JSON",
    "导出 Markdown": "Export Markdown",
    "删除会话": "Delete Session",
    "确认删除": "Confirm Delete",
    "没有历史任务可加载": "No task history to load",
    "当前会话无消息": "No messages in current session",
    "加载历史任务": "Load Task History",
    "历史任务已加载": "Task history loaded",
    "CIF 数量": "CIF Count",
    "DOS 数量": "DOS Count",
    "清空缓存 (Clear Cache)": "Clear Cache",
    "清理缓存并重新加载模型组件": "Clear cache and reload model components",
    "缓存已清理，正在重新加载...": "Cache cleared, reloading...",
    "IMCs 科研平台": "IMCs Research Platform",
    "多智能体科研助手": "Multi-agent research assistant",
}


@st.cache_data(show_spinner=False)
def translate_ui_text(text: str, target_lang: str = "en") -> str:
    if not text:
        return text
    if not HAS_TRANSLATOR:
        return text
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception:
        return text

def ui_text(text: str) -> str:
    lang = st.session_state.get("ui_lang", "zh")
    if lang == "zh":
        return text
    if text in UI_TRANSLATIONS:
        return UI_TRANSLATIONS[text]
    if st.session_state.get("auto_translate_ui"):
        return translate_ui_text(text, target_lang="en")
    return text

def init_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_session_id' not in st.session_state:
        st.session_state.chat_session_id = None
    if 'last_loaded_session_id' not in st.session_state:
        st.session_state.last_loaded_session_id = None
    if 'current_task' not in st.session_state:
        st.session_state.current_task = None
    if 'agents_loaded' not in st.session_state:
        st.session_state.agents_loaded = False
    if 'active_plan' not in st.session_state:
        st.session_state.active_plan = None
    if 'active_task_id' not in st.session_state:
        st.session_state.active_task_id = None
    if 'task_status' not in st.session_state:
        st.session_state.task_status = None
    if 'task_polling' not in st.session_state:
        st.session_state.task_polling = False
    if 'selected_material' not in st.session_state:
        st.session_state.selected_material = None
    if 'last_task_id' not in st.session_state:
        st.session_state.last_task_id = None
    if 'auto_resume_last_task' not in st.session_state:
        st.session_state.auto_resume_last_task = False
    if 'last_auto_resumed_task_id' not in st.session_state:
        st.session_state.last_auto_resumed_task_id = None


