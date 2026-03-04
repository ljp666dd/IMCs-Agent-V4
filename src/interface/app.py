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
from src.interface.ui_api import *
from src.interface.ui_components import *
from src.interface.ui_agents import *
from src.interface.ui_pages import *

st.set_page_config(
    page_title="IMCs Research System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with modern scientific research theme
st.markdown("""
<style>
    /* ========== 全局字体和变量 ========== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --primary-color: #60a5fa;
        --primary-dark: #3b82f6;
        --secondary-color: #a78bfa;
        --accent-color: #22d3ee;
        --success-color: #34d399;
        --warning-color: #fbbf24;
        --error-color: #f87171;
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-tertiary: #334155;
        --text-primary: #ffffff;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --border-color: #475569;
    }
    
    /* ========== 主标题样式 ========== */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 1.5rem 0;
        letter-spacing: -0.02em;
        text-shadow: 0 0 40px rgba(59, 130, 246, 0.3);
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #e2e8f0;
        text-align: center;
        margin-top: -0.5rem;
        letter-spacing: 0.05em;
    }
    
    /* ========== 智能体卡片 ========== */
    .agent-card {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 1.5rem;
        border-radius: 16px;
        margin: 0.75rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .agent-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-color: rgba(59, 130, 246, 0.5);
    }
    
    .agent-card-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .agent-card-desc {
        font-size: 0.9rem;
        color: #e2e8f0;
        line-height: 1.6;
    }
    
    /* ========== 指标卡片 ========== */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.6) 100%);
        border: 1px solid var(--border-color);
        padding: 1.25rem;
        border-radius: 12px;
        border-left: 4px solid var(--primary-color);
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        border-left-color: var(--accent-color);
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.7) 100%);
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--primary-color);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #e2e8f0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    
    /* ========== 状态指示器 ========== */
    .api-status {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.375rem 0.875rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.025em;
    }
    
    .api-ok {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .api-fail {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .api-pending {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    /* ========== 功能标签 ========== */
    .feature-badge {
        display: inline-flex;
        align-items: center;
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.25) 0%, rgba(167, 139, 250, 0.25) 100%);
        color: #e0e7ff;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.15rem;
        border: 1px solid rgba(167, 139, 250, 0.4);
        transition: all 0.2s ease;
    }
    
    .feature-badge:hover {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, rgba(139, 92, 246, 0.3) 100%);
        border-color: rgba(139, 92, 246, 0.5);
        transform: scale(1.05);
    }
    
    /* ========== 科研仪表板样式 ========== */
    .dashboard-section {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-title::before {
        content: '';
        display: inline-block;
        width: 4px;
        height: 1.25rem;
        background: linear-gradient(180deg, var(--primary-color), var(--secondary-color));
        border-radius: 2px;
    }
    
    /* ========== 快捷入口按钮 ========== */
    .quick-action-btn {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    
    .quick-action-btn:hover {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
        border-color: rgba(59, 130, 246, 0.5);
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.2);
    }
    
    .quick-action-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .quick-action-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text-primary);
    }
    
    /* ========== 聊天界面样式 ========== */
    .chat-container {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1rem;
    }
    
    .chat-message {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        max-width: 85%;
        line-height: 1.6;
    }
    
    .chat-message-user {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    .chat-message-assistant {
        background: rgba(51, 65, 85, 0.8);
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        border-bottom-left-radius: 4px;
    }
    
    /* ========== 数据表格增强 ========== */
    .dataframe {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
    }
    
    /* ========== 进度条样式 ========== */
    .progress-bar {
        height: 8px;
        background: rgba(51, 65, 85, 0.5);
        border-radius: 4px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    /* ========== 图标动画 ========== */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .icon-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    .icon-spin {
        animation: spin 1s linear infinite;
    }
    
    /* ========== 侧边栏优化 ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.98) 0%, rgba(30, 41, 59, 0.98) 100%);
    }
    
    /* 侧边栏导航文字增强可读性 */
    [data-testid="stSidebar"] .stRadio > label {
        background: transparent;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        transition: all 0.2s ease;
        color: #ffffff !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label span {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label:hover {
        background: rgba(96, 165, 250, 0.15);
    }
    
    /* 侧边栏所有文字增强 */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* 侧边栏标题样式 */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* 侧边栏 markdown 文字 */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown strong {
        color: #ffffff !important;
    }
    
    /* 侧边栏按钮样式 - 增强可读性 */
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stButton > button > div,
    [data-testid="stSidebar"] .stButton > button > div > p,
    [data-testid="stSidebar"] .stButton > button span,
    [data-testid="stSidebar"] button[kind="secondary"],
    [data-testid="stSidebar"] button[kind="primary"] {
        color: #ffffff !important;
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.35) 0%, rgba(167, 139, 250, 0.35) 100%) !important;
        border: 1px solid rgba(96, 165, 250, 0.5) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] button[kind="secondary"]:hover,
    [data-testid="stSidebar"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.5) 0%, rgba(167, 139, 250, 0.5) 100%) !important;
        border-color: rgba(96, 165, 250, 0.7) !important;
        color: #ffffff !important;
    }
    
    /* 侧边栏指标样式 */
    [data-testid="stSidebar"] [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #60a5fa !important;
        font-weight: 600 !important;
    }
    
    /* ========== 分隔线 ========== */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 1.5rem 0;
    }
    
    /* ========== 代码块样式 ========== */
    code {
        font-family: 'JetBrains Mono', monospace;
        background: rgba(51, 65, 85, 0.5);
        padding: 0.125rem 0.375rem;
        border-radius: 4px;
        font-size: 0.85em;
    }
    
    /* ========== 工具提示 ========== */
    .tooltip {
        position: relative;
        cursor: help;
    }
    
    .tooltip::after {
        content: attr(data-tip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: var(--bg-tertiary);
        color: var(--text-primary);
        padding: 0.5rem 0.75rem;
        border-radius: 6px;
        font-size: 0.75rem;
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.2s ease;
    }
    
    .tooltip:hover::after {
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)


# ========== Sidebar ==========

def render_sidebar():
    """Render sidebar with navigation and status."""
    with st.sidebar:
        # 平台 Logo 和标题
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔬</div>
            <div style="font-size: 1.25rem; font-weight: 700; background: linear-gradient(135deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">IMCs 科研平台</div>
            <div style="font-size: 0.75rem; color: #94a3b8;">Multi-Agent Research System</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        if "ui_lang" not in st.session_state:
            st.session_state.ui_lang = "zh"
        if "auto_translate_ui" not in st.session_state:
            st.session_state.auto_translate_ui = False

        lang_choice = st.selectbox(
            "🌐 语言 / Language",
            ["中文", "English"],
            index=0 if st.session_state.ui_lang == "zh" else 1,
            key="ui_lang_select",
        )
        st.session_state.ui_lang = "zh" if lang_choice == "中文" else "en"
        st.checkbox(ui_text("自动翻译 UI 文本"), key="auto_translate_ui")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # 导航菜单 - 添加图标
        pages = [
            ("home", "🏠 首页", "🏠 Home"),
            ("chat", "💬 智能对话", "💬 Chat"),
            ("data", "📊 数据分析", "📊 Data Analysis"),
            ("ml", "🧠 ML 训练", "🧠 ML Training"),
            ("lit", "📚 文献库", "📚 Literature"),
            ("api", "🔌 API 状态", "🔌 API Status"),
            ("strategy", "📈 策略反馈", "📈 Strategy"),
            ("robot", "🔁 闭环迭代", "🔁 Closed-loop"),
            ("settings", "⚙️ 设置", "⚙️ Settings"),
        ]
        labels = [p[1] if st.session_state.ui_lang == "zh" else p[2] for p in pages]
        if "page" not in st.session_state:
            st.session_state.page = "home"
        default_label = labels[0]
        label_to_key = {
            (p[1] if st.session_state.ui_lang == "zh" else p[2]): p[0] for p in pages
        }
        for lbl, key in label_to_key.items():
            if key == st.session_state.page:
                default_label = lbl
                break
        page_label = st.radio(
            ui_text("导航"),
            labels,
            index=labels.index(default_label),
            label_visibility="collapsed",
        )
        page = label_to_key.get(page_label, "home")
        st.session_state.page = page
        if st.checkbox(ui_text("📋 Evaluation"), key="nav_evaluation"):
            page = "evaluation"

        st.markdown("---")

        st.markdown(ui_text("### 功能模块"))
        st.markdown("""
        **ML Agent**
        <span class="feature-badge">CGCNN</span>
        <span class="feature-badge">SchNet</span>
        <span class="feature-badge">Transformer</span>
        """, unsafe_allow_html=True)

        st.markdown("""
        **Theory Agent**
        <span class="feature-badge">MP</span>
        <span class="feature-badge">AFLOW</span>
        <span class="feature-badge">Catalysis-Hub</span>
        """, unsafe_allow_html=True)

        st.markdown("""
        **Experiment Agent**
        <span class="feature-badge">LSV</span>
        <span class="feature-badge">EIS</span>
        <span class="feature-badge">RDE</span>
        """, unsafe_allow_html=True)

        st.markdown("""
        **Literature Agent**
        <span class="feature-badge">Semantic Scholar</span>
        <span class="feature-badge">arXiv</span>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(ui_text("### 数据统计"))
        stats = get_data_stats()
        st.metric(ui_text("CIF 数量"), f"{stats.get('n_cifs', 0)}")
        st.metric(ui_text("DOS 数量"), f"{stats.get('n_dos', 0)}")
        st.markdown("---")
        if st.button(
            ui_text("清空缓存 (Clear Cache)"),
            help=ui_text("清理缓存并重新加载模型组件"),
        ):
            st.cache_resource.clear()
            import sys
            import importlib
            modules_to_reload = ["src.agents.core.ml_agent", "src.agents.core"]
            for m in modules_to_reload:
                if m in sys.modules:
                    try:
                        importlib.reload(sys.modules[m])
                    except Exception:
                        pass
            st.success(ui_text("缓存已清理，正在重新加载..."))
            st.rerun()
        return page

# ========== Main ==========

def main():
    """Main application."""
    init_session_state()
    
    # Check for stale agents (ensure KNOWN_FEATURES is present)
    # This fixes the issue where cached MLAgent instance uses old logic (leaking targets as features)
    agents = load_agents()
    if agents and 'ml' in agents:
        ml_agent = agents['ml']
        need_reset = False
        
        # Check 1: Version/Attribute check
        if not hasattr(ml_agent, "KNOWN_FEATURES"):
            need_reset = True
            
        # Check 2: Missing models check (AdaBoost/ElasticNet)
        if not need_reset:
            try:
                trad_models = ml_agent.get_traditional_models()
                if "AdaBoost" not in trad_models or "ElasticNet" not in trad_models:
                    need_reset = True
            except:
                pass
        
        if need_reset:
            st.warning("🔄 检测到核心算法组件更新 (Feature Safety & New Models)，正在自动重置环境...")
            st.cache_resource.clear()
            st.rerun()
    
    page = render_sidebar()
    if page == "evaluation":
        render_evaluation()
        return

    if page == "home":
        render_home()
    elif page == "chat":
        render_chat()
    elif page == "data":
        render_data_analysis()
    elif page == "ml":
        render_ml_training()
    elif page == "lit":
        render_literature()
    elif page == "api":
        render_api_status()
    elif page == "strategy":
        render_strategy_feedback()
    elif page == "robot":
        render_closed_loop()
    elif page == "settings":
        render_settings()

if __name__ == "__main__":
    main()
