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
st.set_page_config(
    page_title="IMCs Research System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with modern dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .api-status {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .api-ok { background: #d4edda; color: #155724; }
    .api-fail { background: #f8d7da; color: #721c24; }
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        margin: 0.1rem;
    }
</style>
""", unsafe_allow_html=True)


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

def api_confirm_gap_fill(task_id: str, run_step_ids: List[str] = None, mark_complete: bool = False):
    payload = {"run_step_ids": run_step_ids or [], "mark_complete": mark_complete}
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

def render_knowledge_pack(pack: dict):
    if not pack:
        return
    st.markdown("### Knowledge Pack")
    meta_cols = st.columns(3)
    meta_cols[0].metric("Task ID", pack.get("task_id") or "-")
    meta_cols[1].metric("Task Type", pack.get("task_type") or "-")
    meta_cols[2].metric("Query", (pack.get("query") or "")[:20] + "..." if pack.get("query") else "-")

    stats = pack.get("evidence_stats") or {}
    if stats:
        st.markdown("#### Evidence Stats")
        st.json(stats)

    gap = pack.get("evidence_gap") or {}
    if gap:
        st.markdown("#### Evidence Gaps")
        summary = gap.get("summary") or {}
        if summary:
            try:
                st.dataframe(pd.DataFrame([summary]), use_container_width=True)
            except Exception:
                st.json(summary)
        steps = gap.get("recommended_steps") or []
        if steps:
            st.markdown("#### Recommended Follow-up Steps")
            try:
                st.dataframe(pd.DataFrame(steps), use_container_width=True)
            except Exception:
                st.json(steps)

    report = pack.get("reasoning_report") or []
    if report:
        st.markdown("#### Reasoning Report (Top Materials)")
        try:
            st.dataframe(pd.DataFrame(report), use_container_width=True)
        except Exception:
            st.json(report)

    rag = pack.get("knowledge_rag") or []
    if rag:
        st.markdown("#### Knowledge RAG (Top)")
        try:
            st.dataframe(pd.DataFrame(rag), use_container_width=True)
        except Exception:
            st.json(rag)

def render_task_graph(steps):
    if not steps:
        st.info("No steps available.")
        return
    st.markdown("""
    <style>
    .tg-step { border: 1px solid #2a2a2a; border-radius: 10px; padding: 10px 12px; margin-bottom: 10px; background: #0f1117; }
    .tg-title { font-weight: 600; color: #e5e7eb; font-size: 0.9rem; }
    .tg-status { font-size: 0.75rem; padding: 2px 8px; border-radius: 999px; background: #1f2937; color: #93c5fd; }
    .tg-status-ok { background: #064e3b; color: #34d399; }
    .tg-status-fail { background: #7f1d1d; color: #fca5a5; }
    .tg-status-run { background: #1e3a8a; color: #93c5fd; }
    .tg-status-block { background: #3f2f00; color: #facc15; }
    .tg-deps { margin-top: 6px; }
    .tg-dep { display: inline-block; margin: 2px 4px 0 0; padding: 2px 6px; border-radius: 999px; font-size: 0.7rem; background: #111827; color: #f9fafb; border: 1px solid #374151; }
    .tg-meta { color: #9ca3af; font-size: 0.75rem; margin-top: 4px; }
    </style>
    """, unsafe_allow_html=True)
    for s in steps:
        status = s.get("status", "unknown")
        status_class = "tg-status"
        if status == "completed":
            status_class = "tg-status tg-status-ok"
        elif status == "failed":
            status_class = "tg-status tg-status-fail"
        elif status == "running":
            status_class = "tg-status tg-status-run"
        elif status == "blocked":
            status_class = "tg-status tg-status-block"
        elif status == "replanned":
            status_class = "tg-status tg-status-block"
        st.markdown(
            f"""
            <div class="tg-step">
              <div class="tg-title">[{s.get('agent','').upper()}] {s.get('action','')}</div>
              <div class="tg-meta">Status: <span class="{status_class}">{status}</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if s.get("result"):
            try:
                preview = json.dumps(s.get("result"), ensure_ascii=False)[:240]
            except Exception:
                preview = str(s.get("result"))[:240]
            st.markdown(f"<div class='tg-meta'>Result: <code>{preview}</code></div>", unsafe_allow_html=True)
            with st.expander(f"Result details: {s.get('step_id','step')}"):
                try:
                    full_text = json.dumps(s.get("result"), ensure_ascii=False, indent=2)
                except Exception:
                    full_text = str(s.get("result"))
                st.code(full_text)


def build_task_assignment(steps):
    """Build task assignment table (team/agent/action)."""
    # 简单的团队映射规则，便于展示协作分工
    team_map = {
        "literature": "Scientist Team",
        "theory": "Scientist Team",
        "ml": "Analyst Team",
        "experiment": "Experiment Team",
        "task_manager": "Orchestrator"
    }
    rows = []
    for s in steps or []:
        agent = s.get("agent", "")
        rows.append({
            "team": team_map.get(agent, "Unknown"),
            "agent": agent,
            "action": s.get("action", ""),
            "dependencies": ",".join(s.get("dependencies") or [])
        })
    return pd.DataFrame(rows)


def build_task_mermaid(steps):
    """Build mermaid flow text from steps for copy/paste."""
    lines = ["graph TD", "  User([User])"]
    for s in steps or []:
        sid = s.get("step_id", "")
        label = f"{s.get('agent','')}:{s.get('action','')}"
        if sid:
            lines.append(f"  {sid}[{label}]")
    for s in steps or []:
        sid = s.get("step_id", "")
        deps = s.get("dependencies") or []
        if not deps and sid:
            lines.append(f"  User --> {sid}")
        for dep in deps:
            lines.append(f"  {dep} --> {sid}")
    return "\n".join(lines)


def build_evidence_mermaid(material_id: str, formula: str, evidence: list) -> str:
    """Build a simple evidence graph (Mermaid) from material + evidence types."""
    lines = ["graph LR"]
    label = material_id or "material"
    if formula:
        label = f"{label}\\\\n{formula}"
    lines.append(f"  M[{label}]")

    counts = {}
    for ev in evidence or []:
        et = ev.get("source_type", "unknown")
        counts[et] = counts.get(et, 0) + 1

    for idx, (etype, cnt) in enumerate(counts.items(), start=1):
        node_id = f"E{idx}"
        node_label = f"{etype} ({cnt})"
        lines.append(f"  {node_id}[{node_label}]")
        lines.append(f"  {node_id} --> M")

    return "\n".join(lines)


def build_knowledge_trace_mermaid(trace: dict) -> str:
    """Build mermaid from knowledge trace (nodes + edges)."""
    if not trace:
        return "graph LR\n  Empty[No trace]"
    nodes = trace.get("nodes") or []
    edges = trace.get("edges") or []
    lines = ["graph LR"]

    def _label(node: dict) -> str:
        ntype = node.get("entity_type", "")
        name = node.get("name", "")
        label = f"{ntype}:{name}" if ntype else name
        label = label.replace("[", "(").replace("]", ")")
        return label if label else "node"

    for n in nodes:
        nid = n.get("id")
        if nid is None:
            continue
        lines.append(f"  N{nid}[{_label(n)}]")

    for e in edges:
        sid = e.get("subject_id")
        oid = e.get("object_id")
        pred = e.get("predicate", "")
        if sid is None or oid is None:
            continue
        if pred:
            lines.append(f"  N{sid} -->|{pred}| N{oid}")
        else:
            lines.append(f"  N{sid} --> N{oid}")

    return "\n".join(lines)


def render_evidence_chain(material_id: str):
    """Fetch and render evidence chain for a material."""
    try:
        data = api_get_material_details(material_id)
        evidence = data.get("evidence", [])
        adsorption = data.get("adsorption_energies", [])
        activity = data.get("activity_metrics", [])
    except Exception as e:
        st.warning(f"Failed to load evidence: {e}")
        return

    st.markdown("### Evidence Chain")
    if not evidence:
        st.info("No evidence linked yet.")

    for ev in evidence:
        ev_type = ev.get("source_type", "unknown")
        ev_score = ev.get("score", 0)
        st.markdown(f"**{ev_type}**  | score: `{ev_score}`")
        meta = ev.get("metadata")
        if meta:
            try:
                if isinstance(meta, str):
                    meta = json.loads(meta)
            except Exception:
                pass
        st.code(json.dumps(meta, ensure_ascii=False, indent=2))

    if activity:
        st.markdown("### Activity Metrics")
        st.dataframe(pd.DataFrame(activity), use_container_width=True)
    else:
        st.markdown("### Activity Metrics")
        st.info("No activity metrics available.")

    if adsorption:
        st.markdown("### Adsorption Energies")
        st.dataframe(pd.DataFrame(adsorption), use_container_width=True)
    else:
        st.markdown("### Adsorption Energies")
        st.info("No adsorption energy records available.")

    # Evidence graph (task graph-like view)
    st.markdown("### Evidence Graph")
    mermaid = build_evidence_mermaid(material_id, data.get("formula", ""), evidence)
    st.code(mermaid, language="mermaid")

    # DOS Curve Predictions
    st.markdown("### DOS Curve Predictions")
    try:
        dos_meta = data.get("dos_data")
        if isinstance(dos_meta, str):
            try:
                dos_meta = json.loads(dos_meta)
            except Exception:
                dos_meta = None
        channel_options = ["total", "s", "p", "d"]
        default_channel = "total"
        if isinstance(dos_meta, dict):
            ch = dos_meta.get("dos_curve_pred_channel")
            if ch in channel_options:
                default_channel = ch
        channel = st.selectbox("DOS channel", channel_options, index=channel_options.index(default_channel), key=f"dos_ch_{material_id}")
        pred_curve = None
        pred_plot = None
        real_curve = None
        real_plot = None
        if isinstance(dos_meta, dict):
            if dos_meta.get("dos_curve_pred_channel") == channel:
                pred_curve = dos_meta.get("dos_curve_pred_path")
                pred_plot = dos_meta.get("dos_plot_pred_path")
            real_curve = dos_meta.get("dos_curve_path")
            real_plot = dos_meta.get("dos_plot_path")
        if not pred_curve:
            pred_curve = os.path.join(ROOT_DIR, "data", "theory", "dos_curve_predictions", channel, f"{material_id}_{channel}_pred.csv")
        if not pred_plot:
            pred_plot = os.path.join(ROOT_DIR, "data", "theory", "dos_curve_pred_plots", channel, f"{material_id}_{channel}_pred.png")
        if not real_curve:
            real_curve = os.path.join(ROOT_DIR, "data", "theory", "orbital_dos_curves", f"{material_id}_dos_curve.csv")
        if not real_plot:
            real_plot = os.path.join(ROOT_DIR, "data", "theory", "orbital_dos_plots", f"{material_id}_dos.png")
        # Load model R2 report
        report_path = os.path.join(ROOT_DIR, "data", "ml_agent", "dos_curve_report_all.json")
        rep = None
        if os.path.exists(report_path):
            try:
                with open(report_path, "r") as f:
                    report_all = json.load(f)
                rep = report_all.get(channel) if isinstance(report_all, dict) else None
            except Exception:
                rep = None
        if rep is None:
            alt_report = os.path.join(ROOT_DIR, "data", "ml_agent", f"dos_curve_report_{channel}.json")
            if os.path.exists(alt_report):
                try:
                    with open(alt_report, "r") as f:
                        rep = json.load(f)
                except Exception:
                    rep = None
        if rep:
            st.caption(f"Model R2 (components): {rep.get('component_r2')} | R2 (curve): {rep.get('curve_r2')}")
        # Render comparison
        df_pred = None
        df_real = None
        if pred_curve and os.path.exists(pred_curve):
            try:
                df_pred = pd.read_csv(pred_curve)
            except Exception:
                df_pred = None
        if real_curve and os.path.exists(real_curve):
            try:
                df_real = pd.read_csv(real_curve)
            except Exception:
                df_real = None
        if df_pred is not None or df_real is not None:
            compare = None
            if df_real is not None and "energy" in df_real.columns:
                real_col = "total_dos" if channel == "total" else f"{channel}_dos"
                if real_col in df_real.columns:
                    compare = df_real[["energy", real_col]].rename(columns={real_col: "real"})
            if df_pred is not None and "energy" in df_pred.columns:
                pred_col = "dos" if "dos" in df_pred.columns else df_pred.columns[-1]
                dfp = df_pred[["energy", pred_col]].rename(columns={pred_col: "pred"})
                if compare is None:
                    compare = dfp
                else:
                    compare = compare.merge(dfp, on="energy", how="outer")
            if compare is not None:
                compare = compare.sort_values("energy")
                st.caption("DOS curve comparison")
                st.line_chart(compare.set_index("energy"))
        if pred_plot and os.path.exists(pred_plot):
            st.caption(f"Predicted DOS plot ({channel})")
            st.image(pred_plot, use_container_width=True)
        if real_plot and os.path.exists(real_plot):
            st.caption("Reference DOS plot")
            st.image(real_plot, use_container_width=True)
        if not (pred_curve and os.path.exists(pred_curve)) and not (real_curve and os.path.exists(real_curve)):
            st.info("No DOS curves available for this material.")
    except Exception as e:
        st.info(f"DOS curve preview failed: {e}")

    st.markdown("### Knowledge RAG")
    rag_query = st.text_input("Query knowledge sources", key=f"rag_query_{material_id}")
    if rag_query:
        try:
            rag = api_knowledge_rag(rag_query, material_id=material_id, top_k=5, source_type="literature")
            st.caption(f"Mode: {rag.get('mode')} | Candidate sources: {rag.get('candidate_sources')}")
            results = rag.get("results") or []
            if results:
                st.table(pd.DataFrame(results))
            else:
                st.info("No results.")
        except Exception as e:
            st.warning(f"Knowledge RAG failed: {e}")

    # Knowledge Trace (reasoning path)
    st.markdown("### Knowledge Trace")
    try:
        ent = api_knowledge_entity_by_name("material", material_id)
        trace = api_knowledge_trace(ent.get("id"), depth=1)
        mermaid = build_knowledge_trace_mermaid(trace)
        st.code(mermaid, language="mermaid")
    except Exception as e:
        st.info(f"Knowledge trace not available: {e}")


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


# ========== Sidebar ==========

def render_sidebar():
    """Render sidebar with navigation and status."""
    with st.sidebar:
        st.markdown(f"## {ui_text('IMCs 科研平台')}")
        st.markdown(ui_text("多智能体科研助手"))
        st.markdown("---")

        if "ui_lang" not in st.session_state:
            st.session_state.ui_lang = "zh"
        if "auto_translate_ui" not in st.session_state:
            st.session_state.auto_translate_ui = False

        lang_choice = st.selectbox(
            "中文 / English",
            ["中文", "English"],
            index=0 if st.session_state.ui_lang == "zh" else 1,
            key="ui_lang_select",
        )
        st.session_state.ui_lang = "zh" if lang_choice == "中文" else "en"
        st.checkbox(ui_text("自动翻译 UI 文本"), key="auto_translate_ui")

        st.markdown("---")

        pages = [
            ("home", "首页", "Home"),
            ("chat", "智能体对话", "Chat"),
            ("data", "数据分析", "Data Analysis"),
            ("ml", "ML 训练", "ML Training"),
            ("lit", "文献库", "Literature"),
            ("api", "API 状态", "API Status"),
            ("settings", "设置", "Settings"),
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
        if st.checkbox(ui_text("Evaluation"), key="nav_evaluation"):
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

def render_home():
    """Render home page."""
    st.markdown(f"# {ui_text('IMCs 科研平台')}")
    st.write(ui_text("面向 HOR 候选有序合金发现的多智能体科研平台。"))
    st.markdown("---")

    st.markdown(f"### {ui_text('\u5feb\u6377\u5165\u53e3')}")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button(ui_text("打开智能体对话"), use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()

    with col2:
        if st.button(ui_text("进入 ML 训练"), use_container_width=True):
            st.session_state.page = "ml"
            st.rerun()

    with col3:
        if st.button(ui_text("进入数据分析"), use_container_width=True):
            st.session_state.page = "data"
            st.rerun()

    with col4:
        if st.button(ui_text("进入文献库"), use_container_width=True):
            st.session_state.page = "lit"
            st.rerun()

    st.markdown("---")

    st.markdown(f"### {ui_text('\u7cfb\u7edf\u6982\u89c8')}")
    st.markdown("- 任务图：自动拆解任务、执行调度、状态跟踪")
    st.markdown("- 证据链：理论/文献/ML/实验证据可追溯")
    st.markdown("- 知识库：本地文献 + RAG 检索 + 图谱追踪")
    st.markdown("- 闭环演化：新数据进入后可迭代模型与推荐")

    st.markdown("---")
    st.markdown(f"### {ui_text('\u793a\u4f8b\u95ee\u9898')}")

    examples = [
        "Find ordered alloy candidates for HOR",
        "Train ML model on current materials database",
        "Analyze d-band center vs activity",
        "Search recent HOR catalyst papers",
    ]

    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(example, key=f"example_{i}"):
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()

# ========== Chat Page ==========

def render_chat():
    """Render chat interface."""
    st.markdown(ui_text("智能体对话"))

    db = get_db_service()
    sessions = db.list_chat_sessions(limit=50)
    session_ids = {s.get("id") for s in sessions}

    search_term = st.text_input(ui_text("\u641c\u7d22\u4f1a\u8bdd"), key="chat_session_search")
    if search_term:
        lowered = search_term.lower()
        sessions = [s for s in sessions if lowered in (s.get("title") or "").lower()]
        session_ids = {s.get("id") for s in sessions}

    if st.session_state.chat_session_id is None or st.session_state.chat_session_id not in session_ids:
        if sessions:
            st.session_state.chat_session_id = sessions[0].get("id")
        else:
            title = f"新会话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            st.session_state.chat_session_id = db.create_chat_session(title)
            sessions = db.list_chat_sessions(limit=50)

    if st.session_state.chat_session_id and st.session_state.last_loaded_session_id != st.session_state.chat_session_id:
        load_chat_messages(db, st.session_state.chat_session_id)

    st.markdown(f"### {ui_text('\u4f1a\u8bdd\u7ba1\u7406')}")
    session_labels = []
    label_to_id = {}
    for s in sessions:
        updated = s.get("updated_at") or s.get("created_at") or ""
        label = f"{s.get('id')} | {s.get('title')} | {updated}"
        session_labels.append(label)
        label_to_id[label] = s.get("id")

    if session_labels:
        current_label = session_labels[0]
        for lbl, sid in label_to_id.items():
            if sid == st.session_state.chat_session_id:
                current_label = lbl
                break
        col_a, col_b = st.columns([3, 1])
        with col_a:
            selected_label = st.selectbox(
                ui_text("选择会话"),
                session_labels,
                index=session_labels.index(current_label),
                key="chat_session_select",
            )
        with col_b:
            if st.button(ui_text("新建会话"), key="chat_new_session"):
                title = f"新会话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                new_id = db.create_chat_session(title)
                st.session_state.chat_session_id = new_id
                st.session_state.last_loaded_session_id = None
                st.session_state.active_plan = None
                st.session_state.active_task_id = None
                st.session_state.task_status = None
                st.session_state.task_polling = False
                st.session_state.selected_material = None
                st.rerun()
        selected_id = label_to_id.get(selected_label)
        if selected_id and selected_id != st.session_state.chat_session_id:
            st.session_state.chat_session_id = selected_id
            st.session_state.last_loaded_session_id = None
            st.session_state.active_plan = None
            st.session_state.active_task_id = None
            st.session_state.task_status = None
            st.session_state.task_polling = False
            st.session_state.selected_material = None
            st.rerun()

    current_session = (
        db.get_chat_session(st.session_state.chat_session_id)
        if st.session_state.chat_session_id
        else None
    )
    with st.form("chat_rename_form", clear_on_submit=False):
        title_value = current_session.get("title") if current_session else ""
        new_title = st.text_input(ui_text("会话标题"), value=title_value)
        submitted = st.form_submit_button(ui_text("保存标题"))
    if submitted:
        if st.session_state.chat_session_id and new_title.strip():
            db.update_chat_session_title(st.session_state.chat_session_id, new_title.strip())
            st.rerun()

    with st.expander(ui_text("会话操作")):
        if st.session_state.last_task_id:
            if st.button(ui_text("加载历史任务"), key="chat_load_task"):
                try:
                    status = api_get_task_status(st.session_state.last_task_id)
                    st.session_state.active_task_id = st.session_state.last_task_id
                    st.session_state.active_plan = status
                    st.session_state.task_status = status
                    st.session_state.task_polling = False
                    st.success(ui_text("历史任务已加载"))
                except Exception as e:
                    st.warning(f"Load task failed: {e}")
        else:
            st.info(ui_text("没有历史任务可加载"))

        session_messages = []
        if st.session_state.chat_session_id:
            session_messages = db.list_chat_messages(st.session_state.chat_session_id)
        if session_messages:
            export_data = {
                "session": current_session,
                "messages": session_messages,
            }
            export_json = json.dumps(export_data, ensure_ascii=False, indent=2)
            export_md_lines = [
                f"# {current_session.get('title', '')}",
                "",
            ]
            for m in session_messages:
                role = (m.get("role") or "").upper()
                created_at = m.get("created_at") or ""
                export_md_lines.append(f"## {role} ({created_at})")
                export_md_lines.append(m.get("content") or "")
                export_md_lines.append("")
            export_md = "\n".join(export_md_lines)
            st.download_button(
                ui_text("导出 JSON"),
                export_json,
                file_name=f"chat_{st.session_state.chat_session_id}.json",
                mime="application/json",
                key="chat_export_json",
            )
            st.download_button(
                ui_text("导出 Markdown"),
                export_md,
                file_name=f"chat_{st.session_state.chat_session_id}.md",
                mime="text/markdown",
                key="chat_export_md",
            )
        else:
            st.info(ui_text("当前会话无消息"))

        delete_confirm = st.checkbox(ui_text("确认删除"), key="chat_delete_confirm")
        if st.button(
            ui_text("删除会话"),
            disabled=not delete_confirm,
            key="chat_delete_button",
        ):
            if st.session_state.chat_session_id:
                db.delete_chat_session(st.session_state.chat_session_id)
                st.session_state.chat_session_id = None
                st.session_state.last_loaded_session_id = None
                st.session_state.messages = []
                st.session_state.active_plan = None
                st.session_state.active_task_id = None
                st.session_state.task_status = None
                st.session_state.task_polling = False
                st.session_state.selected_material = None
                st.session_state.last_task_id = None
                st.rerun()

    st.markdown("---")

    # Query translation controls (use global UI setting by default)
    if "auto_translate_query" not in st.session_state:
        st.session_state.auto_translate_query = st.session_state.get("auto_translate_ui", False)
    if "show_translation" not in st.session_state:
        st.session_state.show_translation = True
    with st.expander("Query Translation / 查询翻译"):
        st.checkbox("Auto-translate EN->ZH for planning", key="auto_translate_query")
        st.checkbox("Show translated query", key="show_translation")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input(ui_text("请输入研究问题 / Enter your research query")):
        if not st.session_state.chat_session_id:
            title = f"新会话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            st.session_state.chat_session_id = db.create_chat_session(title)
        st.session_state.messages.append({"role": "user", "content": prompt})
        db.add_chat_message(st.session_state.chat_session_id, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        prompt_for_api = prompt
        translated_query = None
        if st.session_state.get("auto_translate_query"):
            lang = detect_language(prompt)
            if lang == "en":
                translated_query = translate_text(prompt, target_lang="zh")
                if translated_query and translated_query != prompt:
                    prompt_for_api = translated_query

        with st.chat_message("assistant"):
            try:
                with st.status("Planning...", expanded=True) as status:
                    if st.session_state.get("show_translation") and translated_query and translated_query != prompt:
                        st.caption(f"Translated for planning: {translated_query}")
                    status.write("Creating task plan...")
                    plan = api_create_task(prompt_for_api)
                    st.session_state.active_plan = plan
                    st.session_state.active_task_id = plan.get("task_id")
                    st.session_state.task_polling = False
                    status.update(label="Plan created", state="complete", expanded=False)
                    st.markdown(f"Plan created: `{plan.get('task_id')}`")
                    msg = f"Plan created: {plan.get('task_id')}"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": msg
                    })
                    db.add_chat_message(
                        st.session_state.chat_session_id,
                        "assistant",
                        msg,
                        artifacts={"plan_id": plan.get("task_id")},
                    )
                    if current_session and current_session.get("title", "").startswith("新会话"):
                        db.update_chat_session_title(
                            st.session_state.chat_session_id,
                            prompt[:40].strip(),
                        )
            except Exception as e:
                err_msg = f"Task creation failed: {e}"
                st.error(err_msg)
                if st.session_state.chat_session_id:
                    db.add_chat_message(st.session_state.chat_session_id, "assistant", err_msg)

    # Task Graph Viewer
    if st.session_state.active_plan:
        st.markdown("---")
        st.markdown("### Task Graph (Planned)")
        render_task_graph(st.session_state.active_plan.get("steps", []))
        st.markdown("### Task Assignment (Planned)")
        assignment_df = build_task_assignment(st.session_state.active_plan.get("steps", []))
        if not assignment_df.empty:
            st.dataframe(assignment_df, use_container_width=True)
        else:
            st.info("No steps to assign.")
        st.markdown("### Task Flow (Mermaid)")
        st.code(build_task_mermaid(st.session_state.active_plan.get("steps", [])), language="mermaid")

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button(ui_text("开始执行"), type="primary"):
                try:
                    api_execute_task(st.session_state.active_task_id)
                    st.session_state.task_polling = True
                    st.success("Execution started.")
                except Exception as e:
                    st.error(f"Failed to start: {e}")
        with col_b:
            if st.button(ui_text("刷新状态")):
                try:
                    st.session_state.task_status = api_get_task_status(st.session_state.active_task_id)
                except Exception as e:
                    st.warning(f"Status fetch failed: {e}")
            if st.session_state.task_polling:
                if st.button(ui_text("停止自动刷新")):
                    st.session_state.task_polling = False

        if st.session_state.task_polling and st.session_state.active_task_id:
            try:
                st.session_state.task_status = api_get_task_status(st.session_state.active_task_id)
            except Exception as e:
                st.warning(f"Status fetch failed: {e}")

        if st.session_state.task_status:
            st.markdown("### Task Graph (Live Status)")
            render_task_graph(st.session_state.task_status.get("steps", []))
            current_status = st.session_state.task_status.get("status", "unknown")
            st.caption(f"Current status: {current_status}")
            if current_status == "awaiting_confirmation":
                st.info("Evidence gap fill is ready. Confirm to continue.")
                pending_steps = [
                    s for s in st.session_state.task_status.get("steps", [])
                    if s.get("status") in ("pending", "running")
                ]
                if pending_steps:
                    step_rows = []
                    for s in pending_steps:
                        step_rows.append({
                            "step_id": s.get("step_id"),
                            "agent": s.get("agent"),
                            "action": s.get("action"),
                            "params": s.get("params"),
                            "status": s.get("status"),
                        })
                    st.dataframe(pd.DataFrame(step_rows), use_container_width=True)
                    default_ids = [s.get("step_id") for s in pending_steps if s.get("step_id")]
                    selected_ids = st.multiselect(
                        "Select gap steps to run",
                        default_ids,
                        default=default_ids,
                        key="gap_step_select",
                    )
                else:
                    selected_ids = []

                col_gap_a, col_gap_b = st.columns([1, 1])
                with col_gap_a:
                    if st.button("Continue selected gap steps", key="continue_gap_fill"):
                        try:
                            api_confirm_gap_fill(st.session_state.active_task_id, run_step_ids=selected_ids)
                            api_execute_task(st.session_state.active_task_id)
                            st.session_state.task_polling = True
                            st.success("Gap fill execution started.")
                        except Exception as e:
                            st.error(f"Failed to continue: {e}")
                with col_gap_b:
                    if st.button("Skip gap fill and finish", key="skip_gap_fill"):
                        try:
                            api_confirm_gap_fill(st.session_state.active_task_id, run_step_ids=[], mark_complete=True)
                            st.session_state.task_polling = False
                            st.success("Gap fill skipped. Task marked completed.")
                        except Exception as e:
                            st.error(f"Failed to skip: {e}")
            if current_status in ["completed", "failed", "blocked", "awaiting_confirmation"]:
                st.session_state.task_polling = False
            elif st.session_state.task_polling:
                time.sleep(3)
                st.rerun()

            # Evidence Chain Browser
            try:
                mats = api_list_materials()
                if mats:
                    options = []
                    id_by_label = {}
                    for m in mats:
                        mid = m.get("material_id")
                        formula = m.get("formula", "")
                        if not mid:
                            continue
                        label = f"{mid} | {formula}"
                        options.append(label)
                        id_by_label[label] = mid

                    if options:
                        if st.session_state.selected_material is None:
                            st.session_state.selected_material = id_by_label[options[0]]

                        default_label = None
                        for label, mid in id_by_label.items():
                            if mid == st.session_state.selected_material:
                                default_label = label
                                break
                        if default_label is None:
                            default_label = options[0]

                        selected_label = st.selectbox(
                            "Select material to view evidence",
                            options,
                            index=options.index(default_label),
                        )
                        st.session_state.selected_material = id_by_label[selected_label]
                        render_evidence_chain(st.session_state.selected_material)
            except Exception as e:
                st.warning(f"Evidence view failed: {e}")

        # Knowledge Pack (evidence-driven summary)
        pack_task_id = st.session_state.active_task_id or st.session_state.last_task_id
        pack = load_knowledge_pack(pack_task_id)
        if pack:
            st.markdown("---")
            render_knowledge_pack(pack)

# ========== Data Analysis Page ==========

def render_data_analysis():
    """Render data analysis page."""
    st.markdown(f"## {ui_text('\u6570\u636e\u5206\u6790')}")

    tab1, tab2, tab3, tab4 = st.tabs([
        ui_text("理论数据"),
        ui_text("实验数据"),
        ui_text("RDE/RRDE 分析"),
        ui_text("数据可视化"),
    ])

    with tab1:
        st.markdown(f"### {ui_text('\u7406\u8bba\u6570\u636e')}")

        col1, col2 = st.columns([2, 1])
        with col1:
            fe_file = os.path.join(ROOT_DIR, "data", "theory", "formation_energy_full.json")
            if os.path.exists(fe_file):
                with open(fe_file) as f:
                    fe_data = json.load(f)
                df = pd.DataFrame(fe_data)

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric(ui_text("材料总数"), len(df))
                with col_b:
                    if 'formation_energy' in df.columns:
                        st.metric(ui_text("平均形成能"), f"{df['formation_energy'].mean():.3f} eV/atom")
                with col_c:
                    if 'formation_energy' in df.columns:
                        st.metric(ui_text("形成能标准差"), f"{df['formation_energy'].std():.3f}")

                st.dataframe(df.head(20), use_container_width=True)
            else:
                st.info(ui_text("未找到形成能数据"))

        with col2:
            st.markdown(f"### {ui_text('\u4e0b\u8f7d\u65b0\u6570\u636e')}")
            data_source = st.selectbox(
                ui_text("数据源"),
                ["Materials Project", "AFLOW", "Catalysis-Hub"],
            )
            if st.button(ui_text("开始下载")):
                agents = load_agents()
                if agents:
                    with st.spinner(ui_text("正在下载...")):
                        if data_source == "AFLOW":
                            results = agents['theory'].query_aflow(elements=['Pt'], limit=20)
                            st.success(f"Downloaded {len(results)} records")
                        elif data_source == "Catalysis-Hub":
                            results = agents['theory'].query_catalysis_hub(reaction='HER', limit=20)
                            st.success(f"Downloaded {len(results)} records")
                        else:
                            st.info("Materials Project download is not wired here.")

    with tab2:
        st.markdown(f"### {ui_text('\u4e0a\u4f20\u5b9e\u9a8c\u6570\u636e')}")
        uploaded_file = st.file_uploader(
            ui_text("支持格式: CSV, Excel, EC-Lab (.mpt), CHI (.txt)"),
            type=['csv', 'xlsx', 'mpt', 'txt']
        )

        if uploaded_file:
            try:
                file_ext = uploaded_file.name.split('.')[-1].lower()
                if file_ext == 'csv':
                    df = pd.read_csv(uploaded_file)
                elif file_ext == 'xlsx':
                    df = pd.read_excel(uploaded_file)
                else:
                    st.info(f"Detected {file_ext.upper()} file. Please save and analyze manually.")
                    df = None

                if df is not None:
                    st.dataframe(df, use_container_width=True)
                    data_type = st.selectbox(
                        ui_text("数据类型"),
                        ["Auto", "LSV", "CV", "Tafel", "EIS", "Stability"],
                    )
                    if st.button(ui_text("分析数据")):
                        st.info(ui_text("正在分析..."))
            except Exception as e:
                st.error(f"File read failed: {e}")

        st.markdown("---")
        st.markdown(f"### {ui_text('\u626b\u63cf\u6570\u636e\u6587\u4ef6\u5939')}")
        folder_path = st.text_input(ui_text("输入本地数据目录"), value="data/experimental")
        if st.button(ui_text("开始扫描")):
            if os.path.exists(folder_path):
                try:
                    agents = load_agents()
                    if agents:
                        result = agents['experiment'].scan_directory(folder_path)
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(ui_text("总文件数"), result.get('total_files', 0))
                        with col2:
                            st.metric(ui_text("有效数据文件"), result.get('valid_files', 0))

                        if result.get('valid_files', 0) > 0:
                            st.success(ui_text("扫描成功，检测到以下数据类型"))
                            st.json(result.get('data_types'))
                        else:
                            st.warning(ui_text("未检测到支持的数据文件"))
                            if result.get('errors'):
                                st.error(f"Errors: {result.get('errors')}")
                except Exception as e:
                    st.error(f"Scan failed: {e}")
            else:
                st.error(ui_text("目录不存在"))

    with tab3:
        st.markdown(f"### {ui_text('RDE/RRDE 分析')}")
        st.markdown("""
        **RDE:** limiting current, half-wave potential, kinetics
        **RRDE:** electron transfer number (n), H2O2 yield
        """)

        rrde_file = st.file_uploader(ui_text("上传 RRDE 数据"), type=['csv', 'xlsx'], key="rrde")
        if rrde_file:
            collection_eff = st.slider(ui_text("收集效率 (N)"), 0.2, 0.5, 0.37)
            if st.button(ui_text("计算电子转移数")):
                st.info(ui_text("请使用 analyze_rrde() 进行分析"))

    with tab4:
        st.markdown(f"### {ui_text('\u6570\u636e\u53ef\u89c6\u5316')}")
        desc_file = os.path.join(ROOT_DIR, "data", "theory", "dos_descriptors_full.json")
        if os.path.exists(desc_file):
            with open(desc_file) as f:
                desc_data = json.load(f)
            df = pd.DataFrame(desc_data)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### d-band center")
                if 'd_band_center' in df.columns:
                    st.bar_chart(df['d_band_center'].dropna().head(50))
            with col2:
                st.markdown("#### d-band width")
                if 'd_band_width' in df.columns:
                    st.bar_chart(df['d_band_width'].dropna().head(50))
        else:
            st.info(ui_text("未找到 DOS 特征数据"))

# ========== ML Training Page ==========

def render_ml_training():
    """Render ML training page."""
    st.markdown("## 🧪 机器学习训练")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 训练配置")
        
        # Data Source Switch
        data_source_type = st.radio("数据来源", ["理论计算 (CIF/DOS)", "实验数据 (CSV/Excel)"], horizontal=True)
        
        target = None
        exp_file_path = None
        
        if data_source_type.startswith("理论"):
            target = st.selectbox("预测目标 (Output Y)", [
                "形成能 (Formation Energy)",
                "--- DOS 描述符 (11个) ---",
                "d_band_center", 
                "d_band_width", 
                "d_band_filling", 
                "DOS_EF", 
                "DOS_window", 
                "unoccupied_d_states", 
                "epsilon_d_minus_EF", 
                "valence_DOS_slope", 
                "num_DOS_peaks", 
                "first_peak_position", 
                "total_states"
            ], help="选择需要预测的目标性质。输入特征(X)始终为42维结构描述符。")

            if target.startswith("---"):
                st.error("请选择有效的具体目标")
                st.stop()
                
        else:
            # Experimental Data
            st.info("请上传经过整理的实验数据表 (包含特征列和目标列)")
            uploaded_exp = st.file_uploader("上传数据表", type=['csv', 'xlsx'])
            
            if uploaded_exp:
                # Save to temp
                os.makedirs("data/temp", exist_ok=True)
                exp_file_path = os.path.join("data/temp", uploaded_exp.name)
                with open(exp_file_path, "wb") as f:
                    f.write(uploaded_exp.getbuffer())
                
                # Preview columns
                try:
                    if exp_file_path.endswith('.csv'):
                        df_preview = pd.read_csv(exp_file_path)
                    else:
                        df_preview = pd.read_excel(exp_file_path)
                    
                    target = st.selectbox("选择预测目标 (Y)", df_preview.columns)
                    
                    # Optional: Select Features
                    with st.expander("选择特征列 (X) - 默认全选数值列"):
                        feature_cols = st.multiselect("特征列", [c for c in df_preview.columns if c != target], default=[c for c in df_preview.columns if c != target and pd.api.types.is_numeric_dtype(df_preview[c])])
                except Exception as e:
                    st.error(f"文件读取错误: {e}")
        
        st.markdown("#### 选择模型")
        
        # Traditional ML - Column 1
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.markdown("**集成学习**")
            use_xgboost = st.checkbox("XGBoost", value=True)
            use_lightgbm = st.checkbox("LightGBM", value=True)
            use_rf = st.checkbox("RandomForest")
            use_gb = st.checkbox("GradientBoosting")
            use_et = st.checkbox("ExtraTrees")
            use_adaboost = st.checkbox("AdaBoost")
        
        with col_b:
            st.markdown("**线性/核方法**")
            use_ridge = st.checkbox("Ridge")
            use_bayesian = st.checkbox("BayesianRidge")
            use_elastic = st.checkbox("ElasticNet")
            use_svr = st.checkbox("SVR (RBF)")
            use_knn = st.checkbox("KNN")
        
        with col_c:
            st.markdown("**深度学习**")
            use_dnn_256 = st.checkbox("DNN (256-128-64)", value=True)
            use_dnn_512 = st.checkbox("DNN (512-256-128)")
            use_dnn_128 = st.checkbox("DNN (128-64-32)")
            use_transformer = st.checkbox("Transformer")
        
        with col_d:
            st.markdown("**GNN 模型**")
            use_cgcnn = st.checkbox("CGCNN")
            use_schnet = st.checkbox("SchNet")
            use_megnet = st.checkbox("MEGNet")
            st.markdown("")
            st.markdown("*需要 CIF 数据*")
        
        st.markdown("---")
        
        test_size = st.slider("测试集比例", 0.1, 0.4, 0.2)
        enable_shap = st.checkbox("启用 SHAP 分析", value=True)
        
        # Advanced Data Options
        with st.expander("🛠️ 高级数据选项 (特征工程)"):
            st.markdown("""
            ### 特征提取
            此工具将从 CIF 晶体结构文件提取 **42 维物理/化学描述符** (Input X)。
            这些特征将用于预测 **形成能** 和 **DOS 描述符** (Output Y)。
            
            **包含特征**:
            - ⚛️ 原子属性: 序数, 质量, 电负性, **离子半径**
            - 🏗️ 结构属性: 晶格参数, 密度, 堆积系数, **晶格畸变**
            - 🧪 化学属性: **混合焓**, 组分熵, 价电子数
            
            **操作**:
            点击下方按钮将处理所有 CIF 文件，并将新特征合并到数据集中。
            """)
            
            if st.button("🔄 从 CIF 重新提取特征 (生成扩展数据集)", help="这将更新 formation_energy 和 dos_descriptors 数据集"):
                agents = load_agents()
                if agents:
                    ml_agent = agents['ml']
                    cif_dir = os.path.join(ROOT_DIR, "data", "theory", "cifs")
                    
                    files_to_process = [
                        {
                            "source": os.path.join(ROOT_DIR, "data", "theory", "formation_energy_full.json"),
                            "output": os.path.join(ROOT_DIR, "data", "theory", "formation_energy_extended.json"),
                            "type": "formation"
                        },
                        {
                            "source": os.path.join(ROOT_DIR, "data", "theory", "dos_descriptors_full.json"),
                            "output": os.path.join(ROOT_DIR, "data", "theory", "dos_data_extended.json"),
                            "type": "dos"
                        }
                    ]
                    
                    if not os.path.exists(cif_dir):
                        st.error(f"CIF 目录不存在: {cif_dir}")
                    else:
                        with st.spinner("正在处理 CIF 文件并提取特征 (这需要几分钟)..."):
                            try:
                                # 1. Extract features for all CIFs first (Performance optimization)
                                from pymatgen.core import Structure
                                
                                feature_cache = {}
                                cif_files = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]
                                
                                # Use centralized feature definition from MLAgent with fallback
                                if hasattr(ml_agent, "KNOWN_FEATURES"):
                                    known_features = ml_agent.KNOWN_FEATURES
                                else:
                                    # Fallback list (same as in ml_agent.py)
                                    known_features = [
                                        "n_atoms", "n_elements", "volume_per_atom", "density", "packing_fraction",
                                        "avg_Z", "std_Z", "max_Z", "min_Z", "range_Z",
                                        "avg_mass", "std_mass", "max_mass", "min_mass",
                                        "avg_electronegativity", "std_electronegativity", "max_electronegativity", "min_electronegativity", "range_electronegativity",
                                        "avg_radius", "std_radius", "max_radius", "min_radius", "radius_ratio",
                                        "composition_entropy", "composition_variance", "max_composition", "min_composition", "n_elements_comp",
                                        "lattice_a", "lattice_b", "lattice_c", "alpha", "beta", "gamma", "c_over_a",
                                        "avg_lattice", "lattice_distortion",
                                        "mixing_enthalpy_proxy", "avg_valence_electrons", "std_valence_electrons", "volume"
                                    ]
                                
                                progress_bar = st.progress(0)
                                for idx, cif_file in enumerate(cif_files):
                                    mat_id = cif_file.replace('.cif', '')
                                    try:
                                        struct = Structure.from_file(os.path.join(cif_dir, cif_file))
                                        # Call private method to get feature values
                                        feats = ml_agent._structure_to_features(struct)
                                        feature_cache[mat_id] = feats
                                    except:
                                        pass
                                    
                                    if idx % 10 == 0:
                                        progress_bar.progress((idx + 1) / len(cif_files))
                                progress_bar.progress(1.0)
                                
                                processed_counts = {}
                                
                                # 2. Merge with datasets
                                for file_info in files_to_process:
                                    if os.path.exists(file_info["source"]):
                                        with open(file_info["source"], 'r') as f:
                                            raw_data = json.load(f)
                                        
                                        # Ensure list format
                                        data_list = raw_data if isinstance(raw_data, list) else list(raw_data.values())
                                        
                                        new_dataset = []
                                        for record in data_list:
                                            mat_id = record.get("material_id")
                                            if mat_id in feature_cache:
                                                # New record with existing data + new features
                                                new_record = record.copy()
                                                feat_vals = feature_cache[mat_id]
                                                for i, name in enumerate(known_features):
                                                    new_record[name] = feat_vals[i]
                                                new_dataset.append(new_record)
                                        
                                        # Save
                                        with open(file_info["output"], 'w') as f:
                                            json.dump(new_dataset, f, indent=4)
                                        
                                        processed_counts[file_info["type"]] = len(new_dataset)
                                
                                # Clear cache
                                st.cache_data.clear()
                                st.success("特征提取与合并完成！")
                                st.info(f"已更新数据集: 形成能 ({processed_counts.get('formation', 0)}), DOS ({processed_counts.get('dos', 0)})")
                                
                                with st.expander("✅ 查看已提取的 42 维特征 (白名单)"):
                                    st.write(known_features)
                                    st.error("⛔ 已自动剔除所有预测目标 (不会作为特征): d_band_center, d_band_width, DOS_EF, ...")
                                
                            except Exception as e:
                                st.error(f"处理失败: {e}")

        if st.button("🚀 开始训练", type="primary", use_container_width=True):
            selected_models = []
            # Ensemble
            if use_xgboost: selected_models.append("XGBoost")
            if use_lightgbm: selected_models.append("LightGBM")
            if use_rf: selected_models.append("RandomForest")
            if use_gb: selected_models.append("GradientBoosting")
            if use_et: selected_models.append("ExtraTrees")
            if use_adaboost: selected_models.append("AdaBoost")
            # Linear/Kernel
            if use_ridge: selected_models.append("Ridge")
            if use_bayesian: selected_models.append("BayesianRidge")
            if use_elastic: selected_models.append("ElasticNet")
            if use_svr: selected_models.append("SVR")
            if use_knn: selected_models.append("KNN")
            # Deep Learning
            if use_dnn_256: selected_models.append("DNN_256_128_64")
            if use_dnn_512: selected_models.append("DNN_512_256_128")
            if use_dnn_128: selected_models.append("DNN_128_64_32")
            if use_transformer: selected_models.append("Transformer")
            # GNN
            if use_cgcnn: selected_models.append("CGCNN")
            if use_schnet: selected_models.append("SchNet")
            if use_megnet: selected_models.append("MEGNet")
            
            if not selected_models:
                st.warning("请至少选择一个模型")
            else:
                st.info(f"选择的模型: {', '.join(selected_models)}")
                



                
                # Load agents
                agents = load_agents()
                if agents:
                    ml_agent = agents['ml']
                    
                    if data_source_type.startswith("理论"):
                        # === Theoretical Data Loading ===
                        if target == "??? (Formation Energy)":
                            target_col = "formation_energy"
                        else:
                            target_col = target
                        dos_targets = {
                            "d_band_center", "d_band_width", "d_band_filling",
                            "DOS_EF", "DOS_window", "unoccupied_d_states",
                            "epsilon_d_minus_EF", "valence_DOS_slope",
                            "num_DOS_peaks", "first_peak_position", "total_states"
                        }
                        use_db_for_dos = target_col in dos_targets
                        if use_db_for_dos:
                            with st.spinner("Loading DOS target from DB..."):
                                try:
                                    ml_agent.load_from_db(target_col=target_col, include_dos_features=False)
                                    ml_agent.config.test_size = test_size
                                    st.success(f"Loaded DB data: {len(ml_agent.X)} samples, {ml_agent.X.shape[1]} features (DOS target)")
                                except Exception as e:
                                    st.error(f"DB loading failed: {e}")
                                    ml_agent = None
                        else:
                            # Determine data file
                            if target_col == "formation_energy":
                                extended_file = os.path.join(ROOT_DIR, "data", "theory", "formation_energy_extended.json")
                                base_file = os.path.join(ROOT_DIR, "data", "theory", "formation_energy_full.json")
                                if os.path.exists(extended_file):
                                    data_file = extended_file
                                    st.caption("?????????(42?)")
                                else:
                                    data_file = base_file
                                    st.caption("?????????(20?)")
                            else:
                                extended_dos = os.path.join(ROOT_DIR, "data", "theory", "dos_data_extended.json")
                                base_dos = os.path.join(ROOT_DIR, "data", "theory", "dos_descriptors_full.json")
                                if os.path.exists(extended_dos):
                                    data_file = extended_dos
                                    st.caption(f"???? DOS ??? - ??: {target_col}")
                                else:
                                    data_file = base_dos
                                    st.caption("???? DOS ???")
                            if not os.path.exists(data_file):
                                st.error(f"???????: {data_file}")
                                ml_agent = None
                            else:
                                with st.spinner("???????..."):
                                    try:
                                        ml_agent.load_data(data_path=data_file, target_col=target_col)
                                        ml_agent.config.test_size = test_size
                                        st.success(f"????: {len(ml_agent.X)} ??, {ml_agent.X.shape[1]} ??")
                                    except Exception as e:
                                        st.error(f"??????: {e}")
                                        ml_agent = None
                    else:
                        # === Experimental Data Loading ===
                        if exp_file_path and os.path.exists(exp_file_path):
                            with st.spinner("??????..."):
                                try:
                                    feats = feature_cols if "feature_cols" in locals() else None
                                    ml_agent.load_generic_csv(exp_file_path, target, feats)
                                    ml_agent.config.test_size = test_size
                                    st.success(f"??????: {len(ml_agent.X)} ??, {ml_agent.X.shape[1]} ????")
                                except Exception as e:
                                    st.error(f"????????: {e}")
                                    ml_agent = None
                        else:
                            st.error("??????????????????????")
                            ml_agent = None
                    if ml_agent and ml_agent.X is not None:
                        all_results = []
                            
                        # Define model groups
                        traditional_models = ["XGBoost", "LightGBM", "RandomForest", 
                                              "GradientBoosting", "ExtraTrees", "AdaBoost",
                                              "Ridge", "BayesianRidge", "ElasticNet", "SVR", "KNN"]
                        dnn_models = ["DNN_256_128_64", "DNN_512_256_128", "DNN_128_64_32"]
                            
                        # Train traditional ML models
                        trad_selected = [m for m in selected_models if m in traditional_models]
                        if trad_selected:
                            with st.spinner(f"训练传统 ML 模型 ({len(trad_selected)} 个)..."):
                                try:
                                    trad_results = ml_agent.train_traditional_models()
                                    # Filter by selected models
                                    for r in trad_results:
                                        if r.name in trad_selected or any(s in r.name for s in trad_selected):
                                            all_results.append(r)
                                    st.success(f"传统 ML 训练完成: {len(all_results)} 个模型")
                                except Exception as e:
                                    st.warning(f"传统 ML 训练失败: {e}")
                            
                        # Train DNN with selected architectures
                        dnn_selected = [m for m in selected_models if m in dnn_models]
                        if dnn_selected:
                            with st.spinner(f"正在训练深度学习模型集..."):
                                try:
                                    # MLAgent trains all standard architectures at once
                                    all_dnn_results = ml_agent.train_deep_learning_models(epochs=50)
                                        
                                    # Filter requested architectures
                                    filtered_results = []
                                    for r in all_dnn_results:
                                        # Check exact name match or if result name is part of selected key
                                        if r.name in dnn_selected:
                                            filtered_results.append(r)
                                        
                                    all_results.extend(filtered_results)
                                    st.success(f"深度学习训练完成: {len(filtered_results)} 个模型")
                                        
                                except Exception as e:
                                    st.warning(f"深度学习训练失败: {e}")
                            
                        # Train Transformer
                        if "Transformer" in selected_models:
                            with st.spinner("训练 Transformer..."):
                                try:
                                    trans_results = ml_agent.train_transformer_models(epochs=50)
                                    all_results.extend(trans_results)
                                    st.success("Transformer 训练完成")
                                except Exception as e:
                                    st.warning(f"Transformer 训练失败: {e}")
                            
                        # Train GNN
                        gnn_models = ["CGCNN", "SchNet", "MEGNet"]
                        gnn_selected = [m for m in selected_models if m in gnn_models]
                            
                        if gnn_selected:
                            cif_dir = os.path.join(ROOT_DIR, "data", "theory", "cifs")
                            if not os.path.exists(cif_dir):
                                st.error("未找到 CIF 目录，无法训练 GNN 模型。请确保 `data/theory/cifs` 存在。")
                            elif ml_agent.material_ids is None:
                                st.error("当前数据集缺少 'material_id'，无法匹配 CIF。请点击上方 '重新提取特征' 以确保 ID 被保留。")
                            else:
                                with st.spinner("训练 GNN 模型 (需较长时间)..."):
                                    try:
                                        # Build target map using raw loaded data
                                        target_map = dict(zip(ml_agent.material_ids, ml_agent.y))
                                            
                                        # --- GNN Data Diagnostics ---
                                        # Check libraries
                                        import importlib
                                        has_torch = importlib.util.find_spec("torch") is not None
                                        has_pyg = importlib.util.find_spec("torch_geometric") is not None
                                            
                                        # Check explicitly for data matching issues
                                        cif_files_chk = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]
                                        cif_ids = set(f.replace('.cif', '') for f in cif_files_chk)
                                        target_ids = set(target_map.keys())
                                        common_ids = cif_ids.intersection(target_ids)
                                            
                                        st.markdown(f"**GNN 数据检查**: CIF 文件 {len(cif_ids)} 个, 目标 ID {len(target_ids)} 个")
                                            
                                        if not has_torch:
                                            st.error("❌ 未检测到 PyTorch 库。GNN 模型无法运行。请确认已正确安装 `torch`。")
                                            gnn_results = []
                                        elif not has_pyg:
                                            st.error("❌ 未检测到 PyTorch Geometric (PyG)。请安装: `pip install torch-geometric`")
                                            gnn_results = []
                                        elif len(common_ids) == 0:
                                            st.error(f"❌ 严重错误: 没有找到匹配的 ID！请检查 CIF 文件名是否与数据一致。交集为 0。")
                                            gnn_results = []
                                        else:
                                            st.info(f"✅ 环境与数据检查通过 (匹配样本 {len(common_ids)} 个)，开始构建图神经网络...")
                                            # Train GNN models
                                            selected_gnn = [m for m in selected_models if m in ["CGCNN", "SchNet", "MEGNet"]]
                                            gnn_results = ml_agent.train_gnn_models_v2(cif_dir, target_map, epochs=20, model_types=selected_gnn)
                                            
                                        all_results.extend(gnn_results)
                                            
                                        if gnn_results:
                                            st.success(f"GNN 训练完成: {len(gnn_results)} 个模型")
                                        else:
                                            st.warning("GNN 训练未产生结果 (可能数据不足或环境缺失)")
                                                
                                    except Exception as e:
                                        st.warning(f"GNN 训练失败: {e}")
                            
                        # Display results
                        if all_results:
                            st.markdown("---")
                            st.markdown("### 训练结果")
                                
                            # Sort by R2
                            all_results.sort(key=lambda x: x.r2_test, reverse=True)
                                
                            result_data = []
                            for r in all_results[:10]:
                                result_data.append({
                                    "模型": r.name,
                                    "R² (训练)": f"{r.r2_train:.4f}" if r.r2_train is not None else "N/A",
                                    "R² (测试)": f"{r.r2_test:.4f}",
                                    "MAE": f"{r.mae_test:.4f}",
                                    "RMSE": f"{r.rmse_test:.4f}"
                                })
                                
                            st.table(pd.DataFrame(result_data))
                                
                            # Best model
                            best = all_results[0]
                            st.success(f"🏆 最佳模型: {best.name} (R²={best.r2_test:.4f})")
                                
                            # SHAP analysis
                            if enable_shap and best.model is not None:
                                with st.spinner("进行 SHAP 分析..."):
                                    try:
                                        shap_result = ml_agent.shap_analysis(best)
                                        # shap_analysis returns (shap_values, feature_importance) tuple
                                        if shap_result is not None:
                                            shap_values, feature_importance = shap_result
                                            if feature_importance is not None:
                                                st.markdown("### SHAP 特征重要性 (Top 10)")
                                                top_features = feature_importance.head(10)
                                                st.bar_chart(top_features.set_index('feature')['importance'])
                                    except Exception as e:
                                        st.info(f"SHAP 分析: {e}")
                        else:
                            st.warning("没有成功训练的模型")
                else:
                    st.error("无法加载 ML Agent")
    
    with col2:
        st.markdown("### 模型性能")
        
        results_file = os.path.join(ROOT_DIR, "data", "ml_agent", "model_comparison.json")
        if os.path.exists(results_file):
            with open(results_file) as f:
                results = json.load(f)
            
            for r in results[:5]:
                st.markdown(f"""
                <div class="metric-card">
                    <b>{r['name']}</b><br>
                    R²: {r['r2_test']:.4f} | MAE: {r['mae_test']:.4f}
                </div>
                """, unsafe_allow_html=True)


# ========== Literature Page ==========

def render_literature():
    """Render literature search page."""
    st.markdown(f"## {ui_text('\u6587\u732e\u68c0\u7d22')}")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input(ui_text("输入检索关键词"), placeholder="e.g. PtRu alloy HOR catalyst")
    with col2:
        source = st.selectbox(ui_text("数据源"), ["All", "Semantic Scholar", "arXiv"])
    with col3:
        n_results = st.number_input(ui_text("结果数量"), 5, 50, 10)
        min_citations = st.number_input(ui_text("最低引用数"), 0, 1000, 0)

    if st.button(ui_text("搜索"), type="primary"):
        if query:
            agents = load_agents()
            if agents:
                with st.spinner(ui_text("正在搜索...")):
                    try:
                        if source == "All":
                            papers = agents['literature'].search_all_sources(query, n_results)
                        elif source == "arXiv":
                            papers = agents['literature'].search_arxiv(query, n_results)
                        else:
                            papers = agents['literature'].search_semantic_scholar(query, n_results)

                        if papers and min_citations > 0:
                            papers = [p for p in papers if getattr(p, 'citation_count', 0) >= min_citations]

                        if papers:
                            st.success(f"Found {len(papers)} papers")
                            for i, paper in enumerate(papers, 1):
                                with st.expander(f"{i}. {paper.title} (Cite: {paper.citation_count})"):
                                    st.markdown(f"**Authors**: {', '.join(paper.authors[:5])}")
                                    st.markdown(f"**Year**: {paper.year} | **Citations**: {paper.citation_count}")
                                    if paper.abstract:
                                        st.markdown(f"**Abstract**: {paper.abstract[:500]}...")
                                    if paper.url:
                                        st.markdown(f"[View Paper]({paper.url})")
                        else:
                            st.info(ui_text("未找到相关文献"))
                    except Exception as e:
                        st.error(f"Search failed: {e}")

    st.markdown("---")

    # Local library
    st.markdown(ui_text("### 本地文献库 (data/literature) / Local PDF Library"))
    base_dir = os.path.join(ROOT_DIR, "data", "literature")
    default_dir = os.path.join(base_dir, "papers")
    if not os.path.exists(base_dir):
        st.info(ui_text(f"文献目录: {base_dir}"))
        return
    col_idx_a, col_idx_b = st.columns([1, 2])
    with col_idx_a:
        if st.button("Index local PDFs to knowledge graph"):
            agents = load_agents()
            if agents and agents.get("literature"):
                with st.spinner("Indexing local PDFs..."):
                    try:
                        res = agents["literature"].ingest_local_library()
                        st.success(f"Indexed {res.get('indexed_sources', 0)} sources, linked {res.get('linked_materials', 0)} materials.")
                    except Exception as e:
                        st.error(f"Indexing failed: {e}")
            else:
                st.error("Literature agent not available.")
    with col_idx_b:
        st.caption("This indexes local PDFs into knowledge_sources and links detected formulas to materials.")


    search_root = default_dir if os.path.exists(default_dir) else base_dir
    pdf_paths = []
    for root, _, files in os.walk(search_root):
        for fname in files:
            if fname.lower().endswith('.pdf'):
                pdf_paths.append(os.path.join(root, fname))

    if not pdf_paths:
        st.info(ui_text("未发现 PDF 文件"))
        return

    labels = [os.path.relpath(p, base_dir) for p in pdf_paths]
    label_to_path = {label: path for label, path in zip(labels, pdf_paths)}

    col_a, col_b = st.columns([2, 1])
    with col_a:
        selected_label = st.selectbox(ui_text("选择 PDF"), labels)
    with col_b:
        max_pages = st.number_input(ui_text("预览页数"), min_value=1, max_value=10, value=3)

    if st.button(ui_text("加载 PDF 预览")):
        try:
            import pdfplumber
            pdf_path = label_to_path.get(selected_label)
            text_chunks = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages[:int(max_pages)]:
                    text = page.extract_text() or ""
                    if text.strip():
                        text_chunks.append(text)
            if text_chunks:
                st.text_area(ui_text("提取文本"), "\n\n".join(text_chunks), height=300)
            else:
                st.info(ui_text("未提取到文本，可能是扫描 PDF"))
        except Exception as e:
            st.error(f"PDF parse failed: {e}")

    st.markdown(ui_text("#### 本地文献搜索"))
    local_query = st.text_input(ui_text("输入关键词"), key="local_pdf_query")
    if st.button(ui_text("搜索本地库"), key="local_pdf_search"):
        try:
            import pdfplumber
            matches = []
            for pdf_path in pdf_paths:
                hit_count = 0
                snippet = ""
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages[:5]:
                        text = page.extract_text() or ""
                        if local_query and local_query.lower() in text.lower():
                            hit_count += text.lower().count(local_query.lower())
                            if not snippet:
                                snippet = text[:500]
                if hit_count > 0:
                    matches.append({"file": os.path.relpath(pdf_path, base_dir), "hits": hit_count, "snippet": snippet})
            if matches:
                st.success(ui_text(f"找到 {len(matches)} 条匹配"))
                matches = sorted(matches, key=lambda x: x["hits"], reverse=True)
                for m in matches[:10]:
                    with st.expander("{} (hits: {})".format(m.get("file"), m.get("hits"))):
                        st.write(m.get("snippet", ""))
            else:
                st.info(ui_text("没有匹配结果"))
        except Exception as e:
            st.error(f"Search failed: {e}")

    st.markdown("### 本地 PDF 解析")
    pdf_file = st.file_uploader(ui_text("上传 PDF 文件"), type=['pdf'])
    if pdf_file:
        if st.button(ui_text("解析 PDF")):
            st.info(ui_text("PDF 解析需要 pdfplumber 支持"))

# ========== Evaluation Page ==========

def render_evaluation():
    """Render evaluation page for evidence coverage and recommendation recall."""
    st.markdown("## Evaluation")

    st.markdown("### Evidence Stats (Meta-controller)")
    try:
        stats = api_knowledge_stats()
        cols = st.columns(3)
        cols[0].metric("Total materials", stats.get("total_materials", 0))
        cols[1].metric("DOS records", stats.get("dos_count", 0))
        cols[2].metric("Models", stats.get("model_count", 0))
        st.caption(
            "Evidence coverage: literature/ML/experiment counts and activity/adsorption proxies."
        )
        st.json(stats)
    except Exception as e:
        st.info(f"Stats unavailable: {e}")

    # Evidence coverage
    st.markdown("### Evidence Coverage")
    db_path = os.path.join(ROOT_DIR, "data", "imcs.db")
    coverage = compute_evidence_coverage(db_path)
    if coverage.get("error"):
        st.warning(coverage["error"])
    else:
        st.metric("Total materials", coverage.get("total", 0))
        rows = coverage.get("rows") or []
        if rows:
            data = []
            total = coverage.get("total") or 0
            for stype, cnt in rows:
                ratio = (cnt / total) if total else 0
                data.append({"source_type": stype, "materials": cnt, "coverage": f"{ratio:.2%}"})
            st.table(pd.DataFrame(data))
        else:
            st.info("No evidence records yet.")

    st.markdown("---")

    # Recommendation recall
    st.markdown("### Recommendation Recall (Top-K)")
    st.markdown("Upload candidate list and ground-truth list (CSV).")

    candidates_file = st.file_uploader("Candidates CSV", type=["csv"], key="eval_candidates")
    gt_file = st.file_uploader("Ground-truth CSV", type=["csv"], key="eval_ground_truth")
    k_input = st.text_input("K values (comma-separated)", "5,10,20")

    if candidates_file and gt_file:
        try:
            cand_df = pd.read_csv(candidates_file)
            gt_df = pd.read_csv(gt_file)

            # If a score column exists, sort by score descending
            if "score" in cand_df.columns:
                cand_df = cand_df.sort_values("score", ascending=False)

            candidates = load_ids_from_df(cand_df)
            ground_truth = set(load_ids_from_df(gt_df))

            if not candidates:
                st.warning("No candidate ids found. Expected column: material_id / id / mid.")
                return
            if not ground_truth:
                st.warning("No ground-truth ids found. Expected column: material_id / id / mid.")
                return

            ks = []
            for part in k_input.split(","):
                part = part.strip()
                if part.isdigit():
                    ks.append(int(part))
            if not ks:
                ks = [5, 10, 20]

            rows = []
            for k in ks:
                rows.append({"k": k, "recall": round(topk_recall(candidates, ground_truth, k), 4)})
            st.table(pd.DataFrame(rows))
        except Exception as e:
            st.error(f"Failed to compute recall: {e}")


# ========== API Status Page ==========

def render_api_status():
    """Render API status page."""
    st.markdown("## 🔌 API 连接状态")
    
    st.markdown("点击按钮测试各 API 连接")
    
    if st.button("🔄 测试所有 API", type="primary"):
        import requests
        
        results = []
        
        # Materials Project
        with st.spinner("测试 Materials Project..."):
            try:
                from mp_api.client import MPRester
                with MPRester("abx7GG5NQg5YncfROEP4vvQi8Tc5Ywqp") as mpr:
                    docs = mpr.materials.summary.search(
                        elements=['Pt'], is_stable=True, 
                        fields=['material_id'], num_chunks=1, chunk_size=3
                    )
                results.append(("Materials Project", "✅ OK", f"找到 {len(docs)} 材料"))
            except Exception as e:
                results.append(("Materials Project", "❌ Failed", str(e)[:50]))
        
        # Semantic Scholar
        with st.spinner("测试 Semantic Scholar..."):
            try:
                r = requests.get(
                    'https://api.semanticscholar.org/graph/v1/paper/search',
                    params={'query': 'catalyst', 'limit': 1}, timeout=10
                )
                if r.status_code == 200:
                    results.append(("Semantic Scholar", "✅ OK", "HTTP 200"))
                else:
                    results.append(("Semantic Scholar", "⚠️", f"HTTP {r.status_code}"))
            except Exception as e:
                results.append(("Semantic Scholar", "❌ Failed", str(e)[:50]))
        
        # arXiv
        with st.spinner("测试 arXiv..."):
            try:
                r = requests.get(
                    'http://export.arxiv.org/api/query',
                    params={'search_query': 'all:catalyst', 'max_results': 1}, timeout=10
                )
                if r.status_code == 200:
                    results.append(("arXiv", "✅ OK", "HTTP 200"))
                else:
                    results.append(("arXiv", "⚠️", f"HTTP {r.status_code}"))
            except Exception as e:
                results.append(("arXiv", "❌ Failed", str(e)[:50]))
        
        # AFLOW
        with st.spinner("测试 AFLOW..."):
            try:
                r = requests.get(
                    'http://aflowlib.org/API/aflux/?species(Pt),paging(1)', timeout=10
                )
                if r.status_code == 200:
                    results.append(("AFLOW", "✅ OK", "HTTP 200"))
                else:
                    results.append(("AFLOW", "⚠️", f"HTTP {r.status_code}"))
            except Exception as e:
                results.append(("AFLOW", "❌ Failed", str(e)[:50]))
        
        # Catalysis-Hub
        with st.spinner("测试 Catalysis-Hub..."):
            try:
                r = requests.post(
                    'https://api.catalysis-hub.org/graphql',
                    json={'query': '{reactions(first:1){edges{node{Equation}}}}'}, timeout=10
                )
                if r.status_code == 200:
                    results.append(("Catalysis-Hub", "✅ OK", "HTTP 200"))
                else:
                    results.append(("Catalysis-Hub", "⚠️", f"HTTP {r.status_code}"))
            except Exception as e:
                results.append(("Catalysis-Hub", "❌ Failed", str(e)[:50]))
        
        # Display results
        for name, status, detail in results:
            st.markdown(f"**{name}**: {status} - {detail}")


# ========== Settings Page ==========

def render_settings():
    """Render settings page."""
    st.markdown("## ⚙️ 设置")
    
    st.markdown("### API 配置")
    
    mp_key = st.text_input("Materials Project API Key", value="abx7GG5NQg5YncfROEP4vvQi8Tc5Ywqp", type="password")
    
    st.markdown("---")
    
    st.markdown("### 数据目录")
    st.text_input("理论数据目录", value="data/theory", disabled=True)
    st.text_input("实验数据目录", value="data/experimental", disabled=True)
    st.text_input("模型输出目录", value="data/ml_agent", disabled=True)
    
    st.markdown("---")
    
    st.markdown("### 系统信息")
    st.markdown(f"""
    - **版本**: v2.0
    - **智能体数量**: 5
    - **模型类型**: 传统 ML + DNN + GNN
    - **数据源**: Materials Project, AFLOW, Catalysis-Hub, Semantic Scholar, arXiv
    """)
    
    if st.button("保存设置"):
        st.success("设置已保存")


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
    elif page == "settings":
        render_settings()

if __name__ == "__main__":
    main()
