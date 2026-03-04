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

    before_stats = pack.get("evidence_stats_before_gap") or {}
    after_stats = pack.get("evidence_stats_after_gap") or {}
    delta_stats = pack.get("evidence_stats_delta") or {}
    if before_stats or after_stats:
        st.markdown("#### Evidence Coverage Delta")
        rows = []
        keys = set(before_stats.keys()) | set(after_stats.keys()) | set(delta_stats.keys())
        for key in sorted(keys):
            before_val = before_stats.get(key)
            after_val = after_stats.get(key)
            delta_val = delta_stats.get(key)
            percent = None
            try:
                if isinstance(before_val, (int, float)) and isinstance(delta_val, (int, float)) and before_val != 0:
                    percent = round((delta_val / before_val) * 100.0, 2)
            except Exception:
                percent = None
            rows.append({
                "metric": key,
                "before": before_val,
                "after": after_val,
                "delta": delta_val,
                "delta_%": percent,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

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

    strategy_stats = load_strategy_stats() or {}
    strategy_feedback = load_strategy_feedback(pack.get("task_id") or "")
    if strategy_stats or strategy_feedback:
        st.markdown("#### Strategy Feedback")
        if strategy_feedback:
            st.caption("Latest feedback for this task")
            try:
                st.json(strategy_feedback)
            except Exception:
                pass
        evidence_types = strategy_stats.get("evidence_types") or {}
        if evidence_types:
            rows = []
            for key, info in evidence_types.items():
                if not isinstance(info, dict):
                    continue
                rows.append({
                    "evidence_type": key,
                    "attempts": info.get("attempts"),
                    "gains": info.get("gains"),
                    "score": info.get("score"),
                })
            if rows:
                df = pd.DataFrame(rows).sort_values(by="score", ascending=False)
                st.dataframe(df, use_container_width=True)
        else:
            st.caption("No strategy stats found yet.")

    ranking_before = pack.get("ranking_before_gap") or []
    ranking_after = pack.get("ranking_after_gap") or []
    ranking_current = pack.get("ranking_current") or []
    ranking_metric = pack.get("ranking_metric") or ""
    if ranking_before or ranking_after or ranking_current:
        st.markdown("#### Candidate Ranking")
        if ranking_metric:
            st.caption(f"Metric: {ranking_metric}")
        if ranking_before:
            st.markdown("Before gap fill")
            try:
                st.dataframe(pd.DataFrame(ranking_before), use_container_width=True)
            except Exception:
                st.json(ranking_before)
        if ranking_after:
            st.markdown("After gap fill")
            try:
                st.dataframe(pd.DataFrame(ranking_after), use_container_width=True)
            except Exception:
                st.json(ranking_after)
        if not ranking_before and not ranking_after and ranking_current:
            try:
                st.dataframe(pd.DataFrame(ranking_current), use_container_width=True)
            except Exception:
                st.json(ranking_current)
        if ranking_before and ranking_after:
            before_map = {r.get("material_id"): r for r in ranking_before if r.get("material_id")}
            after_map = {r.get("material_id"): r for r in ranking_after if r.get("material_id")}
            rows = []
            for mid in sorted(set(before_map.keys()) | set(after_map.keys())):
                b = before_map.get(mid, {})
                a = after_map.get(mid, {})
                rb = b.get("rank")
                ra = a.get("rank")
                delta_rank = None
                if isinstance(rb, int) and isinstance(ra, int):
                    delta_rank = rb - ra
                rows.append({
                    "material_id": mid,
                    "rank_before": rb,
                    "score_before": b.get("score"),
                    "rank_after": ra,
                    "score_after": a.get("score"),
                    "rank_delta": delta_rank,
                })
            st.markdown("Ranking delta (positive = improved)")
            try:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            except Exception:
                st.json(rows)

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

    eval_metrics = pack.get("evaluation_metrics") or {}
    if eval_metrics:
        st.markdown("#### Evaluation Metrics")
        top_cols = st.columns(3)
        top_cols[0].metric("Plan Status", eval_metrics.get("plan_status") or "-")
        top_cols[1].metric("Candidate Count", eval_metrics.get("candidate_count") or 0)
        top_cols[2].metric("Model Count", eval_metrics.get("model_count") or 0)
        if eval_metrics.get("best_model_r2") is not None:
            st.metric("Best Model R2", eval_metrics.get("best_model_r2"))

        hit_cols = st.columns(2)
        hit_cols[0].metric("Ranking Evidence Hit Rate", eval_metrics.get("ranking_evidence_hit_rate") or 0)
        hit_cols[1].metric("Ranking Activity Hit Rate", eval_metrics.get("ranking_activity_hit_rate") or 0)

        coverage = eval_metrics.get("evidence_coverage_by_key") or {}
        if coverage:
            st.markdown("Coverage by evidence key")
            try:
                cov_df = pd.DataFrame(
                    [{"evidence_key": k, "coverage": v} for k, v in coverage.items()]
                )
                st.dataframe(cov_df, use_container_width=True)
                st.bar_chart(cov_df.set_index("evidence_key")["coverage"])
            except Exception:
                st.json(coverage)

        try:
            st.json(eval_metrics)
        except Exception:
            st.write(eval_metrics)

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
        elif status == "skipped":
            status_class = "tg-status tg-status-ok"
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
            # 自定义渲染: Gemini 主管深入剖析与 Active Learning
            is_recommendation = False
            result_data = s.get("result", {})
            if isinstance(result_data, dict) and "candidates" in result_data and "reasoning" in result_data:
                is_recommendation = True
                
            if is_recommendation:
                st.markdown("---")
                reasoning_text = result_data.get("reasoning", "")
                if "⚠️ [主动发现 - Active Learning] ⚠️" in reasoning_text:
                    st.error("🚨 **主动验证请求 (Active Learning Triggered)** 🚨")
                    st.markdown(reasoning_text.replace("⚠️ [主动发现 - Active Learning] ⚠️", ""))
                    
                    st.markdown("#### 高不确定度候选材料 (High Variance Candidates)")
                    cands = result_data.get("candidates", [])
                    if cands:
                        df_cands = []
                        for c in cands:
                            formula = c.get("formula", "")
                            props = c.get("properties", {})
                            pred_Score = props.get("predicted_activity")
                            uncert = props.get("uncertainty")
                            reason = c.get("active_learning_reason", "")
                            df_cands.append({
                                "结构式": formula,
                                "预测活性极值": pred_Score,
                                "不确定度(方差)": uncert,
                                "触发原因": reason
                            })
                        st.dataframe(pd.DataFrame(df_cands), use_container_width=True)
                else:
                    st.success("✅ **最终专家推荐报告 (Director Report)**")
                    st.markdown(reasoning_text)
                    
                    st.markdown("#### 顶尖候选材料 (Top Recommended Candidates)")
                    cands = result_data.get("candidates", [])
                    if cands:
                        df_cands = []
                        for c in cands:
                            formula = c.get("formula", "")
                            props = c.get("properties", {})
                            pred_Score = props.get("predicted_activity")
                            df_cands.append({
                                "结构式": formula,
                                "综合推荐评分": pred_Score,
                                "形成能(eV)": props.get("formation_energy")
                            })
                        st.dataframe(pd.DataFrame(df_cands), use_container_width=True)
                        
            with st.expander(f"Result details: {s.get('step_id','step')}"):
                try:
                    full_text = json.dumps(result_data, ensure_ascii=False, indent=2)
                except Exception:
                    full_text = str(result_data)
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


