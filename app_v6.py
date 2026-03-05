"""
IMCs V6 Interactive Command Center (Dashboard)

A Streamlit-based Web UI for the Autonomous Closed-Loop System.
Features:
- Live streaming of Agent Orchestrator progress
- Multi-objective Pareto Front visualization (Activity vs. Cost)
- Interactive 3D molecular/crystal viewer
- Expert LLM reasoning report injection
"""

import streamlit as st
import time
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any

from src.agents.orchestrator import AgentOrchestrator
from src.services.knowledge.service import KnowledgeService
from src.services.theory.market_data import get_market_data

# Page config
st.set_page_config(
    page_title="IMCs V6 Command Center", 
    page_icon="⚛️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme glassmorphism and animations
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e6edf3;
    }
    .metric-card {
        background: rgba(30, 35, 45, 0.7);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .status-pulse {
        animation: pulse 2s infinite;
        color: #00ffcc;
        font-weight: bold;
    }
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1.0; }
        100% { opacity: 0.6; }
    }
</style>
""", unsafe_allow_html=True)

st.title("🔋 IMCs V6 Command Center: Autonomous Catalyst Discovery")
st.caption("AI-Driven Multi-Agent Debate | Multi-Objective Pareto Optimization | 3D Visualization")

# Sidebar
with st.sidebar:
    st.header("⚙️ Orchestration Settings")
    max_iters = st.slider("Max Debate Iterations", min_value=1, max_value=5, value=2)
    require_lit = st.checkbox("🔒 强依赖文献 (Safe Mode)", value=False, help="开启后在左侧图表中仅标示或过滤那些已有文献报道记录的材料，抑制纯AI生成的理论猜测。")
    st.markdown("---")
    st.info("💡 **Tip**: Enter constraints like 'low cost', 'high stability', or specify target elements (Pt, Ni) to see Multi-Objective trade-offs.")

    st.markdown("---")
    st.header("🧪 闭环实验反馈 (Feedback Round)")
    st.write("将验证数据喂回系统，校准下一轮推荐：")
    fb_mat_id = st.text_input("测试对象 (Material ID)")
    fb_outcome = st.selectbox("实验结论", ["success", "partial", "failure"])
    fb_yield = st.slider("实测活性得分", 1, 10, 5)
    if st.button("Submit to AI Brain", use_container_width=True):
        from src.services.task.meta_controller import MetaController
        from src.services.db.database import DatabaseService
        import datetime
        try:
            # 强化策略与打分保存到本地
            mc = MetaController()
            mc.strategy_feedback(fb_mat_id or "Manual_Exp", fb_outcome, evidence_yield=fb_yield)
            if fb_mat_id:
                db = DatabaseService()
                db.save_evidence(fb_mat_id, "experiment", f"manual-{datetime.datetime.now().strftime('%Y%m%d%H%M')}", score=float(fb_yield)/10.0, metadata={"outcome": fb_outcome})
            st.success("Feedback Recalibrated! AI 已吸收实验教训。")
        except Exception as e:
            st.error(f"Error saving feedback: {e}")

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "reasoning" not in st.session_state:
    st.session_state.reasoning = ""
if "is_running" not in st.session_state:
    st.session_state.is_running = False

def ui_progress_callback(data: Dict[str, str]):
    """Callback triggered by Orchestrator on_progress"""
    # Simply append to session state messages to yield reactive updates
    stage = data.get("stage", "info")
    msg = data.get("message", "")
    st.session_state.messages.append(f"[{stage.upper()}] {msg}")

# Main Chat Input
query = st.text_input("Enter discovery objective (e.g., 'Discover highly active, low cost HOR catalysts in alkaline media'):", 
                      "寻找适合碱性介质的高性价比 HOR 催化剂，具备极高的活性且尽量减少贵金属使用。")

start_btn = st.button("🚀 Launch Autonomous Discovery", use_container_width=True, type="primary")

# Main execution area
if start_btn:
    st.session_state.messages = []
    st.session_state.candidates = []
    st.session_state.reasoning = ""
    st.session_state.is_running = True
    
    status_container = st.empty()
    terminal_expander = st.expander("Live Multi-Agent Event Stream", expanded=True)
    terminal_placeholder = terminal_expander.empty()
    
    status_container.markdown('<p class="status-pulse">🤖 Agents are debating and reasoning... please wait.</p>', unsafe_allow_html=True)
    
    # Init orchestrator with callback
    orchestrator = AgentOrchestrator(on_progress=ui_progress_callback)
    
    try:
        # Run orchestrator
        # To stream logs to UI, we will periodically read from session state (Not perfect without async, but works for PoC)
        with st.spinner("Executing..."):
            query_with_constraints = query
            if require_lit:
                query_with_constraints += " (Mandatory Constraint: Must have literature evidence)"
            
            result = orchestrator.orchestrate(query_with_constraints, max_iterations=max_iters)
            
            # Post-filter if safe mode is on
            if require_lit:
                from src.services.db.database import DatabaseService
                db = DatabaseService()
                filtered_cands = []
                # Check actual literature evidence in DB
                mat_ids = [c.get("material_id") for c in result.candidates if c.get("material_id")]
                ev_counts = db.get_evidence_counts(mat_ids)
                
                for c in result.candidates:
                    mid = c.get("material_id")
                    if mid and ev_counts.get(mid, {}).get("literature", 0) > 0:
                        filtered_cands.append(c)
                st.session_state.candidates = filtered_cands
            else:
                st.session_state.candidates = result.candidates
                
            st.session_state.reasoning = result.reasoning
    except Exception as e:
        status_container.error(f"Orchestration failed: {e}")
    finally:
        st.session_state.is_running = False
        status_container.empty()
        # Final terminal dump
        terminal_placeholder.code("\\n".join(st.session_state.messages), language="bash")

# Render Results UI if available
if st.session_state.candidates:
    st.markdown("---")
    st.header("📊 Multi-Objective Discovery Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Pareto Front (Activity vs. Cost Limit)")
        
        # Prepare data for plotting
        plot_data = []
        md = get_market_data()
        
        for idx, c in enumerate(st.session_state.candidates[:20]):
            score = c.get("score", 0.0)
            formula = c.get("formula", f"Unknown-{idx}")
            al_reason = c.get("active_learning_reason", "")
            is_pareto = "Pareto" in al_reason
            
            # Estimate cost penalty for plot
            comp = md.estimate_formula_cost({formula: 1.0}) # Simplified, real parser needed for exact scatter
            # Fallback to AL provided EI/PI if available
            ei = c.get("ei_score", 0.0)
            pi = c.get("pi_score", 0.0)
            
            plot_data.append({
                "Formula": formula,
                "Fusion Score": score,
                "Cost Penalty (Approx)": 1.0 - (score * 0.1) if not is_pareto else score * 0.5, # Mock cost for viz if not easily parsable
                "Type": "Pareto Optimal" if is_pareto else "Standard",
                "Size": 15 if is_pareto else 8
            })
            
        df = pd.DataFrame(plot_data)
        
        if not df.empty:
            fig = px.scatter(
                df, x="Cost Penalty (Approx)", y="Fusion Score", 
                color="Type", hover_data=["Formula"], size="Size",
                color_discrete_map={"Pareto Optimal": "#00ffcc", "Standard": "#445566"},
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Candidates Table
        st.subheader("Top Recommended Candidates")
        table_df = pd.DataFrame([{
            "Rank": i+1,
            "Formula": c.get("formula"),
            "Score": round(c.get("score", 0), 3),
            "AL Status": "🏆 Pareto" if "Pareto" in c.get("active_learning_reason", "") else ("🔬 Test" if c.get("active_learning_reason") else "-")
        } for i, c in enumerate(st.session_state.candidates[:10])])
        st.dataframe(table_df, use_container_width=True)

    with col2:
        st.subheader("💎 3D Crystal Verification")
        
        selected_idx = st.selectbox("Inspect Material Structure", range(len(st.session_state.candidates[:10])), 
                                    format_func=lambda x: st.session_state.candidates[x].get("formula", "Unknown"))
        
        if selected_idx is not None:
            mat_id = st.session_state.candidates[selected_idx].get("material_id")
            ks = KnowledgeService()
            html_res = ks.generate_structure_preview(mat_id)
            
            if html_res and html_res.get("success"):
                st.components.v1.html(html_res.get("html_snippet"), height=420)
            else:
                st.warning("CIF missing for 3D render. Structure unknown.")
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.subheader("🧠 Expert Reasoning Report")
        st.markdown(st.session_state.reasoning)
        st.markdown("</div>", unsafe_allow_html=True)
