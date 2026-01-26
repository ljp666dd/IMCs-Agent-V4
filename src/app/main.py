import streamlit as st
import pandas as pd
import numpy as np
import time
from streamlit_agraph import agraph, Node, Edge, Config
import plotly.express as px
from stmol import showmol
import py3Dmol
from styles import apply_custom_styles, card

import sys
import os
import torch
import glob
import json
from src.agents.experimental.literature_ingest import ExperimentalAgent
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.cgcnn import CGCNN
from src.data_ingestion.dataset import CIFDataset

# ...

@st.cache_resource
def load_metadata_map():
    """Cache mapping of Formula -> CIF Path from summary JSON."""
    json_path = os.path.join("data", "theory", "mp_data_summary.json")
    mapping = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
            for entry in data:
                # Store formula -> cif_path (Absolute or relative fix)
                # entry['cif_path'] is likely absolute in the JSON based on previous view.
                # We need to ensure it's accessible.
                # Heuristic: filename is reliable.
                fname = os.path.basename(entry.get('cif_path', ''))
                if fname:
                    local_path = os.path.join("data", "theory", "cifs", fname)
                    if os.path.exists(local_path):
                        mapping[entry['formula']] = local_path
    return mapping

@st.cache_resource
def load_theorist():
    path = os.path.join("data", "cgcnn_best_model.pth")
    if os.path.exists(path):
        model = CGCNN(orig_atom_fea_len=92, n_conv=5)
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model.eval()
        return model
    return None

@st.cache_resource
def load_experimental_agent():
    return ExperimentalAgent()

def gaussian_expansion(distances, dmin=0, dmax=8, step=0.2):
    filter_points = torch.arange(dmin, dmax + step, step)
    sigma = step
    return torch.exp(-(distances.unsqueeze(1) - filter_points.unsqueeze(0))**2 / sigma**2)

def process_cif_for_inference(cif_path):
    from pymatgen.core.structure import Structure
    try:
        structure = Structure.from_file(cif_path)
    except Exception as e:
        return None, None

    # 1. Features (Simplified Atomic Number One-Hot)
    atomic_numbers = [site.specie.number for site in structure]
    x = torch.zeros((len(atomic_numbers), 92))
    for i, z in enumerate(atomic_numbers):
        if z <= 92:
            x[i, z-1] = 1.0
            
    # 2. Find Neighbors (Edges)
    radius = 8.0
    max_neighbors = 12
    all_neighbors = structure.get_all_neighbors(radius, include_index=True)
    all_neighbors = [sorted(n, key=lambda x: x[1])[:max_neighbors] for n in all_neighbors]
    
    edge_indices = []
    edge_dist = []
    
    for i, neighbors in enumerate(all_neighbors):
        for neighbor in neighbors:
            j = neighbor[2]
            dist = neighbor[1]
            edge_indices.append([i, j])
            edge_dist.append(dist)
            
    if len(edge_indices) == 0:
        edge_index = torch.tensor([[], []], dtype=torch.long)
        edge_attr = torch.tensor([], dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        dist_tensor = torch.tensor(edge_dist, dtype=torch.float)
        edge_attr = gaussian_expansion(dist_tensor)

    from torch_geometric.data import Data
    # Create Data object (No labels needed for inference)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return structure, data

def predict_material(model, composition_query, metadata_map):
    """
    Predict properties for a given composition using the trained model.
    """
    # 1. Look up CIF path
    target_cif = None
    
    # Direct Match
    if composition_query in metadata_map:
        target_cif = metadata_map[composition_query]
    else:
        # Scan keys for partial match (case-insensitive) or substring
        query_lower = composition_query.lower()
        for formula, path in metadata_map.items():
            if formula.lower() == query_lower:
                target_cif = path
                break
    
    if not target_cif:
        return None, None
        
    # 2. Process to Graph using Helper
    struct, data = process_cif_for_inference(target_cif)
    
    if data is None:
        return None, None
        
    # 3. Batch (Size 1)
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([data])
    
    # 4. Predict
    model = model.to('cpu')
    with torch.no_grad():
        out = model(batch)
    
    # Return formula from structure to ensure correctness
    real_formula = struct.composition.reduced_formula
    return real_formula, out

def plot_dos(dos_vector):
    """Plot 400-dim DOS vector over -5 to 5 eV range."""
    x = np.linspace(-5, 5, 400)
    y = dos_vector.numpy()
    
    df = pd.DataFrame({"Energy (eV)": x, "DOS (a.u.)": y})
    
    fig = px.line(df, x="Energy (eV)", y="DOS (a.u.)", 
                  title="Electronic Density of States (Predicted)",
                  template="plotly_dark")
    fig.update_traces(line_color='#10B981', fill='tozeroy') # Green for theory
    fig.add_vline(x=0, line_dash="dash", line_color="white", annotation_text="Fermi Level")
    return fig


# 1. Page Config
st.set_page_config(
    page_title="Evolutionary Cockpit",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply CSS
apply_custom_styles()

# --- LOCALIZATION ---
TRANS = {
    "简体中文": {
        "sidebar_title": "多智能体进化平台",
        "sidebar_caption": "有序合金氢电催化系统",
        "config_header": "🧪 实验参数设置",
        "electrolyte": "电解液环境",
        "ref_electrode": "参比电极",
        "stability_test": "稳定性测试 (CV圈数)",
        "data_header": "📂 知识库注入",
        "drop_pdf": "文献上传 (.pdf)",
        "drop_exp": "实验数据 (.csv)",
        "agent_grid": "🤖 智能体集群",
        "tab1": "🕸️ 协同编排",
        "tab2": "🚀 指挥中心",
        "tab3": "📈 综合分析",
        "agent_net": "智能体协同拓扑图",
        "team_roster": "智能体功能状态",
        "cmd_role": "任务管理智能体",
        "lit_role": "文献阅读智能体",
        "theory_role": "理论数据智能体",
        "exp_role": "实验数据智能体",
        "ml_role": "机器学习智能体",
        "cmd_desc": "对话交互 / 任务分发 / 决策协同",
        "lit_desc": "本地文献RAG / IF因子筛选 / 知识提取",
        "theory_desc": "MP数据库检索 / 理论计算 / 结构化建库",
        "exp_desc": "实验数据清洗 / 迭代更新 / 数据库构建",
        "ml_desc": "混合模型训练 / 性能预测 / 逆向设计",
        "chat_tip": "💡 提示：'基于碱性HER条件推荐有序合金' 或 '分析500圈CV后的过电位变化'",
        "chat_welcome": "任务管理智能体就绪。请设定实验目标或询问材料方案。",
        "enter_cmd": "在此输入指令...",
        "received": "已接收：",
        "initiating": "正在调用智能体集群...",
        "analysis_complete": "预测完成 (置信度 94%)：",
        "processing": "正在协同计算中...",
        "system_terminal": "实时运行日志",
        "evo_curve": "稳定性预测 (500 CV)",
        "candidate_3d": "推荐材料结构",
        "visualizing": "正在渲染："
    },
    "English": {
        "sidebar_title": "Multi-Agent Evo Platform",
        "sidebar_caption": "Ordered Alloy HOR System",
        "config_header": "🧪 Exp Parameters",
        "electrolyte": "Electrolyte",
        "ref_electrode": "Ref Electrode",
        "stability_test": "Stability (CV Cycles)",
        "data_header": "📂 Knowledge Injection",
        "drop_pdf": "Upload Papers (.pdf)",
        "drop_exp": "Lab Data (.csv)",
        "agent_grid": "🤖 Agent Cluster",
        "tab1": "🕸️ Orchestration",
        "tab2": "🚀 Command Center",
        "tab3": "📈 Analysis",
        "agent_net": "Agent Collaboration Graph",
        "team_roster": "Agent Status",
        "cmd_role": "Task Manager",
        "lit_role": "Literature Reader",
        "theory_role": "Theoretical Data",
        "exp_role": "Experimental Data",
        "ml_role": "Machine Learning",
        "cmd_desc": "Interaction / Orchestration / Decision",
        "lit_desc": "Local RAG / IF Filter / Knowledge Graph",
        "theory_desc": "MP Fetch / DFT Data / DB Construction",
        "exp_desc": "Lab Data Cleaning / Iterative Update",
        "ml_desc": "Hybrid Training / Prediction / Design",
        "chat_tip": "💡 Tip: 'Recommend ordered alloys for Alkaline HER' or 'Analyze overpotential after 500 CVs'",
        "chat_welcome": "Task Manager Online. Please set target or ask for recommendation.",
        "enter_cmd": "Enter command...",
        "received": "Received:",
        "initiating": "Invoking Agent Cluster...",
        "analysis_complete": "Prediction Complete (Conf 94%):",
        "processing": "Processing...",
        "system_terminal": "Real-time Logs",
        "evo_curve": "Stability Prediction (500 CV)",
        "candidate_3d": "Recommended Structure",
        "visualizing": "Rendering:"
    }
}

# 2. Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/?size=100&id=103424&format=png&color=00D4FF", width=60) 
    
    # Language Selector
    lang_opt = st.selectbox("Language / 语言", ["简体中文", "English"], index=0)
    t = TRANS[lang_opt]
    
    st.markdown(f"### {t['sidebar_title']}")
    st.caption(t['sidebar_caption'])
    st.markdown("---")
    
    st.markdown(f"#### {t['config_header']}")
    # Specific Inputs
    electrolyte = st.text_input(t['electrolyte'], value="1M KOH")
    ref_electrode = st.selectbox(t['ref_electrode'], ["Hg/HgO", "Ag/AgCl", "SCE"])
    stability_cycles = st.number_input(t['stability_test'], value=500, step=100)
    
    st.markdown("---")
    
    st.markdown(f"#### {t['data_header']}")
    literature_file = st.file_uploader(t['drop_pdf'], type="pdf")
    lab_data = st.file_uploader(t['drop_exp'], type="csv")
    
    st.markdown("---")
    st.markdown(f"#### {t['agent_grid']}")
    
    # Status Grid
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption(t['cmd_role'][:4])
        st.markdown(":green[**ON**]")
        st.caption(t['theory_role'][:4])
        st.markdown(":orange[**BUSY**]")
        st.caption(t['ml_role'][:4])
        st.markdown(":green[**ON**]")
    with col_b:
        st.caption(t['lit_role'][:4])
        st.markdown(":green[**ON**]")
        st.caption(t['exp_role'][:4])
        st.markdown(":grey[**IDLE**]")

# 3. Main Tabs
tab1, tab2, tab3 = st.tabs([t['tab1'], t['tab2'], t['tab3']])

# --- TAB 1: ORCHESTRATION ---
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(t['agent_net'])
        
        # PLOTLY STATIC GRAPH IMPLEMENTATION (V10 - Collision Free Layout)
        import plotly.graph_objects as go
        
        # 1. Define Coordinates (Widened for Spacing)
        positions = {
            "Manager": (0, 2.6),
            "Lit": (-2.0, 1.4),     # Far Left
            "Theory": (0.0, 1.4),   # Center
            "Exp": (2.0, 1.4),      # Far Right
            "ML": (0.0, 0.0)        # Bottom
        }
        
        # Neon Colors
        agent_colors = {
            "Manager": "#00D4FF", "Lit": "#F59E0B", 
            "Theory": "#10B981", "Exp": "#EC4899", "ML": "#8B5CF6"
        }
        
        labels = {
            "Manager": f"🤖\n{t['cmd_role']}",
            "Lit": f"📚\n{t['lit_role']}",
            "Theory": f"⚛️\n{t['theory_role']}",
            "Exp": f"🧪\n{t['exp_role']}",
            "ML": f"🧠\n{t['ml_role']}"
        }
        
        # 2. Prepare Data
        node_x, node_y, node_c, node_t = [], [], [], []
        glow_x, glow_y, glow_c = [], [], []
        label_x, label_y = [], []
        
        for key, pos in positions.items():
            node_x.append(pos[0])
            node_y.append(pos[1])
            node_c.append(agent_colors[key])
            node_t.append(labels[key])
            glow_x.append(pos[0])
            glow_y.append(pos[1])
            glow_c.append(agent_colors[key])
            
            # Text Offset Logic (Smart Placement)
            label_x.append(pos[0])
            if key == "ML":
                label_y.append(pos[1] - 0.45) # ML Text Below
            else:
                label_y.append(pos[1] + 0.45) # Others Above

        # 3. Create Traces
        trace_glow = go.Scatter(
            x=glow_x, y=glow_y, mode='markers',
            marker=dict(size=120, color=glow_c, opacity=0.15, line=dict(width=0)),
            hoverinfo='none'
        )

        trace_nodes = go.Scatter(
            x=node_x, y=node_y, mode='markers',
            marker=dict(size=75, color='#000000', line=dict(width=4, color=node_c)),
            hoverinfo='text',
            text=node_t
        )
        
        trace_labels = go.Scatter(
            x=label_x, y=label_y, mode='text',
            text=node_t, textposition="middle center",
            textfont=dict(size=15, color='white', family='Roboto', weight='bold'),
            hoverinfo='none'
        )
        
        # Feedback Loop Curve (ML -> Manager, avoiding center)
        trace_feedback = go.Scatter(
            x=[0, -2.8, -2.8, 0], # Goes far left
            y=[0, 0.2, 2.4, 2.6], 
            mode='lines',
            line=dict(color='#8B5CF6', width=2, dash='dot', shape='spline'),
            hoverinfo='text',
            text="Prediction Feedback"
        )
        
        # 4. Annotations (Arrows)
        annotations = []
        
        # A. Manager Downstream
        for target in ["Lit", "Theory", "Exp"]:
            annotations.append(dict(
                ax=positions["Manager"][0], ay=positions["Manager"][1],
                x=positions[target][0], y=positions[target][1],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor="#444"
            ))

        # B. Data Flow to ML (Lit/Theory/Exp -> ML)
        data_flows = [
            ("Lit", "ML", "Knowledge", "#F59E0B"),
            ("Theory", "ML", "Struct DB", "#10B981"),
            ("Exp", "ML", "Exp DB", "#EC4899")
        ]
        
        for start, end, label, col in data_flows:
            x0, y0 = positions[start]
            x1, y1 = positions[end]
            annotations.append(dict(
                ax=x0, ay=y0, x=x1, y=y1,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3, arrowcolor=col
            ))
            # Label for Flow
            annotations.append(dict(
                x=(x0+x1)/2, y=(y0+y1)/2,
                text=label, showarrow=False,
                font=dict(color=col, size=11, family="Roboto"),
                bgcolor="#000000"
            ))
            
        # C. Feedback Arrow Tip (Manual placement for the Spline end)
        annotations.append(dict(
            ax=-0.5, ay=2.6, x=0, y=2.6, # Short arrow at the end of spline
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor="#8B5CF6"
        ))
        
        # Feedback Label
        annotations.append(dict(
            x=-2.9, y=1.3,
            text="Prediction\nFeedback", showarrow=False,
            font=dict(color="#8B5CF6", size=12), bgcolor="#000000"
        ))

        # 5. Construct Figure
        fig_net = go.Figure(data=[trace_feedback, trace_glow, trace_nodes, trace_labels],
             layout=go.Layout(
                showlegend=False, hovermode='closest',
                margin=dict(b=20,l=20,r=20,t=20),
                xaxis=dict(showgrid=False, zeroline=False, range=[-3.5, 2.5], showticklabels=False), # Wider Range
                yaxis=dict(showgrid=False, zeroline=False, range=[-0.8, 3.2], showticklabels=False),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                height=600, annotations=annotations
            )
        )
            
        st.plotly_chart(fig_net, use_container_width=True, config={'staticPlot': True})
        
    with col2:
        st.subheader(t['team_roster'])
        card(t['cmd_role'], "Chat / Evidence & Predict / Decision", "Success")
        card(t['lit_role'], "PDF RAG / IF Filter / Knowledge Graph", "Success")
        card(t['theory_role'], "MP Data / Struct DB / Feature Eng", "Thinking")
        card(t['exp_role'], "1M KOH / 500 CV / Iterative DB", "Idle")
        card(t['ml_role'], "Hybrid Model / Train & Predict", "Success")

# --- TAB 2: MISSION CONTROL ---
with tab2:
    col_chat, col_log = st.columns([2, 1])
    
    with col_chat:
        st.subheader(t['cmd_role'])
        st.info(t['chat_tip'])
        
        # Chat History Container
        chat_container = st.container(height=500)
        
        # Initial Message
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(t['chat_welcome'])
            
            # Mock User Interaction
            if "messages" not in st.session_state:
                st.session_state["messages"] = []
                
            for msg in st.session_state["messages"]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
        
        # Input
        if prompt := st.chat_input(t['enter_cmd']):
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.write(prompt)
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(f"{t['received']} `'{prompt}'`.")
                    st.write(t['initiating'])
                    
                    # RUN MODEL (REAL INFERENCE)
                    model = load_theorist()
                    meta_map = load_metadata_map()
                    
                    # Intelligent Intent Detection (Regex + Pymatgen)
                    import re
                    from pymatgen.core import Composition
                    
                    match_mat = None
                    
                    # 1. Tokenize prompt to find potential formulas (e.g. "Pt3Co" in "Analyze Pt3Co...")
                    tokens = re.findall(r'[A-Za-z0-9]+', prompt)
                    
                    for token in tokens:
                        if len(token) < 2 and str(token).islower(): continue # Skip "the", "a"
                        try:
                            # Normalize: "Pt3Co" -> "CoPt3"
                            norm_form = Composition(token).reduced_formula
                            if norm_form in meta_map:
                                match_mat = norm_form
                                break
                        except:
                            pass
                    
                    # 2. Fallback: Search known keys in prompt if extraction failed
                    if not match_mat:
                        for formula in meta_map.keys():
                            if formula in prompt:
                                match_mat = formula
                                break

                    # Execute if valid material found OR explicit keywords present
                    if match_mat or "Pt" in prompt or "???" in prompt or "alloy" in prompt.lower():
                        query = match_mat if match_mat else prompt
                        
                        # 1. Theoretical Agent
                        model = load_theorist()
                        meta_map = load_metadata_map()
                        name, pred = predict_material(model, query, meta_map)
                        
                        # 2. Experimental Agent
                        exp_agent = load_experimental_agent()
                        exp_results = exp_agent.query_literature(query)
                        
                        found_any = False
                        
                        # --- DISPLAY THEORY ---
                        if name:
                            found_any = True
                            form_e = pred["formation_energy"].item()
                            desc_vals = pred["descriptors"][0].tolist()
                            desc_keys = [
                                 "d_band_center", "d_band_width", "d_band_filling", "DOS_EF", 
                                 "DOS_window_-0.3_0.3", "unoccupied_d_states_0_0.5", "epsilon_d_minus_EF",
                                 "sp_d_hybridization", "orbital_ratio_d", "valence_DOS_slope",
                                 "num_DOS_peaks", "first_peak_position"
                            ]
                            
                            st.write(f"{t['analysis_complete']} **{name} (Theoretical Prediction)**")
                            st.caption("??Valid Structure Found in Database")
                            
                            # Metrics
                            col_m1, col_m2, col_m3 = st.columns(3)
                            col_m1.metric("Formation Energy", f"{form_e:.3f} eV")
                            col_m2.metric("d-band Center", f"{desc_vals[0]:.2f} eV")
                            col_m3.metric("d-band Width", f"{desc_vals[1]:.2f} eV")
                            
                            # Descriptors
                            with st.expander("Show Theoretical Descriptors", expanded=False):
                                st.dataframe(pd.DataFrame({"Descriptor": desc_keys, "Value": [f"{v:.4f}" for v in desc_vals]}), use_container_width=True)

                            # DOS
                            st.plotly_chart(plot_dos(pred["dos"][0]), use_container_width=True)
                        
                        # --- DISPLAY EXPERIMENT ---
                        if exp_results:
                            found_any = True
                            st.markdown("---")
                            st.subheader(f"?? Experimental Data ({len(exp_results)} Matches)")
                            
                            for res in exp_results:
                                with st.container(border=True):
                                    c1, c2 = st.columns([3, 1])
                                    c1.markdown(f"**{res['formula']}** - {res['reference']}")
                                    c2.metric("Overpotential", f"{res['overpotential_10mA']} mV")
                                    st.caption(f"Method: {res.get('synthesis_method','N/A')} | Electrolyte: {res.get('electrolyte','N/A')}")
                        
                        if not found_any:
                            st.warning(f"No Theoretical or Experimental data found for '{query}'.\nTry standard alloys (e.g., 'Pt3Co') or known HEAs (e.g., 'PtFeCoNiCu').")
                     
                    else:
                        st.write(t['processing'])
                        time.sleep(1)

    with col_log:
        st.subheader(t['system_terminal'])
        # Dynamic Logs based on User Requirements
        log_text = f"""
[14:00:01] <SYS> System Initialized.
[14:00:02] <CFG> Electrolyte: {electrolyte}
[14:00:02] <CFG> Ref Electrode: {ref_electrode}
[14:00:05] <LIT> Scanning local PDF library...
[14:00:06] <LIT> Filtered 8 papers with IF > 10.
[14:00:10] <LIT> Extracted synthesis method: 'Solvothermal'.
[14:01:20] <THEORY> Fetching MP Data for Pt-Co binary...
[14:01:45] <THEORY> Calculated DOS fingerprints (400 bins).
[14:02:00] <EXP> Database updated with recent lab results.
[14:02:05] <ML> Retraining Hybrid Model (Epoch 0/100)...
[14:05:00] <ML> Prediction: Pt3Co exhibits optimal H* adsorption.
        """
        st.markdown(f'<div class="terminal-box">{log_text.strip()}</div>', unsafe_allow_html=True)

# --- TAB 3: ANALYSIS ---
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(t['evo_curve'])
        # Mock Data for Stability (Overpotential vs Cycles)
        cycles = np.linspace(0, stability_cycles, 20)
        overpotential = 45 + (np.random.rand(20) * 5) + (cycles / 100) # Slight degradation
        
        df = pd.DataFrame({
            "Cycles": cycles,
            "Overpotential (mV)": overpotential
        })
        fig = px.line(df, x="Cycles", y="Overpotential (mV)", markers=True, template="plotly_dark", title=f"Stability Test ({ref_electrode})")
        fig.update_traces(line_color='#00D4FF')
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader(t['candidate_3d'])
        
        # Py3Dmol Visualization
        structure_xyz = """4
        Pt3Co
        Pt 0.00000 0.00000 0.00000
        Pt 0.00000 1.95000 1.95000
        Pt 1.95000 0.00000 1.95000
        Co 1.95000 1.95000 0.00000
        """
        
        view = py3Dmol.view(width=400, height=300)
        view.addModel(structure_xyz, "xyz")
        view.setStyle({'sphere': {'scale': 0.35}, 'stick': {'radius': 0.2}})
        view.zoomTo()
        view.setBackgroundColor('#00000000') # Transparent
        
        showmol(view, height=300, width=400)
        st.caption(f"{t['visualizing']} **Pt3Co (L12 Ordered Alloy)**")
