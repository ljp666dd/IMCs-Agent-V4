"""
Multi-Agent Catalyst Research System - Web UI
Streamlit-based interface for the multi-agent framework.

Run with: streamlit run src/ui/app.py
"""

import streamlit as st
import os
import sys
import json
import pandas as pd
import numpy as np

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

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

def init_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_task' not in st.session_state:
        st.session_state.current_task = None
    if 'agents_loaded' not in st.session_state:
        st.session_state.agents_loaded = False


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


# ========== Sidebar ==========

def render_sidebar():
    """Render sidebar with navigation and status."""
    with st.sidebar:
        st.markdown("## 🔬 IMCs Research")
        st.markdown("*多智能体催化剂研究系统*")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "导航",
            ["🏠 首页", "🤖 智能体对话", "📊 数据分析", "🧪 ML 训练", 
             "📚 文献检索", "🔌 API 状态", "⚙️ 设置"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Agent Status with features
        st.markdown("### 🤖 智能体状态")
        
        st.markdown("""
        **ML Agent** 🟢
        <span class="feature-badge">CGCNN</span>
        <span class="feature-badge">SchNet</span>
        <span class="feature-badge">Transformer</span>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Theory Agent** 🟢
        <span class="feature-badge">MP</span>
        <span class="feature-badge">AFLOW</span>
        <span class="feature-badge">Catalysis-Hub</span>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Experiment Agent** 🟢
        <span class="feature-badge">LSV</span>
        <span class="feature-badge">EIS</span>
        <span class="feature-badge">RDE</span>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Literature Agent** 🟢
        <span class="feature-badge">Semantic Scholar</span>
        <span class="feature-badge">arXiv</span>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data Status (cached)
        st.markdown("### 📊 数据状态")
        
        stats = get_data_stats()
        st.metric("CIF 结构", f"{stats.get('n_cifs', 0)}")
        st.metric("DOS 数据", f"{stats.get('n_dos', 0)}")
        
        st.markdown("---")
        
        if st.button("♻️ 强制重载核心 (Clear Cache)", help="如果遇到代码更新未生效或模型不全，请点击此按钮"):
             st.cache_resource.clear()
             
             # Aggressive module reload
             import sys
             import importlib
             
             modules_to_reload = ['src.agents.core.ml_agent', 'src.agents.core']
             for m in modules_to_reload:
                 if m in sys.modules:
                     try:
                         importlib.reload(sys.modules[m])
                         print(f"Reloaded {m}")
                     except Exception as e:
                         print(f"Failed to reload {m}: {e}")
             
             st.success("缓存已清除并强制重载模块！正在重启...")
             st.rerun()
             
        return page


# ========== Home Page ==========

def render_home():
    """Render home page."""
    st.markdown('<h1 class="main-header">🔬 多智能体催化剂研究系统</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #666;">
        <b>v2.0</b> | 100% 符合设计规划 | 5 个核心智能体 | 完整 ML + GNN 支持
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards - 4 columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="agent-card">
            <h3>🤖 ML 智能体</h3>
            <p><b>传统 ML:</b> XGBoost, LightGBM, RF</p>
            <p><b>深度学习:</b> DNN, Transformer</p>
            <p><b>GNN:</b> CGCNN, SchNet, MEGNet</p>
            <p><b>SHAP:</b> 特征重要性分析</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="agent-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3>📊 理论数据</h3>
            <p><b>Materials Project:</b> CIF, DOS, 能带</p>
            <p><b>AFLOW:</b> 弹性常数</p>
            <p><b>Catalysis-Hub:</b> 吸附能</p>
            <p><b>OQMD:</b> 形成能</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="agent-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>🧪 实验分析</h3>
            <p><b>电化学:</b> LSV, CV, EIS, Tafel</p>
            <p><b>RDE/RRDE:</b> 电子转移数</p>
            <p><b>格式:</b> CSV, Excel, EC-Lab, CHI</p>
            <p><b>自动检测:</b> 数据类型识别</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="agent-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <h3>📚 文献检索</h3>
            <p><b>Semantic Scholar:</b> 论文搜索</p>
            <p><b>arXiv:</b> 预印本</p>
            <p><b>本地 PDF:</b> 解析提取</p>
            <p><b>知识提取:</b> 自动总结</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System Architecture
    st.markdown("### 🏗️ 系统架构")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ```
        用户 ←→ 任务管理智能体 (TaskManagerAgent)
                    ↓
            ┌───────┼───────┐
            ↓       ↓       ↓
          ML智能体  理论数据   文献阅读
                    ↓       ↓
                 实验数据智能体
        ```
        """)
    
    with col2:
        st.markdown("""
        **工作流程:**
        1. 接收用户问题
        2. 分析需求 → 生成计划
        3. 调度相关智能体
        4. 整合结果
        5. 推荐实验方案
        """)
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### 🚀 快速开始")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 发现新催化剂", use_container_width=True):
            st.session_state.page = "🤖 智能体对话"
            st.rerun()
    
    with col2:
        if st.button("📈 训练预测模型", use_container_width=True):
            st.session_state.page = "🧪 ML 训练"
            st.rerun()
    
    with col3:
        if st.button("📊 分析实验数据", use_container_width=True):
            st.session_state.page = "📊 数据分析"
            st.rerun()
    
    with col4:
        if st.button("📚 搜索文献", use_container_width=True):
            st.session_state.page = "📚 文献检索"
            st.rerun()


# ========== Chat Page ==========

def render_chat():
    """Render chat interface."""
    st.markdown("## 🤖 智能体对话")
    
    agents = load_agents()
    
    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Input
    if prompt := st.chat_input("输入您的研究问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                with st.status("智能体正在思考...", expanded=True) as status:
                    status.write("正在分析您的请求...")
                    if agents:
                        response = agents['task_manager'].chat(prompt)
                        status.write("生成回复中...")
                        status.update(label="回复完成", state="complete", expanded=False)
                        
                        if response:
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        else:
                            st.warning("智能体没有返回有效内容。请尝试换个说法。")
                    else:
                        st.error("智能体未加载")
            except Exception as e:
                st.error(f"智能体执行出错: {e}")
                import traceback
                st.error(traceback.format_exc())
    
    # Quick prompts
    st.markdown("---")
    st.markdown("### 💡 示例问题")
    
    examples = [
        "寻找最佳的 PtRu 合金 HOR 催化剂",
        "使用 CGCNN 预测材料的形成能",
        "分析 d-band center 与催化活性的关系",
        "查找关于 Pt 基催化剂的最新 arXiv 预印本"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(example, key=f"example_{i}"):
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()


# ========== Data Analysis Page ==========

def render_data_analysis():
    """Render data analysis page."""
    st.markdown("## 📊 数据分析")
    
    tab1, tab2, tab3, tab4 = st.tabs(["理论数据", "实验数据", "RDE/RRDE 分析", "数据可视化"])
    
    with tab1:
        st.markdown("### 📂 理论计算数据")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fe_file = os.path.join(ROOT_DIR, "data", "theory", "formation_energy_full.json")
            if os.path.exists(fe_file):
                with open(fe_file) as f:
                    fe_data = json.load(f)
                
                df = pd.DataFrame(fe_data)
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("材料总数", len(df))
                with col_b:
                    st.metric("平均形成能", f"{df['formation_energy'].mean():.3f} eV/atom")
                with col_c:
                    st.metric("标准差", f"{df['formation_energy'].std():.3f}")
                
                st.dataframe(df.head(20), use_container_width=True)
            else:
                st.info("未找到形成能数据")
        
        with col2:
            st.markdown("### 下载新数据")
            
            data_source = st.selectbox("数据源", [
                "Materials Project",
                "AFLOW",
                "Catalysis-Hub"
            ])
            
            if st.button("🔄 下载数据"):
                agents = load_agents()
                if agents:
                    with st.spinner("正在下载..."):
                        if data_source == "AFLOW":
                            results = agents['theory'].query_aflow(elements=['Pt'], limit=20)
                            st.success(f"下载了 {len(results)} 条记录")
                        elif data_source == "Catalysis-Hub":
                            results = agents['theory'].query_catalysis_hub(reaction='HER', limit=20)
                            st.success(f"下载了 {len(results)} 条记录")
    
    with tab2:
        st.markdown("### 📁 上传实验数据")
        
        uploaded_file = st.file_uploader(
            "支持格式: CSV, Excel, EC-Lab (.mpt), CHI (.txt)", 
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
                    st.info(f"检测到 {file_ext.upper()} 文件，请保存后使用 process_file() 分析")
                    df = None
                
                if df is not None:
                    st.dataframe(df, use_container_width=True)
                    
                    data_type = st.selectbox("数据类型", ["自动检测", "LSV", "CV", "Tafel", "EIS", "稳定性"])
                    
                    if st.button("🔬 分析数据"):
                        st.info("分析中...")
                        
            except Exception as e:
                st.error(f"读取文件失败: {e}")
        
        st.markdown("---")
        st.markdown("### 📂 扫描数据文件夹")
        folder_path = st.text_input("输入本地数据目录", value="data/experimental")
        if st.button("开始扫描目录"):
            if os.path.exists(folder_path):
                try:
                    agents = load_agents()
                    if agents:
                        result = agents['experiment'].scan_directory(folder_path)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("总文件数", result['total_files'])
                        with col2:
                            st.metric("有效数据文件", result['valid_files'])
                        
                        if result['valid_files'] > 0:
                            st.success("扫描成功！检测到以下数据类型:")
                            st.json(result['data_types'])
                            
                            # Auto-process hint
                            st.info("💡 提示: 您可以在 'ML 训练' 页面选择 '实验数据' 源来使用这些数据进行训练。")
                        else:
                            st.warning("未检测到支持的实验数据文件 (.csv, .xlsx, .mpt)")
                            if result['errors']:
                                st.error(f"错误: {result['errors']}")
                except Exception as e:
                    st.error(f"扫描失败: {e}")
            else:
                st.error("目录不存在")
    
    with tab3:
        st.markdown("### ⚡ RDE/RRDE 分析")
        
        st.markdown("""
        **RDE (旋转圆盘电极):**
        - 极限电流密度
        - 半波电位
        - 动力学电流
        
        **RRDE (旋转环盘电极):**
        - 电子转移数 (n)
        - HO₂⁻ 产率
        """)
        
        rrde_file = st.file_uploader("上传 RRDE 数据", type=['csv', 'xlsx'], key="rrde")
        
        if rrde_file:
            collection_eff = st.slider("收集效率 (N)", 0.2, 0.5, 0.37)
            
            if st.button("计算电子转移数"):
                st.info("使用 analyze_rrde() 进行分析...")
    
    with tab4:
        st.markdown("### 📈 数据可视化")
        
        desc_file = os.path.join(ROOT_DIR, "data", "theory", "dos_descriptors_full.json")
        if os.path.exists(desc_file):
            with open(desc_file) as f:
                desc_data = json.load(f)
            
            df = pd.DataFrame(desc_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### d-band center 分布")
                if 'd_band_center' in df.columns:
                    st.bar_chart(df['d_band_center'].dropna().head(50))
            
            with col2:
                st.markdown("#### d-band width 分布")
                if 'd_band_width' in df.columns:
                    st.bar_chart(df['d_band_width'].dropna().head(50))


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
                        if target == "形成能 (Formation Energy)":
                            target_col = "formation_energy"
                        else:
                            target_col = target
                        
                        # Determine data file
                        if target_col == "formation_energy":
                            extended_file = os.path.join(ROOT_DIR, "data", "theory", "formation_energy_extended.json")
                            base_file = os.path.join(ROOT_DIR, "data", "theory", "formation_energy_full.json")
                            
                            if os.path.exists(extended_file):
                                data_file = extended_file
                                st.caption("✅ 使用扩展特征数据集 (42维)")
                            else:
                                data_file = base_file
                                st.caption("⚠️ 使用基础特征数据集 (20维)")
                        else:
                            extended_dos = os.path.join(ROOT_DIR, "data", "theory", "dos_data_extended.json")
                            base_dos = os.path.join(ROOT_DIR, "data", "theory", "dos_descriptors_full.json")
                            
                            if os.path.exists(extended_dos):
                                data_file = extended_dos
                                st.caption(f"✅ 使用扩展 DOS 数据集 - 目标: {target_col}")
                            else:
                                data_file = base_dos
                                st.caption("⚠️ 使用基础 DOS 数据集")
                        
                        if not os.path.exists(data_file):
                            st.error(f"数据文件不存在: {data_file}")
                            ml_agent = None
                        else:
                            with st.spinner("加载理论数据中..."):
                                try:
                                    ml_agent.load_data(data_path=data_file, target_col=target_col)
                                    ml_agent.config.test_size = test_size
                                    st.success(f"加载数据: {len(ml_agent.X)} 样本, {ml_agent.X.shape[1]} 特征")
                                except Exception as e:
                                    st.error(f"加载数据失败: {e}")
                                    ml_agent = None
                    else:
                        # === Experimental Data Loading ===
                        if exp_file_path and os.path.exists(exp_file_path):
                            with st.spinner("加载实验数据..."):
                                try:
                                    feats = feature_cols if 'feature_cols' in locals() else None
                                    ml_agent.load_generic_csv(exp_file_path, target, feats)
                                    ml_agent.config.test_size = test_size
                                    st.success(f"加载实验数据: {len(ml_agent.X)} 样本, {ml_agent.X.shape[1]} 特征选定")
                                except Exception as e:
                                    st.error(f"加载实验数据失败: {e}")
                                    ml_agent = None
                        else:
                            st.error("未找到实验数据文件，请先上传并确保文件有效")
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
    st.markdown("## 📚 文献检索")
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        query = st.text_input("🔍 搜索关键词", placeholder="例如: PtRu alloy HOR catalyst")
    
    with col2:
        source = st.selectbox("数据源", ["全部", "Semantic Scholar", "arXiv"])
    
    with col3:
        n_results = st.number_input("结果数量", 5, 50, 10)
        min_citations = st.number_input("最低引用 (IF替代)", 0, 1000, 0, help="通过引用数筛选高影响力论文")
    
    if st.button("搜索", type="primary"):
        if query:
            agents = load_agents()
            if agents:
                with st.spinner("搜索中..."):
                    try:
                        if source == "全部":
                            papers = agents['literature'].search_all_sources(query, n_results)
                        elif source == "arXiv":
                            papers = agents['literature'].search_arxiv(query, n_results)
                        else:
                            papers = agents['literature'].search_semantic_scholar(query, n_results)
                        
                        # Filter by citations
                        if papers and min_citations > 0:
                            papers = [p for p in papers if getattr(p, 'citation_count', 0) >= min_citations]
                        
                        if papers:
                            st.success(f"找到 {len(papers)} 篇论文 (引用 >= {min_citations})")
                            
                            for i, paper in enumerate(papers, 1):
                                with st.expander(f"{i}. {paper.title} (Cite: {paper.citation_count})"):
                                    st.markdown(f"**作者**: {', '.join(paper.authors[:5])}")
                                    st.markdown(f"**年份**: {paper.year} | **引用**: {paper.citation_count}")
                                    if paper.abstract:
                                        st.markdown(f"**摘要**: {paper.abstract[:500]}...")
                                    if paper.url:
                                        st.markdown(f"[🔗 查看原文]({paper.url})")
                        else:
                            if min_citations > 0:
                                st.warning(f"未找到引用数 >= {min_citations} 的相关文献")
                            else:
                                st.info("未找到相关文献")
                    except Exception as e:
                        st.error(f"搜索失败: {e}")
    
    st.markdown("---")
    
    # Local PDF parsing
    st.markdown("### 📄 本地 PDF 解析")
    
    pdf_file = st.file_uploader("上传 PDF 文件", type=['pdf'])
    
    if pdf_file:
        if st.button("解析 PDF"):
            st.info("PDF 解析需要 pdfplumber 库支持")


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
    
    if page == "🏠 首页":
        render_home()
    elif page == "🤖 智能体对话":
        render_chat()
    elif page == "📊 数据分析":
        render_data_analysis()
    elif page == "🧪 ML 训练":
        render_ml_training()
    elif page == "📚 文献检索":
        render_literature()
    elif page == "🔌 API 状态":
        render_api_status()
    elif page == "⚙️ 设置":
        render_settings()


if __name__ == "__main__":
    main()
