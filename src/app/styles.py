import streamlit as st

def apply_custom_styles():
    st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=JetBrains+Mono:wght@400&display=swap');
        
        /* Main Page Background */
        .stApp {
            background-color: #0E1117;
        }

        /* HIDE STREAMLIT HEADER/FOOTER */
        header[data-testid="stHeader"] {
            background-color: transparent !important;
            visibility: hidden !important;
        }
        #MainMenu {
            visibility: hidden;
        }
        footer {
            visibility: hidden;
        }
        .stDeployButton {
            display: none;
        }

        /* Content Alignment */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* HEADERS - Cyan */
        h1, h2, h3, h4, h5, h6 {
            color: #00D4FF !important;
            font-family: 'Roboto', sans-serif;
            font-weight: 700;
            text-shadow: none !important;
        }
        
        /* BODY TEXT - White */
        .stMarkdown, p, span, div, li {
            color: #FFFFFF;
            font-family: 'Roboto', sans-serif;
        }
        
        /* --- INPUTS & WIDGETS CRITICAL FIX --- */
        
        /* 1. Reset all Input Backgrounds to Dark Grey */
        .stTextInput input, 
        .stNumberInput input, 
        .stSelectbox div[data-baseweb="select"], 
        .stTextArea textarea {
            background-color: #262730 !important;
            color: #FFFFFF !important;
            caret-color: #FFFFFF !important; /* Cursor color */
            border: 1px solid #4B5563 !important;
            border-radius: 5px !important;
        }

        /* 2. NUCLEAR DROPDOWN FIX (V6) */
        
        /* The container of the dropdown list */
        div[data-baseweb="popover"] {
            background-color: #000000 !important;
            border: 1px solid #333 !important;
        }
        
        /* The list itself */
        div[data-baseweb="menu"],
        ul[data-baseweb="menu"],
        div[role="listbox"] {
            background-color: #000000 !important;
        }
        
        /* The items in the list */
        li[data-baseweb="menu-item"],
        li[role="option"],
        div[role="option"] {
            background-color: #000000 !important; 
            color: #FFFFFF !important;
        }
        
        /* The Text inside the items */
        li[data-baseweb="menu-item"] div,
        li[data-baseweb="menu-item"] span,
        div[role="option"] div,
        div[role="option"] span {
            color: #FFFFFF !important;
        }
        
        /* HOVER / FOCUS States */
        li[data-baseweb="menu-item"]:hover,
        div[role="option"]:hover,
        li[role="option"]:hover,
        li[aria-selected="true"] {
            background-color: #333333 !important;
            color: #00D4FF !important;
        }
        
        /* Fix for Inner Divs inheriting wrong colors */
        li[data-baseweb="menu-item"]:hover > div,
        div[role="option"]:hover > div {
             color: #00D4FF !important;
        }
        
        /* 3. Helper Text (Labels) */
        label, .stWidgetLabel, .stCaption {
            color: #E0E0E0 !important;
            font-weight: 500 !important;
        }

        /* 4. Fix "Drag and Drop" Cloud Icon Area */
        .stFileUploader div[data-testid="stFileUploaderDropzone"] {
            background-color: #000000 !important;
            border: 1px dashed #00D4FF !important;
        }
        
        .stFileUploader div[data-testid="stFileUploaderDropzone"] div,
        .stFileUploader div[data-testid="stFileUploaderDropzone"] span,
        .stFileUploader div[data-testid="stFileUploaderDropzone"] small {
             color: #FFFFFF !important;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #000000 !important;
            border-right: 1px solid #222;
        }
        
        /* Sidebar Text - Force White & Bold */
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }
        
        /* NUCLEAR FILE UPLOADER FIX */
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background-color: #000000 !important;
            border: 1px dashed #00D4FF !important;
        }
        
        /* Force EVERY element inside the dropzone to be white */
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
             color: #FFFFFF !important;
        }

        /* Fix 'Browse files' Button specifically */
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
            color: #FFFFFF !important;
            border-color: #FFFFFF !important;
            background-color: transparent !important;
        }
        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover {
            color: #00D4FF !important;
            border-color: #00D4FF !important;
        }

        /* SIDEBAR INPUTS BLACK BACKGROUND */
        section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
            background-color: #000000 !important;
            border-color: #333 !important;
        }

        /* 5. Metrics */
        div[data-testid="stMetricValue"] {
            color: #00D4FF !important; 
            background-color: transparent !important;
        }
        
        /* 6. Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
            border-bottom: 1px solid #333;
        }
        .stTabs [data-baseweb="tab"] {
            color: #AAAAAA !important; /* Inactive */
            background-color: transparent !important;
        }
        .stTabs [aria-selected="true"] {
            color: #FFFFFF !important; /* Active */
            border-bottom-color: #00D4FF !important;
        }

        /* CARDS */
        .solid-card {
            background-color: #1F2937; /* Grey-800 */
            border: 1px solid #374151;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
        }
        
        /* TERMINAL */
        .terminal-box {
            background-color: #000000 !important;
            color: #00FF00 !important;
            font-family: 'JetBrains Mono', monospace;
            border: 1px solid #333;
            padding: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

def card(title, content, status="Idle"):
    color_map = {
        "Idle": "#6B7280",
        "Thinking": "#F59E0B", 
        "Success": "#10B981",
        "Error": "#EF4444"
    }
    color = color_map.get(status, "#6B7280")
    
    st.markdown(f"""
    <div class="solid-card" style="border-left: 5px solid {color};">
        <h4 style="margin: 0; color: #FFFFFF !important;">{title}</h4>
        <p style="font-size: 12px; color: {color} !important; margin-bottom: 5px;">● {status}</p>
        <p style="font-size: 14px; color: #CCCCCC !important; margin: 0;">{content}</p>
    </div>
    """, unsafe_allow_html=True)
