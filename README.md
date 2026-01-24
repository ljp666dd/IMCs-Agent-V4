# Intelligent Materials Analysis System (IMCs) v3.3

**IMCs** 是一个基于多智能体架构 (Multi-Agent System) 的科研辅助平台，专为**有序合金 (Intermetallic Compounds)** 的电化学性能研究设计。系统集成了理论计算 (DFT)、机器学习 (ML)、实验数据处理与文献分析功能，旨在加速催化剂的筛选与发现过程。

## 🌟 核心特性 (v3.3 Features)

- **多智能体协作**: 集成 Task, Theory, ML, Experiment, Literature 五大智能体，由 Service 层驱动。
- **现代化架构**: 采用 **FastAPI (Backend)** + **Next.js (Frontend)** 的前后端分离设计。
- **数据管理**: 内置 **SQLite** 数据库，支持 CIF 结构、实验数据 (CV/LSV) 与 ML 模型的持久化存储。
- **自动化工作流**: 
  - 自动从 Materials Project 下载结构并计算 d-band center。
  - 自动解析实验文件夹，提取过电位与 Tafel 斜率。
  - 自动进行机器学习模型训练 (XGBoost/GNN) 并评估特征重要性 (SHAP)。

## 🏗️ 系统架构

```mermaid
graph TD
    User[Web Frontend (Next.js)] <--> API[Backend API (FastAPI)]
    API <--> Task[Task Manager Service]
    
    subgraph Services
        Task --> Planner
        Task --> Executor
        Executor --> Theory[Theory Service]
        Executor --> ML[ML Service]
        Executor --> Exp[Experiment Service]
        Executor --> Lit[Literature Service]
    end
    
    Services --> DB[(SQLite Database)]
```

## 🚀 快速开始 (Quick Start)

### 1. 环境准备 (Prerequisites)
- Python 3.9+
- Node.js 18+
- Materials Project API Key

### 2. 启动后端 (Backend)
```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 启动 API 服务
python src/api/main.py
```
> 服务将运行在 `http://localhost:8000`，API 文档见 `/docs`。

### 3. 启动前端 (Frontend)
```bash
cd src/ui/web

# 安装 NPM 依赖
npm install

# 启动开发服务器
npm run dev
```
> 访问 `http://localhost:3000` 使用平台。

## 📂 项目结构

```bash
IMCs/
├── src/
│   ├── agents/         # 智能体外观层 (Agent Facades)
│   ├── services/       # 核心业务逻辑 (Core Services)
│   │   ├── chemistry/  # 特征提取
│   │   ├── db/         # 数据库操作
│   │   ├── experiment/ # 实验数据解析
│   │   ├── literature/ # 文献搜索与分析
│   │   ├── ml/         # 机器学习引擎
│   │   ├── task/       # 任务规划器
│   │   └── theory/     # MP API 与物理计算
│   ├── core/           # 基础设施 (Logger, Config)
│   ├── api/            # REST API (FastAPI)
│   └── ui/web/         # Web 前端 (Next.js)
├── data/               # 本地数据缓存 (GitIgnored)
└── tests/              # 单元测试
```

## ⚠️ 注意事项
- 本项目包含大文件（如训练数据、模型权重），已在 `.gitignore` 中配置。发布前请检查。
- 使用前请在 `config.yaml` 或环境变量中配置您的 MP API Key。

## 📜 License
MIT License