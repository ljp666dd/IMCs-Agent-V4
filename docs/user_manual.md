# IMCs 用户手册（Streamlit 主入口）

本手册面向使用者，强调“多智能体协作 + 任务图 + 证据链”的科研体验。当前首个可展示场景为 **HOR 候选有序合金材料发现**。

## 1. 快速开始

### 1.1 环境准备
- Python 3.10+、pip
- 在项目根目录配置 `.env`（必填）：
  ```ini
  MP_API_KEY=your_materials_project_api_key
  IMCS_SECRET_KEY=your_production_secret_key
  IMCS_API_URL=http://localhost:8000
  ```

### 1.2 一键启动（推荐）
双击 `start_imcs.bat`：
- 后端 API：`http://localhost:8000`
- Streamlit UI：`http://localhost:8501`

### 1.3 手动启动
```powershell
# Backend API
python src/api/main.py

# Streamlit UI
streamlit run src/interface/app.py
```

---

## 2. 界面导航概览
侧边栏主要入口：
- **🏠 首页**：系统概览与功能卡片
- **🤖 智能体对话**：输入任务、生成/执行任务图、查看证据链与推荐结果
- **📊 数据分析**：理论/实验数据加载与可视化
- **🧪 ML 训练**：模型训练、特征选择与预测
- **📚 文献检索**：文献检索与摘要归纳
- **🔌 API 状态**：后端在线状态与健康检查
- **⚙️ 设置**：缓存清理与配置提示

---

## 3. HOR 候选有序合金发现（可展示场景）
建议演示流程如下：
1. 进入 **智能体对话** 页面
2. 输入问题：`发现 HOR 候选有序合金材料` (无需手动限定元素，系统已预设元素集)
3. 系统生成 **任务图**（文献 → 理论 → ML → 推荐）
4. 点击执行，观察步骤状态变化（pending / running / completed / failed / blocked / retrying）
5. 在推荐区域查看候选材料与排序
6. 在“Evidence Chain”区域选择材料，查看 **理论/文献/ML/实验**证据

---

## 4. 任务图与证据链说明
### 4.1 任务图（Task Graph）
- 每个任务由多个 **步骤（Step）** 组成
- 每个步骤含有 **依赖关系**（dependencies）
- 状态含义：
  - `pending`：等待执行
  - `running`：执行中
  - `completed`：完成
  - `failed`：失败
  - `blocked`：依赖未满足
  - `retrying`：失败后重试中

### 4.2 证据链（Evidence Chain）
证据来源类型：
- `theory`：理论计算/数据库证据（如 formation_energy）
- `literature`：文献证据（标题/年份等元信息）
- `ml_prediction`：模型预测证据
- `experiment`：实验数据证据（若已上传）

证据会被汇总到材料实体上，形成可追溯的科研链路。

---

## 5. 数据输入与训练
### 5.1 理论数据
- 依赖 Materials Project（需要 `MP_API_KEY`）
- 可通过任务图或 API 下载数据到 `data/`

### 5.2 实验数据
- 支持 CSV/Excel 上传
- 支持 LSV/CV/Tafel/EIS 等基本电化学数据解析
- 数据会存储于 `data/experimental`

### 5.3 ML 训练
- 支持传统 ML 与深度学习（视配置）
- 可选择数据源（理论/实验）并训练模型

---

## 6. 常见问题
- **API 红色 / 无法连接**：确认 `python src/api/main.py` 已启动，或检查 `IMCS_API_URL`
- **材料数据为空**：检查 Materials Project Key，或先执行数据下载
- **任务卡住**：可点击侧边栏“Clear Cache”，或重启服务
- **证据链为空**：需要先完成任务执行，或上传实验数据
---

## 7. 更新记录
- 2026-01-25：补充任务图/证据链说明，新增进度与计划文档 `docs/progress_plan.md`
- 2026-01-25: 支持中文任务触发（HOR/合金场景）
- 2026-01-25: 证据链增加吸附能与 DOS 描述符基础支持
- 2026-01-25: Added MARS benchmark doc `docs/mars_benchmark.md`
- 2026-01-25: Added examples and evaluation folders for reproducible demos
- 2026-01-25: Added idea upgrade doc `docs/idea_upgrade.md`


## 8. Literature-Driven HOR Pipeline (One Command)

Run a full pipeline in one command:
**online literature harvest -> seed CSV -> metrics + LSV generation -> import -> LSV analysis**

```powershell
python src/tools/harvest_literature_hor_seed.py --query "HOR ordered alloy catalyst" --limit 15 --max-pdfs 5 --persist --run-all --seeded
```

Outputs:
- `data/experimental/literature_hor_seed.csv`
- `data/experimental/literature_activity_metrics.csv`
- `data/experimental/literature_rde_lsv/`
