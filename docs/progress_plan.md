<!-- ASCII padding for tooling compatibility: 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -->


# IMCs 进度与计划记录





记录日期：2026-01-25  


目标：围绕“HOR 候选有序合金发现”，建立可迭代进化的多智能体科研平台（任务图 + 证据链 + 实验闭环）。





---





## 1. 当前进度概览（已具备）


- **主入口**：Streamlit UI 已作为默认入口（`src/interface/app.py`）


- **任务图**：支持 DAG 执行、依赖调度、状态记录，已落库（`plans`/`plan_steps`）


- **证据链**：支持 `theory / literature / ml_prediction / experiment` 证据类型并展示


- **数据库**：Schema v4.0 已升级，支持用户/任务/证据/模型元数据


- **理论数据**：Materials Project 下载与本地缓存，元素范围已在 `TheoryDataConfig.elements` 限定


- **ML 管线**：支持传统 ML + 深度学习基础训练、模型持久化、预测结果回写


- **文献检索**：Semantic Scholar + arXiv 基础检索与摘要结构化





---





## 2. 与需求对齐情况（当前 vs 目标）


已对齐：


- **元素限定**：在 `src/agents/core/theory_agent.py` 的 `TheoryDataConfig.elements`


- **证据来源**：理论/文献/ML 已可写入证据链





存在差距：


- **活性代理指标**：CatHub 吸附能 (H*/OH*) 尚未入库与材料关联


- **DOS/轨道 DOS**：可抓取但未写入 DB，也未纳入 ML 特征


- **模型规模**：传统 ML 目前 9 +（XGBoost/LightGBM 可选）；深度学习 3；GNN 3 但未接入主链路


- **可解释性**：SHAP 入口未完全生效；GNN 解释方法待补齐


- **实验闭环**：实验数据训练模型与反馈机制未形成闭环





---





## 3. 关键风险与现实约束


- **CatHub 表面数据 ↔ 体相材料映射**天然不完美，需要弱关联或人工确认


- **轨道 DOS 图**过高维度，不宜直接入模，需描述符或降维


- **GNN 可解释性**不能直接使用 SHAP，需要替代解释方法





---





## 4. 下一阶段计划（面向你的需求）


**P0 基线打通**  


- 中文意图识别 → HOR 任务图  


- 元素表贯通任务图 / 理论检索 / ML / 推荐  





**P1 活性代理指标**  


- CatHub H*/OH* 拉取、入库、弱关联  


- 证据链评分加入活性代理指标权重  





**P2 电子结构特征**  


- DOS/轨道 DOS 描述符提取 → 入库  


- ML 特征融合（DOS 描述符 + 结构特征）  





**P3 模型与解释性**  


- 传统 ML 扩展至 12  


- 深度学习扩展至 5  


- GNN 三模型接入主链路  


- 修复 SHAP 入口 + GNN 解释替代方案  





**P4 实验闭环**  


- 实验数据入库 → 实验 ML 训练 → 推荐迭代  


- 增加“是否需要补数据”的决策逻辑  





---





## 5. 迭代验收指标（建议）


- 任务图可完整执行（含依赖/失败/重试）


- 证据链至少含 3 类来源（理论/文献/ML）


- HOR 推荐能够输出 Top-N 候选及解释


- 实验数据可被引入并影响下一轮推荐





---


## 6. Update Log
- 2026-01-25: Added idea upgrade doc `docs/idea_upgrade.md`
- 2026-01-25: Added Chinese intent triggers for HOR/alloy discovery
- 2026-01-25: Added adsorption energy proxy (CatHub H*/OH*) and DOS descriptor storage
- 2026-01-25: Added task assignment/mermaid output, examples templates, and evaluation scripts
- 2026-01-26: Added Knowledge Core tables (knowledge_entities/sources/relations)
- 2026-01-26: Added Knowledge Service API (/knowledge)
- 2026-01-26: Added knowledge backfill script (src/tools/migrate_knowledge.py)
- 2026-01-26: Added KnowledgeRAG (graph-filtered retrieval over knowledge_sources)
- 2026-01-26: Synced evidence writes into Knowledge Core
- 2026-01-26: Added Knowledge Trace visualization (mermaid) in Streamlit
- 2026-01-26: Added task-level Knowledge RAG summary into executor results
- 2026-01-26: Added dataset snapshot storage and reasoning report output
- 2026-01-26: Added task report export endpoint and snapshot export
- 2026-01-26: Added knowledge quality scoring in reasoning report

---

## 7. MARS Benchmark Summary
- Benchmarked local MARS reference: `docs/mars_benchmark.md`
- Near-term: task assignment table + Mermaid output, case logs, error replan, evaluation scripts
- Mid-term: lightweight knowledge graph (literature-material-evidence)
- Long-term: experiment platform interface / automation


---

## Update Log (2026-01-26)
- Added CSV templates in data/templates and adsorption import tool (import_adsorption_energies.py).

- Added activity_metrics table and DB APIs; evidence chain now includes activity metrics and adsorption energies.
- Added knowledge API endpoints for activity/adsorption ingestion.
- Added CLI tool src/tools/import_activity_metrics.py for batch import.
- Improved English intent detection with langdetect to ensure task planning triggers.
\n## Update Log (2026-01-26)
- Added persistent chat sessions/messages (DB + UI session manager).
- Added session export/delete and task history reload in chat UI.
- Added session search in chat UI.
- Added file-level upgrade checklist: docs/upgrade_checklist.md.
- 2026-01-26: Added meta-controller (evidence-driven planning + follow-ups) and /knowledge/stats; evaluation UI shows evidence stats.
- 2026-01-26: Added P2 tools (DOS backfill + synthetic activity generator) and P2 test checklist.
- 2026-01-27: Enforced element restriction across theory API, ML training, evidence stats, and import tools (TheoryDataConfig.elements only).
