<!-- ASCII padding for tooling compatibility: 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -->
# IMCs 开发手册

本手册面向开发与维护人员，聚焦架构、数据流、任务图执行与演进规划。

## 1. 架构概览
- **主入口**：Streamlit UI（`src/interface/app.py`）
- **后端**：FastAPI（`src/api/main.py`），提供任务图与数据访问 API
- **智能体**：位于 `src/agents/core/`，由 `TaskManagerAgent` 统一调度
- **任务图**：`TaskPlanner` 生成、`PlanExecutor` 执行（见 `src/services/task/`）
- **数据库**：SQLite（默认 `data/imcs.db`），Schema 在 `src/services/db/schema.sql`
- **Next.js**：`src/ui/web` 为可选 UI，不作为当前主入口

---

## 2. 目录结构（关键路径）
- `src/api/`：FastAPI 路由与入口
- `src/interface/`：Streamlit 前端
- `src/agents/core/`：核心智能体（ML / Theory / Literature / Experiment）
- `src/services/`：任务编排、数据库、ML 数据管理等
- `src/tools/`：数据库迁移脚本
- `data/`：SQLite 与数据缓存
- `docs/`：文档与架构图

---

## 3. 环境与运行
```powershell
pip install -r requirements.txt
python src/api/main.py
streamlit run src/interface/app.py
```

**配置文件**：
- `.env`：运行所需密钥与 API 地址
- `config.ini`：LLM 与 CGCNN 相关参数

---

## 4. 数据库与迁移
- 默认数据库：`data/imcs.db`
- Schema 定义：`src/services/db/schema.sql`
- 迁移脚本：
- `python src/tools/migrate_db.py`（v4.0 Schema）
- `python src/tools/migrate_m3.py`
- `python src/tools/migrate_m4.py`
- `python src/tools/migrate_knowledge.py` (Knowledge Core backfill)

建议已有旧库时先运行迁移，再启动服务。

---

## 5. 任务图与执行器开发
**任务图生成**：
- `src/services/task/planner.py` 负责将自然语言请求转换为 DAG 计划
- `TaskStep.dependencies` 用于声明依赖关系

**任务执行**：
- `src/services/task/executor.py` 负责依赖调度、状态记录、重试与证据汇总
- `DatabaseService.log_plan_step()` 记录任务图执行日志

**API 暴露**：
- `src/api/routers/tasks.py` 提供 `/tasks/create`、`/tasks/execute/{id}`、`/tasks/{id}` 等接口

---

## 6. 证据链与推荐逻辑
- Evidence 写入位置：`src/services/task/executor.py`
- Evidence 表：`evidence`（`source_type` = theory / literature / ml_prediction / experiment）
- UI 读取路径：`/theory/materials/batch` → Streamlit Evidence Chain 组件

当前推荐逻辑以 **formation_energy + evidence score** 进行加权排序（可进一步扩展）。

---

## 7. 下一步演进方向（7 项）
1. **任务图治理**：加入超时、取消、并发队列、失败恢复与可重放
2. **证据链质量**：文献去重、证据打分标准化、来源可信度分层
3. **HOR 候选筛选链路**：引入可配置筛选规则、Top-N 过滤与多目标排序
4. **实验闭环**：实验数据上传 → 结构化 → 入库 → 反哺训练
5. **数据治理与版本**：dataset hash、模型注册、可复现实验脚本
6. **可解释性与可视化**：SHAP/d-band/DOS 解释面板与任务图交互
7. **部署与安全**：鉴权、密钥轮换、日志审计、CI/CD 与容器化

---

## 8. 开发建议
- 新增任务类型时，优先扩展 `TaskType` 与 `TaskPlanner` 规则模板
- 新增证据来源时，确保更新 `evidence` 写入与 UI 展示逻辑
- 优先保持 API 与 Streamlit UI 解耦，便于替换前端
---


## 9. Update Log
- 2026-01-25: Added progress/plan doc `docs/progress_plan.md`
- 2026-01-25: Added Chinese intent triggers and inline code comments
- 2026-01-25: Added adsorption energy table and DOS descriptor storage
- 2026-01-25: Added task assignment/mermaid output, examples, and evaluation scripts
- 2026-01-25: Added MARS benchmark doc `docs/mars_benchmark.md`
- 2026-01-26: Added Knowledge Core schema + Knowledge Service API
- 2026-01-26: Added knowledge backfill script `src/tools/migrate_knowledge.py`
- 2026-01-26: Added KnowledgeRAG (graph-filtered retrieval) and evidence sync
- 2026-01-26: Added Knowledge Trace visualization in Streamlit
- 2026-01-26: Added task-level Knowledge RAG summary in executor
- 2026-01-26: Added dataset snapshot tables + reasoning report output
- 2026-01-26: Added task report export endpoint and knowledge scoring

---

## 10. MARS Benchmark
- See `docs/mars_benchmark.md` for details
- Suggested focus: plan output + replan policy + case logs + evaluation
- 2026-01-25: Added idea upgrade doc `docs/idea_upgrade.md`


## 11. Literature-Driven HOR Pipeline (One Command)

One command for: **online literature harvest -> seed CSV -> metrics + LSV generation -> import -> LSV analysis**

```powershell
python src/tools/harvest_literature_hor_seed.py --query "HOR ordered alloy catalyst" --limit 15 --max-pdfs 5 --persist --run-all --seeded
```

Outputs:
- `data/experimental/literature_hor_seed.csv`
- `data/experimental/literature_activity_metrics.csv`
- `data/experimental/literature_rde_lsv/`


## 12. DOS Automation Pipeline

One command to fill missing DOS, render orbital curves/plots, and report coverage:

```powershell
python src/tools/auto_dos_pipeline.py --limit 200 --batch-size 20 --merge-pdos --update-db
```

Outputs:
- `data/theory/orbital_pdos.json`
- `data/theory/orbital_dos_curves/`
- `data/theory/orbital_dos_plots/`
- `docs/dos_coverage_report.json`


## 13. DOS Curve Prediction (Structure -> DOS Curve)

Train curve model (PCA + Ridge):
```powershell
python src/tools/train_dos_curve_model.py --channel total --n-components 20
```

Predict curves (optional plots):
```powershell
python src/tools/predict_dos_curves.py --only-missing --update-db --plot
```

Outputs:
- `data/ml_agent/dos_curve_model.pkl`
- `data/ml_agent/dos_curve_report.json`
- `data/theory/dos_curve_predictions/`
- `data/theory/dos_curve_pred_plots/`


## 14. DOS Curve Multi-Channel Pipeline

Train and predict for multiple channels (total/s/p/d):

```powershell
python src/tools/auto_dos_curve_pipeline.py --channels total,s,p,d --n-components 20 --only-missing --plot --update-db
```

Outputs:
- `data/ml_agent/dos_curve_model_total.pkl` (and s/p/d)
- `data/ml_agent/dos_curve_report_all.json`
- `data/theory/dos_curve_predictions/<channel>/`
- `data/theory/dos_curve_pred_plots/<channel>/`
