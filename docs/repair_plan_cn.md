<!-- ASCII padding for tooling compatibility: 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -->
# IMCs 智能体系统修复计划书（P0→P3）

日期：2026-03-02  
范围：`C:\Users\Administrator\Desktop\课题2-有序合金-HOR\IMCs\IMCs`（本仓库）

本文件用于把“已识别问题 → 修复路线 → 验收标准 → 回滚策略”固化下来，避免修复过程中遗忘或跑偏。  
执行原则：**P0 优先（稳定性/闭环/可恢复）→ P1（智能化评估/策略库）→ P2（实验闭环/自进化）**。

---

## 1. 问题摘要（按优先级）

### P0（稳定性与闭环）
- **任务状态机/恢复不够强**：计划/步骤状态更新为“追加日志”后，调用方需要稳定取“最新态”；执行重入要幂等。
- **执行链路可能割裂**：同一个 `task_id` 若存在多个执行入口（API/Chat/UI/脚本）走不同引擎，会导致状态/结果口径不一致、排障困难。
- **混合中英输入识别不稳**：中英混合关键词容易误路由，导致任务类型与计划不符合预期。
- **LLM 超时与降级不统一**：云端不可用/超时下的回退路径需要明确且可配置。
- **工程可测性受干扰**：pytest cache 目录权限/遗留目录导致 `rg` 扫描报错与 pytest cache 警告，影响回归效率。

### P1（智能化升级）
- **缺少系统化评估套件**：Top‑K、证据覆盖、RAG 质量、任务成功率/耗时等指标未形成一键评估与对比。
- **失败策略库缺失**：retry/replan/降级策略分散在代码中，缺少统一配置与可解释日志。
- **GraphRAG/图谱增强偏弱**：现有 RAG 可用，但图谱化证据整合与可解释性仍不足。

### P2（先进能力）
- **实验平台协议未标准化**：robot/middleware 仍偏接口层，缺少可复用 schema 与幂等回调约束。
- **真实闭环学习未形成流水线**：实验数据→入库→训练→再推荐 的端到端链路需要可重复验证。

---

## 2. 已完成修复（截至 2026-03-02）

> 说明：下列为已落地改动与新增测试（便于对照回归）。

- Evidence Gap：支持 `$agent` 依赖占位符解析并用于 gap 步骤落盘/追加。  
  关键文件：`src/services/task/executor.py`
- 任务执行：`/tasks/execute/{task_id}` 优先从 DB 还原 plan，并用懒加载 agent registry 创建执行器。  
  关键文件：`src/api/routers/tasks.py`
- DB：新增 `update_plan_step_status` 以“追加日志”的方式记录步骤状态，保留历史；allowed-elements 过滤为空时回退。  
  关键文件：`src/services/db/database.py`
- LLM：新增硬超时与云端开关。  
  环境变量：`IMCS_LLM_TIMEOUT`（秒，默认 30）、`IMCS_ENABLE_CLOUD_LLM`（默认 1）  
  关键文件：`src/services/llm/expert_reasoning.py`
- QueryParser：修复 `HOR` → `H` 误抽元素，使用 `\bor\b` 避免把 `or` 当成字母序列的一部分。  
  关键文件：`src/agents/query_parser.py`
- 静态模板：为 discovery / analysis 模板补齐 `knowledge_pack` 作为后置步骤。  
  关键文件：`src/services/task/planner.py`
- API：版本与健康检查对齐；CORS origins/credentials 改为读取环境变量，并加入启动告警。  
  环境变量：`IMCS_API_VERSION`、`IMCS_CORS_ORIGINS`、`IMCS_CORS_ALLOW_CREDENTIALS`  
  关键文件：`src/api/main.py`
- 新增测试：Evidence gap 确认闭环用例；混合中英任务识别用例。  
  关键文件：`tests/test_gap_confirmation_flow.py`、`tests/test_planner_multilang.py`
- 评估套件：新增一键跑分脚本，输出 JSON/Markdown 报告用于对比迭代效果。  
  关键文件：`src/tools/run_eval_suite.py`、`src/services/task/eval_suite.py`、`tests/test_eval_suite.py`
- Robot 回调协议：引入幂等回调事件表（`callback_id`/`payload_hash` 去重），重复回调不再重复入库指标；并对指标值做 finite 校验；提供事件查询接口 `/robot/task_events/{task_id}`。  
  关键文件：`src/api/routers/robot.py`、`src/services/db/schema.sql`、`src/services/db/database.py`、`tests/test_robot_callback_idempotency.py`
- 闭环产物复用：Streamlit 新增“闭环迭代”页面展示 iteration Top‑N，并支持一键把 Top‑N 转为新 TaskPlan 自动执行推荐（`/robot/iteration_to_taskplan`）。  
  关键文件：`src/interface/app.py`、`src/api/routers/robot.py`、`src/services/task/executor.py`、`tests/test_iteration_topn_workflow.py`

### 2.1 现状评估（EvalSuite / DB 快照，2026-03-02）

- 评估报告：`data/analysis/eval_report_20260302_222142.md`（recent=100）
  - Plan 状态：completed=57、failed=1、awaiting_confirmation=23、pending=16、executing=3
  - 口径提示：当前 `success_rate=completed/plan_count` 会把 “awaiting_confirmation/pending” 计入分母，偏悲观；终态成功率 `57/(57+1)=98.3%`
  - 耗时分布（completed）：P50≈54s，P90≈452s（均值易被长任务拉高）
- DB 总体（`data/imcs.db`）：plans=147，completed=73，awaiting_confirmation=31，pending=29，failed=10，executing=4
- 证据链数据质量：evidence 总行数=4884，其中孤儿 evidence（material 不存在）=155（≈3.2%）
  - `ml_prediction` evidence 多为孤儿（fake id），导致“有效覆盖”的统计偏低（allowed set 下仅 1 个 material）

---

## 3. 后续修复路线（建议按顺序执行）

### 3.1 P0-1：测试与开发基线（先把回归跑顺）

- [x] 处理/规避根目录 `pytest-cache-files-*` 访问拒绝问题（避免 `rg` 搜索与 pytest 缓存警告干扰）
- [x] 固化一套“最小回归命令集”（单测 + 关键路径用例）
- [ ] 统一日志与异常输出：关键路径必须可定位（plan_id、step_id、agent、action、status）

验收：
- `rg` 在仓库根目录搜索不再报 `os error 5`
- `python -m pytest` 关键用例可稳定跑完（允许第三方依赖弃用告警，但不允许缓存/权限导致的随机失败）

### 3.2 P0-2：统一执行链路与口径（减少“同任务多引擎”风险）

- [ ] 梳理所有执行入口（API/Chat/UI/脚本）与各自落库逻辑
- [ ] 明确“单一执行引擎/单一真相来源”（建议：DB 计划 + `PlanExecutor`），其它入口仅做薄封装
- [ ] 对重复执行请求做幂等（同 `task_id` 不重复跑已完成步骤）

验收：
- 任意入口查询同一 `task_id` 的 `status/steps/results` 口径一致
- 重复触发执行不会导致步骤重复写入/重复执行

### 3.3 P0-3：状态机与恢复机制加强（断点恢复可靠）

- [ ] 把“取步骤最新态/聚合结果”的 reduce 逻辑收敛到 DB 查询层（避免每个调用方各写一套）
- [ ] 明确依赖满足规则（例如 `skipped` 是否算依赖满足、`awaiting_confirmation` 的出入边界）
- [ ] 为“重启恢复执行”补齐回归用例（模拟中断、重启、继续）

验收：
- 重启后可继续执行，不丢步骤、不重跑已完成步骤
- 状态转换覆盖齐全：pending→running→completed/failed/skipped/awaiting_confirmation

### 3.4 P0-4：Evidence Gap 确认闭环增强（四条路径都闭环）

- [ ] 覆盖四种确认路径：全部 skip、部分 run、参数覆盖、`mark_complete=true`
- [ ] API 返回中明确下一步（继续执行/已完成/仍需确认）

验收：
- 新增用例全部通过；前端/调用方无需“猜测”下一步

### 3.5 P1-1：失败策略库 + 统一降级（从“散落逻辑”到“可配置策略”）

- [ ] 定义失败分类（网络失败/数据缺失/模型缺失/超时/格式错误…）
- [ ] 提供策略模板（retry、replan、跳过、缩小范围、改用本地 mock、人工确认）
- [ ] 输出“采取了哪条策略”的结构化日志

验收：
- 故障注入下系统表现可预测且可解释

### 3.6 P1-2：评估套件（用指标驱动迭代）

- [x] 最小指标：Top‑K、证据覆盖增量、RAG 命中/引用率、成功率、平均耗时
- [x] 一键跑分脚本 + 报告落盘（JSON/Markdown）

验收：
- 每次改动可生成可对比报告（不靠“看起来更聪明”）

### 3.7 P2：实验平台协议与闭环学习（把接口变成流水线）

- [x] 定义实验任务/结果 schema（幂等回调、数据质量校验）
- [x] 形成“实验回传 → 入库 → 重训/重排 → 新候选”的可重复流程

验收：
- 使用模拟回调即可完成一轮闭环；真实接入只需替换 transport

### 3.8 P3-1：评估口径补齐（把“blocked”与“失败”分开）

- [x] EvalSuite 指标拆分：`terminal_success_rate`、`blocked_ratio`（awaiting_confirmation/pending/executing）、`p50/p90 duration`
- [x] 报告中标注“是否有 knowledge_pack”（避免 candidates/rag 为空时误判）
- [ ] （可选）按 plan_id 前缀/时间窗过滤（避免混入历史演示/测试任务）

验收：
- `run_eval_suite.py` 输出的 summary 能解释“为什么看起来成功率低”（blocked vs failed）

### 3.9 P3-2：证据链数据质量（防止孤儿/提升可解释性）

- [x] DB 连接启用 `PRAGMA foreign_keys=ON`（至少保证新增 evidence 不再产生孤儿）
- [x] 新增健康检查：统计孤儿 evidence/孤儿 activity/孤儿 adsorption，并在 EvalSuite 报告中展示
- [x] 提供一次性清理脚本（可选）：删除孤儿 evidence 或补齐 stub material（按策略选择）

验收：
- 新写入 evidence 都能在 `/theory/materials/{material_id}` evidence chain 中可追溯
- `ml_prediction` 不再出现大量 fake id 覆盖“虚高/虚低”

### 3.10 P3-3：减少 `awaiting_confirmation` / `pending` 堵塞（提升吞吐）

- [ ] Streamlit/Chat 增加“待确认任务列表”与一键处理（run/skip/mark_complete）
- [ ] 为 `pending`/`executing` 增加“恢复/重试”入口（基于 DB plan restore + PlanExecutor）
- [ ] （可选）给 Evidence Gap 增加可配置的自动策略（小步快跑：自动跑 1 轮、超出转人工）

验收：
- recent=100 中 blocked_ratio 明显下降；同等输入下 completed 数提升

---

## 4. 回归测试建议（最小集合）

- `python -m pytest tests/test_gap_confirmation_flow.py -q`
- `python -m pytest tests/test_planner_multilang.py -q`
- `python -m pytest tests/test_intelligent_planning.py -q`
- （可选）覆盖 API/协议相关的单测：`tests/test_chat_api.py`、`tests/test_protocol.py`

---

## 5. 回滚/降级开关（建议保留）

- LLM：`IMCS_ENABLE_CLOUD_LLM=0` 强制离线 mock；`IMCS_LLM_TIMEOUT=<seconds>` 控制超时
- CORS/版本：`IMCS_CORS_ORIGINS`、`IMCS_CORS_ALLOW_CREDENTIALS`、`IMCS_API_VERSION`

---

## 6. 更新记录（手动维护）

- 2026-03-02：建立修复计划书；记录已完成项与后续路线。
- 2026-03-02：完成 P0-1~P0-4：pytest 基线（禁用 cacheprovider + 忽略不可访问目录）、统一 PlanExecutor 工厂、DB latest steps 查询、gap 确认接口语义修复与新增回归测试。
- 2026-03-02：完成 P1-1：引入失败分类与策略库（`configs/failure_policies.json`），PlanExecutor 统一按策略执行 replan/skip/降级，并新增回归测试。
- 2026-03-02：完成 P1-2：新增一键评估脚本 `src/tools/run_eval_suite.py`（输出 JSON/Markdown 报告到 `data/analysis/`），并补齐 DB `list_plans/get_plan_last_step_created_at` 与回归测试 `tests/test_eval_suite.py`。
- 2026-03-02：完成 P2-1：robot 回调幂等与数据校验（新增 `robot_task_events` 表、`callback_id` 支持、重复回调跳过副作用）及回归测试 `tests/test_robot_callback_idempotency.py`。
- 2026-03-02：完成 P2-2：补齐 robot 事件查询接口（`GET /robot/task_events/{task_id}`），并将迭代重训/重排结果写入事件流（`iteration_completed`）。
- 2026-03-02：完成 P2-3：iteration Top‑N 一键生成新 TaskPlan 并自动执行推荐（`POST /robot/iteration_to_taskplan`）；Streamlit 新增闭环页展示 Top‑N，并新增 seed 预测执行器动作 `ml.seed_predictions`。
- 2026-03-02：补齐 pytest 回归稳定性：`pytest.ini` 默认只收集 `tests/`（避免扫描不可访问目录），并修复 `tests/integration_test.py` 的 DB fixture 以规避 WinError 5 的临时目录问题；`python -m pytest` 全量用例通过。
- 2026-03-02：完成 P3-1：EvalSuite 输出新增 `terminal_success_rate/blocked_ratio/p50&p90 duration/knowledge_pack_present`，并补齐回归测试 `tests/test_eval_suite.py`。
- 2026-03-02：完成 P3-2（部分）：SQLite 连接启用 `foreign_keys=ON`，DB 新增 `get_data_integrity_stats()` 并接入 EvalSuite；补齐外键约束下的回归清理与 `save_activity_metric/save_adsorption_energy` 的 stub material 兜底。
- 2026-03-02：补齐一次性孤儿清理脚本 `src/tools/cleanup_orphans.py`（默认 dry-run，`--apply` 才修改 DB，并自动备份）。
