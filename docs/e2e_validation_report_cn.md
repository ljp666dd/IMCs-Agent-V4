<!-- ASCII padding for tooling compatibility: 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -->
# IMCs E2E 验证报告（本地快速验证）

日期：2026-01-29

## 1. 验证目标
- 验证“理论数据 → ML 训练 → 候选排序 → Knowledge Pack 写入”的闭环是否可用
- 验证 CIF 与 formation_energy 的一致性，确保 ML 可训练
- 验证证据链排序写入与 UI 展示数据准备

## 2. 关键修复与改动
- 修复 CIF 被 formation_energy 覆盖清空问题（UPSERT + COALESCE）
  - 文件：`src/services/db/database.py`
- 修复 formation_energy=0.0 被误判为 None 导致不写入
  - 文件：`src/agents/core/theory_agent.py`
- MP 搜索控制参数支持：`MP_MAX_ELEMENTS` / `MP_SEARCH_ALL_ELEMENTS`
  - 文件：`src/services/theory/mp_client.py`

## 3. 数据准备结果
- CIF 已下载并写入 DB
  - `cif_count = 120`
  - `cif_with_fe = 120`
- CIF 目录示例：`data/theory/cifs/mp-126.cif`

## 4. E2E 验证执行
### 4.1 任务配置（快速验证计划）
- 任务类型：`catalyst_discovery`
- 描述：`HOR ordered alloy candidates (quick e2e verify 2)`
- 步骤：
  1) literature.search
  2) literature.extract_knowledge
  3) theory.download (cif + formation_energy)
  4) ml.train
  5) task_manager.recommend
  6) task_manager.knowledge_pack

### 4.2 执行结果摘要
- 任务状态：`completed`
- ML 训练：成功（有 R2 输出）
- Ranking 输出：`ranking_current = True`
- Ranking 指标：`formation_energy`

### 4.3 Knowledge Pack 输出
- 文件路径：`data/tasks/knowledge_e2e_verify_20260129_084616.json`
- 关键字段：
  - `ranking_current`（Top-N 排名）
  - `ranking_metric = formation_energy`
  - `candidate_material_ids`

## 5. 发现的问题与处置
- 早期问题：ML 无法训练（X/y 为空）
  - 原因：CIF 路径被 formation_energy 更新覆盖为 NULL
  - 已修复：`save_material` 改为 UPSERT + COALESCE

## 6. 下一步验证建议（面向 UI）
- 启动后端 + Streamlit，运行一个 HOR 任务
- 在“Knowledge Pack”区域查看 `Candidate Ranking`
- 若出现 `awaiting_confirmation`，执行补齐并观察排名变化

---

如需将此报告合并到总体评估文档或生成英文版，请告诉我。
