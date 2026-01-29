<!-- ASCII padding for tooling compatibility: 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -->
# IMCs 智能化评估与对比报告（对标 MARS 等多智能体系统）

日期：2026-01-28

> 说明：本对比基于本地 `bin/MARS-main.zip` 与项目现有实现；未联网检索其它项目源码，仅做功能层面与工程方法的对照。

---

## 1. 总体结论（简要）

- **当前智能化程度**：处于 **L2~L3**（证据链驱动 + 部分自适应规划）。
- **优势**：面向 HOR 场景的“理论/文献/实验/ML”打通较完整；证据链与知识包已经可展示；能生成任务图与补齐证据的第二阶段。
- **短板**：
  - 自动化仍依赖“明确触发 + 手动确认”，对复杂查询的“自我拆解 + 自适应重规划”还不够强。
  - 与 MARS 相比，缺少系统化评估套件与“案例日志库/失败修复策略库”。
  - 与机器人/实验平台联动仍是接口层面的规划，未形成标准协议与闭环。

---

## 2. 与 MARS 的功能对比（简表）

| 维度 | MARS（本地 ZIP 观察） | IMCs 当前状态 | 差距/可借鉴 |
| --- | --- | --- | --- |
| 任务编排 | 多团队协作编排，显式计划输出 | 有任务图与 DAG 执行，但策略较固定 | 引入“任务分配表 + Mermaid 输出 + 失败重规划策略库” |
| 知识驱动 | GraphRAG/知识图谱为核心 | 有知识源/关系表 + RAG，但图谱应用较轻 | 引入轻量 GraphRAG 或更强证据驱动检索 |
| 评估体系 | RAG/多智能体评估脚本较全 | 评估脚本较少，主要是 Coverage 展示 | 建立 Top-K 召回、证据覆盖、任务成功率评估 |
| 失败处理 | 有“err-deal”示例与恢复流程 | 有基本 retry/replan，但不体系化 | 增加失败策略模板与回退路径 |
| 实验联动 | 有 middleware/robot 平台接口 | 目前仅规划，没有标准协议 | 先定义实验平台 API 协议，再逐步联动 |

> 结论：IMCs 在“面向 HOR 的数据链路”更贴近材料方向，但在“多智能体调度策略 + 评估体系 + 失败恢复”上仍弱于 MARS。

---

## 3. 智能化程度分级评估（L1~L5）

- **L1（流程自动化）**：已具备。任务图、自动执行、数据库落盘。
- **L2（证据链驱动）**：已具备。证据链写入、Knowledge Pack 展示。
- **L3（自适应规划）**：部分具备。已有 meta-controller + evidence gap 分析，但仍需人工确认。
- **L4（闭环实验驱动）**：尚未实现。实验平台未联动，更多是数据导入。
- **L5（自我演化）**：尚未实现。未形成持续学习/策略优化闭环。

**综合判断**：L2~L3。

---

## 4. 功能清单与使用方式（中文）

### 4.1 系统启动

- 一键启动：双击 `start_imcs.bat`
- 手动启动：
  ```powershell
  python src/api/main.py
  streamlit run src/interface/app.py
  ```

### 4.2 任务图与多智能体协作

- UI：进入“智能体对话” → 输入问题 → 生成任务图 → 点击执行
- API：
  - 创建任务：`POST /tasks/create`
  - 执行任务：`POST /tasks/execute/{task_id}`

### 4.3 证据链（Evidence Chain）

- UI：在任务执行后，选择材料查看证据链
- 数据来源：`theory / literature / experiment / ml_prediction / adsorption_energy`
- API：`GET /theory/materials/{material_id}`

### 4.4 Knowledge Pack（证据驱动总结）

- 自动生成：任务执行结束后写入 `data/tasks/knowledge_{task_id}.json`
- UI 展示：Chat 页面底部“Knowledge Pack”区域
- 包含：evidence stats / reasoning report / RAG 摘要

### 4.5 文献管理

- 文献入库（本地）：UI → Literature 页 → “Index local PDFs”
- 在线检索（Semantic Scholar / arXiv）：由 LiteratureAgent 自动调用
- HOR 文献种子生成（命令行）：
  ```powershell
  python src/tools/harvest_literature_hor_seed.py --query "HOR ordered alloy" --limit 15 --max-pdfs 5 --persist --run-all --seeded
  ```

### 4.6 理论数据

- 自动调用 Materials Project API：TaskManager / TheoryAgent
- 元素限制：`TheoryDataConfig.elements`
- 命令行：
  ```powershell
  python src/tools/fill_missing_dos_from_mp.py
  ```

### 4.7 实验数据处理（LSV / RDE）

- 数据放入：`data/experimental/rde_lsv/`
- 自动解析与指标提取：ExperimentAgent → `analyze_rde_series()`
- 指标：Jk、J0、Tafel、MA 等

### 4.8 机器学习训练

- UI：ML Training 页面
- 训练数据：从 DB 自动加载
- DOS 作为输出性质：支持预测 d 带中心等

### 4.9 DOS 曲线预测

- 自动化流水线：
  ```powershell
  python src/tools/auto_dos_pipeline.py --limit 200 --batch-size 20 --merge-pdos --update-db
  python src/tools/auto_dos_curve_pipeline.py --channels total,s,p,d --n-components 20 --only-missing --plot --update-db
  ```
- UI：Evidence Chain → DOS Curve Predictions

### 4.10 证据链补齐（第二阶段）

- 任务执行结束后 → 状态 `awaiting_confirmation`
- UI 显示补齐步骤 → 勾选确认 → 点击继续执行
- 可直接“跳过补齐并完成”
- 重启后仍可恢复继续（已支持从 DB 恢复 plan）

---

## 5. 下一步细化计划（建议按优先级）

### P0（稳定性与闭环必备）
1. **完善证据补齐确认逻辑**
   - 允许编辑补齐步骤参数（例如文献检索 query）
2. **多语言输入稳定化**
   - 修复中文/英文混合输入导致的任务识别不稳定
3. **任务恢复机制加强**
   - 计划步骤持久化完整（params/results/version）

### P1（智能化升级）
1. **评估套件**
   - 增加 Top-K 召回、证据覆盖、RAG 质量评估
2. **GraphRAG 轻量化**
   - 先做“文献-材料-指标”简易图谱 + RAG 增强
3. **失败策略库**
   - 多智能体执行失败的回退/替代策略

### P2（先进能力）
1. **实验平台联动协议**
   - 定义实验任务 API（合成/测试/回传）
2. **真实实验数据闭环学习**
   - 实验数据驱动 ML 迭代
3. **自我演化策略**
   - 系统自动调整任务规划策略

---

## 6. 你可以立即做的验证动作

1. 运行一个“HOR 有序合金候选”任务 → 看任务图
2. 执行完成后 → 进入 Evidence Chain / Knowledge Pack
3. 观察是否出现 `awaiting_confirmation` → 勾选补齐步骤 → 继续
4. 检查 evidence 覆盖数量变化（文献/理论/实验/ML）

---

如果你希望，我可以把“计划恢复 + 证据补齐确认 + 评估套件”整理成单独的 README 与操作脚本。

---

## 补充：证据补齐参数编辑与自动闭环

- **补齐参数可编辑**：当任务进入 `awaiting_confirmation`，UI 会显示每个补齐步骤的 JSON 参数编辑框。
- **自动实验闭环**：`experiment.process` 若成功处理 RDE 数据，会自动追加 `ML(train)` 以 activity_metric 为目标。
- **证据覆盖对比**：补齐前/后会生成 coverage 对比表，便于评估收益。
- **候选排序对比**：补齐前/后基于 ML 预测输出 Top-N 排序，并展示排名变化。

## 补充：robot 回调触发迭代（准备级）

- `POST /robot/result_callback` 支持 `auto_iterate=true`，自动触发基于实验指标的 ML 重训。
- `result.metrics` 中包含的指标会自动入库为 activity_metrics（来源标记为 robot）。

