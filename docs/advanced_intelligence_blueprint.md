# 🎓 IMCs Phase 2 Master Plan: The Autonomous Catalyst Discovery Group

**Date**: 2026-02-27
**Status**: **Ready for Execution**
**Objective**: 彻底摆脱僵化的“算分排序”流水线，将 IMCs 升级为一个"懂化学、知敬畏"的自治化 HOR 催化剂发现课题组。

---

## 1. 核心痛点与演进愿景 (Vision)

### 🚨 As-Is (现状的“伪智能”)
目前系统运转为一个**盲目的装配流水线**：
1.  **黑盒算分**：系统背后仅是调用预定义好的 Scikit-Learn (XGB/RF) 模型输出一个单纯的分数（如 $j_0$ 的数值），然后按步就班地从高到低排序推荐。
2.  **不懂化学**：模型内部既不懂 Sabatier 原理，也不懂 d-band 理论。哪怕它给出了一个吸附能极差但活性极高（过拟合）的荒谬结果，系统也是“深信不疑”。
3.  **单向执行**：Agent 之间只是数据的搬运工（Theory 出数据 $\rightarrow$ ML 算分 $\rightarrow$ 结束），面对没见过的新材料，系统没有“感知自身不确定性”并主动求证的能力。

### 🌟 To-Be (真正的“科研课题组”)
我们要将系统从 **"Toolbox" (自动工具箱)** 升级为 **"Research Group" (自治课题组)**：
1.  **物理约束直觉**：预测中枢（ML Agent）必须在底层就被铸入化学规律。它得知道违反 Sabatier 火山图的预测是“错误”的。
2.  **主动学习 (Active Learning)**：系统面对没见过的高分材料时，不会直接死板推荐，而是能“感知不确定性”（高方差），主动中断流程，向实验 Agent 发出 **"亟需实验验证"** 的请求。
3.  **大模型溯源 (LLM Reasoning)**：让推荐从冷冰冰的数值变为人类专家式的科学论述链（Chain-of-Thought）。

---

## 2. 团队重构：各 Agent 的“化” (Agent Specialization)

每个智能实体必须具备“主观能动性”与深度的专业性协同：

### 🧠 A. The Director (Orchestrator & MetaController)
*   **Role**: PI（课题组长）。不再只是发派任务，而是做“决策”。
*   **Upgrade**:
    *   **Uncertainty Handling**: 当 ML Agent 提交了一份高活性但“方差极大”的报告时，Director 不会盲目写入最终推荐榜，而是触发 **"Active Exploration Protocol"**。
    *   **LLM 整合**: 会同所有 Agent 的报告，手写出一份有依据的“结题报告”（解释这到底为什么是个好催化剂）。

### 🧪 B. The Analyst (ML Agent)
*   **Role**: 计算催化专家。不再用 `sklearn` 调用完事。
*   **Upgrade**:
    *   **Physics-Informed Loss**: 在 PyTorch 层重写损失函数。在拟合实验数据的同时，对偏离 $d\text{-band center} \sim -2.0\text{ eV}$ 或 $\Delta G_H \sim 0$ 的预测进行严重惩罚 (Volcano Penalty)。
    *   **Ensemble & Variance**: 使用模型集成（MC Dropout 或 Deep Ensembles）。同时输出“预测中位数”和“预测不确定度 (std)”。

### 📖 C. The Librarian (Literature Agent)
*   **Role**: 情报局。
*   **Upgrade（已完成）**: 已在 Phase 1 实现了从文献中收割真实活性作为锚点。现在需要能够在遇到异常值（Conflict）时，去数据库里找寻“这是否有先例”。

### 🧮 D. The Simulator (Theory Agent)
*   **Role**: 第一性原理打工仔。
*   **Upgrade（已完成）**: 已在 Phase 1 从 Materials Project 补齐了 1000+ 最优材料的真实电子态（`d_band_center` 等组分特征）。

---

## 3. 核心智能协方案：The "Active Learning" Protocol

这是一个典型的课题组运转实况（当用户输入 "推荐 HOR 合金" 时发生）：

1.  **Simulator (Theory)** 把 1800 多个有电子特征的候选材料推给 **Analyst (ML)**。
2.  **Analyst** 基于带物理惩罚的 PyTorch 模型进行预测。它发现 `Pt3Ni` 的活性极高，同时方差很低（它很自信）；但又发现了一个新型组合 `RuRe`，预测活性很高，但方差极大（不确定）。
3.  **Analyst** 将报告交给 **Director**：
    *   *Pt3Ni ➡️ 高活性, 高置信度*
    *   *RuRe ➡️ 高活性, 低置信度*
4.  **Director** 作为 PI 进行研判：
    *   把 `Pt3Ni` 打包，交由大语言模型（LLM）写出 "可解释推荐报告" 准备推送给用户。
    *   把 `RuRe` 扣下，触发 **主动学习警报**。通知 **Exp Agent (或提示真实用户)**：“这个体系极具潜力，但我们缺乏数据，请优先合成该材料并测定其 LSV/CV。”

---

## 4. 实施路线图 (Execution Roadmap)

基于目前第一阶段（数据底座建设已完成：DOS与文献活性已齐备），我们可以直接切入核心“大脑”的构建。

*   [ ] **Phase 2.1: 物理感知模型构建 (Physics-Informed ML)**
    *   创建 `src/models/hor_physics_model.py`。
    *   实现支持不确定性估计（Ensemble/MC Dropout）的 PyTorch 网络。
    *   编写融入 $d$-band 吸附理论与火山图规律的联合损失函数。
*   [ ] **Phase 2.2: ML Agent 升级 (Uncertainty & Interpretation)**
    *   改造 `MLAgent`，放弃单独的 scikit-learn 流水线。
    *   训练并保存上述模型（使用第一阶段下载的真实 `d_band_center` 和 `activity_metrics`）。
    *   增加 SHAP 等可解释性导出。
*   [ ] **Phase 2.3: Director 决策升级 (Orchestrator Refactoring)**
    *   改造 `src/agents/fusion.py`。不再单纯依赖硬编码权重和求和排序。
    *   引入 LLM 推理接口，处理 ML 传来的（Score, Uncertainty）二元数据。
*   [ ] **Phase 2.4: 主动学习反馈引入 (Active Suggestion)**
    *   在 UI 和报告输出层引入 "实验建议清单 (Experiment Proposals)" 面板，反馈那些“高活性但存在盲区”的候选材料。

---

**下一步**：
如果这份蓝图符合你对“系统智能化”的期望，接下来我们将立即执行 **Phase 2.1** —— 开始手写融合化学常识的 PyTorch `hor_physics_model.py`。
