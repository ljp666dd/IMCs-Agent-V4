# 理论智能体 (Theorist Agent) 运行逻辑全解

## 1. 核心任务
理论智能体的目标是**替代昂贵的 DFT 计算**。它通过阅读晶体结构文件 (CIF)，毫秒级预测出该材料的稳定性、电子结构和催化活性。

## 2. 数据流向 (Data Pipeline)

### 输入端 (Input)
*   **晶体结构**: `data/theory/cifs/*.cif` (1086 个文件)
    *   包含：原子坐标、晶胞参数、空间群信息 (L10/L12/B2 标记)。

### 处理端 (The "Brain" - Multi-Task CGCNN)
模型同时学习三个任务（多任务学习）：

1.  **宏观稳定性 (Formation Energy)**
    *   **标签来源**: Mentals Project (`mp_data_summary.json`)
    *   **目的**: 过滤掉不稳定的材料（形成能 > 0.05 eV/atom 的不要）。
    
2.  **微观电子结构 (Electronic Structure)**
    *   **标签来源**: `dos_features.json` (刚刚抓取的)
    *   **核心**: **DOS 指纹 (400维向量)** + **d-band center (数值)**。
    *   **深度逻辑**: 结构决定电子结构。CGCNN 将学习如何通过原子排列推算出 DOS 曲线的形状。
    *   **物理意义**: 预测出 d-band center 后，我们可以利用 Nørskov 的 *d-band theory* 间接推导吸附能。

3.  **位点特异性活性 (Site-Specific Adsorption)**
    *   **模型能力**: 我们的 CGCNN 有一个 `site_activity_head`，可以输出每一个原子的吸附能（解决您提到的“不同位点吸附能不同”）。
    *   **⚠️ 数据缺口**: 目前我们**没有** H/OH 的吸附能标签 (Ground Truth)。
    *   **应对策略**: 暂时让模型先学前两个任务（形成能+DOS）。等未来有少量实验/DFT数据时，再用**迁移学习**微调这个 Head。

### 输出端 (Output)
当 Planner Agent 询问：“Pt3Co 的性能如何？”
Theorist Agent 运行模型并回答：
*   **形成能**: -0.32 eV/atom (稳定)
*   **d-band center**: -2.14 eV (适中)
*   **DOS 预测**: [展示预测的 DOS 曲线]
*   **综合评价**: "推荐尝试，结构稳定且电子结构有利于 HER。"

---

## 3. 下一步的代码调整
为了实现上述逻辑，我们需要对代码做微调，把刚刚下载的 DOS 数据利用起来：

1.  **修改 `dataset.py`**:
    *   之前只读了 `formation_energy`。
    *   **修改后**: 同时读取 `dos_features.json`，将 DOS 指纹和 d-band center 作为训练的 Target。

2.  **修改 `train_cgcnn.py`**:
    *   **Loss 函数升级**: $Loss = Loss_{formation} + 0.5 \times Loss_{DOS}$
    *   这样模型就会被迫去理解电子结构，从而变得更聪明。

您是否认可这个运行逻辑和代码修改计划？
