# Project Resource Summary / 项目资源归纳

截止当前，针对 **“有序合金 - 碱性 HER”** 多智能体系统，我们已完成以下核心组件的构建：

## 1. 目录结构与关键文件 (File Structure)

项目根目录: `IMCs/`
*   `config.ini`: 系统配置文件 (Gemini API, CGCNN 目标)。
*   `requirements.txt`: Python 依赖清单。

### `src/` (源代码)
*   **`src/data_ingestion/` (数据层)**
    *   `init_db.py`: 初始化 SQLite 数据库 (`experiments` 表, `theory` 表)。
    *   `fetch_mp_data.py`: **[已完成]** 自动抓取 Materials Project 数据的脚本 (已配置为下载 L10/L12/B2/B22 等有序结构)。
    *   `fetch_dos_data.py`: **[新建]** 抓取完整 DOS 数据并计算描述符的脚本。
    *   `dataset.py`: **[核心]** PyTorch Geometric 数据集加载器 (Reader -> Graph)。
    *   `verify_cifs_for_cgcnn.py`: 数据校验脚本 (通过率 100%)。
*   **`src/models/` (模型层)**
    *   `cgcnn.py`: **[核心]** 实现了 Multi-Head CGCNN 模型 (预测形成能 + 位点活性 + DOS 指纹)。
    *   `external/cgcnn/`: 克隆的官方 CGCNN 代码库 (作为对比基准)。
*   **`src/utils/` (工具层)**
    *   `dos_processing.py`: **[物理增强]** 包含 400-bin 指纹生成器和 11 种电子结构描述符 (d-band center 等) 计算器。
*   **`src/agents/` (智能体层)**
    *   `train_cgcnn.py`: 模型训练脚本 (目前默认训练 Formation Energy)。

### `data/` (数据层)
*   `her_catalysts.db`: 本地 SQLite 数据库。
*   `theory/cifs/`: **[已就绪]** 存放了 **1086** 个有序合金的 CIF 结构文件。
*   `theory/mp_data_summary.json`: 对应这 1086 个材料的基础元数据。
*   `theory/dos_features.json`: (等待运行 `fetch_dos_data.py` 生成)。

---

## 2. 数据流向 (Data Flow)
1.  **Ingestion**: `fetch_mp_data.py` -> 下载 CIF -> `data/theory/cifs/`
2.  **Processing**: `fetch_dos_data.py` -> 下载 DOS -> 调用 `dos_processing.py` -> 生成指纹 -> `data/theory/dos_features.json`
3.  **Training**: `train_cgcnn.py` -> 调用 `dataset.py` (读取 CIF) + 读取 JSON -> 训练 `cgcnn.py` 模型。

---

## 3. 下一步行动 (Action Items)
1.  **必做**: 运行 [`fetch_dos_data.py`](file:///c:/Users/Administrator/Desktop/课题2-有序合金-HOR/IMCs/IMCs/src/data_ingestion/fetch_dos_data.py) 以获取电子结构数据。
2.  **验证**: 运行训练脚本，观察模型是否收敛。
3.  **扩展**: 编写 `Experimental Agent` 开始处理实验室数据。
