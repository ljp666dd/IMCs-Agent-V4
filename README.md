# Intelligent Materials Catalyst System (IMCs) v4.0

An agentic AI platform for **Hydrogen Oxidation Reaction (HOR)** catalyst discovery, leveraging Multi-Agent Collaboration, Machine Learning, and High-Throughput Theory.

## 🚀 Quick Start (Windows)

We provide a **One-Click Launcher** for Windows users.

1.  **Start System**:
    Double-click `start_imcs.bat` in the root directory.
    -   Starts Backend API (Port 8000)
    -   Starts Streamlit UI (Port 8501)

2.  **Manual Launch**:
    ```powershell
    # Backend
    python src/api/main.py
    
    # Streamlit UI
    streamlit run src/interface/app.py
    ```

---

## 📚 Literature-Driven HOR Pipeline (One Command)

This pipeline runs: **online literature harvest → seed CSV → metrics + LSV generation → import → LSV analysis**.

```powershell
python src/tools/harvest_literature_hor_seed.py --query "HOR ordered alloy catalyst" --limit 15 --max-pdfs 5 --persist --run-all --seeded
```

Outputs:
- `data/experimental/literature_hor_seed.csv`
- `data/experimental/literature_activity_metrics.csv`
- `data/experimental/literature_rde_lsv/` (LSV curves + manifest)

---

## 🛡️ Security & Configuration

**CRITICAL**: You must configure secrets before running.

1.  **Environment File**:
    The system reads from `.env` in the root directory.
    We have automatically created a default one for you.

2.  **Keys**:
    Edit `.env` to set your real keys:
    ```ini
    MP_API_KEY=your_materials_project_api_key
    IMCS_SECRET_KEY=your_production_secret_key
    IMCS_API_URL=http://localhost:8000
    ```

---

## 🏗️ Architecture (v4.0)

The system uses a modern Service-Oriented Architecture:

-   **Frontend**: Streamlit (Primary UI) + Next.js (Optional)
-   **Backend**: FastAPI - REST Wrappers for Agents.
-   **Database**: SQLite (`data/imcs.db`) with User/Session management.
-   **Agents**:
    -   `LiteratureAgent`: Semantic Scholar/ArXiv search.
    -   `TheoryAgent`: Materials Project API + Local Cache.
    -   `MLAgent`: Scikit-Learn/PyTorch + Joblib Persistence.
    -   `ExperimentAgent`: LSV/CV Data Parsing.
    -   `TaskManager`: Interactive Planning State Machine.

## 🧪 Scientific Validation

-   **Data Leakage Prevention**: Strict Train/Test splitting before scaling.
-   **Reproducibility**: Models are serialized to `data/ml_agent/models/`.
-   **Evidence**: Results link data sources (Literature -> Theory -> Experiment).

## 📚 Docs

-   `docs/user_manual.md`
-   `docs/developer_manual.md`
-   `docs/api_reference.md`
-   `docs/mars_benchmark.md`
-   `docs/idea_upgrade.md`

## 🧩 Examples & Evaluation

-   `examples/hor-logs/`
-   `examples/err-deal/`
-   `evaluate/`

## 🛠️ Development

**Install Dependencies:**
```powershell
pip install -r requirements.txt
```

**Run Tests:**
```powershell
python -m pytest tests/
```
