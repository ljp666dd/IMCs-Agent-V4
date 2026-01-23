# 🔬 IMCs: Multi-Agent Catalyst Research System

A comprehensive multi-agent AI framework for intermetallic compound (IMC) catalyst discovery and analysis, specifically designed for hydrogen evolution/oxidation reaction (HER/HOR) catalysts.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## 🌟 Overview

This system integrates multiple specialized AI agents to assist researchers in:
- **Catalyst Discovery**: Screening materials from theoretical databases
- **Property Prediction**: ML-based prediction of formation energy, DOS, d-band descriptors
- **Experiment Analysis**: Processing electrochemical test data (LSV, CV, EIS, Tafel)
- **Literature Research**: Extracting knowledge from scientific papers
- **Experiment Recommendation**: AI-driven suggestions for next experiments

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Task Manager Agent                        │
│               (Central Orchestrator)                         │
└────────────┬─────────────┬────────────┬────────────┬────────┘
             │             │            │            │
      ┌──────▼──────┐ ┌────▼────┐ ┌─────▼─────┐ ┌────▼────┐
      │ ML Agent    │ │ Theory  │ │ Experiment│ │Literature│
      │             │ │ Agent   │ │ Agent     │ │ Agent   │
      │ • XGBoost   │ │         │ │           │ │         │
      │ • LightGBM  │ │ • CIF   │ │ • LSV     │ │ • Search│
      │ • DNN       │ │ • DOS   │ │ • CV      │ │ • Extract│
      │ • SHAP      │ │ • Props │ │ • EIS     │ │ • Report│
      └─────────────┘ └─────────┘ └───────────┘ └─────────┘
```

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/IMCs.git
cd IMCs

# Install dependencies
pip install -r requirements.txt

# Optional: Install additional ML libraries
pip install xgboost lightgbm shap
```

### Requirements

- Python 3.10+
- PyTorch
- torch-geometric
- pymatgen
- mp-api (Materials Project API)
- scikit-learn
- numpy, pandas, scipy

## 🚀 Quick Start

### 1. ML Agent - Property Prediction

```python
from src.agents.core import MLAgent, MLAgentConfig

# Initialize agent
agent = MLAgent(MLAgentConfig(output_dir="data/ml_agent"))

# Load and train
X, y = agent.load_data_from_cifs(
    cif_dir="data/theory/cifs",
    target_file="data/theory/formation_energy_full.json"
)

# Train all models with SHAP analysis
top3 = agent.train_all(X, y, include_deep_learning=True)

# Best model: XGBoost (R² = 0.90)
```

### 2. Theory Agent - Materials Project Data

```python
from src.agents.core import TheoryDataAgent, TheoryDataConfig

agent = TheoryDataAgent(TheoryDataConfig(
    api_key="YOUR_MP_API_KEY",
    output_dir="data/theory"
))

# Check available data
info = agent.list_available_data()

# Download structures
agent.download_structures(limit=100)
```

### 3. Experiment Agent - Electrochemical Data

```python
from src.agents.core import ExperimentDataAgent

agent = ExperimentDataAgent()

# Process LSV data
result = agent.process_file("my_lsv_data.csv")

# Result: overpotential, current density, Tafel slope
```

### 4. Literature Agent - Knowledge Extraction

```python
from src.agents.core import LiteratureAgent

agent = LiteratureAgent()

# Search for papers
papers = agent.search_catalyst("PtRu", "HOR")

# Generate report
report = agent.generate_report("PtRu HOR Catalyst")
```

### 5. Task Manager - Full Orchestration

```python
from src.agents.core import TaskManagerAgent

# Create research system
system = TaskManagerAgent()

# Analyze request and create plan
response = system.chat("Find the best PtRu alloy for HOR")

# Execute plan
results = system.confirm_and_execute()
```

## 📊 Model Performance

### Formation Energy Prediction (2093 materials)

| Model | R² | MAE (eV/atom) |
|-------|-----|---------------|
| **XGBoost** | **0.90** | **0.056** |
| LightGBM | 0.90 | 0.059 |
| GradientBoosting | 0.90 | 0.059 |
| RandomForest | 0.86 | 0.067 |

### DOS Descriptor Prediction (1642 materials)

| Descriptor | R² |
|------------|-----|
| total_states | 0.95 |
| d_band_filling | 0.85 |
| d_band_center | 0.81 |
| d_band_width | 0.75 |

## 📁 Project Structure

```
IMCs/
├── src/
│   ├── agents/
│   │   ├── core/                    # Core agents
│   │   │   ├── ml_agent.py          # ML models + SHAP
│   │   │   ├── theory_agent.py      # Materials Project
│   │   │   ├── experiment_agent.py  # Electrochemical
│   │   │   ├── literature_agent.py  # Paper search
│   │   │   └── task_manager.py      # Orchestrator
│   │   └── ...
│   ├── models/                      # Neural network models
│   └── data_ingestion/              # Data fetching scripts
├── data/
│   ├── theory/                      # Theoretical data
│   │   ├── cifs/                    # Crystal structures
│   │   ├── formation_energy_full.json
│   │   └── orbital_pdos.json
│   ├── experimental/                # Experiment data
│   └── ml_agent/                    # Trained models
└── README.md
```

## 🔧 Configuration

### Materials Project API Key

```python
# Set in TheoryDataConfig
config = TheoryDataConfig(api_key="YOUR_API_KEY")
```

### ML Agent Configuration

```python
config = MLAgentConfig(
    output_dir="data/ml_agent",
    test_size=0.2,
    random_state=42,
    n_cv_folds=5,
    shap_samples=100
)
```

## 📈 SHAP Feature Importance

The ML Agent provides SHAP-based interpretability:

| Feature | Importance |
|---------|------------|
| std_electronegativity | 0.125 |
| min_atomic_number | 0.061 |
| max_atomic_number | 0.026 |
| avg_electronegativity | 0.020 |

**Key Insight**: Electronegativity difference is the most important factor for formation energy prediction.

## 🎯 Use Cases

1. **Catalyst Screening**: Screen thousands of materials for optimal HER/HOR catalysts
2. **Property Prediction**: Predict formation energy, d-band center, DOS
3. **Experiment Planning**: Get AI recommendations for next experiments
4. **Data Analysis**: Process and analyze electrochemical test results

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@software{imcs_multiagent,
  title = {IMCs: Multi-Agent Catalyst Research System},
  year = {2026},
  url = {https://github.com/yourusername/IMCs}
}
```

## 📄 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines.