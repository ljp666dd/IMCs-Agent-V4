# IMCs Upgrade Checklist (File-Level)

Date: 2026-01-26
Scope: Multi-agent research platform (Streamlit UI + FastAPI + DB + agents)

## P0 - Stability and Persistence (Now)
- UI chat persistence
  - src/services/db/schema.sql (chat_sessions, chat_messages)
  - src/tools/migrate_db.py (ensure tables exist)
  - src/services/db/database.py (CRUD for chat sessions/messages)
  - src/interface/app.py (session picker, rename, export/delete, history load)
- UI import/runtime stability
  - src/interface/app.py (ensure ROOT_DIR added to sys.path before local imports)
  - src/interface/app.py (avoid Streamlit duplicate element IDs)
- Task graph recovery
  - src/interface/app.py (store plan_id in artifacts; reload last task)

## P1 - Task Graph + Evidence Chain (Next)
- Evidence-chain schema standardization
  - src/services/db/database.py (consistent evidence metadata)
  - src/services/task/executor.py (attach evidence per step)
  - src/services/task/planner.py (step output format)
- UI evidence binding to task/session
  - src/interface/app.py (task report panel per session)
  - src/api/routers/tasks.py (report endpoint to include evidence chain)

## P2 - Data Pipeline and Activity Metrics
- Activity metrics ingestion
  - data/templates/activity_metrics_template.csv
  - src/tools/import_activity_metrics.py
  - src/api/routers/knowledge.py (activity endpoints)
- Adsorption data ingestion
  - data/templates/adsorption_energies_template.csv
  - src/tools/import_adsorption_energies.py
- DOS/PDOS descriptors
  - src/agents/core/theory_agent.py (extract features)
  - src/services/db/database.py (store descriptors)
  - src/agents/core/ml_agent.py (feature use)

## P3 - ML Training + Explainability
- Stable ML training flow
  - src/interface/app.py (always train when dataset present)
  - src/agents/core/ml_agent.py (log metrics + save model card)
- Explainability
  - src/agents/core/ml_agent.py (SHAP pipeline)
  - src/agents/core/gnn_agent.py (GNN explainability method)

## P4 - UX and Internationalization
- UI text cleanup
  - src/interface/app.py (remove garbled strings)
  - docs/user_manual.md (sync UI labels)
- English query planning
  - src/services/task/planner.py (language detection)
  - src/interface/app.py (EN->ZH query translation)

## P5 - Evaluation and Benchmarks
- Automated evaluation
  - evaluate/ (top-k recall, evidence coverage)
  - src/interface/app.py (evaluation page)
- Baseline comparison
  - docs/mars_benchmark.md
  - docs/ui_smoke_test.md

## Immediate Next Actions
1) Verify Streamlit boots after chat persistence changes
2) Run a full chat session: create -> rename -> export -> delete
3) Create a task in chat and reload via "Load Task History"
4) Decide which UI pages to clean next (home/data/ml)
