# Streamlit UI Smoke Test Report

Date: 2026-01-26  
Scope: UI functionality in `src/interface/app.py` and its backend dependencies.

## Test Setup
- Backend started locally at `http://127.0.0.1:8000` (FastAPI).
- Streamlit UI not interacted via browser (headless checks + API verification).
- Database: existing `data/imcs.db`.

## Verified Working (via API)
- `GET /health` → OK
- `GET /theory/materials` → OK (materials present)
- `POST /tasks/create` → OK (plan created)
- `POST /tasks/execute/{task_id}` → OK (execution started)
- `GET /tasks/{task_id}` → OK (status visible)
- `GET /tasks/{task_id}/report` → OK (report + evidence_chain fields present)
- `POST /knowledge/rag` → OK (returns empty results when no sources)
- `GET /knowledge/trace/1` → OK
- `POST /knowledge/activity` + `GET /knowledge/activity/{material_id}` → OK
- `POST /knowledge/adsorption` + `GET /knowledge/adsorption/{material_id}` → OK

## UI Features Likely Working (requires backend running)
- **Chat / Task Graph**: plan creation, execution, status polling, task graph rendering.
- **Evidence Chain**: material selection, evidence list, knowledge trace & RAG (if sources exist).
- **Dataset Snapshot / Report download**: report endpoint works, UI can download JSON.

## Not Working / Blocked (observed or expected)
1) **Literature search (Semantic Scholar)**  
   - API returned HTTP 429 (rate limit). UI search likely fails or shows error.
2) **API Status → AFLOW**  
   - Timeout during connectivity test.
3) **API Status → Materials Project**  
   - Not validated in this run; may fail if API key invalid or network blocks.
4) **Local PDF parsing**  
   - UI only displays提示“需要 pdfplumber”，未实现解析流程（功能为占位）。
5) **ML Training (GNN/SHAP)**  
   - Highly data-dependent; if CIF filenames and target IDs mismatch, UI shows errors (not fully tested).

## Known Limitations / Notes
- External APIs are unstable (rate limits, timeouts); results depend on network.
- UI text contains visible mojibake in source view; check file encoding if UI中文字显示异常.

## Next Checks (if you want)
- Run Streamlit UI and click through each page to confirm widgets display.
- Verify Materials Project API key in Settings / API Status page.
- Test Literature search with low-frequency queries to avoid rate limit.


## User-Reported Issues
- Chat input in English not triggering plan creation reliably.
- ML training only loads data for theoretical source but does not train.
- UI lacks built-in EN/ZH translation helper.

## Fixes Applied (Pending UI retest)
- Strengthened English intent detection and forced EN when keywords detected.
- Moved ML training execution block to run for both theoretical and experimental sources.
- Added translation controls in Chat (auto-translate EN?ZH) and optional translator dependency.
