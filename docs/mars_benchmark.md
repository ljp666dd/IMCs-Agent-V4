<!-- ASCII padding for tooling compatibility: 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -->
# MARS Benchmark & Improvement Checklist

Source: local `MARS-main.zip` extracted under `_mars_ref/MARS-main`.

## 1. MARS Key Features (Observed)
- **Multi-team orchestration**: Orchestrator + Scientist / Engineer / Robot / Analyst teams
- **Structured plan output**: task assignment table + Mermaid flow + HUMAN/START/TERMINATE signals
- **Knowledge-driven**: GraphRAG + knowledge base tools (`graphrag/`)
- **Error handling & replanning**: `examples/err-deal/` provides re-plan/tool-failure cases
- **Case logs**: `examples/fig5-mars-logs/` contains real task logs
- **Evaluation suite**: `evaluate/` includes RAG/multi-agent eval scripts
- **Automation interface**: `middleware/` and `robot_platform.py` enable experiment platform integration

## 2. What IMCs Can Borrow (Actionable)
1) **Structured plan output**
   - Add task assignment table + Mermaid plan output in TaskManager/UI

2) **Error handling & replanning**
   - Add explicit failure policies (re-try / data-source swap / ask-human)

3) **Case logs & reproducible demos**
   - Add `examples/hor-logs/` with full HOR runs and outputs

4) **Evaluation suite**
   - Add `evaluate/` for Top-N recall, evidence coverage, and RAG quality

5) **Lightweight knowledge graph**
   - Start with literature -> material -> evidence relations

6) **Experiment platform interface (optional)**
   - Define a stable experiment API protocol first, integrate hardware later

## 3. Suggested File Mapping in IMCs
- Plan output: `src/agents/core/task_manager.py` + UI render
- Replan policies: `src/services/task/executor.py`
- Case logs: `examples/hor-logs/`, `examples/err-deal/`
- Evaluation: `evaluate/` scripts and datasets
- Knowledge graph: `knowledge_graph/` or `graphrag/` (optional)

## 4. Deferred Items (High Cost)
- Full robotics/middleware integration
- Full GraphRAG pipeline (Neo4j, ingestion, ops)

## 5. Near-term Order
1) Case logs + error/replan examples
2) Evaluation scripts (Top-N, evidence coverage, RAG)
3) Task assignment table + Mermaid output
4) Lightweight knowledge graph
