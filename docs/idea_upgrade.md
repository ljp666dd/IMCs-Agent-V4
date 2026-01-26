# Idea Upgrade: IMCs vs MARS

## 1. Your Original Idea (IMCs)
- Multi-agent collaboration for HOR candidate discovery
- Task graph + evidence chain as core product
- Theory data + literature + ML for recommendation
- Iterative loop with experimental data and model retraining

## 2. MARS Idea (Reference)
- Multi-agent teams with explicit role separation and orchestration
- Structured planning output (assignment table + Mermaid flow)
- Error handling and replanning patterns
- Case logs and reproducible demos
- Evaluation scripts and knowledge graph (GraphRAG)
- Optional robotic platform integration

## 3. Upgraded IMCs Concept (Merged)
1) **Structured Planning Output**
   - Add task assignment table + Mermaid flow for every plan
   - Keep task graph execution status in UI

2) **Robustness & Replanning**
   - Add explicit failure policy per step (retry / alternate source / human-in-loop)
   - Maintain error case logs for reproducibility

3) **Case Library & Evaluation Suite**
   - Add `examples/hor-logs/` and `examples/err-deal/`
   - Add `evaluate/` for Top-N recall and evidence coverage

4) **Evidence-driven Iteration**
   - Tie recommendation confidence to evidence coverage
   - Trigger data acquisition when evidence is insufficient

5) **Lightweight Knowledge Graph**
   - Link literature -> material -> evidence -> recommendation
   - Start with SQLite + relations, evolve to GraphRAG when needed

6) **Experiment Interface (Future)**
   - Define experiment API contract first
   - Integrate hardware platform only when ready

## 4. What Changes First (Ordered)
1. Task assignment table + Mermaid output (done)
2. Case logs + replan examples (done: templates)
3. Evaluation scripts (done: basic)
4. Evidence-driven triggers and gap analysis (next)
5. Lightweight knowledge graph (later)
6. Experiment platform interface (future)
