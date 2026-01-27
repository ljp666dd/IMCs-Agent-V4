# P2 测试清单（DOS + ML 特征 + 虚拟实验数据）

本清单用于验证 P2 功能：
- DOS / 轨道 DOS 描述符已入库
- ML 训练特征包含 DOS 描述符
- 虚拟实验活性指标可导入并作为证据链
- 元素范围严格限定为 TheoryDataConfig.elements

## 1）DOS 描述符回填到数据库
1. 执行：
   - `python src/tools/backfill_dos_descriptors.py`
2. 验证计数（任选其一）：
   - API：`GET http://localhost:8000/knowledge/stats`，确认 `dos_count` > 0
   - SQLite：
     - `python - <<'PY'`
       `import sqlite3; c=sqlite3.connect('data/imcs.db');`
       `cur=c.cursor(); cur.execute('SELECT COUNT(*) FROM materials WHERE dos_data IS NOT NULL');`
       `print(cur.fetchone()[0]); c.close()`
       `PY`
3. 抽查一条：
   - `python - <<'PY'`
     `import sqlite3, json;`
     `c=sqlite3.connect('data/imcs.db'); cur=c.cursor();`
     `cur.execute('SELECT material_id, dos_data FROM materials WHERE dos_data IS NOT NULL LIMIT 1');`
     `mid, data = cur.fetchone(); print(mid, json.loads(data).keys()); c.close()`
     `PY`

## 2）ML 训练使用 DOS 特征（DB 训练路径）
1. 在 Streamlit Chat 发起任务：
   - `Train ML model on current materials database`
2. 确认训练完成并保存模型：
   - API：`GET /knowledge/stats`，确认 `model_count` > 0
   - 或检查 `data/ml_agent/models/` 目录下是否有新模型文件
3.（可选）检查特征列是否包含 `dos_` 前缀：
   - 在 `models` 表中查看 `feature_cols`

## 3）生成并导入虚拟活性指标
1. 生成 CSV：
   - `python src/tools/generate_synthetic_activity_metrics.py --rows 30`
   - 输出：`data/experimental/synthetic_activity_metrics.csv`
2. 导入：
   - `python src/tools/import_activity_metrics.py data/experimental/synthetic_activity_metrics.csv`
3. 验证：
   - API：`GET /knowledge/stats`，确认 `activity_rows` > 0 且 `activity_materials` > 0
   - SQLite：`SELECT COUNT(*) FROM activity_metrics;`

## 4）证据链验证（UI）
1. 打开 Streamlit。
2. 选择一个在虚拟 CSV 中出现的 material_id。
3. Evidence Chain 中应看到：
   - Activity Metrics 表格有数据
   - evidence 类型包含 `activity_metric`

## 5）元素范围验证（TheoryDataConfig）
1. 在生成的 CSV 中任选一个 material_id。
2. 检查其公式元素是否全部属于 TheoryDataConfig.elements。
3. 尝试导入一个“非金属元素材料”的 material_id，应被导入工具跳过。
