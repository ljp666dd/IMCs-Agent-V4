<!-- ASCII padding for tooling compatibility: 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -->
# IMCs API 说明

默认 Base URL（本地）：
```text
http://localhost:8000
```

返回数据为 JSON；请求体除上传外均为 JSON。

---

## 1. Health
**GET** `/health`  
返回：
```json
{ "status": "ok", "version": "3.3" }
```

---

## 2. Tasks（任务图）

### 2.1 创建任务
**POST** `/tasks/create`  
请求：
```json
{ "query": "发现 HOR 候选有序合金材料" }
```
返回：
```json
{
  "task_id": "task_20260125_000000",
  "task_type": "catalyst_discovery",
  "description": "...",
  "steps": [{ "step_id": "step_1", "agent": "literature", "action": "search", "dependencies": [] }],
  "status": "pending"
}
```

### 2.2 对话模式
**POST** `/tasks/chat`  
请求：
```json
{ "message": "Find catalysts for HOR..." }
```
返回示例（可能是 plan 或 chat）：
```json
{ "type": "chat", "content": "..." }
```

### 2.3 执行任务
**POST** `/tasks/execute/{task_id}`  
返回：
```json
{ "message": "Task execution started", "task_id": "task_..." }
```

### 2.4 查询任务状态
**GET** `/tasks/{task_id}`  
返回：
```json
{
  "task_id": "task_...",
  "status": "completed",
  "steps": [ { "step_id": "step_1", "status": "completed" } ],
  "results": { "step_1": { "message": "..." } }
}
```

---

## 3. Theory（理论数据）

### 3.1 元素检索
**POST** `/theory/search`  
请求：
```json
{ "elements": ["Pt", "Ru"], "limit": 20 }
```

### 3.2 下载 CIF（异步）
**POST** `/theory/download/cif`  
请求：
```json
{ "elements": ["Pt", "Ru"], "limit": 50 }
```
说明：当前实现会使用配置中的默认元素，`elements` 可能被忽略。

### 3.3 数据状态
**GET** `/theory/status`

### 3.4 材料列表
**GET** `/theory/materials`

### 3.5 材料详情
**GET** `/theory/materials/{material_id}`

### 3.6 批量材料详情
**POST** `/theory/materials/batch`  
请求：
```json
{ "material_ids": ["mp-1", "mp-2"], "include_cif": false }
```

---

## 4. ML（机器学习）

### 4.1 训练模型（异步）
**POST** `/ml/train`  
请求：
```json
{ "model_types": ["traditional", "deep_learning"], "epochs": 100 }
```

### 4.2 模型列表
**GET** `/ml/models`

### 4.3 预测
**POST** `/ml/predict`  
请求：
```json
{ "features": [0.1, 0.2, 0.3], "model_name": null }
```

---

## 5. Experiment（实验数据）

### 5.1 上传实验文件
**POST** `/experiment/upload?method=lsv`  
表单字段：`file`（CSV/Excel）  
返回：
```json
{ "message": "File processed successfully", "analysis": { "...": "..." } }
```

---

## 6. Auth（可选）
当依赖安装完成时启用。

### 6.1 注册
**POST** `/auth/register`  
请求：
```json
{ "username": "user1", "password": "pass123" }
```

### 6.2 获取 Token
**POST** `/auth/token`  
表单字段：`username`, `password`

### 6.3 当前用户
**GET** `/auth/users/me`  
Header：`Authorization: Bearer <token>`

---

## 7. 常见错误码
- `400` 参数错误或模型未训练
- `404` 任务/材料不存在
- `500` 后端执行异常
---

## 8. 更新记录
- 2026-01-25：补充进度与计划文档链接（非 API 变更）
- 2026-01-25: 无 API 变更，仅记录中文任务触发规则


## 11. Literature-Driven Pipeline (CLI)
This pipeline is CLI-based (not API) and runs:
**online literature harvest -> seed CSV -> metrics + LSV generation -> import -> LSV analysis**

```powershell
python src/tools/harvest_literature_hor_seed.py --query "HOR ordered alloy catalyst" --limit 15 --max-pdfs 5 --persist --run-all --seeded
```

Outputs:
- `data/experimental/literature_hor_seed.csv`
- `data/experimental/literature_activity_metrics.csv`
- `data/experimental/literature_rde_lsv/`
