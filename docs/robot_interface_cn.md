<!-- ASCII padding for tooling compatibility: 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -->
# IMCs Middleware / Robot 接入准备规范（v0.1）

日期：2026-01-28

> 目标：仅提供标准接口与数据结构，**不绑定任何硬件平台**。后续可由外部 middleware 通过 HTTP 接入。

---

## 1. 接口总览

- 提交任务：`POST /robot/submit_task`
- 查询状态：`GET /robot/task_status/{task_id}`
- 回传结果：`POST /robot/result_callback`
- 列出任务：`GET /robot/tasks`

Base URL:
```
http://localhost:8000
```

---

## 2. 任务数据结构（建议）

### 2.1 任务请求（RobotTaskRequest）
```json
{
  "task_type": "synthesis | test | characterization",
  "payload": {
    "material": {
      "formula": "PtRu",
      "composition": {"Pt": 0.5, "Ru": 0.5}
    },
    "conditions": {
      "temperature_K": 298,
      "electrolyte": "0.1M KOH",
      "device": "RDE",
      "rpm": [400, 900, 1600, 2500]
    },
    "target_metrics": ["exchange_current_density", "mass_activity"]
  },
  "external_id": "middleware-12345"
}
```

### 2.2 状态返回（RobotTaskStatus）
```json
{
  "task_id": 12,
  "status": "queued | running | completed | failed",
  "result": {"files": ["path/to/csv"]},
  "external_id": "middleware-12345"
}
```

### 2.3 回调结果（RobotResultCallback）
```json
{
  "task_id": 12,
  "status": "completed",
  "result": {
    "metrics": {
      "exchange_current_density": 0.38,
      "mass_activity": 1.25
    },
    "artifacts": ["data/experimental/rde_lsv/PtRu_LSV_1600rpm.csv"]
  }
}
```

---

## 3. IMCs 当前实现（仅接入准备）

- 已实现：
  - 任务入库与状态存储（`robot_tasks`）
  - 基本查询与回调接口
- 未实现：
  - 实际执行调度
  - 与硬件平台的通信驱动

---

## 4. 推荐对接流程（外部 middleware）

1. Middleware 调用 `POST /robot/submit_task` 提交任务
2. Middleware 执行真实实验
3. Middleware 将结果回调到 `POST /robot/result_callback`
4. IMCs 将结果落库，并触发“实验数据迭代优化”流程

---

如需将该接口升级为完整执行闭环，可继续扩展：
- 任务队列调度（Celery/Redis）
- 任务状态 Webhook
- 数据自动归档与解析

---

## 5. 回调触发迭代（可选）

在 `POST /robot/result_callback` 中加入 `auto_iterate=true`：

```json
{
  "task_id": 12,
  "status": "completed",
  "auto_iterate": true,
  "metric_name": "exchange_current_density",
  "result": {
    "metrics": {
      "exchange_current_density": 0.38,
      "mass_activity": 1.25
    },
    "material": {"formula": "PtRu"}
  }
}
```

系统会：
1) 将 metrics 写入 activity_metrics
2) 自动训练基于该指标的 ML 模型（准备级）

