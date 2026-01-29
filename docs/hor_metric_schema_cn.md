<!-- ASCII padding for tooling compatibility: 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ -->
# HOR 指标模板（规范字段）

日期：2026-01-29

本模板用于统一 HOR 相关性能指标的字段、单位与含义，确保文献/实验/ML 使用一致名称。

---

## 1. 规范指标字段

| 字段名 | 缩写 | 单位 | 含义 |
| --- | --- | --- | --- |
| Jk_ref | Jk | A/cm2 | 参考电位下的动力学电流密度 |
| exchange_current_density | J0 | A/cm2 | 交换电流密度 |
| mass_activity | MA | A/mg | 质量活性 |
| overpotential_10mA | η | mV | 10 mA/cm2 过电位 |
| tafel_slope | b | mV/dec | Tafel 斜率 |

---

## 2. 使用说明

- **实验数据处理**：RDE/LSV 解析后统一写入上述字段。
- **文献抽取**：提取后需要映射为上述字段名。
- **模型训练**：`activity_metric:<字段名>` 作为目标列。

---

## 3. 配置文件

- 机器可读版本：`configs/hor_metric_schema.json`

---
