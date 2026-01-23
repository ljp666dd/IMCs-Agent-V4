# IMP-MAS Future Roadmap (v3.3+)

基于 v3.3 (Web 平台化) 状态，制定以下演进计划。

## Phase 6: 前后端对接 (Frontend Integration) [Immediate]
- [ ] **API Client**: 在前端实现 `src/lib/api.ts` 封装 Axios 请求。
- [ ] **Task UI**: 制作任务对话框与进度条组件。
- [ ] **Dashboard**: 制作数据概览仪表盘 (Charts)。

## Phase 7: 高级电化学分析 (Advanced Analysis)
- [ ] **K-L Plot**: 自动拟合转速与电流关系。
- [ ] **Tafel Advanced**: 自动识别 Tafel 区间。

## Phase 8: 生产环境部署 (Deployment)
- [ ] **Dockerize**: 为 Backend 和 Frontend 编写 `Dockerfile`。
- [ ] **PostgreSQL**: 迁移 SQLite 至 Postgres (Docker Compose)。
- [ ] **Nginx**: 配置反向代理。
