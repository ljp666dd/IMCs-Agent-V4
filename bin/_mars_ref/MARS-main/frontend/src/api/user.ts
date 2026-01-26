import type { ProjectListResult } from './model/detailModel'
import { request } from '../utils/request'
// 登录获取Token
export function authLogin(data: any) {
  return request.post<ProjectListResult>({
    url: `/login`,
    data
  })
}
// 发送聊天请求
export function chat(data: any) {
  return request.post<ProjectListResult>({
    url: `/chat`,
    data
  })
}
// 获取模型列表
export function getModelList() {
  return request.get<ProjectListResult>({
    url: `/model`
  })
}
// 获取历史消息列表
export function getSessionsList() {
  return request.get<ProjectListResult>({
    url: `/sessions`
  })
}

// 获取单条历史消息的聊天历史
export function getSessionsInfoList(session_uuid: any) {
  return request.get<ProjectListResult>({
    url: `/history/${session_uuid}`
  })
}
