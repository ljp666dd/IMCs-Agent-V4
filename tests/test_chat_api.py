"""
IMCs 智能对话功能测试脚本
通过 API 测试智能体对话功能
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_chat_api():
    print("="*60)
    print("IMCs 智能对话功能测试")
    print("="*60)

    # 测试1: 创建任务 - 查询材料
    print("\n[Test 1] 创建任务: 查询数据库中的材料")
    try:
        res = requests.post(
            f"{API_BASE}/tasks/create",
            json={"query": "查询数据库中有哪些材料？"},
            timeout=30
        )
        if res.status_code == 200:
            data = res.json()
            task_id = data.get("task_id")
            print(f"  任务ID: {task_id}")
            print(f"  响应: {json.dumps(data, indent=2, ensure_ascii=False)[:500]}")
        else:
            print(f"  失败: {res.status_code}")
            task_id = None
    except Exception as e:
        print(f"  异常: {e}")
        task_id = None

    # 测试2: 执行任务
    if task_id:
        print("\n[Test 2] 执行任务")
        try:
            res = requests.post(f"{API_BASE}/tasks/execute/{task_id}", timeout=120)
            if res.status_code == 200:
                data = res.json()
                print(f"  状态: {data.get('status')}")
                result = data.get("result", {})
                if result:
                    result_str = str(result)
                    print(f"  结果预览: {result_str[:1000]}...")
            else:
                print(f"  状态码: {res.status_code}")
                print(f"  响应: {res.text[:500]}")
        except Exception as e:
            print(f"  执行异常: {e}")

    # 测试3: 更复杂的查询
    print("\n[Test 3] 创建任务: 推荐HOR催化剂")
    try:
        res = requests.post(
            f"{API_BASE}/tasks/create",
            json={"query": "推荐适合HOR反应的有序合金催化剂候选材料"},
            timeout=30
        )
        if res.status_code == 200:
            data = res.json()
            task_id2 = data.get("task_id")
            print(f"  任务ID: {task_id2}")
            
            # 执行
            print("  执行中...")
            res2 = requests.post(f"{API_BASE}/tasks/execute/{task_id2}", timeout=180)
            if res2.status_code == 200:
                data2 = res2.json()
                print(f"  状态: {data2.get('status')}")
                result2 = data2.get("result", {})
                if result2:
                    result_str2 = str(result2)
                    print(f"  结果: {result_str2[:1500]}")
            else:
                print(f"  执行状态码: {res2.status_code}")
                print(f"  响应: {res2.text[:500]}")
    except Exception as e:
        print(f"  异常: {e}")

    # 测试4: 查看已执行任务历史
    print("\n[Test 4] 查询任务历史")
    try:
        res = requests.get(f"{API_BASE}/tasks/list", timeout=10)
        if res.status_code == 200:
            tasks = res.json()
            print(f"  找到 {len(tasks)} 个任务")
            for t in tasks[:5]:
                print(f"    - {t.get('task_id')}: {t.get('status')}")
        else:
            print(f"  状态码: {res.status_code}")
    except Exception as e:
        print(f"  异常: {e}")

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    test_chat_api()
