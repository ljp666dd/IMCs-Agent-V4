"""
IMCs HOR催化剂推荐对话测试
深入测试智能体对话功能和智能化程度
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_hor_recommendation():
    print("="*70)
    print("IMCs HOR催化剂推荐对话测试")
    print("="*70)
    
    query = "推荐HOR有序合金催化剂"
    print(f"\n查询: {query}")
    print("-"*70)
    
    # Step 1: 创建任务
    print("\n[Step 1] 创建任务...")
    try:
        res = requests.post(
            f"{API_BASE}/tasks/create",
            json={"query": query},
            timeout=60
        )
        if res.status_code == 200:
            data = res.json()
            task_id = data.get("task_id")
            task_type = data.get("task_type")
            steps = data.get("steps", [])
            
            print(f"  任务ID: {task_id}")
            print(f"  任务类型: {task_type}")
            print(f"  执行步骤数: {len(steps)}")
            
            print("\n  执行计划:")
            for i, step in enumerate(steps):
                print(f"    {i+1}. [{step.get('agent')}] {step.get('action')}")
                params = step.get('params', {})
                if params:
                    for k, v in params.items():
                        print(f"       - {k}: {v}")
                deps = step.get('dependencies', [])
                if deps:
                    print(f"       依赖: {deps}")
        else:
            print(f"  创建失败: {res.status_code}")
            print(f"  响应: {res.text[:500]}")
            return
    except Exception as e:
        print(f"  异常: {e}")
        return
    
    # Step 2: 执行任务
    print("\n[Step 2] 执行任务（可能需要较长时间）...")
    try:
        start_time = time.time()
        res = requests.post(
            f"{API_BASE}/tasks/execute/{task_id}", 
            timeout=300  # 5分钟超时
        )
        elapsed = time.time() - start_time
        
        print(f"  执行耗时: {elapsed:.1f}秒")
        
        if res.status_code == 200:
            data = res.json()
            status = data.get("status")
            result = data.get("result", {})
            
            print(f"  状态: {status}")
            
            if result:
                print("\n  执行结果:")
                result_str = json.dumps(result, indent=2, ensure_ascii=False)
                # 截断过长的输出
                if len(result_str) > 2000:
                    print(f"{result_str[:2000]}\n  ... (结果过长已截断)")
                else:
                    print(result_str)
        else:
            print(f"  执行失败: {res.status_code}")
            print(f"  响应: {res.text[:1000]}")
    except requests.exceptions.Timeout:
        print("  任务执行超时（5分钟）")
    except Exception as e:
        print(f"  执行异常: {e}")
    
    # Step 3: 获取任务详情
    print("\n[Step 3] 获取任务最终状态...")
    try:
        res = requests.get(f"{API_BASE}/tasks/{task_id}", timeout=30)
        if res.status_code == 200:
            data = res.json()
            print(f"  任务状态: {data.get('status')}")
            steps = data.get('steps', [])
            if steps:
                print("\n  各步骤状态:")
                for step in steps:
                    step_id = step.get('step_id')
                    agent = step.get('agent')
                    status = step.get('status')
                    error = step.get('error')
                    print(f"    {step_id} [{agent}]: {status}")
                    if error:
                        print(f"      错误: {error[:100]}")
    except Exception as e:
        print(f"  异常: {e}")
    
    print("\n" + "="*70)
    print("测试完成")
    print("="*70)

if __name__ == "__main__":
    test_hor_recommendation()
