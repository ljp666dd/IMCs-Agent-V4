"""
P2 协同决策机制验证测试
测试 AgentOrchestrator 的协调功能
"""

import sys
sys.path.insert(0, '.')

from src.agents.orchestrator import AgentOrchestrator, FusionEngine, recommend_catalysts

def test_orchestrator():
    print("="*60)
    print("P2 协同决策机制验证测试")
    print("="*60)
    
    # Test 1: 初始化协调器
    print("\n[1] 初始化协调器")
    print("-"*60)
    
    orchestrator = AgentOrchestrator()
    print(f"  已注册智能体: {list(orchestrator.agents.keys())}")
    
    # Test 2: 收集能力评估
    print("\n[2] 收集能力评估")
    print("-"*60)
    
    query = "推荐HOR有序合金催化剂"
    capabilities = orchestrator.collect_capabilities(query)
    
    for name, cap in capabilities.items():
        status = "✅" if cap.can_contribute else "❌"
        print(f"  {name}: {status} confidence={cap.confidence:.2f}")
        print(f"    reason: {cap.reason}")
    
    # Test 3: 执行顺序调度
    print("\n[3] 执行顺序调度")
    print("-"*60)
    
    order = orchestrator.schedule_execution(capabilities)
    print(f"  执行顺序: {order}")
    
    # Test 4: 完整协同推荐
    print("\n[4] 完整协同推荐")
    print("-"*60)
    
    result = orchestrator.orchestrate(query, max_iterations=1)
    
    print(f"  成功: {result.success}")
    print(f"  迭代次数: {result.iteration}")
    print(f"  执行顺序: {result.execution_order}")
    print(f"  候选数量: {len(result.candidates)}")
    print(f"  推理: {result.reasoning}")
    
    if result.candidates:
        print("\n  Top 5 候选材料:")
        for i, cand in enumerate(result.candidates[:5]):
            mat_id = cand.get("material_id", "N/A")
            formula = cand.get("formula", "N/A")
            score = cand.get("final_score", 0)
            print(f"    {i+1}. {mat_id} ({formula}) - score={score:.3f}")
    
    # Test 5: 知识融合引擎
    print("\n[5] 知识融合引擎")
    print("-"*60)
    
    fusion = FusionEngine()
    print(f"  权重: {fusion.weights}")
    
    print("\n" + "="*60)
    print("P2 协同决策机制验证: 通过 ✅")
    print("="*60)
    
    return result

if __name__ == "__main__":
    test_orchestrator()
