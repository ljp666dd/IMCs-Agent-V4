"""
IMCs 智能化规划验证测试
测试 MetaController 的资源感知和智能降级功能
"""

import sys
sys.path.insert(0, '.')

from src.services.task.planner import TaskPlanner
from src.services.task.types import TaskType
from src.services.task.meta_controller import MetaController

def test_intelligent_planning():
    print("="*60)
    print("IMCs 智能化规划验证测试")
    print("="*60)
    
    # Get current system stats
    mc = MetaController()
    stats = mc.get_stats()
    
    print("\n[1] 系统资源状态")
    print("-"*60)
    print(f"  材料总数: {stats.get('total_materials', 0)}")
    print(f"  活性指标行数: {stats.get('activity_rows', 0)}")
    print(f"  实验证据数: {stats.get('evidence_by_source', {}).get('experiment', 0)}")
    print(f"  ML模型数: {stats.get('model_count', 0)}")
    
    # Test decision making
    print("\n[2] 智能决策测试")
    print("-"*60)
    decisions = mc.decide(TaskType.PERFORMANCE_ANALYSIS, "HOR催化剂分析", stats)
    print(f"  has_experiment_data: {decisions.get('has_experiment_data')}")
    print(f"  has_theory_data: {decisions.get('has_theory_data')}")
    print(f"  has_ml_model: {decisions.get('has_ml_model')}")
    
    # Test query planning
    print("\n[3] 查询规划测试")
    print("-"*60)
    
    planner = TaskPlanner()
    queries = [
        ("推荐HOR有序合金催化剂", "第一轮推荐（无实验数据）"),
        ("分析我的LSV实验数据", "实验数据分析"),
    ]
    
    test_passed = True
    for query, desc in queries:
        task_type = planner.analyze_request(query)
        plan = planner.create_plan(query)
        has_exp = any(s.agent == "experiment" for s in plan.steps)
        agents = [s.agent for s in plan.steps]
        
        print(f"\n  查询: {query}")
        print(f"  场景: {desc}")
        print(f"  类型: {task_type.value}")
        print(f"  智能体: {agents}")
        print(f"  包含实验步骤: {'是' if has_exp else '否'}")
        
        # For first-round recommendation, should NOT have experiment step
        if "推荐" in query and has_exp:
            print(f"  [FAIL] 第一轮推荐不应包含实验步骤！")
            test_passed = False
        elif "推荐" in query and not has_exp:
            print(f"  [OK] 智能降级成功")
    
    print("\n" + "="*60)
    if test_passed:
        print("智能化规划验证: 通过")
    else:
        print("智能化规划验证: 失败")
    print("="*60)
    
    return test_passed

if __name__ == "__main__":
    test_intelligent_planning()
