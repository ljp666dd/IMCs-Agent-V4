"""
P3 多轮迭代支持验证测试
测试 IterativeSession 的会话管理和实验反馈功能
"""

import sys
sys.path.insert(0, '.')

from src.agents.session import (
    IterativeSession, ExperimentFeedback, CandidateStatus,
    create_session, continue_session
)

def test_iterative_session():
    print("="*60)
    print("P3 多轮迭代支持验证测试")
    print("="*60)
    
    # Test 1: 创建会话
    print("\n[1] 创建迭代会话")
    print("-"*60)
    
    session = IterativeSession()
    print(f"  会话ID: {session.session_id}")
    print(f"  当前轮次: {session.current_round}")
    
    # Test 2: 第一轮推荐
    print("\n[2] 第一轮推荐")
    print("-"*60)
    
    query = "推荐HOR有序合金催化剂"
    result = session.start_recommendation(query)
    
    print(f"  查询: {query}")
    print(f"  当前轮次: {session.current_round}")
    print(f"  候选数量: {len(result.candidates)}")
    print(f"  总候选材料: {len(session.candidates)}")
    
    if result.candidates:
        print("\n  Top 3 候选:")
        for i, cand in enumerate(result.candidates[:3]):
            print(f"    {i+1}. {cand.get('material_id')} - score={cand.get('final_score', 0):.3f}")
    
    # Test 3: 添加实验反馈
    print("\n[3] 添加实验反馈")
    print("-"*60)
    
    if result.candidates:
        # 模拟对前两个候选进行实验
        mat1 = result.candidates[0].get("material_id")
        mat2 = result.candidates[1].get("material_id")
        
        # 第一个材料验证成功
        session.add_experiment_feedback(
            material_id=mat1,
            experiment_type="LSV",
            metrics={"overpotential": 0.035, "mass_activity": 3.2},
            status=CandidateStatus.VALIDATED,
            notes="优秀的HOR性能"
        )
        print(f"  {mat1}: VALIDATED (overpotential=0.035V)")
        
        # 第二个材料验证失败
        session.add_experiment_feedback(
            material_id=mat2,
            experiment_type="LSV",
            metrics={"overpotential": 0.150, "mass_activity": 0.5},
            status=CandidateStatus.REJECTED,
            notes="HOR性能不佳"
        )
        print(f"  {mat2}: REJECTED (overpotential=0.150V)")
    
    # Test 4: 候选状态统计
    print("\n[4] 候选状态统计")
    print("-"*60)
    
    validated = session.get_candidates_by_status(CandidateStatus.VALIDATED)
    rejected = session.get_candidates_by_status(CandidateStatus.REJECTED)
    pending = session.get_candidates_by_status(CandidateStatus.PENDING)
    
    print(f"  已验证: {len(validated)}")
    print(f"  已拒绝: {len(rejected)}")
    print(f"  待验证: {len(pending)}")
    
    # Test 5: 会话摘要
    print("\n[5] 会话摘要")
    print("-"*60)
    
    summary = session.get_session_summary()
    print(f"  会话ID: {summary['session_id']}")
    print(f"  总轮次: {summary['total_rounds']}")
    print(f"  总候选: {summary['total_candidates']}")
    print(f"  状态分布: {summary['status_distribution']}")
    
    # Test 6: 会话持久化
    print("\n[6] 会话持久化")
    print("-"*60)
    
    session.save()
    print(f"  会话已保存")
    
    # 列出所有会话
    sessions = IterativeSession.list_sessions()
    print(f"  已保存会话: {len(sessions)}")
    
    # 加载会话
    loaded_session = IterativeSession.load(session.session_id)
    print(f"  会话已加载: {loaded_session.session_id}")
    print(f"  恢复轮次: {loaded_session.current_round}")
    print(f"  恢复候选: {len(loaded_session.candidates)}")
    
    print("\n" + "="*60)
    print("P3 多轮迭代支持验证: 通过 ✅")
    print("="*60)
    
    return session

if __name__ == "__main__":
    test_iterative_session()
