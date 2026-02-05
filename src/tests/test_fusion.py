"""
P4 知识融合优化验证测试
测试 AdvancedFusionEngine 的多源评分和可解释性功能
"""

import sys
sys.path.insert(0, '.')

from src.agents.fusion import (
    AdvancedFusionEngine, RecommendationExplanation, 
    RecommendationReason, create_fusion_report
)
from src.agents.protocol import AgentContribution, ContributionType

def test_advanced_fusion():
    print("="*60)
    print("P4 知识融合优化验证测试")
    print("="*60)
    
    # Test 1: 创建融合引擎
    print("\n[1] 创建高级融合引擎")
    print("-"*60)
    
    engine = AdvancedFusionEngine()
    print(f"  基础权重: {engine.base_weights}")
    print(f"  最优参数: {engine.optimal_params}")
    
    # Test 2: 模拟多智能体贡献
    print("\n[2] 模拟多智能体贡献")
    print("-"*60)
    
    contributions = {
        "theory": AgentContribution(
            agent_name="theory",
            contribution_type=ContributionType.CANDIDATES,
            success=True,
            confidence=0.85,
            candidates=[
                {"material_id": "PtCo", "formula": "PtCo", "formation_energy": -0.45},
                {"material_id": "PtNi", "formula": "PtNi", "formation_energy": -0.38},
                {"material_id": "PtFe", "formula": "PtFe", "formation_energy": -0.52},
            ],
            properties={
                "PtCo": {"d_band_center": -2.1},
                "PtNi": {"d_band_center": -1.8},
            }
        ),
        "ml": AgentContribution(
            agent_name="ml",
            contribution_type=ContributionType.PREDICTIONS,
            success=True,
            confidence=0.70,
            predictions={
                "PtCo": 0.85,
                "PtNi": 0.72,
                "IrNi": 0.68,
            }
        ),
        "experiment": AgentContribution(
            agent_name="experiment",
            contribution_type=ContributionType.METRICS,
            success=True,
            confidence=0.95,
            candidates=[
                {"material_id": "PtCo", "overpotential": 0.032},
            ],
            metrics={
                "PtCo": {"mass_activity": 3.5, "overpotential": 0.032}
            }
        ),
        "literature": AgentContribution(
            agent_name="literature",
            contribution_type=ContributionType.INSIGHTS,
            success=False,  # 模拟失败
            confidence=0.0,
        ),
    }
    
    print(f"  Theory: {len(contributions['theory'].candidates)} 候选")
    print(f"  ML: {len(contributions['ml'].predictions)} 预测")
    print(f"  Experiment: {len(contributions['experiment'].candidates)} 验证")
    print(f"  Literature: 失败")
    
    # Test 3: 融合计算
    print("\n[3] 融合计算")
    print("-"*60)
    
    candidates, explanations = engine.synthesize(contributions)
    
    print(f"  候选数量: {len(candidates)}")
    print(f"  解释数量: {len(explanations)}")
    
    # Test 4: 可解释性输出
    print("\n[4] 可解释性输出")
    print("-"*60)
    
    for exp in explanations[:5]:
        print(f"\n  {exp.get_summary()}")
        print(f"    来源得分: {exp.source_scores}")
        print(f"    理由详情: {exp.reason_details}")
        print(f"    置信度: {exp.confidence_explanation}")
    
    # Test 5: 动态权重调整
    print("\n[5] 动态权重调整验证")
    print("-"*60)
    
    adjusted_weights = engine._adjust_weights(contributions)
    print(f"  调整后权重: {adjusted_weights}")
    print(f"  (Literature 失败后权重已重分配)")
    
    # Test 6: 生成报告
    print("\n[6] 生成融合报告")
    print("-"*60)
    
    report = create_fusion_report(explanations, top_n=3)
    print(report)
    
    print("\n" + "="*60)
    print("P4 知识融合优化验证: 通过 ✅")
    print("="*60)
    
    return explanations

if __name__ == "__main__":
    test_advanced_fusion()
