"""
IMCs 系统智能化综合评估
评估各智能体能力和系统整体智能化程度
"""

import sys
sys.path.insert(0, '.')

from src.agents.orchestrator import AgentOrchestrator
from src.agents.session import IterativeSession
from src.agents.fusion import AdvancedFusionEngine

def evaluate_system():
    print("="*70)
    print("IMCs 多智能体协同系统 - 智能化综合评估报告")
    print("="*70)
    
    # 初始化
    orchestrator = AgentOrchestrator()
    
    print("\n" + "="*70)
    print("第一部分: 各智能体能力评估")
    print("="*70)
    
    test_queries = [
        ("推荐HOR有序合金催化剂", "催化剂发现"),
        ("分析PtCo的理论性质", "材料分析"),
        ("预测新材料性能", "ML预测"),
    ]
    
    agent_scores = {name: {"total": 0, "count": 0} for name in orchestrator.agents}
    
    for query, desc in test_queries:
        print(f"\n测试查询: {query} ({desc})")
        print("-"*50)
        
        capabilities = orchestrator.collect_capabilities(query)
        
        for name, cap in capabilities.items():
            status = "✅" if cap.can_contribute else "❌"
            conf = cap.confidence
            agent_scores[name]["total"] += conf
            agent_scores[name]["count"] += 1
            
            print(f"  {name:12s}: {status} conf={conf:.2f} | {cap.reason[:40]}...")
    
    print("\n" + "-"*70)
    print("智能体能力汇总:")
    print("-"*70)
    
    for name, scores in agent_scores.items():
        avg = scores["total"] / scores["count"] if scores["count"] > 0 else 0
        status = orchestrator.agents[name].get_resource_status()
        
        # 能力等级
        if avg >= 0.8:
            level = "🌟 优秀"
        elif avg >= 0.6:
            level = "✅ 良好"
        elif avg >= 0.4:
            level = "⚠️ 一般"
        else:
            level = "❌ 需提升"
        
        print(f"\n  {name.upper()}")
        print(f"    平均置信度: {avg:.2f} ({level})")
        print(f"    数据资源: {status.data_count} 条")
        print(f"    模型资源: {status.model_count} 个")
        print(f"    可用状态: {'是' if status.is_available else '否'}")
    
    # 第二部分: 系统智能化程度
    print("\n" + "="*70)
    print("第二部分: 系统智能化程度评估")
    print("="*70)
    
    capabilities_matrix = {
        "资源自感知": {
            "score": 0.9,
            "evidence": "各智能体实现 get_resource_status()，可报告数据/模型数量",
            "gap": "缺少实时性能监控"
        },
        "能力评估": {
            "score": 0.85,
            "evidence": "各智能体实现 assess_capability()，返回置信度和理由",
            "gap": "评估逻辑基于规则，非学习型"
        },
        "协同决策": {
            "score": 0.8,
            "evidence": "AgentOrchestrator 按置信度智能排序执行",
            "gap": "缺少并行执行优化"
        },
        "动态重规划": {
            "score": 0.7,
            "evidence": "orchestrate() 支持失败后重规划",
            "gap": "重规划策略较简单"
        },
        "多轮迭代": {
            "score": 0.85,
            "evidence": "IterativeSession 支持实验反馈和状态跟踪",
            "gap": "迭代优化算法待增强"
        },
        "知识融合": {
            "score": 0.8,
            "evidence": "AdvancedFusionEngine 多源加权 + 属性评分",
            "gap": "权重自适应学习"
        },
        "可解释性": {
            "score": 0.85,
            "evidence": "RecommendationExplanation 提供多维度推荐理由",
            "gap": "可视化呈现待加强"
        },
        "LLM集成": {
            "score": 0.3,
            "evidence": "当前主要依赖规则和模板",
            "gap": "需要集成大语言模型增强理解"
        },
    }
    
    total_score = sum(c["score"] for c in capabilities_matrix.values())
    avg_score = total_score / len(capabilities_matrix)
    
    print(f"\n{'能力维度':<15} {'得分':>6} {'说明'}")
    print("-"*70)
    
    for cap, data in capabilities_matrix.items():
        score = data["score"]
        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
        print(f"{cap:<15} {score:.2f}  [{bar}] {data['evidence'][:35]}...")
    
    print("-"*70)
    print(f"{'系统综合得分':<15} {avg_score:.2f}  [{avg_score*100:.0f}/100分]")
    
    # 智能化等级
    if avg_score >= 0.85:
        level = "Level 5: 高度智能 (Highly Intelligent)"
    elif avg_score >= 0.7:
        level = "Level 4: 智能协同 (Intelligent Collaboration)"
    elif avg_score >= 0.5:
        level = "Level 3: 基础智能 (Basic Intelligence)"
    elif avg_score >= 0.3:
        level = "Level 2: 自动化 (Automation)"
    else:
        level = "Level 1: 规则驱动 (Rule-Based)"
    
    print(f"\n系统智能化等级: {level}")
    
    # 第三部分: 提升计划
    print("\n" + "="*70)
    print("第三部分: 下一步提升计划")
    print("="*70)
    
    improvements = [
        {
            "priority": "P0",
            "area": "LLM集成",
            "current": 0.3,
            "target": 0.8,
            "tasks": [
                "集成本地LLM (如 Ollama/Qwen)",
                "实现自然语言查询理解",
                "增强文献智能体的语义分析",
            ],
            "effort": "高 (2-3周)"
        },
        {
            "priority": "P1",
            "area": "实验智能体增强",
            "current": 0.6,
            "target": 0.9,
            "tasks": [
                "集成 ElecChemTool 数据处理",
                "自动解析电化学数据文件",
                "实现实验-理论数据关联",
            ],
            "effort": "中 (1-2周)"
        },
        {
            "priority": "P1",
            "area": "ML模型训练",
            "current": 0.4,
            "target": 0.85,
            "tasks": [
                "训练HOR活性预测模型",
                "实现在线学习/增量更新",
                "添加不确定性量化",
            ],
            "effort": "高 (2-3周)"
        },
        {
            "priority": "P2",
            "area": "可视化与交互",
            "current": 0.5,
            "target": 0.85,
            "tasks": [
                "优化推荐结果可视化",
                "添加材料对比视图",
                "实现交互式参数调整",
            ],
            "effort": "中 (1-2周)"
        },
        {
            "priority": "P2",
            "area": "权重自适应学习",
            "current": 0.3,
            "target": 0.8,
            "tasks": [
                "基于用户反馈调整融合权重",
                "实现贝叶斯优化自动调参",
            ],
            "effort": "中 (1周)"
        },
    ]
    
    for imp in improvements:
        print(f"\n[{imp['priority']}] {imp['area']}")
        print(f"    当前: {imp['current']:.0%} → 目标: {imp['target']:.0%}")
        print(f"    工作量: {imp['effort']}")
        print("    任务:")
        for task in imp['tasks']:
            print(f"      - {task}")
    
    print("\n" + "="*70)
    print("评估完成")
    print("="*70)
    
    return {
        "agent_scores": agent_scores,
        "system_score": avg_score,
        "level": level,
        "improvements": improvements
    }

if __name__ == "__main__":
    evaluate_system()
