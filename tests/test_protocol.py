"""
P1 协议接口验证测试
测试各智能体的 assess_capability 和 get_resource_status 功能
"""

import sys
sys.path.insert(0, '.')

from src.agents.protocol import (
    AgentProtocol, AgentCapability, ResourceStatus, 
    AgentContribution, QueryContext, ContributionType
)
from src.agents.protocol_impl import (
    TheoryAgentProtocolMixin, MLAgentProtocolMixin,
    ExperimentAgentProtocolMixin, LiteratureAgentProtocolMixin
)

def test_protocol_definitions():
    print("="*60)
    print("P1 协议接口验证测试")
    print("="*60)
    
    # Test 1: Protocol class definitions
    print("\n[1] 协议类定义测试")
    print("-"*60)
    
    cap = AgentCapability(
        can_contribute=True,
        confidence=0.8,
        contribution_types=[ContributionType.CANDIDATES],
        reason="测试"
    )
    print(f"  AgentCapability: {cap.to_dict()}")
    
    status = ResourceStatus(
        agent_name="test",
        data_count=100,
    )
    print(f"  ResourceStatus: {status.to_dict()}")
    
    contrib = AgentContribution(
        agent_name="test",
        contribution_type=ContributionType.PREDICTIONS,
        success=True,
    )
    print(f"  AgentContribution: created successfully")
    
    ctx = QueryContext(
        user_query="推荐HOR催化剂",
        task_type="catalyst_discovery"
    )
    print(f"  QueryContext: {ctx.user_query}")
    
    print("  [OK] 所有协议类定义正常")


def test_theory_agent_protocol():
    print("\n[2] TheoryAgent 协议测试")
    print("-"*60)
    
    # 创建带有混入的测试类
    from src.agents.core.theory_agent import TheoryDataAgent, TheoryDataConfig
    
    class TheoryAgentWithProtocol(TheoryAgentProtocolMixin, TheoryDataAgent):
        pass
    
    agent = TheoryAgentWithProtocol(TheoryDataConfig())
    
    # Test assess_capability
    cap = agent.assess_capability("推荐HOR有序合金催化剂")
    print(f"  assess_capability:")
    print(f"    can_contribute: {cap.can_contribute}")
    print(f"    confidence: {cap.confidence}")
    print(f"    reason: {cap.reason}")
    
    # Test get_resource_status
    status = agent.get_resource_status()
    print(f"  get_resource_status:")
    print(f"    data_count: {status.data_count}")
    print(f"    data_coverage: {status.data_coverage}")
    
    print("  [OK] TheoryAgent 协议实现正常")


def test_ml_agent_protocol():
    print("\n[3] MLAgent 协议测试")
    print("-"*60)
    
    from src.agents.core.ml_agent import MLAgent
    
    class MLAgentWithProtocol(MLAgentProtocolMixin, MLAgent):
        pass
    
    agent = MLAgentWithProtocol()
    
    cap = agent.assess_capability("预测材料性能")
    print(f"  assess_capability:")
    print(f"    can_contribute: {cap.can_contribute}")
    print(f"    confidence: {cap.confidence}")
    print(f"    reason: {cap.reason}")
    
    status = agent.get_resource_status()
    print(f"  get_resource_status:")
    print(f"    model_count: {status.model_count}")
    
    print("  [OK] MLAgent 协议实现正常")


def test_collaboration_scenario():
    print("\n[4] 协同场景测试")
    print("-"*60)
    
    from src.agents.core.theory_agent import TheoryDataAgent, TheoryDataConfig
    from src.agents.core.ml_agent import MLAgent
    
    class TheoryAgentWithProtocol(TheoryAgentProtocolMixin, TheoryDataAgent):
        pass
    
    class MLAgentWithProtocol(MLAgentProtocolMixin, MLAgent):
        pass
    
    # 创建智能体
    agents = {
        "theory": TheoryAgentWithProtocol(TheoryDataConfig()),
        "ml": MLAgentWithProtocol(),
    }
    
    query = "推荐HOR有序合金催化剂"
    
    # 收集能力评估
    print(f"  查询: {query}")
    print(f"  能力评估:")
    
    for name, agent in agents.items():
        cap = agent.assess_capability(query)
        status = "✅" if cap.can_contribute else "❌"
        print(f"    {name}: {status} (confidence={cap.confidence:.2f}) - {cap.reason}")
    
    # 排序决定执行顺序
    sorted_agents = sorted(
        [(name, agent.assess_capability(query)) for name, agent in agents.items()],
        key=lambda x: (-x[1].confidence, x[1].estimated_time_seconds)
    )
    
    print(f"\n  执行顺序:")
    for i, (name, cap) in enumerate(sorted_agents):
        if cap.can_contribute:
            print(f"    {i+1}. {name} (confidence={cap.confidence:.2f})")
    
    print("\n  [OK] 协同场景测试通过")


def main():
    test_protocol_definitions()
    test_theory_agent_protocol()
    test_ml_agent_protocol()
    test_collaboration_scenario()
    
    print("\n" + "="*60)
    print("P1 协议接口验证: 全部通过 ✅")
    print("="*60)


if __name__ == "__main__":
    main()
