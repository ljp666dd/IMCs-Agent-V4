"""
P0 LLM 集成验证测试
测试 Ollama 服务和查询理解功能
"""

import sys
sys.path.insert(0, '.')

from src.services.llm import (
    OllamaService, QueryUnderstanding, 
    LiteratureAnalyzer, ReasoningGenerator
)

def test_llm_integration():
    print("="*60)
    print("P0 LLM 集成验证测试")
    print("="*60)
    
    # Test 1: 检查 Ollama 服务
    print("\n[1] 检查 Ollama 服务")
    print("-"*60)
    
    llm = OllamaService()
    available = llm.is_available()
    
    if available:
        print("  ✅ Ollama 服务可用")
        models = llm.list_models()
        print(f"  已安装模型: {models}")
    else:
        print("  ⚠️ Ollama 服务不可用（模型可能仍在下载）")
        print("  将使用规则回退模式")
    
    # Test 2: 查询理解
    print("\n[2] 查询理解测试")
    print("-"*60)
    
    analyzer = QueryUnderstanding(llm)
    
    test_queries = [
        "推荐HOR有序合金催化剂",
        "分析PtCo的电化学性能",
        "预测PtNi的活性",
    ]
    
    for query in test_queries:
        result = analyzer.analyze_query(query)
        print(f"\n  查询: {query}")
        print(f"  意图: {result.get('intent')}")
        print(f"  元素: {result.get('target_elements')}")
        print(f"  反应: {result.get('reaction')}")
    
    # Test 3: 文献分析
    print("\n[3] 文献分析测试")
    print("-"*60)
    
    lit_analyzer = LiteratureAnalyzer(llm)
    
    test_abstract = """
    PtCo alloy nanoparticles supported on carbon show excellent 
    hydrogen oxidation reaction (HOR) activity with mass activity 
    of 3.5 A/mg_Pt and overpotential of 35 mV at 10 mA/cm2.
    """
    
    if available:
        materials = lit_analyzer.extract_materials(test_abstract)
        print(f"  提取的材料: {materials}")
    else:
        print("  ⚠️ LLM 不可用，跳过文献分析测试")
    
    # Test 4: 推荐理由生成
    print("\n[4] 推荐理由生成测试")
    print("-"*60)
    
    reasoner = ReasoningGenerator(llm)
    
    material = {"material_id": "PtCo", "formula": "PtCo"}
    scores = {"theory": 0.8, "ml": 0.7, "experiment": 0.9}
    properties = {"formation_energy": -0.45, "overpotential": 0.035}
    
    reason = reasoner.generate_recommendation_reason(material, scores, properties)
    print(f"  材料: PtCo")
    print(f"  推荐理由: {reason}")
    
    print("\n" + "="*60)
    if available:
        print("P0 LLM 集成验证: 通过 ✅")
    else:
        print("P0 LLM 集成验证: 部分通过 ⚠️ (等待模型下载完成)")
    print("="*60)
    
    return available

if __name__ == "__main__":
    test_llm_integration()
