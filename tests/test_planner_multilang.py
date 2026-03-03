import sys
sys.path.insert(0, ".")

from src.services.task.planner import TaskPlanner
from src.services.task.types import TaskType


def test_planner_multilang_classification():
    planner = TaskPlanner()
    cases = [
        ("推荐HOR催化剂候选", TaskType.CATALYST_DISCOVERY),
        ("Recommend HOR catalysts with ML model", TaskType.CATALYST_DISCOVERY),
        ("分析HOR LSV performance", TaskType.PERFORMANCE_ANALYSIS),
        ("文献综述 HOR 催化", TaskType.LITERATURE_REVIEW),
        ("预测HOR活性 使用 ML model", TaskType.PROPERTY_PREDICTION),
    ]
    for query, expected in cases:
        assert planner.analyze_request(query) == expected
