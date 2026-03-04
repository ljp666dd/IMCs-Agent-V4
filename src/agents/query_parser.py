# -*- coding: utf-8 -*-
"""
IMCs Query Parser - 用户意图解析器

从自然语言查询中提取结构化约束：
- 目标元素 (target_elements)
- 目标反应 (target_reaction)
- 性能约束 (constraints)
- 资源提示 (resource_hints)
"""

import re
import json
from typing import Dict, List, Optional, Set
from src.core.logger import get_logger

logger = get_logger(__name__)

# 周期表元素 (专注于催化相关的过渡金属和主族元素)
PERIODIC_TABLE = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Hf", "Ta", "W", "Re", "Os", "Ir",
    "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
}

# 常见反应类型映射
REACTION_PATTERNS = {
    "HOR": [r"\bHOR\b", r"hydrogen\s+oxidation", r"氢氧化", r"氢气氧化"],
    "HER": [r"\bHER\b", r"hydrogen\s+evolution", r"析氢"],
    "OER": [r"\bOER\b", r"oxygen\s+evolution", r"析氧"],
    "ORR": [r"\bORR\b", r"oxygen\s+reduction", r"氧还原"],
}

# 性能指标关键词
PERFORMANCE_KEYWORDS = {
    "high_activity": ["高活性", "high activity", "excellent activity", "极高活性", "优异活性"],
    "low_overpotential": ["低过电位", "low overpotential", "小过电位"],
    "high_stability": ["高稳定性", "high stability", "stable", "耐久", "durability"],
    "low_cost": ["低成本", "low cost", "cheap", "经济", "非贵金属", "non-precious"],
    "high_selectivity": ["高选择性", "high selectivity"],
}

# 约束动词
CONSTRAINT_VERBS = {
    "include": ["含", "包含", "含有", "including", "containing", "with"],
    "exclude": ["不含", "排除", "不要", "without", "excluding", "no "],
    "prefer": ["优先", "prefer", "偏好", "重点"],
}


from src.services.llm.expert_reasoning import get_expert_reasoning

class LLMQueryParser:
    """使用 LLM 辅助提取意图的高级解析器"""
    
    def __init__(self, fallback_parser):
        self.fallback_parser = fallback_parser
        self.llm_service = get_expert_reasoning()
        
    def parse(self, query: str) -> dict:
        """优先使用 LLM 解析意图，解析失败时回退到正则解析"""
        if not self.llm_service.available:
            logger.info("LLM not available, using fallback regex parser")
            return self.fallback_parser.parse(query)
            
        prompt = f"""
你是一个电催化材料领域的意图解析专家。
请从以下用户输入中提取关键的查询结构约束，并严格以JSON格式输出，不带任何Markdown标记和额外文字。

用户输入: "{query}"

JSON结构及字段说明:
{{
    "target_elements": ["Pt", "Ru", ...],  // 提取用户提到的所有的化学元素符号（必须转换为元素符号，如铂->Pt）
    "target_reaction": "HOR", // HOR / HER / OER / ORR。默认为 HOR
    "constraints": {{
        "high_activity": ["高活性"], // 从输入中提取要求高性能的关键词
        "low_cost": ["便宜", "非贵金属"] // 或其他要求
    }},
    "resource_hints": {{
        "exclude_elements": ["Fe", "Ni"], // 用户要求不含或排除的元素符号
        "only_elements": ["Pt", "Co"]     // 用户要求只使用的元素符号
    }}
}}
如果某个字段在用户的输入中完全没有体现，则返回空数组或空字典（对 reaction 则默认为 HOR）。
请直接输出纯JSON字符串。
"""
        try:
            logger.info("Attempting LLM intent parsing...")
            response = self.llm_service._call_with_timeout(
                self.llm_service._call_openai_compatible if self.llm_service.backend in ("deepseek", "siliconflow") else self.llm_service._call_gemini, 
                prompt
            )
            
            if response:
                # 清理可能的 markdown 格式
                cleaned = response.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                
                parsed_json = json.loads(cleaned)
                
                # 做基本的 fallback 合并，避免模型输出缺失核心字段
                fallback_result = self.fallback_parser.parse(query)
                
                if "target_elements" not in parsed_json or not isinstance(parsed_json["target_elements"], list):
                    parsed_json["target_elements"] = fallback_result["target_elements"]
                if "target_reaction" not in parsed_json:
                    parsed_json["target_reaction"] = fallback_result["target_reaction"]
                if "constraints" not in parsed_json:
                    parsed_json["constraints"] = fallback_result["constraints"]
                if "resource_hints" not in parsed_json:
                    parsed_json["resource_hints"] = fallback_result["resource_hints"]
                    
                logger.info(f"LLMQueryParser result: {parsed_json}")
                return parsed_json
        except Exception as e:
            logger.warning(f"LLM parsing failed: {e}. Falling back to regex.")
            
        return self.fallback_parser.parse(query)

class QueryParser:
    """从自然语言查询中提取结构化约束"""

    def parse(self, query: str) -> Dict:
        """
        解析用户查询，返回结构化约束

        Args:
            query: 用户的自然语言查询

        Returns:
            Dict with keys: target_elements, target_reaction, constraints, resource_hints
        """
        result = {
            "target_elements": self._extract_elements(query),
            "target_reaction": self._extract_reaction(query),
            "constraints": self._extract_constraints(query),
            "resource_hints": self._extract_resources(query),
        }
        logger.info(
            f"QueryParser: elements={result['target_elements']}, "
            f"reaction={result['target_reaction']}, "
            f"constraints={list(result['constraints'].keys())}"
        )
        return result

    def _extract_elements(self, query: str) -> List[str]:
        """从查询中提取化学元素"""
        found = []

        # 匹配独立的元素符号 (1-2字母, 首字母大写)
        # 需要排除常见英文单词: I, In, As, No, He, Be, etc.
        false_positives = {"I", "In", "As", "No", "He", "Be", "Am", "At", "If",
                           "H", "C", "N", "O", "S", "P", "F", "Y", "W", "V", "K", "U"}

        for el in PERIODIC_TABLE:
            if el in false_positives and len(el) <= 2:
                # 对这些易误判的元素，要求前后有非字母字符
                pattern = rf"(?<![A-Za-z]){re.escape(el)}(?![a-z])"
                if re.search(pattern, query):
                    # 额外确认: 在化学式或列表上下文中
                    context_pattern = rf"(?:含|包含|with|containing|,\s*|、|/)\s*{re.escape(el)}|{re.escape(el)}\s*(?:,|、|/|和|\bor\b|及)"
                    if re.search(context_pattern, query, re.IGNORECASE):
                        found.append(el)
            else:
                # 标准元素: 只要不被嵌入在更长的单词中
                pattern = rf"(?<![A-Za-z]){re.escape(el)}(?![a-z])"
                if re.search(pattern, query):
                    found.append(el)

        # 中文元素名映射
        cn_elements = {
            "铂": "Pt", "钌": "Ru", "镍": "Ni", "铱": "Ir", "钯": "Pd",
            "金": "Au", "银": "Ag", "铜": "Cu", "铁": "Fe", "钴": "Co",
            "锰": "Mn", "铬": "Cr", "钼": "Mo", "钨": "W", "铼": "Re",
            "锇": "Os", "铑": "Rh", "锡": "Sn", "锗": "Ge", "铝": "Al",
            "钛": "Ti", "锆": "Zr", "铪": "Hf", "钽": "Ta", "铌": "Nb",
            "钒": "V", "锌": "Zn",
        }
        for cn_name, el_symbol in cn_elements.items():
            if cn_name in query and el_symbol not in found:
                found.append(el_symbol)

        return found

    def _extract_reaction(self, query: str) -> str:
        """从查询中提取目标反应类型"""
        for reaction, patterns in REACTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return reaction
        return "HOR"  # 默认为 HOR

    def _extract_constraints(self, query: str) -> Dict[str, List[str]]:
        """提取性能约束"""
        constraints = {}
        lower = query.lower()

        for constraint_key, keywords in PERFORMANCE_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in lower or kw in query:
                    constraints[constraint_key] = constraints.get(constraint_key, [])
                    constraints[constraint_key].append(kw)
                    break

        return constraints

    def _extract_resources(self, query: str) -> Dict[str, any]:
        """提取资源约束提示"""
        hints = {}

        # 检测排除/仅用约束
        exclude_pattern = r"(?:不含|排除|不要|without|excluding|no\s+)[\s,、]*(\w+)"
        ex_match = re.findall(exclude_pattern, query, re.IGNORECASE)
        if ex_match:
            hints["exclude_elements"] = ex_match

        include_pattern = r"(?:只用|仅用|only\s+use|只含)[\s,、]*(\w+)"
        in_match = re.findall(include_pattern, query, re.IGNORECASE)
        if in_match:
            hints["only_elements"] = in_match

        return hints


# Module-level singleton
_regex_parser = QueryParser()
_parser = LLMQueryParser(_regex_parser)


def parse_query(query: str) -> Dict:
    """便捷函数"""
    return _parser.parse(query)
