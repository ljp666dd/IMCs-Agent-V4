"""
IMCs LLM Service - 大语言模型服务封装

提供与 Ollama 本地 LLM 的集成：
1. 查询理解增强
2. 文献语义分析
3. 推荐理由生成
4. 知识抽取

Version: 1.0
"""

import json
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.core.logger import get_logger

logger = get_logger(__name__)

# Ollama API 默认地址
OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass
class LLMConfig:
    """LLM 配置"""
    model: str = "qwen2.5:7b"
    base_url: str = OLLAMA_BASE_URL
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60


class OllamaService:
    """
    Ollama 本地 LLM 服务封装
    """
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self.base_url = self.config.base_url
        self.model = self.config.model
        self._available = None
        
        logger.info(f"OllamaService initialized with model: {self.model}")
    
    def is_available(self) -> bool:
        """检查 Ollama 服务是否可用"""
        if self._available is not None:
            return self._available
        
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            self._available = response.status_code == 200
            if self._available:
                models = response.json().get("models", [])
                logger.info(f"Ollama available with {len(models)} models")
            return self._available
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            self._available = False
            return False
    
    def list_models(self) -> List[str]:
        """列出可用模型"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name") for m in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        return []
    
    def generate(
        self, 
        prompt: str, 
        system: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 用户提示
            system: 系统提示（可选）
            temperature: 温度参数
            max_tokens: 最大生成长度
            
        Returns:
            生成的文本
        """
        if not self.is_available():
            logger.warning("Ollama not available, returning empty")
            return ""
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature or self.config.temperature,
                    "num_predict": max_tokens or self.config.max_tokens,
                }
            }
            
            if system:
                payload["system"] = system
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                logger.error(f"Generate failed: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Generate error: {e}")
            return ""
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None
    ) -> str:
        """
        对话模式
        
        Args:
            messages: 对话消息列表 [{"role": "user/assistant", "content": "..."}]
            temperature: 温度参数
            
        Returns:
            助手回复
        """
        if not self.is_available():
            return ""
        
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature or self.config.temperature,
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                logger.error(f"Chat failed: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return ""


class QueryUnderstanding:
    """
    查询理解增强
    
    使用 LLM 理解用户查询意图
    """
    
    def __init__(self, llm_service: OllamaService = None):
        self.llm = llm_service or OllamaService()
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        分析用户查询
        
        Returns:
            {
                "intent": "catalyst_discovery/analysis/prediction",
                "target_elements": ["Pt", "Co"],
                "target_properties": ["HOR", "activity"],
                "constraints": {...}
            }
        """
        if not self.llm.is_available():
            # 回退到规则解析
            return self._rule_based_analysis(query)
        
        system_prompt = """你是一个材料科学查询分析器。分析用户查询并提取关键信息。
返回 JSON 格式：
{
    "intent": "catalyst_discovery|performance_analysis|property_prediction|literature_search",
    "target_elements": ["元素符号列表"],
    "target_properties": ["目标性质列表"],
    "reaction": "HOR|ORR|HER|OER|CO2RR",
    "constraints": {}
}"""
        
        prompt = f"分析以下查询：\n{query}\n\n返回 JSON："
        
        response = self.llm.generate(prompt, system=system_prompt, temperature=0.1)
        
        try:
            # 提取 JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
        
        return self._rule_based_analysis(query)
    
    def _rule_based_analysis(self, query: str) -> Dict[str, Any]:
        """规则解析回退"""
        query_lower = query.lower()
        
        # 意图识别
        if any(kw in query for kw in ["推荐", "发现", "设计", "recommend"]):
            intent = "catalyst_discovery"
        elif any(kw in query for kw in ["分析", "评估", "analyze"]):
            intent = "performance_analysis"
        elif any(kw in query for kw in ["预测", "predict"]):
            intent = "property_prediction"
        else:
            intent = "general"
        
        # 元素识别
        elements = []
        element_symbols = ["Pt", "Pd", "Ni", "Co", "Fe", "Cu", "Au", "Ir", "Rh", "Ru"]
        for el in element_symbols:
            if el in query or el.lower() in query_lower:
                elements.append(el)
        
        # 反应类型
        reaction = None
        if "HOR" in query.upper() or "氢氧化" in query:
            reaction = "HOR"
        elif "ORR" in query.upper() or "氧还原" in query:
            reaction = "ORR"
        elif "HER" in query.upper() or "析氢" in query:
            reaction = "HER"
        
        return {
            "intent": intent,
            "target_elements": elements,
            "target_properties": [],
            "reaction": reaction,
            "constraints": {}
        }


class LiteratureAnalyzer:
    """
    文献语义分析
    
    使用 LLM 分析文献内容
    """
    
    def __init__(self, llm_service: OllamaService = None):
        self.llm = llm_service or OllamaService()
    
    def extract_materials(self, abstract: str) -> List[Dict[str, Any]]:
        """从摘要中提取材料信息"""
        if not self.llm.is_available():
            return []
        
        system_prompt = """从材料科学论文摘要中提取催化剂材料信息。
返回 JSON 列表：
[{"formula": "材料化学式", "performance": "性能描述", "reaction": "反应类型"}]"""
        
        prompt = f"从以下摘要提取材料信息：\n{abstract}\n\n返回 JSON："
        
        response = self.llm.generate(prompt, system=system_prompt, temperature=0.1)
        
        try:
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
        
        return []
    
    def summarize_paper(self, title: str, abstract: str) -> str:
        """总结论文关键发现"""
        if not self.llm.is_available():
            return abstract[:200] + "..."
        
        prompt = f"""请用一句话总结这篇论文的关键发现：

标题：{title}
摘要：{abstract}

一句话总结："""
        
        return self.llm.generate(prompt, temperature=0.3, max_tokens=100)


class ReasoningGenerator:
    """
    推荐理由生成
    
    使用 LLM 生成可解释的推荐理由
    """
    
    def __init__(self, llm_service: OllamaService = None):
        self.llm = llm_service or OllamaService()
    
    def generate_recommendation_reason(
        self,
        material: Dict[str, Any],
        scores: Dict[str, float],
        properties: Dict[str, Any]
    ) -> str:
        """生成推荐理由"""
        if not self.llm.is_available():
            return self._template_reason(material, scores, properties)
        
        prompt = f"""为以下 HOR 催化剂候选材料生成简短的推荐理由（中文，50字以内）：

材料: {material.get('formula', material.get('material_id'))}
评分: {scores}
属性: {properties}

推荐理由："""
        
        response = self.llm.generate(prompt, temperature=0.5, max_tokens=100)
        return response.strip() if response else self._template_reason(material, scores, properties)
    
    def _template_reason(
        self,
        material: Dict[str, Any],
        scores: Dict[str, float],
        properties: Dict[str, Any]
    ) -> str:
        """模板化理由生成"""
        reasons = []
        
        if scores.get("theory", 0) > 0:
            fe = properties.get("formation_energy")
            if fe:
                reasons.append(f"形成能{fe:.2f}eV")
        
        if scores.get("ml", 0) > 0:
            reasons.append("ML预测高分")
        
        if scores.get("experiment", 0) > 0:
            reasons.append("实验验证")
        
        return "、".join(reasons) if reasons else "多源证据支持"


# 便捷函数
def get_llm_service() -> OllamaService:
    """获取 LLM 服务实例"""
    return OllamaService()


def analyze_user_query(query: str) -> Dict[str, Any]:
    """分析用户查询"""
    analyzer = QueryUnderstanding()
    return analyzer.analyze_query(query)
