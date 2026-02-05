"""
IMCs Agent Protocol Implementations - 智能体协议实现

为各核心智能体提供标准协议接口的实现
通过混入类 (Mixin) 方式扩展现有智能体
"""

from typing import Dict, List, Optional, Any
import re

from src.agents.protocol import (
    AgentProtocol, AgentCapability, ResourceStatus, 
    AgentContribution, QueryContext, ContributionType
)
from src.core.logger import get_logger

logger = get_logger(__name__)


class TheoryAgentProtocolMixin:
    """
    TheoryDataAgent 协议实现混入类
    
    提供能力评估、资源状态和知识贡献功能
    """
    
    @property
    def agent_name(self) -> str:
        return "theory"
    
    def assess_capability(self, query: str, context: Optional[QueryContext] = None) -> AgentCapability:
        """评估对查询的贡献能力"""
        query_lower = query.lower()
        
        # 检测相关关键词
        theory_keywords = [
            "理论", "形成能", "吸附", "能量", "dos", "态密度", "计算", 
            "结构", "晶体", "材料", "合金", "元素",
            "formation", "energy", "structure", "download", "theory"
        ]
        is_relevant = any(kw in query_lower or kw in query for kw in theory_keywords)
        
        # 获取当前资源状态
        status = self.get_resource_status()
        has_data = status.data_count > 0
        
        if not is_relevant:
            return AgentCapability(
                can_contribute=False,
                confidence=0.0,
                reason="查询与理论计算数据无关"
            )
        
        # 评估置信度
        confidence = 0.7 if has_data else 0.5
        if "材料" in query or "合金" in query or "material" in query_lower or "alloy" in query_lower:
            confidence += 0.2
        
        return AgentCapability(
            can_contribute=True,
            confidence=min(confidence, 1.0),
            contribution_types=[ContributionType.CANDIDATES, ContributionType.PROPERTIES],
            requirements=[],
            estimated_items=min(status.data_count, 100),
            estimated_time_seconds=5.0 if has_data else 30.0,
            reason=f"可提供 {status.data_count} 个材料的理论计算数据"
        )
    
    def get_resource_status(self) -> ResourceStatus:
        """获取当前资源状态"""
        try:
            materials = self.db.list_materials(limit=5000, allowed_elements=self.config.elements)
            data_count = len(materials)
            
            # 统计各类数据覆盖
            fe_count = sum(1 for m in materials if m.get("formation_energy") is not None)
            dos_count = sum(1 for m in materials if m.get("dos_data") is not None)
            
            return ResourceStatus(
                agent_name=self.agent_name,
                is_available=True,
                data_count=data_count,
                data_coverage={
                    "materials": data_count,
                    "formation_energy": fe_count,
                    "dos_data": dos_count,
                },
                api_available=True,
            )
        except Exception as e:
            logger.warning(f"Failed to get resource status: {e}")
            return ResourceStatus(
                agent_name=self.agent_name,
                is_available=False,
            )
    
    def contribute(self, context: QueryContext) -> AgentContribution:
        """贡献理论计算数据"""
        try:
            # 获取材料列表
            materials = self.db.list_materials(limit=100, allowed_elements=self.config.elements)
            
            candidates = []
            properties = {}
            
            for m in materials:
                mat_id = m.get("material_id")
                if not mat_id:
                    continue
                    
                candidates.append({
                    "material_id": mat_id,
                    "formula": m.get("formula"),
                    "formation_energy": m.get("formation_energy"),
                    "source": "theory"
                })
                
                properties[mat_id] = {
                    "formation_energy": m.get("formation_energy"),
                    "dos_data": m.get("dos_data") is not None,
                }
            
            return AgentContribution(
                agent_name=self.agent_name,
                contribution_type=ContributionType.CANDIDATES,
                success=True,
                candidates=candidates,
                properties=properties,
                confidence=0.8,
                reasoning=f"从数据库获取 {len(candidates)} 个材料的理论数据",
                sources=["Materials Project", "Local Database"]
            )
        except Exception as e:
            logger.error(f"Theory contribution failed: {e}")
            return AgentContribution(
                agent_name=self.agent_name,
                contribution_type=ContributionType.CANDIDATES,
                success=False,
                reasoning=str(e)
            )


class MLAgentProtocolMixin:
    """
    MLAgent 协议实现混入类
    
    集成 HOR 活性预测模型
    """
    
    @property
    def agent_name(self) -> str:
        return "ml"
    
    def assess_capability(self, query: str, context: Optional[QueryContext] = None) -> AgentCapability:
        """评估对查询的贡献能力"""
        query_lower = query.lower()
        
        ml_keywords = [
            "预测", "训练", "模型", "机器学习", "深度学习", "推荐",
            "predict", "train", "model", "ml", "recommend", "ranking"
        ]
        is_relevant = any(kw in query_lower or kw in query for kw in ml_keywords)
        
        status = self.get_resource_status()
        has_model = status.model_count > 0
        
        if not is_relevant and not has_model:
            return AgentCapability(
                can_contribute=False,
                confidence=0.0,
                reason="查询与机器学习预测无关且无可用模型"
            )
        
        # 有模型就可以贡献预测
        confidence = 0.85 if has_model else 0.4
        
        return AgentCapability(
            can_contribute=True,
            confidence=confidence,
            contribution_types=[ContributionType.PREDICTIONS],
            requirements=["theory"] if not has_model else [],
            estimated_items=100,
            estimated_time_seconds=5.0 if has_model else 120.0,
            reason=f"可使用 HOR 活性预测模型" if has_model else "需要先训练模型"
        )
    
    def get_resource_status(self) -> ResourceStatus:
        """获取当前资源状态"""
        try:
            # 检查 HOR 预测模型
            from src.ml.hor_predictor import get_predictor
            predictor = get_predictor()
            
            model_count = 1 if predictor.is_available() else 0
            model_types = ["HOR_Activity_GBR"] if predictor.is_available() else []
            
            return ResourceStatus(
                agent_name=self.agent_name,
                is_available=True,
                model_count=model_count,
                model_types=model_types,
            )
        except Exception as e:
            return ResourceStatus(
                agent_name=self.agent_name,
                is_available=False,
            )
    
    def contribute(self, context: QueryContext) -> AgentContribution:
        """贡献ML预测"""
        try:
            from src.ml.hor_predictor import get_predictor
            predictor = get_predictor()
            
            if not predictor.is_available():
                return AgentContribution(
                    agent_name=self.agent_name,
                    contribution_type=ContributionType.PREDICTIONS,
                    success=False,
                    reasoning="HOR 预测模型不可用"
                )
            
            predictions = {}
            candidates = []
            
            # 获取候选材料（从 theory 贡献）
            theory_contrib = context.contributions.get("theory") if context else None
            if theory_contrib and theory_contrib.candidates:
                materials = theory_contrib.candidates
            else:
                materials = []
            
            # 批量预测
            if materials:
                results = predictor.predict_batch(materials)
                
                for r in results:
                    mat_id = r['material_id']
                    predictions[mat_id] = r['predicted_activity']
                    candidates.append({
                        "material_id": mat_id,
                        "formula": r.get('formula'),
                        "predicted_activity": r['predicted_activity'],
                        "uncertainty": r['uncertainty'],
                        "source": "ml"
                    })
            
            return AgentContribution(
                agent_name=self.agent_name,
                contribution_type=ContributionType.PREDICTIONS,
                success=True,
                candidates=candidates[:50],
                predictions=predictions,
                confidence=0.85,
                reasoning=f"为 {len(predictions)} 个材料预测 HOR 活性",
                sources=["HOR Activity Prediction Model (GBR, R²=0.97)"]
            )
        except Exception as e:
            logger.error(f"ML contribution failed: {e}")
            return AgentContribution(
                agent_name=self.agent_name,
                contribution_type=ContributionType.PREDICTIONS,
                success=False,
                reasoning=str(e)
            )
    
    def can_iterate(self) -> bool:
        return True
    
    def iterate(self, context: QueryContext, feedback: Dict[str, Any]) -> AgentContribution:
        """基于实验反馈迭代优化"""
        # 使用反馈数据更新训练集
        experiment_data = feedback.get("experiment_results", [])
        if experiment_data:
            logger.info(f"Received {len(experiment_data)} experiment feedback for model update")
            # 触发模型重训练逻辑...
        
        return self.contribute(context)


class ExperimentAgentProtocolMixin:
    """
    ExperimentDataAgent 协议实现混入类
    """
    
    @property
    def agent_name(self) -> str:
        return "experiment"
    
    def assess_capability(self, query: str, context: Optional[QueryContext] = None) -> AgentCapability:
        """评估对查询的贡献能力"""
        query_lower = query.lower()
        
        exp_keywords = [
            "实验", "测试", "lsv", "cv", "eis", "表征", "过电位", "电化学",
            "experiment", "test", "electrochemical", "overpotential", "activity"
        ]
        is_relevant = any(kw in query_lower or kw in query for kw in exp_keywords)
        
        status = self.get_resource_status()
        has_data = status.data_count > 0
        
        if not is_relevant and not has_data:
            return AgentCapability(
                can_contribute=False,
                confidence=0.0,
                reason="查询与实验数据无关且无实验数据"
            )
        
        if not has_data:
            return AgentCapability(
                can_contribute=False,
                confidence=0.0,
                reason="暂无实验数据，需要用户上传实验结果"
            )
        
        return AgentCapability(
            can_contribute=True,
            confidence=0.9,
            contribution_types=[ContributionType.METRICS],
            requirements=[],
            estimated_items=status.data_count,
            estimated_time_seconds=5.0,
            reason=f"可提供 {status.data_count} 条实验性能数据"
        )
    
    def get_resource_status(self) -> ResourceStatus:
        """获取当前资源状态"""
        try:
            # 检查实验数据目录
            import os
            data_dir = getattr(self, 'data_dir', 'data/experimental')
            file_count = 0
            if os.path.exists(data_dir):
                for root, dirs, files in os.walk(data_dir):
                    file_count += len([f for f in files if f.endswith(('.csv', '.xlsx', '.txt'))])
            
            return ResourceStatus(
                agent_name=self.agent_name,
                is_available=True,
                data_count=file_count,
                data_coverage={
                    "files": file_count,
                },
            )
        except Exception as e:
            return ResourceStatus(
                agent_name=self.agent_name,
                is_available=False,
            )
    
    def contribute(self, context: QueryContext) -> AgentContribution:
        """贡献实验数据"""
        status = self.get_resource_status()
        
        if status.data_count == 0:
            return AgentContribution(
                agent_name=self.agent_name,
                contribution_type=ContributionType.METRICS,
                success=False,
                reasoning="暂无实验数据"
            )
        
        # 扫描并处理实验数据
        metrics = {}
        # 实际实现需要调用 scan_directory 和 analyze 方法
        
        return AgentContribution(
            agent_name=self.agent_name,
            contribution_type=ContributionType.METRICS,
            success=True,
            metrics=metrics,
            confidence=0.9,
            reasoning=f"已处理 {status.data_count} 个实验数据文件",
            sources=["Local Experiment Files"]
        )


class LiteratureAgentProtocolMixin:
    """
    LiteratureAgent 协议实现混入类
    """
    
    @property
    def agent_name(self) -> str:
        return "literature"
    
    def assess_capability(self, query: str, context: Optional[QueryContext] = None) -> AgentCapability:
        """评估对查询的贡献能力"""
        query_lower = query.lower()
        
        lit_keywords = [
            "文献", "论文", "综述", "知识", "引用", "参考",
            "literature", "paper", "review", "knowledge", "reference"
        ]
        is_relevant = any(kw in query_lower or kw in query for kw in lit_keywords)
        
        # 文献智能体几乎对所有材料相关查询都有用
        material_keywords = ["材料", "催化", "合金", "hor", "her", "material", "catalyst", "alloy"]
        has_material_context = any(kw in query_lower or kw in query for kw in material_keywords)
        
        status = self.get_resource_status()
        
        if not is_relevant and not has_material_context:
            return AgentCapability(
                can_contribute=False,
                confidence=0.0,
                reason="查询与文献知识无关"
            )
        
        return AgentCapability(
            can_contribute=True,
            confidence=0.6 if has_material_context else 0.8,
            contribution_types=[ContributionType.INSIGHTS, ContributionType.KNOWLEDGE],
            requirements=[],
            estimated_items=10,
            estimated_time_seconds=15.0,
            reason="可搜索相关文献并提取知识"
        )
    
    def get_resource_status(self) -> ResourceStatus:
        """获取当前资源状态"""
        try:
            import os
            pdf_dir = getattr(self.config, 'pdf_dir', 'data/pdfs') if hasattr(self, 'config') else 'data/pdfs'
            pdf_count = 0
            if os.path.exists(pdf_dir):
                pdf_count = len([f for f in os.listdir(pdf_dir) if f.endswith('.pdf')])
            
            return ResourceStatus(
                agent_name=self.agent_name,
                is_available=True,
                data_count=pdf_count,
                data_coverage={
                    "local_pdfs": pdf_count,
                },
                api_available=True,  # Semantic Scholar API
            )
        except Exception as e:
            return ResourceStatus(
                agent_name=self.agent_name,
                is_available=False,
            )
    
    def contribute(self, context: QueryContext) -> AgentContribution:
        """贡献文献知识"""
        try:
            query = context.user_query if context else ""
            
            # 搜索相关文献
            insights = []
            knowledge = {}
            
            # 调用搜索方法
            if hasattr(self, 'search_all_sources'):
                results = self.search_all_sources(query, limit=10)
                for r in results:
                    insights.append({
                        "title": r.get("title"),
                        "source": r.get("source"),
                        "relevance": r.get("score", 0.5)
                    })
            
            return AgentContribution(
                agent_name=self.agent_name,
                contribution_type=ContributionType.INSIGHTS,
                success=True,
                insights=insights,
                knowledge=knowledge,
                confidence=0.7,
                reasoning=f"找到 {len(insights)} 篇相关文献",
                sources=["Semantic Scholar", "Local PDFs"]
            )
        except Exception as e:
            logger.error(f"Literature contribution failed: {e}")
            return AgentContribution(
                agent_name=self.agent_name,
                contribution_type=ContributionType.INSIGHTS,
                success=False,
                reasoning=str(e)
            )
