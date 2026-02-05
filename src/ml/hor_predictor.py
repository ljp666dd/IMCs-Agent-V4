"""
HOR 活性预测模型 - 加载和推理

提供加载训练好的模型并进行预测的功能
"""

import os
import json
import numpy as np
import joblib
from typing import Dict, List, Any, Optional, Tuple

from src.core.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'models')


class HORActivityPredictor:
    """
    HOR 活性预测器
    
    加载训练好的模型并进行预测
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.loaded = False
        
        self._try_load_model()
    
    def _try_load_model(self):
        """尝试加载模型"""
        model_path = os.path.join(MODEL_DIR, 'hor_activity_model.joblib')
        scaler_path = os.path.join(MODEL_DIR, 'hor_activity_scaler.joblib')
        config_path = os.path.join(MODEL_DIR, 'hor_activity_config.json')
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}")
            return
        
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.feature_columns = config['feature_columns']
            
            self.loaded = True
            logger.info(f"HOR activity model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def is_available(self) -> bool:
        """模型是否可用"""
        return self.loaded and self.model is not None
    
    def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        """
        预测 HOR 活性
        
        Args:
            features: 特征字典，包含 formation_energy, avg_d_electrons 等
            
        Returns:
            (activity_score, uncertainty): 活性评分和不确定性估计
        """
        if not self.is_available():
            return 0.5, 0.5  # 返回默认值
        
        try:
            # 构造特征向量，处理缺失值
            fe = features.get('formation_energy')
            d_e = features.get('avg_d_electrons')
            en = features.get('avg_electronegativity')
            ar = features.get('avg_atomic_radius')
            
            # 检查必要特征是否存在
            if fe is None or d_e is None:
                return 0.5, 0.5
            
            # 使用默认值填充可能缺失的特征
            X = np.array([[
                float(fe) if fe is not None else 0,
                float(d_e) if d_e is not None else 8,
                float(en) if en is not None else 2.0,
                float(ar) if ar is not None else 130,
            ]])
            
            # 检查 NaN
            if np.isnan(X).any():
                return 0.5, 0.5
            
            # 标准化
            X_scaled = self.scaler.transform(X)
            
            # 预测
            prediction = self.model.predict(X_scaled)[0]
            
            # 不确定性估计（基于训练误差）
            uncertainty = 0.04  # 基于测试集 MAE
            
            return float(prediction), uncertainty
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.5, 0.5
    
    def predict_batch(self, materials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量预测
        
        Args:
            materials: 材料列表，每个材料包含 formula 和属性
            
        Returns:
            预测结果列表
        """
        if not self.is_available():
            return []
        
        results = []
        
        for mat in materials:
            # 提取特征
            features = self._extract_features_from_material(mat)
            
            if features:
                score, uncertainty = self.predict(features)
                
                results.append({
                    'material_id': mat.get('material_id', mat.get('formula')),
                    'formula': mat.get('formula'),
                    'predicted_activity': score,
                    'uncertainty': uncertainty,
                    'features': features,
                })
        
        # 按活性评分排序
        results.sort(key=lambda x: x['predicted_activity'], reverse=True)
        
        return results
    
    def _extract_features_from_material(self, material: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """从材料数据提取特征"""
        # 元素属性
        element_props = {
            'Pt': {'d_electrons': 9, 'electronegativity': 2.28, 'atomic_radius': 139},
            'Pd': {'d_electrons': 10, 'electronegativity': 2.20, 'atomic_radius': 137},
            'Ni': {'d_electrons': 8, 'electronegativity': 1.91, 'atomic_radius': 124},
            'Co': {'d_electrons': 7, 'electronegativity': 1.88, 'atomic_radius': 125},
            'Fe': {'d_electrons': 6, 'electronegativity': 1.83, 'atomic_radius': 126},
            'Cu': {'d_electrons': 10, 'electronegativity': 1.90, 'atomic_radius': 128},
            'Au': {'d_electrons': 10, 'electronegativity': 2.54, 'atomic_radius': 144},
            'Ir': {'d_electrons': 7, 'electronegativity': 2.20, 'atomic_radius': 136},
            'Rh': {'d_electrons': 8, 'electronegativity': 2.28, 'atomic_radius': 135},
            'Ru': {'d_electrons': 7, 'electronegativity': 2.20, 'atomic_radius': 134},
        }
        
        formula = material.get('formula', '')
        
        d_electrons = []
        electronegativity = []
        atomic_radius = []
        
        for el, props in element_props.items():
            if el in formula:
                d_electrons.append(props['d_electrons'])
                electronegativity.append(props['electronegativity'])
                atomic_radius.append(props['atomic_radius'])
        
        if not d_electrons:
            return None
        
        return {
            'formation_energy': material.get('formation_energy', 0),
            'avg_d_electrons': np.mean(d_electrons),
            'avg_electronegativity': np.mean(electronegativity),
            'avg_atomic_radius': np.mean(atomic_radius),
        }


# 全局预测器实例
_predictor = None

def get_predictor() -> HORActivityPredictor:
    """获取预测器实例"""
    global _predictor
    if _predictor is None:
        _predictor = HORActivityPredictor()
    return _predictor


def predict_hor_activity(material: Dict[str, Any]) -> Tuple[float, float]:
    """便捷预测函数"""
    predictor = get_predictor()
    features = predictor._extract_features_from_material(material)
    if features:
        return predictor.predict(features)
    return 0.5, 0.5
