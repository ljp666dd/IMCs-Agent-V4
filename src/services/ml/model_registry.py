import os
import json
import shutil
import glob
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.core.logger import get_logger

logger = get_logger(__name__)

class ModelRegistry:
    """模型版本与注册表管理服务"""
    
    def __init__(self, registry_dir: str = "data/ml_agent/models"):
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)
        self.registry_file = os.path.join(self.registry_dir, "registry.json")
        self._load_registry()
        
    def _load_registry(self):
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    self.registry = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load model registry: {e}")
                self.registry = {}
        else:
            self.registry = {}
            
    def _save_registry(self):
        try:
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
            
    def register_model(self, name: str, filepath: str, metrics: Dict[str, float], hyperparameters: Dict[str, Any] = None) -> str:
        """
        注册新模型版本，并根据指标决定是否升级为 best。
        只保留最近的 N 个版本（默认 5 个）。
        """
        if name not in self.registry:
            self.registry[name] = {
                "best_version": None,
                "versions": {}
            }
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"v_{timestamp}"
        
        # 复制模型文件到 registry 目录
        ext = os.path.splitext(filepath)[1]
        dest_filename = f"{name}_{version_id}{ext}"
        dest_path = os.path.join(self.registry_dir, dest_filename)
        
        try:
            shutil.copy2(filepath, dest_path)
            logger.info(f"Copied model {filepath} to {dest_path}")
        except Exception as e:
            logger.error(f"Failed to copy model to registry: {e}")
            return version_id
            
        version_info = {
            "version_id": version_id,
            "filepath": dest_path,
            "metrics": metrics,
            "hyperparameters": hyperparameters or {},
            "created_at": datetime.now().isoformat()
        }
        
        self.registry[name]["versions"][version_id] = version_info
        
        # 判断是否为最优版本 (以 R2 为主要指标，或者 MSE 取最小)
        best_version = self.registry[name].get("best_version")
        is_best = False
        
        if not best_version:
            is_best = True
        else:
            best_metrics = self.registry[name]["versions"][best_version]["metrics"]
            # 默认假设 R2 越大越好，MAE/MSE 越小越好。这里简单用 R2 比较
            current_r2 = metrics.get("r2", -float('inf'))
            best_r2 = best_metrics.get("r2", -float('inf'))
            
            if current_r2 > best_r2:
                is_best = True
                
        if is_best:
            self.registry[name]["best_version"] = version_id
            # 创建/更新 symlink 或 copy 为 _best 文件
            best_symlink = os.path.join(self.registry_dir, f"{name}_best{ext}")
            try:
                shutil.copy2(dest_path, best_symlink)
                logger.info(f"Updated best model reference: {best_symlink}")
            except Exception as e:
                logger.error(f"Failed to update best model symlink: {e}")
        
        # 清理旧版本，只保留最近 5 个
        self._cleanup_old_versions(name, keep=5)
        self._save_registry()
        
        return version_id
        
    def _cleanup_old_versions(self, name: str, keep: int = 5):
        """清理冗余的旧模型版本"""
        versions = self.registry[name]["versions"]
        if len(versions) <= keep:
            return
            
        # 按创建时间排序
        sorted_versions = sorted(
            versions.values(), 
            key=lambda x: x["created_at"], 
            reverse=True
        )
        
        best_version = self.registry[name].get("best_version")
        
        # 保留前 keep 个，以及 best_version
        keep_ids = {v["version_id"] for v in sorted_versions[:keep]}
        if best_version:
            keep_ids.add(best_version)
            
        versions_to_delete = [vid for vid in versions if vid not in keep_ids]
        
        for vid in versions_to_delete:
            v_info = versions.pop(vid)
            # 删除物理文件
            try:
                if os.path.exists(v_info["filepath"]):
                    os.remove(v_info["filepath"])
                    logger.info(f"Deleted old model version file: {v_info['filepath']}")
            except Exception as e:
                logger.warning(f"Failed to delete old model file: {e}")
                
    def get_best_model(self, name: str) -> Optional[Dict[str, Any]]:
        """获取指定最佳模型的信息"""
        if name not in self.registry:
            return None
        best_vid = self.registry[name].get("best_version")
        if not best_vid:
            return None
        return self.registry[name]["versions"].get(best_vid)


# 单例
_registry = None

def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
