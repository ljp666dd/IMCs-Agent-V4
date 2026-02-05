"""
全量数据增强脚本 (v3.0)

功能:
1. 获取完整 DOS 数据 (Projected/Total)
2. 提取多重电子结构特征 (Center, Width, Skewness, Kurtosis, Filling, etc.)
3. 获取 H/OH 吸附能
4. 批量处理与断点续传

元素限定: 33种金属
"""

import sqlite3
import json
import os
import time
import gzip
import numpy as np
import requests
from typing import List, Dict, Any, Tuple
from datetime import datetime
from scipy.integrate import simpson

import sys
# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.logger import get_logger

logger = get_logger(__name__)

# 配置
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'imcs.db')
DOS_RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'theory', 'dos_raw')
MP_API_KEY = os.getenv("MP_API_KEY", "")

# 允许的元素
ALLOWED_ELEMENTS = [
    "Pt", "Pd", "Au", "Ag", "Ir", "Rh", "Ru", "Os",  # 贵金属
    "Ni", "Co", "Fe", "Cu", "Mn", "Cr", "V", "Ti", "Zn", "Sc",  # 3d
    "Mo", "W", "Nb", "Ta", "Zr", "Hf", "Re", "Y",  # 4d/5d
    "Cd", "In", "Sn", "Ga", "Ge", "Al", "La", "Ce"  # 其他
]

os.makedirs(DOS_RAW_DIR, exist_ok=True)


class DOSCalculator:
    """DOS 特征计算器"""
    
    @staticmethod
    def calculate_moments(energies: np.ndarray, densities: np.ndarray, 
                         center: float) -> Tuple[float, float, float]:
        """计算二、三、四阶矩 (Width, Skewness, Kurtosis)"""
        if np.sum(densities) == 0:
            return 0.0, 0.0, 0.0
            
        # 使用 Simpson 积分提高精度
        norm = simpson(densities, x=energies)
        if norm == 0:
            return 0.0, 0.0, 0.0
            
        # 2nd moment (Variance -> Width)
        var = simpson(densities * (energies - center)**2, x=energies) / norm
        width = np.sqrt(var) if var > 0 else 0.0
        
        if width == 0:
            return 0.0, 0.0, 0.0
            
        # 3rd moment (Skewness)
        m3 = simpson(densities * (energies - center)**3, x=energies) / norm
        skewness = m3 / (width**3)
        
        # 4th moment (Kurtosis)
        m4 = simpson(densities * (energies - center)**4, x=energies) / norm
        kurtosis = m4 / (width**4)
        
        return width, skewness, kurtosis

    @staticmethod
    def extract_features(energies: np.ndarray, 
                        total_dos: np.ndarray, 
                        d_dos: np.ndarray = None,
                        fermi_energy: float = 0.0) -> Dict[str, float]:
        """提取所有特征"""
        # 相对能量 E - Ef
        E = energies  # 假设传入的已经是相对能量或者单独处理
        
        features = {}
        
        # --- d-band 特征 (如果存在) ---
        if d_dos is not None and np.sum(d_dos) > 0:
            # 限制积分区间以消除高能噪声 (-15 eV 到 +10 eV)
            mask_e = (E >= -15) & (E <= 10)
            
            if np.any(mask_e):
                E_mask = E[mask_e]
                d_mask = d_dos[mask_e]
                
                # 使用 Simpson 积分
                norm_d = simpson(d_mask, x=E_mask)
                
                if norm_d > 0:
                    # 1. d-band center
                    center = simpson(d_mask * E_mask, x=E_mask) / norm_d
                    features['d_band_center'] = float(center)
                    
                    # 2. Moments
                    width, skew, kurt = DOSCalculator.calculate_moments(E_mask, d_mask, center)
                    features['d_band_width'] = float(width)
                    features['d_band_skewness'] = float(skew)
                    features['d_band_kurtosis'] = float(kurt)
                    
                    # 3. Filling (integral up to Fermi level 0.0)
                    mask_occ = E_mask <= 0.0
                    if np.any(mask_occ):
                        occ_d = simpson(d_mask[mask_occ], x=E_mask[mask_occ])
                        features['d_band_filling'] = float(occ_d / norm_d)
                    else:
                        features['d_band_filling'] = 0.0
                    
                    # 5. Edges
                    # 简单定义：密度下降到峰值的 5% 处的能量
                    threshold = np.max(d_mask) * 0.05
                    mask_sig = d_mask > threshold
                    if np.any(mask_sig):
                        features['d_band_upper_edge'] = float(np.max(E_mask[mask_sig]))
                        features['d_band_lower_edge'] = float(np.min(E_mask[mask_sig]))
        
        # --- 费米面特征 ---
        # 找到 E=0 附近的 DOS 值
        idx_f = np.argmin(np.abs(E))
        features['dos_at_fermi'] = float(total_dos[idx_f])
        
        if d_dos is not None:
             features['d_dos_at_fermi'] = float(d_dos[idx_f])
             
        # --- 带隙 (如果 MP 没给，可以自己估算，但 MP summary 已有 band_gap) ---
        # 这里只做 DOS 形状分析
        
        return features


def fetch_mp_dos(material_ids: List[str], batch_size: int = 50) -> Dict[str, Any]:
    """批量获取 DOS 数据"""
    results = {}
    
    try:
        from mp_api.client import MPRester
    except ImportError:
        logger.error("mp-api not installed")
        return {}

    logger.info(f"Connecting to MP API with key len={len(MP_API_KEY)}...")
    
    with MPRester(MP_API_KEY) as mpr:
        total = len(material_ids)
        for i in range(0, total, batch_size):
            batch_ids = material_ids[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}, items {i+1}-{min(i+batch_size, total)}")
            
            for mat_id in batch_ids:
                try:
                    # 获取 DOS
                    # 尝试获取 Projected DOS (更详细)
                    # 注意: get_dos_by_material_id 可能返回 CompleteDos 或 Dos
                    dos = mpr.get_dos_by_material_id(mat_id)
                    
                    if dos:
                        # 准备数据
                        efermi = dos.efermi
                        energies = dos.energies - efermi  # 相对能量
                        
                        # 总态密度
                        total_densities = dos.total.get_densities() if hasattr(dos, 'total') else \
                                          list(dos.densities.values())[0] # Fallback if simple DOS
                        if np.ndim(total_densities) > 1: # Deal with Spin up/down
                             # Sum spins
                             total_densities = np.sum(total_densities, axis=0)
                        
                        # d-band 态密度提取
                        d_densities = np.zeros_like(total_densities)
                        
                        # 如果有投影数据 (CompleteDos)
                        if hasattr(dos, 'get_element_spd_dos'):
                            try:
                                # 遍历结构中的所有元素
                                for element in dos.structure.composition.elements:
                                    # 获取该元素的轨道投影 DOS {OrbitalType: Dos}
                                    spd_dos = dos.get_element_spd_dos(element)
                                    
                                    for orbital_type, orb_dos in spd_dos.items():
                                        # 检查是否为 d 轨道 (转化为字符串判断，避免导入 Enum)
                                        if 'd' in str(orbital_type).lower():
                                            orb_val = orb_dos.get_densities()
                                            if np.ndim(orb_val) > 1:
                                                orb_val = np.sum(orb_val, axis=0)
                                            # 注意：需确保长度一致，通常是一致的
                                            if len(orb_val) == len(d_densities):
                                                d_densities += orb_val
                            except Exception as e:
                                logger.warning(f"Failed to extract orbital DOS for {mat_id}: {e}")

                        
                        # 计算特征
                        features = DOSCalculator.extract_features(
                            energies, total_densities, d_densities
                        )
                        
                        # 保存原始数据 (压缩 JSON)
                        raw_data = {
                            "material_id": mat_id,
                            "efermi": efermi,
                            "energies": energies.tolist(),
                            "total": total_densities.tolist(),
                            "d_band": d_densities.tolist(),
                            "features": features
                        }
                        
                        file_path = os.path.join(DOS_RAW_DIR, f"{mat_id}_dos.json.gz")
                        with gzip.open(file_path, 'wt', encoding='UTF-8') as f:
                            json.dump(raw_data, f)
                            
                        # 记录结果 (存入 DB 的部分)
                        results[mat_id] = {
                            "features": features,
                            "raw_path": file_path
                        }
                        logger.info(f"✓ {mat_id}: Center={features.get('d_band_center', 'NaN'):.2f}")
                        
                    else:
                        logger.warning(f"✗ {mat_id}: No DOS data found")
                        
                except Exception as e:
                    logger.error(f"Error fetching {mat_id}: {str(e)}")
                
                # Rate limit
                time.sleep(0.5)
                
    return results


def save_to_db(data: Dict[str, Any]):
    """保存特征到数据库"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    count = 0
    for mat_id, item in data.items():
        # 这里我们将特征合并到 dos_data 字段，或者创建一个新字段 raw_dos_path
        # 为了兼容，我们更新 dos_data 存 JSON
        
        # 1. 读取旧数据
        cursor.execute("SELECT dos_data FROM materials WHERE material_id=?", (mat_id,))
        row = cursor.fetchone()
        existing_data = json.loads(row[0]) if row and row[0] else {}
        
        # 2. 合并新特征
        new_features = item['features']
        new_features['raw_file'] = item['raw_path']
        existing_data.update(new_features)
        
        # 3. 更新
        cursor.execute("UPDATE materials SET dos_data=? WHERE material_id=?", 
                      (json.dumps(existing_data), mat_id))
        count += 1
        
    conn.commit()
    conn.close()
    logger.info(f"Saved {count} records to database.")


def get_candidate_ids(limit: int = 10, priority_elements: List[str] = None, exclude_existing: bool = True) -> List[str]:
    """从数据库获取待处理的 ID"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 构建查询
    query = "SELECT material_id, formula FROM materials WHERE material_id LIKE 'mp-%'"
    
    if exclude_existing:
        # 检查 dos_data 是否已经包含 'd_band_kurtosis' (新特征标志)
        query += " AND (dos_data IS NULL OR dos_data NOT LIKE '%d_band_kurtosis%')"
        
    logger.info(f"Executing query: {query}")
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    logger.info(f"Query returned {len(rows)} raw rows.")
    
    candidates = []
    
    # 本地过滤元素 (SQL LIKE 不够灵活)
    priority_set = set(priority_elements) if priority_elements else set(ALLOWED_ELEMENTS)
    
    for mid, formula in rows:
        # 提取元素
        # 简单正则提取
        import re
        elements = set(re.findall(r'([A-Z][a-z]?)', formula))
        
        # 必须是允许的元素
        if not elements.issubset(set(ALLOWED_ELEMENTS)):
            continue
            
        # 如果有优先级，检查是否包含优先级元素
        if priority_elements:
             if not elements.intersection(priority_set):
                 continue
        
        candidates.append(mid)
        
    return candidates[:limit]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full Data Fetcher")
    parser.add_argument("--limit", type=int, default=10, help="Batch size limit")
    parser.add_argument("--batch_type", type=str, default="test", choices=["test", "precious", "3d", "others"], help="Batch type")
    
    args = parser.parse_args()
    
    logger.info(f"Starting DOS fetcher. Batch type: {args.batch_type}, Limit: {args.limit}")
    
    # 确定优先级元素
    priority = None
    if args.batch_type == "precious":
        priority = ["Pt", "Pd", "Au", "Ag", "Ir", "Rh", "Ru", "Os"]
    elif args.batch_type == "3d":
        priority = ["Ni", "Co", "Fe", "Cu", "Mn", "Cr", "V", "Ti", "Zn", "Sc"]
        
    # 获取候选
    ids = get_candidate_ids(limit=args.limit, priority_elements=priority, exclude_existing=True)
    logger.info(f"Found {len(ids)} candidates for processing.")
    
    if not ids:
        logger.info("No candidates found. Exiting.")
        return
        
    # 执行下载
    results = fetch_mp_dos(ids, batch_size=10)
    
    # 保存
    if results:
        save_to_db(results)
    
    logger.info("Batch completed.")

if __name__ == "__main__":
    main()
