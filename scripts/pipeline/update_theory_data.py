"""
重新从理论计算数据库获取符合元素限定的1-5元合金材料数据

数据源:
1. Materials Project - 形成能、DOS、带隙、结构
2. Catalysis-Hub - H吸附能

元素限定: 33种金属元素
"""

import sqlite3
import json
import os
import time
import re
from typing import List, Dict, Set, Any
from collections import defaultdict

# 添加项目路径
import sys
sys.path.insert(0, '.')

from src.core.logger import get_logger

logger = get_logger(__name__)

# MP API Key
MP_API_KEY = os.getenv("MP_API_KEY", "")

# 定义允许的元素 (33种)
ALLOWED_ELEMENTS = [
    # 贵金属
    "Pt", "Pd", "Au", "Ag", "Ir", "Rh", "Ru", "Os",
    # 3d过渡金属
    "Ni", "Co", "Fe", "Cu", "Mn", "Cr", "V", "Ti", "Zn", "Sc",
    # 4d/5d过渡金属
    "Mo", "W", "Nb", "Ta", "Zr", "Hf", "Re", "Y",
    # 其他
    "Cd", "In", "Sn", "Ga", "Ge", "Al", "La", "Ce"
]

DB_PATH = 'data/imcs.db'


def extract_elements(formula: str) -> Set[str]:
    """从化学式提取元素"""
    elements = re.findall(r'([A-Z][a-z]?)', formula)
    return set(elements)


def is_valid_alloy(formula: str, min_elements: int = 1, max_elements: int = 5) -> bool:
    """检查是否为有效的1-5元合金"""
    elements = extract_elements(formula)
    n = len(elements)
    return (elements.issubset(set(ALLOWED_ELEMENTS)) and 
            min_elements <= n <= max_elements)


def clean_database():
    """清理不符合元素限定的材料"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("="*60)
    print("步骤1: 清理不符合元素限定的材料")
    print("="*60)
    
    # 获取所有材料
    cursor.execute('SELECT id, formula FROM materials')
    all_materials = cursor.fetchall()
    print(f"当前总材料数: {len(all_materials)}")
    
    # 找出要删除的材料
    to_delete = []
    to_keep = []
    
    for mat_id, formula in all_materials:
        if formula and is_valid_alloy(formula):
            to_keep.append(mat_id)
        else:
            to_delete.append(mat_id)
    
    print(f"符合条件保留: {len(to_keep)}")
    print(f"不符合条件删除: {len(to_delete)}")
    
    # 批量删除
    if to_delete:
        placeholders = ','.join('?' * len(to_delete))
        cursor.execute(f'DELETE FROM materials WHERE id IN ({placeholders})', to_delete)
        conn.commit()
        print(f"已删除 {len(to_delete)} 条不符合条件的材料")
    
    # 验证
    cursor.execute('SELECT COUNT(*) FROM materials')
    remaining = cursor.fetchone()[0]
    print(f"清理后剩余: {remaining}")
    
    conn.close()
    return remaining


def fetch_from_mp():
    """从 Materials Project 获取数据"""
    print("\n" + "="*60)
    print("步骤2: 从 Materials Project 获取数据")
    print("="*60)
    
    try:
        from mp_api.client import MPRester
    except ImportError:
        print("ERROR: mp-api 未安装, 跳过 MP 数据获取")
        return []
    
    api_key = MP_API_KEY
    if not api_key:
        print("ERROR: MP API Key 未配置")
        return []
    
    print(f"API Key: {api_key[:8]}...")
    print(f"元素范围: {len(ALLOWED_ELEMENTS)} 种")
    
    all_materials = []
    seen_ids = set()
    
    # 请求的字段
    fields = [
        "material_id", 
        "formula_pretty",
        "formation_energy_per_atom",
        "energy_above_hull",
        "band_gap",
        "volume",
        "density",
        "is_stable",
        "nelements",
    ]
    
    try:
        with MPRester(api_key) as mpr:
            # 策略: 按元素数量分批查询
            for n_elements in range(1, 6):  # 1-5元
                print(f"\n查询 {n_elements} 元合金...")
                
                # 按主要元素分组查询
                for i, el in enumerate(ALLOWED_ELEMENTS):
                    try:
                        # 查询包含该元素的材料
                        docs = mpr.materials.summary.search(
                            elements=[el],
                            num_elements=(n_elements, n_elements),
                            is_stable=True,
                            fields=fields,
                            chunk_size=100
                        )
                        
                        count = 0
                        for doc in docs:
                            mat_id = str(doc.material_id)
                            formula = doc.formula_pretty
                            
                            # 检查是否符合元素限定
                            if mat_id not in seen_ids and is_valid_alloy(formula):
                                seen_ids.add(mat_id)
                                all_materials.append({
                                    'material_id': mat_id,
                                    'formula': formula,
                                    'formation_energy': doc.formation_energy_per_atom,
                                    'energy_above_hull': doc.energy_above_hull,
                                    'band_gap': doc.band_gap,
                                    'volume': doc.volume,
                                    'density': doc.density,
                                    'is_stable': doc.is_stable,
                                    'n_elements': n_elements,
                                })
                                count += 1
                        
                        if count > 0:
                            print(f"  [{el}] +{count} (总计: {len(all_materials)})")
                        
                        time.sleep(0.1)  # 避免 API 限制
                        
                    except Exception as e:
                        logger.warning(f"Query for {el} failed: {e}")
                        continue
                
                print(f"{n_elements}元合金: 累计 {len(all_materials)} 种")
                
    except Exception as e:
        logger.error(f"MP 查询失败: {e}")
    
    print(f"\n总共获取: {len(all_materials)} 种材料")
    return all_materials


def save_to_database(materials: List[Dict]):
    """保存到数据库"""
    print("\n" + "="*60)
    print("步骤3: 保存到数据库")
    print("="*60)
    
    if not materials:
        print("没有新数据需要保存")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 获取已存在的 material_id
    cursor.execute('SELECT material_id FROM materials')
    existing_ids = set(row[0] for row in cursor.fetchall())
    print(f"已存在材料: {len(existing_ids)}")
    
    # 插入新材料
    new_count = 0
    update_count = 0
    
    for mat in materials:
        mat_id = mat['material_id']
        
        if mat_id in existing_ids:
            # 更新现有记录
            cursor.execute('''
                UPDATE materials 
                SET formation_energy = ?, 
                    formula = ?
                WHERE material_id = ?
            ''', (mat['formation_energy'], mat['formula'], mat_id))
            update_count += 1
        else:
            # 插入新记录
            cursor.execute('''
                INSERT INTO materials (material_id, formula, formation_energy)
                VALUES (?, ?, ?)
            ''', (mat_id, mat['formula'], mat['formation_energy']))
            new_count += 1
    
    conn.commit()
    
    # 验证
    cursor.execute('SELECT COUNT(*) FROM materials')
    total = cursor.fetchone()[0]
    
    print(f"新增材料: {new_count}")
    print(f"更新材料: {update_count}")
    print(f"数据库总计: {total}")
    
    conn.close()


def analyze_results():
    """分析获取结果"""
    print("\n" + "="*60)
    print("步骤4: 分析结果")
    print("="*60)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 按元素数量统计
    cursor.execute('SELECT formula FROM materials')
    all_formulas = cursor.fetchall()
    
    by_count = defaultdict(int)
    for (formula,) in all_formulas:
        if formula:
            n = len(extract_elements(formula))
            by_count[n] += 1
    
    print("\n按元素数量分布:")
    for n in sorted(by_count.keys()):
        name = {1: '纯金属', 2: '二元', 3: '三元', 4: '四元', 5: '五元'}.get(n, f'{n}元')
        print(f"  {name}: {by_count[n]}")
    
    # 数据完整性
    cursor.execute('SELECT COUNT(*) FROM materials WHERE formation_energy IS NOT NULL')
    with_fe = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM materials WHERE dos_data IS NOT NULL')
    with_dos = cursor.fetchone()[0]
    
    print(f"\n数据完整性:")
    print(f"  有形成能: {with_fe}")
    print(f"  有DOS: {with_dos}")
    
    conn.close()


def main():
    print("="*60)
    print("理论计算数据库更新")
    print(f"元素限定: {len(ALLOWED_ELEMENTS)} 种")
    print(f"合金范围: 1-5 元")
    print("="*60)
    
    # 步骤1: 清理
    remaining = clean_database()
    
    # 步骤2: 获取新数据
    materials = fetch_from_mp()
    
    # 步骤3: 保存
    save_to_database(materials)
    
    # 步骤4: 分析
    analyze_results()
    
    print("\n" + "="*60)
    print("数据更新完成!")
    print("="*60)


if __name__ == "__main__":
    main()
