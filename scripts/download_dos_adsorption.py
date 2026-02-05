"""
下载 DOS 数据并提取描述符
获取 Catalysis-Hub H吸附能数据

DOS 描述符包括:
1. d-band center (d带中心) - HOR活性关键
2. d-band width (d带宽度)
3. d-band filling (d带填充度)
4. Fermi能级态密度
5. 价带/导带边缘
"""

import sqlite3
import json
import os
import time
import numpy as np
from typing import List, Dict, Any, Optional
import requests

import sys
sys.path.insert(0, '.')

from src.core.logger import get_logger

logger = get_logger(__name__)

DB_PATH = 'data/imcs.db'
MP_API_KEY = os.getenv("MP_API_KEY", "")


def calculate_dos_descriptors(energies: np.ndarray, densities: np.ndarray, 
                               fermi_energy: float = 0.0) -> Dict[str, float]:
    """
    从 DOS 数据计算多种描述符
    
    Args:
        energies: 能量数组 (eV)
        densities: 态密度数组
        fermi_energy: 费米能 (eV)
    
    Returns:
        描述符字典
    """
    # 相对于费米能的能量
    E = energies - fermi_energy
    D = np.array(densities)
    
    # 确保数据有效
    if len(E) == 0 or len(D) == 0:
        return {}
    
    # 1. d-band center (一阶矩)
    # 通常取 -10 eV 到 Fermi 能级的区间
    mask_d = (E >= -10) & (E <= 0)
    if np.any(mask_d):
        E_d = E[mask_d]
        D_d = D[mask_d]
        if np.sum(D_d) > 0:
            d_band_center = np.sum(E_d * D_d) / np.sum(D_d)
        else:
            d_band_center = None
    else:
        d_band_center = None
    
    # 2. d-band width (二阶矩的平方根)
    if d_band_center is not None and np.any(mask_d):
        d_band_width = np.sqrt(np.sum((E_d - d_band_center)**2 * D_d) / np.sum(D_d))
    else:
        d_band_width = None
    
    # 3. d-band filling (填充度)
    # 占据态 vs 总态
    mask_occupied = E <= 0
    total_states = np.trapz(D, E)
    occupied_states = np.trapz(D[mask_occupied], E[mask_occupied]) if np.any(mask_occupied) else 0
    d_band_filling = occupied_states / total_states if total_states > 0 else None
    
    # 4. Fermi 能级态密度
    fermi_idx = np.argmin(np.abs(E))
    dos_at_fermi = float(D[fermi_idx]) if fermi_idx < len(D) else None
    
    # 5. 价带边缘 (Fermi 以下最高能量有态密度的位置)
    mask_valence = (E < 0) & (D > 0.1)
    if np.any(mask_valence):
        valence_band_edge = float(np.max(E[mask_valence]))
    else:
        valence_band_edge = None
    
    # 6. 导带边缘 (Fermi 以上最低能量有态密度的位置)
    mask_conduction = (E > 0) & (D > 0.1)
    if np.any(mask_conduction):
        conduction_band_edge = float(np.min(E[mask_conduction]))
    else:
        conduction_band_edge = None
    
    return {
        'd_band_center': d_band_center,
        'd_band_width': d_band_width,
        'd_band_filling': d_band_filling,
        'dos_at_fermi': dos_at_fermi,
        'valence_band_edge': valence_band_edge,
        'conduction_band_edge': conduction_band_edge,
    }


def download_dos_from_mp(material_ids: List[str], batch_size: int = 50):
    """从 Materials Project 下载 DOS 数据"""
    print("="*60)
    print("下载 DOS 数据")
    print("="*60)
    
    try:
        from mp_api.client import MPRester
    except ImportError:
        print("ERROR: mp-api 未安装")
        return {}
    
    if not MP_API_KEY:
        print("ERROR: MP API Key 未配置")
        return {}
    
    print(f"待下载材料数: {len(material_ids)}")
    
    dos_data = {}
    failed = []
    
    try:
        with MPRester(MP_API_KEY) as mpr:
            for i in range(0, len(material_ids), batch_size):
                batch = material_ids[i:i+batch_size]
                print(f"\n批次 {i//batch_size + 1}/{(len(material_ids)-1)//batch_size + 1}")
                
                for mat_id in batch:
                    try:
                        # 获取 DOS
                        dos_doc = mpr.materials.electronic_structure.get_dos_by_material_id(mat_id)
                        
                        if dos_doc and hasattr(dos_doc, 'total'):
                            # 提取总态密度
                            total_dos = dos_doc.total
                            energies = total_dos.energies
                            densities = total_dos.densities
                            fermi = dos_doc.efermi if hasattr(dos_doc, 'efermi') else 0
                            
                            # 计算描述符
                            descriptors = calculate_dos_descriptors(
                                np.array(energies), 
                                np.array(densities),
                                fermi
                            )
                            
                            if descriptors:
                                dos_data[mat_id] = descriptors
                                print(f"  ✓ {mat_id}: d-band={descriptors.get('d_band_center', 'N/A'):.2f} eV" 
                                      if descriptors.get('d_band_center') else f"  ✓ {mat_id}")
                        else:
                            failed.append(mat_id)
                            
                    except Exception as e:
                        failed.append(mat_id)
                        logger.debug(f"DOS for {mat_id} failed: {e}")
                    
                    time.sleep(0.1)  # 避免 API 限制
                
                print(f"  进度: {min(i+batch_size, len(material_ids))}/{len(material_ids)}")
                
    except Exception as e:
        logger.error(f"MP DOS 下载失败: {e}")
    
    print(f"\n成功获取: {len(dos_data)}")
    print(f"失败: {len(failed)}")
    
    return dos_data


def fetch_catalysis_hub_h_adsorption(limit: int = 500) -> List[Dict]:
    """从 Catalysis-Hub 获取 H 吸附能数据"""
    print("\n" + "="*60)
    print("获取 Catalysis-Hub H吸附能数据")
    print("="*60)
    
    url = "https://api.catalysis-hub.org/graphql"
    
    # 查询 H 吸附反应
    query = """
    {
      reactions(first: %d, reactants: "H") {
        edges {
          node {
            Equation
            reactionEnergy
            activationEnergy
            surfaceComposition
            facet
          }
        }
      }
    }
    """ % limit
    
    results = []
    
    try:
        response = requests.post(url, json={"query": query}, timeout=30)
        if response.status_code == 200:
            data = response.json()
            edges = data.get("data", {}).get("reactions", {}).get("edges", [])
            
            for edge in edges:
                node = edge.get("node", {})
                results.append({
                    "equation": node.get("Equation", ""),
                    "reaction_energy": node.get("reactionEnergy"),
                    "activation_energy": node.get("activationEnergy"),
                    "surface": node.get("surfaceComposition", ""),
                    "facet": node.get("facet", ""),
                })
            
            print(f"获取 {len(results)} 条 H 吸附能数据")
        else:
            print(f"请求失败: {response.status_code}")
            
    except Exception as e:
        logger.error(f"Catalysis-Hub 查询失败: {e}")
    
    return results


def save_dos_to_database(dos_data: Dict[str, Dict]):
    """保存 DOS 描述符到数据库"""
    print("\n" + "="*60)
    print("保存 DOS 描述符到数据库")
    print("="*60)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    update_count = 0
    
    for mat_id, descriptors in dos_data.items():
        dos_json = json.dumps(descriptors)
        cursor.execute('''
            UPDATE materials 
            SET dos_data = ?
            WHERE material_id = ?
        ''', (dos_json, mat_id))
        
        if cursor.rowcount > 0:
            update_count += 1
    
    conn.commit()
    
    # 验证
    cursor.execute('SELECT COUNT(*) FROM materials WHERE dos_data IS NOT NULL')
    total_with_dos = cursor.fetchone()[0]
    
    print(f"更新材料数: {update_count}")
    print(f"有 DOS 描述符的材料: {total_with_dos}")
    
    conn.close()


def save_adsorption_to_database(adsorption_data: List[Dict]):
    """保存吸附能数据到数据库"""
    print("\n" + "="*60)
    print("保存吸附能数据到数据库")
    print("="*60)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    insert_count = 0
    
    for item in adsorption_data:
        if item.get('reaction_energy') is not None:
            cursor.execute('''
                INSERT OR REPLACE INTO adsorption_energies 
                (material_id, surface_composition, facet, adsorbate, reaction_energy, activation_energy, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                item.get('surface', ''),
                item.get('surface', ''),
                item.get('facet', ''),
                'H',
                item.get('reaction_energy'),
                item.get('activation_energy'),
                'Catalysis-Hub'
            ))
            insert_count += 1
    
    conn.commit()
    
    # 验证
    cursor.execute('SELECT COUNT(*) FROM adsorption_energies WHERE adsorbate = "H"')
    total_h = cursor.fetchone()[0]
    
    print(f"新增吸附能记录: {insert_count}")
    print(f"H 吸附能总记录数: {total_h}")
    
    conn.close()


def main():
    print("="*60)
    print("DOS 和 H吸附能数据下载")
    print("="*60)
    
    # 获取需要下载 DOS 的材料
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT material_id FROM materials 
        WHERE dos_data IS NULL 
        AND material_id LIKE "mp-%"
        LIMIT 200
    ''')  # 先限制 200 个避免超时
    
    material_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"需要下载 DOS 的材料: {len(material_ids)}")
    
    # 步骤1: 下载 DOS
    dos_data = download_dos_from_mp(material_ids)
    
    # 步骤2: 保存 DOS
    if dos_data:
        save_dos_to_database(dos_data)
    
    # 步骤3: 获取 H 吸附能
    h_adsorption = fetch_catalysis_hub_h_adsorption(limit=500)
    
    # 步骤4: 保存吸附能
    if h_adsorption:
        save_adsorption_to_database(h_adsorption)
    
    # 总结
    print("\n" + "="*60)
    print("下载完成!")
    print("="*60)
    
    # 最终统计
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM materials WHERE dos_data IS NOT NULL')
    with_dos = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM adsorption_energies WHERE adsorbate = "H"')
    h_count = cursor.fetchone()[0]
    
    print(f"有 DOS 描述符的材料: {with_dos}")
    print(f"H 吸附能数据: {h_count}")
    
    conn.close()


if __name__ == "__main__":
    main()
