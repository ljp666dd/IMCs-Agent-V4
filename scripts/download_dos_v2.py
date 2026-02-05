"""
使用正确的 MP API 下载 DOS 数据
"""

import sqlite3
import json
import os
import time
import numpy as np
from typing import List, Dict, Any

import sys
sys.path.insert(0, '.')

from src.core.logger import get_logger

logger = get_logger(__name__)

DB_PATH = 'data/imcs.db'
MP_API_KEY = os.getenv("MP_API_KEY", "")


def calculate_dos_descriptors(energies: np.ndarray, d_dos: np.ndarray, 
                               total_dos: np.ndarray) -> Dict[str, float]:
    """
    从 DOS 数据计算多种描述符
    """
    E = np.array(energies)
    D_d = np.array(d_dos)
    D_total = np.array(total_dos)
    
    descriptors = {}
    
    # 1. d-band center (一阶矩)
    mask_d = (E >= -10) & (E <= 0)
    if np.any(mask_d) and np.sum(D_d[mask_d]) > 0:
        descriptors['d_band_center'] = float(np.sum(E[mask_d] * D_d[mask_d]) / np.sum(D_d[mask_d]))
    
    # 2. d-band width (二阶矩的平方根)
    if 'd_band_center' in descriptors and np.sum(D_d[mask_d]) > 0:
        dc = descriptors['d_band_center']
        descriptors['d_band_width'] = float(np.sqrt(np.sum((E[mask_d] - dc)**2 * D_d[mask_d]) / np.sum(D_d[mask_d])))
    
    # 3. d-band filling
    mask_occupied = E <= 0
    total_d = np.trapz(D_d, E)
    occupied_d = np.trapz(D_d[mask_occupied], E[mask_occupied]) if np.any(mask_occupied) else 0
    if total_d > 0:
        descriptors['d_band_filling'] = float(occupied_d / total_d)
    
    # 4. Fermi 能级态密度
    fermi_idx = np.argmin(np.abs(E))
    descriptors['dos_at_fermi'] = float(D_total[fermi_idx])
    
    # 5. d-band 上边缘
    if np.any(mask_d) and np.sum(D_d[mask_d]) > 0:
        # 找到 d-band 态密度显著的最高能量
        threshold = np.max(D_d) * 0.1
        d_significant = (D_d > threshold) & (E <= 2)
        if np.any(d_significant):
            descriptors['d_band_upper_edge'] = float(np.max(E[d_significant]))
    
    return descriptors


def download_dos_correct():
    """使用正确的 MP API 下载 DOS"""
    print("="*60)
    print("使用正确的 MP API 下载 DOS")
    print("="*60)
    
    try:
        from mp_api.client import MPRester
    except ImportError:
        print("ERROR: mp-api 未安装")
        return {}
    
    if not MP_API_KEY:
        print("ERROR: MP API Key 未配置")
        return {}
    
    # 获取需要下载的材料
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT material_id FROM materials 
        WHERE dos_data IS NULL 
        AND material_id LIKE "mp-%"
        LIMIT 100
    ''')
    material_ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    print(f"待下载材料数: {len(material_ids)}")
    
    dos_data = {}
    success_count = 0
    fail_count = 0
    
    try:
        with MPRester(MP_API_KEY) as mpr:
            for i, mat_id in enumerate(material_ids):
                try:
                    # 使用正确的旧 API 方法
                    dos = mpr.get_dos_by_material_id(mat_id)
                    
                    if dos is not None:
                        efermi = dos.efermi
                        energies = dos.energies - efermi
                        
                        # 提取 d 轨道 DOS
                        s_dos = np.zeros(len(energies))
                        p_dos = np.zeros(len(energies))
                        d_dos = np.zeros(len(energies))
                        
                        pdos = dos.get_element_dos()
                        for element, element_dos in pdos.items():
                            for spin, densities in element_dos.densities.items():
                                # 启发式权重
                                s_dos += densities * 0.1
                                p_dos += densities * 0.3
                                d_dos += densities * 0.6
                        
                        total_dos = s_dos + p_dos + d_dos
                        
                        # 计算描述符
                        descriptors = calculate_dos_descriptors(energies, d_dos, total_dos)
                        
                        if descriptors:
                            dos_data[mat_id] = descriptors
                            success_count += 1
                            dc = descriptors.get('d_band_center')
                            if dc:
                                print(f"  ✓ {mat_id}: d-band center = {dc:.3f} eV")
                            else:
                                print(f"  ✓ {mat_id}")
                    else:
                        fail_count += 1
                        
                except Exception as e:
                    fail_count += 1
                    logger.debug(f"DOS for {mat_id} failed: {e}")
                
                # 进度
                if (i + 1) % 20 == 0:
                    print(f"  进度: {i+1}/{len(material_ids)}, 成功: {success_count}")
                
                time.sleep(0.2)  # 避免 API 限制
                
    except Exception as e:
        logger.error(f"MP DOS 下载失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n成功获取: {success_count}")
    print(f"失败: {fail_count}")
    
    return dos_data


def save_dos_to_database(dos_data: Dict[str, Dict]):
    """保存 DOS 描述符到数据库"""
    if not dos_data:
        print("没有 DOS 数据需要保存")
        return
    
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
    print(f"有 DOS 描述符的材料总数: {total_with_dos}")
    
    # 显示样本
    cursor.execute('SELECT material_id, dos_data FROM materials WHERE dos_data IS NOT NULL LIMIT 3')
    samples = cursor.fetchall()
    print("\n样本:")
    for mat_id, dos_json in samples:
        data = json.loads(dos_json)
        dc = data.get('d_band_center')
        print(f"  {mat_id}: d_band_center = {dc:.3f} eV" if dc else f"  {mat_id}")
    
    conn.close()


def main():
    print("="*60)
    print("DOS 数据下载 (修正版)")
    print("="*60)
    
    # 下载 DOS
    dos_data = download_dos_correct()
    
    # 保存
    save_dos_to_database(dos_data)
    
    print("\n完成!")


if __name__ == "__main__":
    main()
