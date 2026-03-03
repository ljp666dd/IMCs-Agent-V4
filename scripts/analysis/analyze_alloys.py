"""分析理论数据库中的合金数据"""
import sqlite3
import json
import re
from collections import defaultdict

conn = sqlite3.connect('data/imcs.db')
cursor = conn.cursor()

# 定义允许的元素
ALLOWED_ELEMENTS = {
    # 贵金属
    "Pt", "Pd", "Au", "Ag", "Ir", "Rh", "Ru", "Os",
    # 3d过渡金属
    "Ni", "Co", "Fe", "Cu", "Mn", "Cr", "V", "Ti", "Zn", "Sc",
    # 4d/5d过渡金属
    "Mo", "W", "Nb", "Ta", "Zr", "Hf", "Re", "Y",
    # 其他
    "Cd", "In", "Sn", "Ga", "Ge", "Al", "La", "Ce"
}

def extract_elements(formula):
    """从化学式提取元素"""
    elements = re.findall(r'([A-Z][a-z]?)', formula)
    return set(elements)

def is_valid_alloy(formula):
    """检查是否为有效合金（所有元素都在允许列表中）"""
    elements = extract_elements(formula)
    return elements.issubset(ALLOWED_ELEMENTS) and len(elements) >= 1

print('='*70)
print('理论数据库合金分析')
print('='*70)

# 加载所有材料
cursor.execute('SELECT material_id, formula, formation_energy, dos_data FROM materials')
all_materials = cursor.fetchall()

print(f'\n总材料数: {len(all_materials)}')

# 按元素数量分类
by_element_count = defaultdict(list)
valid_alloys = []

for mat in all_materials:
    mat_id, formula, fe, dos = mat
    if not formula:
        continue
    
    elements = extract_elements(formula)
    
    if elements.issubset(ALLOWED_ELEMENTS) and len(elements) >= 1:
        n_elements = len(elements)
        by_element_count[n_elements].append({
            'material_id': mat_id,
            'formula': formula,
            'elements': elements,
            'formation_energy': fe,
            'has_dos': dos is not None
        })
        valid_alloys.append(mat)

print(f'符合元素限定的材料: {len(valid_alloys)}')

# 统计
print('\n' + '='*70)
print('按元素数量分布')
print('='*70)

for n in sorted(by_element_count.keys()):
    materials = by_element_count[n]
    with_fe = sum(1 for m in materials if m['formation_energy'] is not None)
    with_dos = sum(1 for m in materials if m['has_dos'])
    
    element_name = {1: '纯金属', 2: '二元', 3: '三元', 4: '四元', 5: '五元'}.get(n, f'{n}元')
    
    print(f'\n{element_name}合金: {len(materials)} 种')
    print(f'  有形成能: {with_fe}')
    print(f'  有DOS: {with_dos}')
    
    # 显示样本
    print(f'  样本: ', end='')
    samples = [m['formula'] for m in materials[:5]]
    print(', '.join(samples))

# 统计各元素出现频率
print('\n' + '='*70)
print('元素出现频率 (Top 15)')
print('='*70)

element_freq = defaultdict(int)
for n, materials in by_element_count.items():
    for m in materials:
        for el in m['elements']:
            element_freq[el] += 1

sorted_freq = sorted(element_freq.items(), key=lambda x: -x[1])
for el, count in sorted_freq[:15]:
    bar = '█' * (count // 100)
    print(f'  {el:3}: {count:5} {bar}')

# 二元合金详细统计
print('\n' + '='*70)
print('二元合金组合统计 (Top 20)')
print('='*70)

binary_combos = defaultdict(int)
for m in by_element_count[2]:
    combo = tuple(sorted(m['elements']))
    binary_combos[combo] += 1

sorted_combos = sorted(binary_combos.items(), key=lambda x: -x[1])
for combo, count in sorted_combos[:20]:
    print(f'  {combo[0]}-{combo[1]}: {count}')

conn.close()
