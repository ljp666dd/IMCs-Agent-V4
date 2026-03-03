"""分析实验数据来源"""
import sqlite3
import json

conn = sqlite3.connect('data/imcs.db')
cursor = conn.cursor()

print('='*60)
print('实验数据来源分析')
print('='*60)

# 1. 查看 experiments 表的样本
print('\n[1] experiments 表样本')
print('-'*60)
cursor.execute('SELECT id, name, type, material_id, raw_data_path, metadata FROM experiments LIMIT 10')
samples = cursor.fetchall()
for s in samples:
    mat_id = s[3] if s[3] else "null"
    print(f'ID={s[0]:3} name={s[1]:15} type={s[2]:8} mat_id={mat_id}')
    if s[4]:
        print(f'      raw_data_path: {s[4]}')
    if s[5]:
        try:
            meta = json.loads(s[5])
            print(f'      metadata: {list(meta.keys())}')
        except:
            pass

# 2. 统计实验类型
print('\n[2] 实验类型分布')
print('-'*60)
cursor.execute('SELECT type, COUNT(*) FROM experiments GROUP BY type')
types = cursor.fetchall()
for t in types:
    type_name = t[0] if t[0] else "unknown"
    print(f'  {type_name:20}: {t[1]} 条')

# 3. 查看原始数据路径
print('\n[3] 原始数据路径样本')
print('-'*60)
cursor.execute('SELECT DISTINCT raw_data_path FROM experiments WHERE raw_data_path IS NOT NULL LIMIT 5')
paths = cursor.fetchall()
for p in paths:
    print(f'  {p[0]}')

# 4. activity_metrics 来源
print('\n[4] activity_metrics 来源')
print('-'*60)
cursor.execute('SELECT source, COUNT(*) FROM activity_metrics GROUP BY source')
sources = cursor.fetchall()
for s in sources:
    source_name = s[0] if s[0] else "unknown"
    print(f'  {source_name:30}: {s[1]} 条')

# 5. 具体查看 results 内容
print('\n[5] 实验结果内容样本')
print('-'*60)
cursor.execute('SELECT name, results FROM experiments WHERE results IS NOT NULL LIMIT 3')
samples = cursor.fetchall()
for s in samples:
    print(f'\n  样本: {s[0]}')
    try:
        results = json.loads(s[1])
        for k, v in results.items():
            if k != 'jk_by_potential':  # 跳过大数组
                print(f'    {k}: {v}')
    except:
        print(f'    results: {str(s[1])[:100]}...')

conn.close()
