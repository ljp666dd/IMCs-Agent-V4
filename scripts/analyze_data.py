"""分析可用训练数据"""
import sqlite3
import json

conn = sqlite3.connect('data/imcs.db')
cursor = conn.cursor()

print("="*60)
print("可用训练数据分析")
print("="*60)

# 1. materials 表
print("\n[1] materials 表结构")
print("-"*60)
cursor.execute('PRAGMA table_info(materials)')
columns = cursor.fetchall()
for c in columns:
    print(f"  {c[1]:20} {c[2]:15} {'(PK)' if c[5] else ''}")

cursor.execute('SELECT COUNT(*) FROM materials')
print(f"\n  总记录数: {cursor.fetchone()[0]}")

cursor.execute('SELECT COUNT(*) FROM materials WHERE formation_energy IS NOT NULL')
print(f"  有formation_energy: {cursor.fetchone()[0]}")

# 看一些样本
cursor.execute('SELECT material_id, formula, formation_energy FROM materials LIMIT 5')
samples = cursor.fetchall()
print("\n  样本数据:")
for s in samples:
    print(f"    {s[0]:15} {s[1]:20} fe={s[2]}")

# 2. experiments 表
print("\n[2] experiments 表结构")
print("-"*60)
cursor.execute('PRAGMA table_info(experiments)')
columns = cursor.fetchall()
for c in columns:
    print(f"  {c[1]:20} {c[2]:15}")

cursor.execute('SELECT COUNT(*) FROM experiments')
print(f"\n  总记录数: {cursor.fetchone()[0]}")

# 看一些样本
cursor.execute('SELECT id, name, type, results FROM experiments LIMIT 3')
samples = cursor.fetchall()
print("\n  样本数据:")
for s in samples:
    print(f"    ID={s[0]}, name={s[1]}, type={s[2]}")
    if s[3]:
        try:
            results = json.loads(s[3])
            print(f"      results keys: {list(results.keys())[:5]}")
        except:
            print(f"      results: {str(s[3])[:50]}...")

# 3. activity_metrics 表
print("\n[3] activity_metrics 表结构")
print("-"*60)
cursor.execute('PRAGMA table_info(activity_metrics)')
columns = cursor.fetchall()
for c in columns:
    print(f"  {c[1]:20} {c[2]:15}")

cursor.execute('SELECT COUNT(*) FROM activity_metrics')
print(f"\n  总记录数: {cursor.fetchone()[0]}")

# 4. adsorption_energies 表
print("\n[4] adsorption_energies 表结构")
print("-"*60)
cursor.execute('PRAGMA table_info(adsorption_energies)')
columns = cursor.fetchall()
for c in columns:
    print(f"  {c[1]:20} {c[2]:15}")

cursor.execute('SELECT COUNT(*) FROM adsorption_energies')
print(f"\n  总记录数: {cursor.fetchone()[0]}")

# 5. 知识图谱
print("\n[5] knowledge_entities 表结构")
print("-"*60)
cursor.execute('PRAGMA table_info(knowledge_entities)')
columns = cursor.fetchall()
for c in columns:
    print(f"  {c[1]:20} {c[2]:15}")

cursor.execute('SELECT COUNT(*) FROM knowledge_entities')
print(f"\n  总记录数: {cursor.fetchone()[0]}")

conn.close()
