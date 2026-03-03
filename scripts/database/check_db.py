"""检查数据库结构"""
import sqlite3

conn = sqlite3.connect('data/imcs.db')
cursor = conn.cursor()

# 检查可用数据表
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print('可用数据表:', [t[0] for t in tables])

# 检查 materials 表
cursor.execute('SELECT COUNT(*) FROM materials')
mat_count = cursor.fetchone()[0]
print(f'materials 表: {mat_count} 条')

# 检查 experiments 表
cursor.execute('SELECT COUNT(*) FROM experiments')
exp_count = cursor.fetchone()[0]
print(f'experiments 表: {exp_count} 条')

# 查看 experiments 表结构
cursor.execute('PRAGMA table_info(experiments)')
columns = cursor.fetchall()
print('experiments 列:', [c[1] for c in columns])

# 查看 materials 表结构
cursor.execute('PRAGMA table_info(materials)')
mat_columns = cursor.fetchall()
print('materials 列:', [c[1] for c in mat_columns])

# 查看一些数据
cursor.execute('SELECT formula, data FROM materials LIMIT 2')
samples = cursor.fetchall()
for s in samples:
    print(f'  {s[0]}: {str(s[1])[:100]}...')

conn.close()
