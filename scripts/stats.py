"""数据统计"""
import sqlite3
import json
conn = sqlite3.connect('data/imcs.db')
cursor = conn.cursor()

print('='*60)
print('数据完整性统计')
print('='*60)

cursor.execute('SELECT COUNT(*) FROM materials')
total = cursor.fetchone()[0]
print(f'总材料数: {total}')

cursor.execute('SELECT COUNT(*) FROM materials WHERE formation_energy IS NOT NULL')
with_fe = cursor.fetchone()[0]
print(f'有形成能: {with_fe}')

cursor.execute('SELECT COUNT(*) FROM materials WHERE dos_data IS NOT NULL')
with_dos = cursor.fetchone()[0]
print(f'有DOS描述符 (Total): {with_dos}')

cursor.execute("SELECT COUNT(*) FROM materials WHERE dos_data LIKE '%d_band_kurtosis%'")
with_new_features = cursor.fetchone()[0]
print(f'有新特征 (Compact): {with_new_features}')

# DOS描述符样本
print()
print('DOS描述符样本 (New):')
cursor.execute("SELECT material_id, formula, dos_data FROM materials WHERE dos_data LIKE '%d_band_kurtosis%' LIMIT 5")
for row in cursor.fetchall():
    data = json.loads(row[2])
    dc = data.get('d_band_center')
    kurt = data.get('d_band_kurtosis')
    print(f'  {row[0]:12} {row[1]:15} d-band Center: {dc:.2f} eV, Kurtosis: {kurt:.2f}' if dc else f'  {row[0]}')


# H吸附能
print()
cursor.execute("SELECT COUNT(*) FROM adsorption_energies WHERE adsorbate = 'H'")
h_count = cursor.fetchone()[0]
print(f'H吸附能数据: {h_count}')

conn.close()
