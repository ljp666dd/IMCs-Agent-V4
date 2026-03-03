import sqlite3
import numpy as np
import json

DB_PATH = 'data/imcs.db'

def ascii_hist(data, bins=10, title="Histogram"):
    if not data:
        print(f"{title}: No data")
        return
        
    counts, bin_edges = np.histogram(data, bins=bins)
    print(f"\n{title} (N={len(data)}):")
    max_count = max(counts)
    if max_count == 0: return

    for i in range(len(counts)):
        bar_len = int(20 * counts[i] / max_count)
        bar = '#' * bar_len
        range_str = f"{bin_edges[i]:6.2f} - {bin_edges[i+1]:6.2f}"
        print(f"{range_str} | {bar} ({counts[i]})")

def validate():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Adsorption Energies (H*)
    print("--- Adsorption Energy (H*) Validation ---")
    cursor.execute("SELECT reaction_energy FROM adsorption_energies WHERE adsorbate LIKE 'H%' AND reaction_energy IS NOT NULL")
    h_energies = [r[0] for r in cursor.fetchall()]
    ascii_hist(h_energies, bins=15, title="H* Adsorption Energy (eV)")
    
    # Check for extreme outliers
    outliers = [e for e in h_energies if e < -3.0 or e > 1.0]
    if outliers:
        print(f"Warning: {len(outliers)} outliers found (<-3 or >1 eV). Sample: {outliers[:5]}")
    
    # 2. DOS Features (d-band center)
    print("\n--- DOS Features (d-band center) Validation ---")
    cursor.execute("SELECT dos_data FROM materials WHERE dos_data LIKE '%d_band_center%'")
    centers = []
    widths = []
    kurtosis = []
    
    for row in cursor.fetchall():
        try:
            d = json.loads(row[0])
            if 'd_band_center' in d: centers.append(d['d_band_center'])
            if 'd_band_width' in d: widths.append(d['d_band_width'])
            if 'd_band_kurtosis' in d: kurtosis.append(d['d_band_kurtosis'])
        except:
            pass
            
    ascii_hist(centers, bins=15, title="d-band Center (eV)")
    ascii_hist(widths, bins=10, title="d-band Width (eV)")
    ascii_hist(kurtosis, bins=10, title="d-band Kurtosis")

    # Validity check on Center
    # Expected: Mostly negative, around -1.0 to -5.0 eV for transition metals
    positive_centers = [c for c in centers if c > 0]
    if positive_centers:
        print(f"\nWarning: {len(positive_centers)} positive d-band centers found! (Physical impossibility for metals relative to Ef unless empty band)")
        print(f"Sample positive centers: {positive_centers[:5]}")
    else:
        print("\ncheck: All d-band centers are <= 0 eV (Correct).")

    conn.close()

if __name__ == "__main__":
    validate()
