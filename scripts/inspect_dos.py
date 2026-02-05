import gzip
import json
import numpy as np
import os
import matplotlib.pyplot as plt

DOS_FILE = 'data/theory/dos_raw/mp-126_dos.json.gz'

def inspect_mp126():
    with gzip.open(DOS_FILE, 'rt') as f:
        data = json.load(f)
        
    energies = np.array(data['energies'])
    d_band = np.array(data['d_band'])
    
    print(f"Energies range: {energies.min():.2f} to {energies.max():.2f} eV")
    print(f"Fermi level (subtracted): {data['efermi']:.2f} eV")
    print(f"d-band max density: {d_band.max():.2f}")
    
    # Calculate Center manually
    norm = np.trapz(d_band, energies)
    center = np.trapz(d_band * energies, energies) / norm
    print(f"Recalculated Center: {center:.2f} eV")
    
    # Check peak position
    peak_idx = np.argmax(d_band)
    print(f"Peak position: {energies[peak_idx]:.2f} eV")
    
    # Plot simple ASCII
    print("\nVisual Inspection:")
    indices = np.linspace(0, len(energies)-1, 20, dtype=int)
    for i in indices:
        e = energies[i]
        d = d_band[i]
        bar = '*' * int(d * 5)
        print(f"{e:6.2f} | {bar}")

if __name__ == "__main__":
    if os.path.exists(DOS_FILE):
        inspect_mp126()
    else:
        print(f"File not found: {DOS_FILE}")
