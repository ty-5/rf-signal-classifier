import pandas as pd
import json
import numpy as np
import os
from pathlib import Path

def test_all_transmitters(root_dir):
    """Get complete picture of all transmitters across all files"""
    root_path = Path(root_dir)
    
    all_transmitters = set()
    files_by_distance = {}
    
    for data_path in root_path.rglob("*.sigmf-data"):
        filename_parts = data_path.stem.split('_')
        transmitter_id = filename_parts[3]
        distance = data_path.parent.name
        
        all_transmitters.add(transmitter_id)
        
        if distance not in files_by_distance:
            files_by_distance[distance] = set()
        files_by_distance[distance].add(transmitter_id)
    
    print(f"Total unique transmitters: {len(all_transmitters)}")
    print(f"All transmitter IDs: {sorted(all_transmitters)}")
    print(f"\nBreakdown by distance:")
    for distance, transmitters in files_by_distance.items():
        print(f"  {distance}: {len(transmitters)} transmitters - {sorted(transmitters)}")

# Run this to see the full picture
test_all_transmitters(r"C:\Users\tyler\OneDrive\Desktop\ZuLeris\RF-FINGERPRINTING\ORACLE\Dataset-1")
