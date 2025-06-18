import pandas as pd
import json
import numpy as np
import os
from pathlib import Path
#from sigmf import SigMFFile#
# I dont think we need this import^

# should use ~ 1M spf & 512 stride
def load_sigmf_windows(root_dir, samples_per_file=50000, window_size=1024, stride=1024):
    root_path = Path(root_dir)
    windows = []

    for data_path in root_path.rglob("*.sigmf-data"):
        meta_path = data_path.with_suffix(".sigmf-meta")
        if not meta_path.exists():
            continue

        with open(meta_path, "r") as f:
            meta = json.load(f)

        # Always use np.complex128 due to dataset warning
        dtype = np.complex128

        # Load and truncate samples
        with open(data_path, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=dtype)
        raw = raw[:samples_per_file]

    
        # Label is the transmitter ID -- extract from file name
        filename_parts = data_path.stem.split('_')
        transmitter_id = filename_parts[3]
        label = transmitter_id
        # When we eventually use all distances at once, we can have a combined label of id + dist
        # Probably going to have to utilize batch processing for this...

        #distance = data_path.parent.name#
        #label = f"{transmitter_id}_{distance}"#

        # Extract windows
        # Consider using a stride of 512 (50% overlap) for more training exammples
        # May keep seperate columns as complex numbers for CNN input
        for start in range(0, len(raw) - window_size + 1, stride):
            segment = raw[start:start + window_size]
            windows.append({
                "real": segment.real,
                "imag": segment.imag,
                "label": label
            })

    return pd.DataFrame(windows)



df = load_sigmf_windows(r"C:\Users\tyler\OneDrive\Desktop\ZuLeris\RF-FINGERPRINTING\ORACLE\Dataset-1\2ft")
print(df.head())

print("Saving processed data...")
df.to_pickle("rf_fingerprints_2ft.pkl")
print("✅ Data saved successfully!")
print("You can now use the saved data without reprocessing!")

"""
#To explore the data --

print("DATASET OVERVIEW:")
print(f"Total windows: {len(df)}")
print(f"Unique transmitters: {df['label'].nunique()}")
print("Windows per transmitter:")
counts = df['label'].value_counts()
for transmitter, count in counts.head().items():
    print(f"  {transmitter}: {count}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print(f"Shape of first window: {len(df.iloc[0]['real'])}")
print(f"Data type of real part: {type(df.iloc[0]['real'])}")
print(f"Sample real values: {df.iloc[0]['real'][:5]}")

"""