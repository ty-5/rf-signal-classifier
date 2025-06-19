import pandas as pd
import json
import numpy as np
import os
from pathlib import Path
from typing import List

# should use ~ 1M spf & 512 stride
#Corrected to a stride of 512, using 500k samples per file for now.
def parse_sigmf_data(root_dir, samples_per_file=500000, window_size=1024, stride=512):
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

#This function will run the parse data function for all folders in the Oracle dataset directory
def collect_all_distances(root_dir: str, distances: List[str]) -> pd.DataFrame:
    root = Path(root_dir)
    all_data = []
    #Starting from the root and our list of all the distances, which are just strings to be appended to the root file path
    for d in distances:
        #Append the distance folder's name to the root, skip any nonexisting folders
        folder = root / d
        if not folder.exists():
            print(f"Skipping missing folder: {folder}")
            continue
        
        #Parse each folder successfully
        print(f"Parsing {folder.name}")
        df = parse_sigmf_data(folder)
        print(f"-> {len(df)} windows extracted")
        #Append the data frame created for each folder to our list of dataframes
        all_data.append(df)
        
    #Concatenate our list of all data into one dataframe and return it
    return pd.concat(all_data, ignore_index=True)

#This function will simply save dataframe into a pickle file
def save_dataset(df: pd.DataFrame, out_path: str):
    print(f"Saving {len(df)} samples to {out_path}...")
    #Save the dataframe into a pickle file, which is a byte stream optimized for storing data
    df.to_pickle(out_path)
    print("Complete")

distances = ["2ft", "8ft", "14ft", "20ft", "26ft", "32ft", "38ft", "44ft", "50ft", "56ft", "62ft"]
data_directory = r"C:\Users\adamm\PROJECTS\ZuLeris\KRI-16Devices-RawData"

df = collect_all_distances(data_directory, distances)
save_dataset(df, "oracle_rf_baseline.pkl")


#To explore the data --
'''
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
'''