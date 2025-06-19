import pandas as pd
import numpy as np

print("Loading saved RF fingerprint data...")
df = pd.read_pickle(r"Data Processing/oracle_rf_baseline.pkl")

print("✅ Data loaded successfully!")
print(f"Total windows: {len(df)}")
print(f"Unique transmitters: {df['label'].nunique()}")
print("\nFirst few samples:")
print(df.head())

# Now we can work with 'df' for model development

print("\n" + "="*50)
print("DETAILED DATA VALIDATION:")
print("="*50)

# Test 1: Check data shapes
print(f"Shape of first window: {len(df.iloc[0]['real'])}")
print(f"Data type: {type(df.iloc[0]['real'])}")

# Test 2: Check all transmitters are present
print(f"\nAll transmitter IDs:")
for tx_id in sorted(df['label'].unique()):
    count = len(df[df['label'] == tx_id])
    print(f"  {tx_id}: {count} windows")

# Test 3: Verify actual IQ data
print(f"\nSample I values: {df.iloc[0]['real'][:5]}")
print(f"Sample Q values: {df.iloc[0]['imag'][:5]}")

# Test 4: Check memory usage
print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print("\n✅ All tests complete! Data is ready.")