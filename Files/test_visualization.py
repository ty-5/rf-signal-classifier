import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_pickle("rf_fingerprints_2ft.pkl")

# Plot IQ data from 3 different transmitters
transmitters = df['label'].unique()[:3]

fig, axes = plt.subplots(3, 2, figsize=(12, 8))

for i, tx_id in enumerate(transmitters):
    sample = df[df['label'] == tx_id].iloc[0]
    
    # I/Q vs time
    axes[i, 0].plot(sample['real'][:100], label='I')
    axes[i, 0].plot(sample['imag'][:100], label='Q')
    axes[i, 0].set_title(f'TX {tx_id} - I/Q vs Time')
    axes[i, 0].legend()
    
    # Constellation plot
    axes[i, 1].scatter(sample['real'], sample['imag'], alpha=0.3, s=1)
    axes[i, 1].set_title(f'TX {tx_id} - Constellation')

plt.tight_layout()
plt.show()