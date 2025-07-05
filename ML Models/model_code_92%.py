import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import os

# Load dataset
df = pd.read_pickle('/kaggle/input/oracle-1-w-dlabel/oracle_rf_baseline.pkl') # change
print(f"✅ Loaded {len(df):,} samples")
print(f"Sample labels: {df['label'].unique()[:10]}")  # Should show distance labels
print(f"Total unique labels: {df['label'].nunique()}")  # Should be 176 (16×11)

# OracleDataset class
class OracleDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        
        # Map each transmitter ID to numerical index
        self.labels = sorted(self.df['label'].unique())
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        
        print(f"📊 Dataset created:")
        print(f"   Total samples: {len(self.df)}")
        print(f"   Unique transmitters: {len(self.labels)}")
        print(f"   Transmitters: {self.labels}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Use float64
        real = torch.tensor(row['real'], dtype=torch.float64)
        imag = torch.tensor(row['imag'], dtype=torch.float64)
        
        # Stack I/Q data
        iq_data = torch.stack([real, imag])  # shape [2, window_size]
        
        # Get label index
        label_str = row['label']
        label_index = self.label_to_index[label_str]
        
        return iq_data, label_index
    
# RFFingerprinter class
class RFFingerprinter(nn.Module):
    def __init__(self, num_classes=176, input_channels=2, input_length=1024):
        super(RFFingerprinter, self).__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_length = input_length
        
        # architecture
        self.conv1 = nn.Conv1d(2, 16, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(64, num_classes) #change from num_transmitters to num_classes for new classes

        self.double()

    def forward(self, x):
        # Scale input
        x = x * 500  # scaling value
        
        # Conv layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Output
        x = self.fc_out(x)
        
        return x
    
    # Helper function for trans by dist acc
def get_transmitter_accuracy(predictions, targets, dataset_labels):
    """
    Calculate transmitter-only accuracy (ignoring distance)
    """
    pred_transmitters = []
    true_transmitters = []
    
    for pred_idx, true_idx in zip(predictions, targets):
        pred_label = dataset_labels[pred_idx]  # e.g., "3123D52_2ft"
        true_label = dataset_labels[true_idx]
        
        # Extract transmitter ID (part before underscore)
        pred_tx = pred_label.split('_')[0]  # "3123D52"
        true_tx = true_label.split('_')[0]
        
        pred_transmitters.append(pred_tx)
        true_transmitters.append(true_tx)
    
    # Calculate accuracy
    correct = sum(p == t for p, t in zip(pred_transmitters, true_transmitters))
    accuracy = correct / len(predictions)
    
    # Per-transmitter breakdown
    from collections import defaultdict
    tx_correct = defaultdict(int)
    tx_total = defaultdict(int)
    
    for pred_tx, true_tx in zip(pred_transmitters, true_transmitters):
        tx_total[true_tx] += 1
        if pred_tx == true_tx:
            tx_correct[true_tx] += 1
    
    tx_accuracies = {tx: tx_correct[tx]/tx_total[tx] for tx in tx_total.keys()}
    
    return accuracy, tx_accuracies