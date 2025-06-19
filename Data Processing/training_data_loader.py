import torch
from torch.utils.data import Dataset
import pandas as pd

from torch.utils.data import DataLoader

class OracleDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        
        #Map each transmitter ID to an numerical index ex ['3123D7B'] becomes (0, '3123D7B')
        self.labels = sorted(self.df['label'].unique())
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
    
    def __len__(self):
        return len(self.df)
    
    #getitem is necessary for PyTorch to iterate over our dataset and acquire data in the form of (data_value, label_id)
    def __getitem__(self, idx):
        row = self.df.iloc[idx] #iloc = index-location, it returns a row at that index
        
        #Convert the real and imaginary columns from the DF into tensors
        real = torch.tensor(row['real'], dtype=torch.float32)
        imag = torch.tensor(row['imag'], dtype=torch.float32)
        
        #Stack the Interphase/Quadrature data tensors into one
        IQ_Data_tensor = torch.stack([real, imag]) #shape [2, WindowSize]
        
        #Convert the label column to indices
        label_str = row['label']
        label_index = self.label_to_index[label_str]
        
        #Return a tuple of both value the IQ data tensor and the label that goes with it
        return IQ_Data_tensor, label_index
    
    

#Unpickle the data: convert from byte stream to pandas dataframe
df = pd.read_pickle(r"Data Processing/oracle_rf_baseline.pkl")

dataset = OracleDataset(df)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

for batch_x, batch_y in dataloader:
    print("Batch Shape: ", batch_x.shape)
    print("Labels: ", batch_y)
    break;    
        











