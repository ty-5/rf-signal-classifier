'''
This is file replicates ORACLE's CNN implementation to create an RF fingerprinting model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import configs

class CNNFingerprinter(nn.Module):
    def __init__(self, num_transmitters=16, input_channels=2, input_window=128):
        super(CNNFingerprinter, self).__init__()
        
        self.datatype = configs.datatype
        
        #First convolution layer
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=50,
            kernel_size=7,
            groups=2,
            dtype=self.datatype
        )
        self.bn1 = nn.BatchNorm1d(
            num_features=50,
            dtype=self.datatype
        )
        
        #Second convolution layer
        self.conv2 = nn.Conv1d(
            in_channels=50,
            out_channels=50,
            kernel_size=7,
            groups=1,
            dtype=self.datatype
        )
        self.bn2 = nn.BatchNorm1d(
            num_features=50,
            dtype=self.datatype
        )
        
        #Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        #First linear lyaer
        self.fc1 = nn.Linear(
            in_features=50,
            out_features=256,
            dtype=self.datatype
        )
        self.dropout1 = nn.Dropout(p=0.5)
        
        #Second linear lyaer
        self.fc2 = nn.Linear(
            in_features=256,
            out_features=80,
            dtype=self.datatype
        )
        self.dropout2 = nn.Dropout(p=0.5)
        
        #Since the labels are integers from 0-15, softmax will be applied within CrossEntropy(), this linear
        #mapping will suffice in mapping our output to one of the transmitters
        #Softmax classification layer
        self.classifier = nn.Linear(
            in_features=80,
            out_features=num_transmitters,
            dtype=self.datatype
        )
        
    def forward(self, x):
        #Power normalization
        x = x * 100
        
        #First convolution layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        #Second convolution layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        #Pooling
        x = self.global_pool(x) #(batch, 50, 1)
        x = x.squeeze(2)        #(batch, 50)
        
        #First linear lyaer
        x = self.fc1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        
        #Second linear lyaer
        x = self.fc2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        
        #Softmax classification layer
        return self.classifier(x) #Logits for CrossEntropyLoss
        
        
        
    