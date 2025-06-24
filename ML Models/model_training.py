import torch
import torch.nn as nn
from training_data_loader import OracleDataset
from torch.utils.data import DataLoader, random_split
import pandas as pd
from rf_cnn_model_1d3l import RFFingerprinter
import torch.optim as optim
from torchviz import make_dot
import time


#Hyperparameters:
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

#Load the data
df = pd.read_pickle(r"oracle_rf_baseline.pkl")
dataset = OracleDataset(df)

training_data_size = int(0.8 * len(dataset))
testing_data_size = int(0.2 * len(dataset))

training_dataset, testing_dataset = random_split(dataset, lengths=[training_data_size, testing_data_size])

training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
testing_dataloader = DataLoader(testing_dataset, batch_size=BATCH_SIZE, shuffle=False)




def train_model(data: DataLoader, model, optimizer):
    model.train()
    current_loss = 0
    
    #Training loop
    for epoch in range(EPOCHS):
    
        start_time = time.time()
        #X represents features or the IQ_Data_Tensor from training_data_loader.py
        #y represents labels from training_data_loader.py
        for _, (X, y) in enumerate(data):
            X = X.to(device)
            y = y.to(device)    
            
            #Forward propagation
            prediction = model(X)
            
            #Compute the loss/error
            loss = loss_function(prediction, y)
            current_loss += loss.item()

            #Backpropagation
            loss.backward()
            
            optimizer.step()
            
            optimizer.zero_grad()
        
        #torchviz/graphviz has a problem with environmental paths
        #if epoch == 0:
            #dot = make_dot(prediction, params=dict(model.named_parameters()))
            #dot.format = 'png'
            #dot.render("model_visualization")
            
        
        #Keep track of how long each epoch takes
        epoch_time = time.time() - start_time
        
        print(f"Epoch = {epoch + 1}/{EPOCHS}, Loss = {(current_loss / len(data)):.6f}, Epoch time = {epoch_time:.2f} seconds")
        
        #Reset loss
        current_loss = 0

    print("Training Complete, saving weights.")
    torch.save(model.state_dict(), "CNN_weights.pth")


if __name__ == "__main__":
    
    model = RFFingerprinter().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_model(training_dataloader, model, optimizer)
    
    
    

