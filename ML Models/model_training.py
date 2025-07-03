import torch
import torch.nn as nn
from training_data_loader import OracleDataset
from torch.utils.data import DataLoader, random_split
import pandas as pd
from rf_cnn_prototype import RFFingerprinter
import torch.optim as optim
from torchviz import make_dot
import time
import configs


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




def train_model(data: DataLoader, model, optimizer, loss_function, scheduler):
    model.train()
    current_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    #1. represents features or the IQ_Data_Tensor from training_data_loader.py
    #2. represents labels from training_data_loader.py
    for _, (signal_features, labels) in enumerate(data):
        
        signal_features = signal_features.to(device)
        labels = labels.to(device)    
        
        #Clear the gradients from the computational graph
        optimizer.zero_grad()
        
        #Forward propagation
        prediction = model(signal_features)
        
        #Compute loss
        loss = loss_function(prediction, labels)
        
        #Backward propagation
        loss.backward()
        
        #Update the weights
        optimizer.step()
        
        current_loss += loss.item()
        _, predicted_label = torch.max(prediction.detach(), 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted_label == labels).sum().item()
        
        
    return correct_predictions, total_predictions, current_loss


def test_model(data: DataLoader, model, loss_function):
    model.eval()
    current_test_loss = 0
    
    with torch.no_grad():
        for (signal, label) in data:
            signal = signal.to(device)
            label = label.to(device)
            
            output = model(signal)
            current_test_loss += loss_function(output, label).item()
    
    return current_test_loss
            

if __name__ == "__main__":
    
    model = RFFingerprinter().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', #When the quantity stops decreasing
        factor=0.5, #Lower it by 50%
        patience=5, #If the loss does not improve for 5 epochs, then lower learning rate
        verbose=True
    )
    
    #Training loop
    for epoch in range(EPOCHS):
        
        
        start_time = time.time()
        
        #One epoch of training
        correct_predictions, total_predictions, current_loss = train_model(training_dataloader, model, optimizer, loss_function, scheduler)
        
        #One epoch of testing - we'll test as we go to assess the testing loss
        current_test_loss = test_model(testing_dataloader, model, loss_function)
        
        #Track our learning data
        epoch_time = time.time() - start_time
        print(f"Epoch = {epoch + 1}/{EPOCHS}, Training Loss = {(current_loss / len(testing_dataloader)):.6f}, Epoch time = {epoch_time:.2f} seconds")
        training_accuracy = 100 * (correct_predictions/total_predictions)
        print(f"Training Accuracy = {training_accuracy}, Test Loss = {(current_test_loss / len(testing_dataloader)):.6f}")
        
        #Use the scheduler to update the learning rate with respect to the validation/test loss
        scheduler.step(current_test_loss)
    
    print("Training Complete, saving weights.")
    torch.save(model.state_dict(), "RF_Model_Weights.pth")
    
    
    
    

