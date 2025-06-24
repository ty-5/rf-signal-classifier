from model_training import testing_dataloader, loss_function
import torch
from torch.utils.data import DataLoader
from rf_cnn_model_1d3l import RFFingerprinter

device = "cuda" if torch.cuda.is_available else "cpu"

#Instantiate model
model = RFFingerprinter()

def test_model(model, dataloader: DataLoader):
    model.eval()
    with torch.no_grad():
        for (X, y) in dataloader:
            X = X.to(device), y = y.to(device)
            
            prediction = model(X)
            predicted_labels = torch.argmax(prediction, dim=1)
            test_loss += loss_function(prediction, y).item()
    
    return test_loss/len(testing_dataloader)
        
    
#INCOMPLETE  

#Testing loop


#Visuals