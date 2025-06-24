import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset



# Import existing classes
from training_data_loader import OracleDataset
from rf_cnn_model_1d3l import RFFingerprinter

def train_rf_fingerprinter():
    """
    Complete training pipeline for RF fingerprinting CNN
    """
    
    print("🚀 RF FINGERPRINTING CNN TRAINING")
    print("=" * 50)
    
    # Create output directory for training results
    output_dir = "training_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}/")
    print()
    
    # STEP 1: Load and prepare data
    print("📁 Step 1: Loading RF data...")
    df = pd.read_pickle(r"oracle_rf_baseline.pkl")
    print(f"✅ Loaded {len(df)} windows from {df['label'].nunique()} transmitters")
    
    # Create dataset
    dataset = OracleDataset(df)
    print(f"✅ Created dataset with {len(dataset)} samples")
    print(f"   Transmitter mapping: {dataset.label_to_index}")
    print()
    
    # STEP 2: Train/Test Split
    print("✂️ Step 2: Creating train/test split...")
    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size  # 20% for testing
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Testing samples: {len(test_dataset)}")
    print()
    
    # STEP 3: Create data loaders
    print("📦 Step 3: Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"✅ Training batches: {len(train_loader)}")
    print(f"✅ Testing batches: {len(test_loader)}")
    print()
    
    # STEP 4: Create model, loss function, and optimizer
    print("🧠 Step 4: Setting up model and training components...")
    
    
    # Model
    model = RFFingerprinter(num_transmitters=16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {total_params:,} parameters")
    
    # Loss function (for classification)
    criterion = nn.CrossEntropyLoss()
    print("✅ Loss function: CrossEntropyLoss")
    
    # Optimizer (Adam is good for most cases)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    print("✅ Optimizer: Adam (learning rate = 0.0001)")
    print()
    
    # STEP 5: Training Loop
    print("🏋️ Step 5: Training the model...")
    
    # Training parameters
    num_epochs = 50
    train_losses = []
    train_accuracies = []
    
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Progress tracking
        start_time = time.time()
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_predictions / total_predictions
        epoch_time = time.time() - start_time
        
        # Store for plotting
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Loss: {epoch_loss:.4f} | "
              f"Accuracy: {epoch_accuracy:.2f}% | "
              f"Time: {epoch_time:.1f}s")
    
    print()
    print("✅ Training completed!")
    print()
    
    # STEP 6: Evaluate on test set
    print("📊 Step 6: Evaluating on test set...")
    
    model.eval()  # Set model to evaluation mode
    test_predictions = []
    test_targets = []
    test_loss = 0.0
    
    with torch.no_grad():  # Don't compute gradients for testing
        for data, targets in test_loader:
            outputs = model(data)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
    
    # Calculate test metrics
    test_loss = test_loss / len(test_loader)
    test_accuracy = accuracy_score(test_targets, test_predictions)
    
    print(f"✅ Test Loss: {test_loss:.4f}")
    print(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print()
    
    # STEP 7: Detailed Results Analysis
    print("🔍 Step 7: Detailed results analysis...")
    
    # Per-transmitter accuracy
    cm = confusion_matrix(test_targets, test_predictions)
    transmitter_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    print("Per-transmitter accuracy:")
    for i, (tx_id, accuracy) in enumerate(zip(dataset.labels, transmitter_accuracies)):
        print(f"  {tx_id}: {accuracy:.2%}")
    
    print()
    
    # Best and worst performing transmitters
    best_idx = np.argmax(transmitter_accuracies)
    worst_idx = np.argmin(transmitter_accuracies)
    
    print(f"🏆 Best performing: {dataset.labels[best_idx]} ({transmitter_accuracies[best_idx]:.2%})")
    print(f"😓 Worst performing: {dataset.labels[worst_idx]} ({transmitter_accuracies[worst_idx]:.2%})")
    print()
    
    # STEP 8: Plot training progress
    print("📈 Step 8: Plotting training progress...")
    
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_progress.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    #plt.show()
    
    print(f"✅ Training plots saved as '{plot_path}'")
    print()
    
    # STEP 9: Save the trained model
    print("💾 Step 9: Saving trained model...")
    
    # Save model weights
    weights_path = os.path.join(output_dir, 'rf_fingerprinter_weights.pth')
    torch.save(model.state_dict(), weights_path)
    
    # Save label mapping for future use
    label_info = {
        'label_to_index': dataset.label_to_index,
        'labels': dataset.labels,
        'test_accuracy': test_accuracy
    }
    
    import pickle
    info_path = os.path.join(output_dir, 'model_info.pkl')
    with open(info_path, 'wb') as f:
        pickle.dump(label_info, f)
    
    print(f"✅ Model weights saved as '{weights_path}'")
    print(f"✅ Model info saved as '{info_path}'")
    print()
    
    # STEP 10: Final Summary
    print("🎯 TRAINING SUMMARY:")
    print("=" * 30)
    print(f"Final Test Accuracy: {test_accuracy:.2%}")
    print(f"Total Training Time: {sum(train_losses)/len(train_losses)*num_epochs:.1f}s")
    print(f"Model Parameters: {total_params:,}")
    print(f"Training Samples: {len(train_dataset)}")
    print(f"Test Samples: {len(test_dataset)}")
    print()
    
    if test_accuracy > 0.8:
        print("🎉 Excellent! Your model is performing well!")
    elif test_accuracy > 0.6:
        print("👍 Good start! Consider training longer or tuning hyperparameters.")
    else:
        print("🤔 Model needs improvement. Try more epochs or different architecture.")
    
    print()
    print("Next steps:")
    print(f"- Find trained model in: {output_dir}/")
    print("- Load model with: model.load_state_dict(torch.load('training_results/rf_fingerprinter_weights.pth'))")
    print("- Use for predictions on new RF data")
    print("- Consider training on more distance folders for better generalization")

def load_trained_model(results_dir="training_results"):
    """
    Helper function to load a trained model for inference
    
    Args:
        results_dir: Directory containing the trained model files
    """

    # Load the model
    model = RFFingerprinter(num_transmitters=16)
    weights_path = os.path.join(results_dir, 'rf_fingerprinter_weights.pth')
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    # Load label mapping
    import pickle
    info_path = os.path.join(results_dir, 'model_info.pkl')
    with open(info_path, 'rb') as f:
        label_info = pickle.load(f)
    
    return model, label_info

if __name__ == "__main__":
    train_rf_fingerprinter()
    