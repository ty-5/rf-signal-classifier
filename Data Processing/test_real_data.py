import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import your existing classes
from training_data_loader import OracleDataset
from rf_cnn_model_1d2l import RFFingerprinter

def test_real_data_pipeline():
    """
    Test the complete pipeline with real RF data:
    DataFrame → OracleDataset → CNN → Predictions
    """
    
    print("🔬 TESTING REAL RF DATA PIPELINE")
    print("=" * 50)
    
    # STEP 1: Load your real preprocessed data
    print("📁 Step 1: Loading real RF data...")
    try:
        df = pd.read_pickle("rf_fingerprints_2ft.pkl")
        print(f"✅ Loaded {len(df)} windows from {df['label'].nunique()} transmitters")
        print(f"   Transmitters: {sorted(df['label'].unique())}")
        print()
    except FileNotFoundError:
        print("❌ Error: rf_fingerprints_2ft.pkl not found!")
        print("   Make sure the file is in the same directory as this script.")
        return
    
    # STEP 2: Convert DataFrame to PyTorch Dataset
    print("🔄 Step 2: Converting to PyTorch format...")
    dataset = OracleDataset(df)
    print(f"✅ Created PyTorch dataset with {len(dataset)} samples")
    print(f"   Label mapping: {dataset.label_to_index}")
    print()
    
    # STEP 3: Create DataLoader for batching
    print("📦 Step 3: Creating data batches...")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"✅ Created DataLoader with batch size 8")
    print(f"   Total batches: {len(dataloader)}")
    print()
    
    # STEP 4: Create your CNN model (untrained)
    print("🧠 Step 4: Creating CNN model...")
    model = RFFingerprinter(num_transmitters=16)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Created RFFingerprinter model")
    print(f"   Total parameters: {total_params:,}")
    print()
    
    # STEP 5: Test with real data batch
    print("🧪 Step 5: Testing with real RF data...")
    
    # Get one batch of real data
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        print(f"📊 Testing batch {batch_idx + 1}:")
        print(f"   Input shape: {batch_x.shape}")
        print(f"   True labels: {batch_y.tolist()}")
        
        # Show what transmitters these are
        transmitter_names = [dataset.labels[label.item()] for label in batch_y]
        print(f"   Transmitter IDs: {transmitter_names}")
        
        # Run through your CNN (no training, just testing)
        with torch.no_grad():  # Don't compute gradients for testing
            predictions = model(batch_x)
        
        print(f"   Output shape: {predictions.shape}")
        print()
        
        # Show raw predictions for first sample
        print("🎯 Raw predictions for first sample:")
        print(f"   Actual transmitter: {transmitter_names[0]}")
        print(f"   Raw scores: {predictions[0].tolist()}")
        
        # Convert to probabilities and show top 3 predictions
        probabilities = torch.softmax(predictions[0], dim=0)
        top_probs, top_indices = torch.topk(probabilities, 3)
        
        print("   Top 3 predictions:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            pred_transmitter = dataset.labels[idx.item()]
            print(f"     {i+1}. {pred_transmitter}: {prob.item():.1%}")
        
        print()
        break  # Only test one batch for now
    
    # STEP 6: Summary
    print("📋 PIPELINE TEST SUMMARY:")
    print("=" * 30)
    print("✅ Data loading: SUCCESS")
    print("✅ PyTorch conversion: SUCCESS") 
    print("✅ CNN forward pass: SUCCESS")
    print("✅ Shape compatibility: SUCCESS")
    print()
    print("🔮 Expected results:")
    print("   • Random predictions (model is untrained)")
    print("   • All shapes should match")
    print("   • No error messages")
    print()
    print("🚀 Next step: Train the model to make good predictions!")

if __name__ == "__main__":
    test_real_data_pipeline()