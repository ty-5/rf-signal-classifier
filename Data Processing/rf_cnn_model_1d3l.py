import torch
import torch.nn as nn
import torch.nn.functional as F

class RFFingerprinter(nn.Module):
    def __init__(self, num_transmitters=16, input_channels=2, input_length=1024):
        """
        RF Fingerprinting CNN with 3 convolutional layers for better discrimination
        
        Args:
            num_transmitters (int): Number of transmitter classes (16 for ORACLE)
            input_channels (int): Number of input channels (2 for I/Q data)
            input_length (int): Length of input signal (1024 from preprocessing)
        """
        super(RFFingerprinter, self).__init__()
        
        # Store parameters
        self.num_transmitters = num_transmitters
        self.input_channels = input_channels
        self.input_length = input_length
        
        # LAYER 1: First Convolutional Layer (Basic pattern detection)
        # Input shape: [batch_size, 2, 1024]
        self.conv1 = nn.Conv1d(
            in_channels=2,      # I and Q channels
            out_channels=16,    # 16 basic feature maps
            kernel_size=7,      # Look at 7 consecutive samples
            stride=1,           # Move 1 sample at a time
            padding=3           # Keep same length
        )
        # Output shape: [batch_size, 16, 1024]
        
        # LAYER 2: Second Convolutional Layer (Intermediate patterns)
        self.conv2 = nn.Conv1d(
            in_channels=16,     # From previous layer
            out_channels=32,    # More feature maps
            kernel_size=5,      # Smaller kernel for finer details
            stride=1,
            padding=2           # Keep same length
        )
        # Output shape: [batch_size, 32, 1024]
        
        # LAYER 3: Third Convolutional Layer (Complex transmitter signatures)
        self.conv3 = nn.Conv1d(
            in_channels=32,     # From previous layer
            out_channels=64,    # Even more feature maps for subtle differences
            kernel_size=3,      # Small kernel for fine-grained patterns
            stride=1,
            padding=1           # Keep same length
        )
        # Output shape: [batch_size, 64, 1024]
        
        # LAYER 4: Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # Output shape: [batch_size, 64, 1]
        
        # Calculate size for fully connected layer
        self.fc_input_size = 64
        
        # LAYER 5: Output Layer
        self.fc_out = nn.Linear(
            in_features=self.fc_input_size,
            out_features=num_transmitters    # 16 transmitter classes
        )
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, 2, 1024]
            
        Returns:
            output: Raw scores for each transmitter [batch_size, 16]
        """
        # Input: [batch_size, 2, 1024]
        
        # Scale up the tiny IQ values for better learning
        x = x * 100  # Convert ~0.01 range to ~1.0 range
        
        # First conv layer - Basic RF pattern detection
        x = self.conv1(x)           # -> [batch_size, 16, 1024]
        x = F.relu(x)               # Activation function
        
        # Second conv layer - Intermediate pattern combinations
        x = self.conv2(x)           # -> [batch_size, 32, 1024]
        x = F.relu(x)
        
        # Third conv layer - Complex transmitter-specific signatures
        x = self.conv3(x)           # -> [batch_size, 64, 1024]
        x = F.relu(x)
        
        # Global pooling - Average across all time samples
        x = self.global_pool(x)     # -> [batch_size, 64, 1]
        
        # Flatten for output layer
        x = x.view(x.size(0), -1)   # -> [batch_size, 64]
        
        # Direct to output classification
        x = self.fc_out(x)          # -> [batch_size, 16]
        
        return x

# Test function to verify architecture
def test_model():
    """Test the model with dummy data to verify shapes"""
    print("Testing RF Fingerprinter Architecture (3 Layers):")
    print("=" * 60)
    
    # Create model
    model = RFFingerprinter()
    
    # Create dummy input (like your real data)
    batch_size = 32
    dummy_input = torch.randn(batch_size, 2, 1024)  # Random IQ data
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():  # Don't compute gradients for testing
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: [32, 16]")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Show architecture progression
    print(f"\nArchitecture progression:")
    print(f"Input:  [batch, 2, 1024]")
    print(f"Conv1:  [batch, 16, 1024] (basic patterns)")
    print(f"Conv2:  [batch, 32, 1024] (intermediate patterns)")
    print(f"Conv3:  [batch, 64, 1024] (complex signatures)")
    print(f"Pool:   [batch, 64, 1]")
    print(f"Output: [batch, 16] (transmitter scores)")
    
    print("\n✅ Model architecture test successful!")
    print(f"📈 Parameter increase: {total_params:,} vs 3,360 (2-layer)")
    
    return model

if __name__ == "__main__":
    # Run test when file is executed
    model = test_model()