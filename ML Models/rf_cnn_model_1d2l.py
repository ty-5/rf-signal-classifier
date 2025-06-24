import torch
import torch.nn as nn
import torch.nn.functional as F

class RFFingerprinter(nn.Module):
    def __init__(self, num_transmitters=16, input_channels=2, input_length=1024):
        """
        RF Fingerprinting CNN for identifying transmitters from IQ data
        
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

        # LAYER 1: First Convolutional Layer
        # Input shape: [batch_size, 2, 1024]
        self.conv1 = nn.Conv1d(
            in_channels=2,      # I and Q channels
            out_channels=16,    # 16 feature maps (filters)
            kernel_size=7,      # Look at 7 consecutive samples
            stride=1,           # Move 1 sample at a time
            padding=3           # Keep same length: (7-1)/2 = 3
        )
        # Output shape: [batch_size, 16, 1024]

         # LAYER 2: Second Convolutional Layer
        self.conv2 = nn.Conv1d(
            in_channels=16,     # From previous layer
            out_channels=32,    # More feature maps
            kernel_size=5,      # Smaller kernel
            stride=1,
            padding=2           # Keep same length: (5-1)/2 = 2
        )
        # Output shape: [batch_size, 32, 1024]

        # LAYER 3: Global Average Pooling (simpler than multiple pools)
        # Will average across the entire (time) dimension
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # Output shape: [batch_size, 32, 1]
        
        # Calculate size for fully connected layer
        # After global pooling: 32 channels * 1 = 32
        self.fc_input_size = 32

        # LAYER 4: Output Layer (direct to classification)
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
        
        # First conv layer
        x = self.conv1(x)           # -> [batch_size, 16, 1024]
        x = F.relu(x)               # Activation function
        
        # Second conv layer  
        x = self.conv2(x)           # -> [batch_size, 32, 1024]
        x = F.relu(x)
        
        # Global pooling (much simpler than multiple pools)
        x = self.global_pool(x)     # -> [batch_size, 32, 1]
        
        # Flatten for output layer
        x = x.view(x.size(0), -1)   # -> [batch_size, 32]
        
        # Direct to output (no intermediate FC layer)
        x = self.fc_out(x)          # -> [batch_size, 16]
        
        return x


# To test model 
'''

def test_model():
    """Test the model with dummy data to verify shapes"""
    print("Testing RF Fingerprinter Architecture:")
    print("=" * 50)
    
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
    
    print("\n✅ Model architecture test successful!")
    
    return model

if __name__ == "__main__":
    # Run test when file is executed
    model = test_model()

'''