import torch
from torch import nn

class CustomNet(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=200, dropout_rate=0.3):  # Add num_classes as an argument
        super(CustomNet, self).__init__()
        
        self.input_channels = input_shape[0]  # Assuming input_shape is in (C, H, W) format
        self.num_classes = num_classes
        
        # Use nn.Sequential to stack conv layers and pooling layers
        self.convNet = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),  # Flatten the output before passing to the fully connected layer
        )
        
        # Calculate the input size for the fully connected layer
        fc1_input_size = self._calculate_fc1_input_size(input_shape)
        
        # Define the fully connected layers
        self.fcNet = nn.Sequential(
            # Multiple fully connected layers
            nn.Linear(fc1_input_size, 1024),  # First fully connected layer
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 512),  # Second fully connected layer
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),  # Final fully connected layer for output
        )

    def _calculate_fc1_input_size(self, input_shape):
        # Dummy input to calculate the shape after convolutions and pooling
        with torch.no_grad():
            x = torch.randn(1, *input_shape)  # Create a dummy input with the provided input shape
            x = self.convNet(x)  # Pass through convNet (Sequential of conv layers)
            print(f"Input to the Fully Connected layer is of size: {x.shape[1]}")
            return x.shape[1]

    def forward(self, x):
        x = self.convNet(x)  # Pass through convNet (with flattening)
        x = self.fcNet(x)  # Pass through fcNet (with dropout)
        return x

if __name__ == "__main__":
    # Example of instantiating the model
    model = CustomNet(input_shape=(3, 224, 224), num_classes=200)
    print(model)
