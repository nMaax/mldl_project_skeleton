import torch
from torch import nn

class CustomNet(nn.Module):
    def __init__(self, num_classes=200):  # Add num_classes as an argument
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=0, stride=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.convNet = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool1,
            self.conv2,
            nn.ReLU(),
            self.pool2,
            self.conv3,
            nn.ReLU(),
            self.pool3,
            self.conv4,
            nn.ReLU(),
            self.pool4,
            self.conv5,
            nn.ReLU(),
            self.pool5,
            nn.Flatten(),
        )

        # Calculate input size for fc1
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)  # Input shape
            x = self.convNet(x)
            print(f"Input to the FFNN of size: {x.shape[1]}")
            fc1_input_size = x.shape[1]

        self.fc1 = nn.Linear(fc1_input_size, num_classes)

        self.net = nn.Sequential(
            self.convNet,
            self.fc1
        )

    def forward(self, x):
        x = self.net(x)
        return x