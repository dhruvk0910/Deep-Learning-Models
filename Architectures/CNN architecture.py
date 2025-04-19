import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        feature_maps = []

        x = F.relu(self.conv1(x))
        feature_maps.append(x)
        x = F.relu(self.conv2(x))
        feature_maps.append(x)
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        feature_maps.append(x)
        x = F.relu(self.conv4(x))
        feature_maps.append(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x, feature_maps
    

model = CIFAR10Net()
summary(model, input_size=(3, 32, 32))