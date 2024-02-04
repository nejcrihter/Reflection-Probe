import torch
import torch.nn as nn
import torch.optim as optim


class ReflectionProbeCNN(nn.Module):
    def __init__(self):
        super(ReflectionProbeCNN, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 6 * 128 * 128 * 3)  # 6 faces, 128x128 resolution, 3 channels

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 6, 3, 128, 128)  # Reshape to 6 faces, 3 channels, 128x128
        return x
