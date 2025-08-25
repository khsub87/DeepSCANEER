import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
#
class Classifier(nn.Module):
    def __init__(self, hidden_size=128, dropout_rate=0.2):
        super(Classifier, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=65, out_channels=hidden_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 64)
        self.bnfc = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

        self.dropoutfc = nn.Dropout(p=dropout_rate)

        # Weight initialization
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x_input):
        if not isinstance(x_input, np.ndarray):
            x_input = x_input.detach().cpu().numpy()

        scaler = StandardScaler()
        x_input = scaler.fit_transform(x_input)

        x_input = torch.from_numpy(x_input).float()
        x_input = x_input.unsqueeze(-1)  # (batch, features, 1)
        x_input = x_input.to(self.conv1.weight.device)

        x = self.conv1(x_input)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        
        # Flatten for FC layers
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.bnfc(x)
        x = self.relu3(x)
        x = self.dropoutfc(x)
        x = self.fc2(x)

        return x.squeeze(1)
