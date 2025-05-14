
# Updated model with slightly deeper architecture for better representation
import torch.nn as nn
import torch.nn.functional as F
import torch

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input layers - common trunk
        self.conv1 = nn.Conv2d(18, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc_bn = nn.BatchNorm1d(1024)
        self.fc_dropout = nn.Dropout(0.3)  # Add dropout for regularization
        
        # Policy head
        self.policy = nn.Linear(1024, 20480)  # Full action space
        
        # Value head
        self.value_fc = nn.Linear(1024, 256)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        # Common layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Shared representation
        x = F.relu(self.fc_bn(self.fc1(x)))
        x = self.fc_dropout(x)
        
        # Policy head
        policy_logits = self.policy(x)
        
        # Value head with additional layer
        value = F.relu(self.value_fc(x))
        value = torch.tanh(self.value(value))
        
        return policy_logits, value