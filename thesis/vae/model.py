import torch
import torch.nn as nn
import torch.nn.functional as F

class Config():
    def __init__(self):
        self.dropout_rate = 0.2

config = Config()

# ---------------------------- MODEL ------------------------
class BaselineConvModel(nn.Module):
    def __init__(self):
        super(BaselineConvModel, self).__init__()
        # Initial 1D Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=1)  # Reduced stride to minimize shrinking
        self.dropout1 = nn.Dropout(config.dropout_rate)

        # Residual Block 1
        self.conv2a = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm1d(64)
        self.conv2b = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=1)  # Reduced stride
        self.dropout2 = nn.Dropout(config.dropout_rate)
        self.shortcut1 = nn.Conv1d(32, 64, kernel_size=1)  # To match dimensions for residual addition

        # Residual Block 2
        self.conv3a = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm1d(128)
        self.conv3b = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm1d(128)
        # No pooling here to preserve sequence length
        self.dropout3 = nn.Dropout(config.dropout_rate)
        self.shortcut2 = nn.Conv1d(64, 128, kernel_size=1)  # To match dimensions for residual addition

        # Residual Block 3
        self.conv4a = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4a = nn.BatchNorm1d(256)
        self.conv4b = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=1)  # Reduced stride
        self.dropout4 = nn.Dropout(config.dropout_rate)
        self.shortcut3 = nn.Conv1d(128, 256, kernel_size=1)  # To match dimensions for residual addition

        # Residual Block 4 (Additional)
        self.conv5a = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn5a = nn.BatchNorm1d(512)
        self.conv5b = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn5b = nn.BatchNorm1d(512)
        # Optional pooling here; consider removing if it shrinks too much
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=1)
        self.dropout5 = nn.Dropout(config.dropout_rate)
        self.shortcut4 = nn.Conv1d(256, 512, kernel_size=1)  # To match dimensions for residual addition

        # Attention Mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(512, 1, kernel_size=1),  # Attention weights
            nn.Softmax(dim=-1)
        )

        # Fully Connected Layers
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(512, 256)
        self.dropout6 = nn.Dropout(config.dropout_rate)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Initial Convolution
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Residual Block 1
        identity1 = self.shortcut1(x)
        out = F.relu(self.bn2a(self.conv2a(x)))
        out = self.bn2b(self.conv2b(out))
        out += identity1  # Residual connection
        out = F.relu(out)
        out = self.pool2(out)
        out = self.dropout2(out)

        # Residual Block 2
        identity2 = self.shortcut2(out)
        out = F.relu(self.bn3a(self.conv3a(out)))
        out = self.bn3b(self.conv3b(out))
        out += identity2  # Residual connection
        out = F.relu(out)
        # No pooling here to preserve sequence length
        out = self.dropout3(out)

        # Residual Block 3
        identity3 = self.shortcut3(out)
        out = F.relu(self.bn4a(self.conv4a(out)))
        out = self.bn4b(self.conv4b(out))
        out += identity3  # Residual connection
        out = F.relu(out)
        out = self.pool3(out)  # Optional pooling with reduced stride
        out = self.dropout4(out)

        # Residual Block 4 (Additional)
        identity4 = self.shortcut4(out)
        out = F.relu(self.bn5a(self.conv5a(out)))
        out = self.bn5b(self.conv5b(out))
        out += identity4  # Residual connection
        out = F.relu(out)
        out = self.pool4(out)  # Optional pooling with reduced stride
        out = self.dropout5(out)

        # Apply Attention
        weights = self.attention(out)
        out = out * weights  # Element-wise multiplication with attention weights

        # Global Average Pooling and Fully Connected Layers
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.dropout6(out)
        out = self.fc2(out)
        return out
# ---------------------------- MODEL -END ------------------------