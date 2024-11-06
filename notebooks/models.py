# The MNIST-1D dataset | 2020
# Sam Greydanus

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import time, copy
import numpy as np

from mnist1d.utils import ObjectView



class LinearBase(nn.Module):

    def __init__(self, input_size, output_size):
        super(LinearBase, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        print("Initialized LinearBase model with {} parameters".format(
            self.count_params()))

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])

    def forward(self, x):
        return self.linear(x)


class MLPBase(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=100):
        super(MLPBase, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        print("Initialized MLPBase model with {} parameters".format(
            self.count_params()))

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])

    def forward(self, x):
        h = self.linear1(x).relu()
        h = h + self.linear2(h).relu()
        return self.linear3(h)


class ConvBase(nn.Module):

    def __init__(self, output_size, channels=25, linear_in=125):
        super(ConvBase, self).__init__()
        self.conv1 = nn.Conv1d(1, channels, 5, stride=2, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.linear = nn.Linear(
            linear_in,
            output_size)  # flattened channels -> 10 (assumes input has dim 50)
        print("Initialized ConvBase model with {} parameters".format(
            self.count_params()))

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])

    def forward(self,
                x,
                verbose=False):  # the print statements are for debugging
        x = x.view(-1, 1, x.shape[-1])
        h1 = self.conv1(x).relu()
        h2 = self.conv2(h1).relu()
        h3 = self.conv3(h2).relu()
        h3 = h3.view(h3.shape[0], -1)  # flatten the conv features
        return self.linear(h3)  # a linear classifier goes on top


class GRUBase(torch.nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size=6,
                 time_steps=40,
                 bidirectional=True):
        super(GRUBase, self).__init__()
        self.gru = torch.nn.GRU(input_size=input_size,
                                hidden_size=hidden_size,
                                batch_first=True,
                                bidirectional=bidirectional)
        flat_size = 2 * hidden_size * time_steps if bidirectional else hidden_size * time_steps
        self.linear = torch.nn.Linear(flat_size, output_size)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        print("Initialized GRUBase model with {} parameters".format(
            self.count_params()))

    def count_params(self):
        return sum([p.view(-1).shape[0] for p in self.parameters()])

    def forward(self, x, h0=None):  # assumes seq has [batch, time]
        x = x.view(*x.shape[:2], 1)  # add a spatial dimension
        k = 2 if self.bidirectional else 1
        h0 = 0 * torch.randn(k, x.shape[0],
                             self.hidden_size) if h0 is None else h0
        h0 = h0.to(x.device)  # GPU support

        output, hn = self.gru(x, h0)
        output = output.reshape(output.shape[0],
                                -1)  # [batch, time*hidden_size]
        return self.linear(output)


####################################################################################
##################               NEW MODELS                     ####################
####################################################################################
# --- Resnet 1D ---
class ResNet1DBase(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=64):
        super(ResNet1DBase, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv1d(1,
                               hidden_size,
                               kernel_size=7,
                               stride=2,
                               padding=3)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        # Residual block
        self.conv2 = nn.Conv1d(hidden_size,
                               hidden_size,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.conv3 = nn.Conv1d(hidden_size,
                               hidden_size,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn3 = nn.BatchNorm1d(hidden_size)

        # Fully connected layer
        self.linear = nn.Linear(hidden_size, output_size)

        print(
            f"Initialized ResNet1DBase model with {self.count_params()} parameters"
        )

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        # Reshape input: [batch_size, 1, input_size] (same as ConvBase)
        x = x.view(x.size(0), 1, -1)

        # Initial conv + batch norm + relu
        out = F.relu(self.bn1(self.conv1(x)))

        # Residual connection
        identity = out
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # Add the skip connection (residual connection)
        out += identity
        out = F.relu(out)

        # Global average pooling to reduce feature map to size of hidden_size
        out = F.adaptive_avg_pool1d(out, 1).view(out.size(0), -1)

        # Final linear layer
        out = self.linear(out)
        return out


# --- Temporal ConvNet ---
class TCNBase(nn.Module):
    def __init__(self, input_size, output_size, num_channels, dropout=0.2):
        super(TCNBase, self).__init__()

        # Initial convolutional layer
        self.initial_conv = nn.Conv1d(1, num_channels[0], kernel_size=7, stride=2, padding=3)
        self.initial_bn = nn.BatchNorm1d(num_channels[0])

        # Residual blocks for TCN
        self.tcn_blocks = nn.ModuleList()
        num_levels = len(num_channels)

        for i in range(num_levels):
            in_channels = num_channels[i - 1] if i > 0 else num_channels[0]
            out_channels = num_channels[i]
            self.tcn_blocks.append(self._build_tcn_block(in_channels, out_channels, dropout))

        # Fully connected layer for output
        self.linear = nn.Linear(num_channels[-1], output_size)

        print(f"Initialized TCNBase model with {self.count_params()} parameters")

    def _build_tcn_block(self, in_channels, out_channels, dropout):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        # Initial convolution + batch normalization + ReLU
        x = x.view(x.size(0), 1, -1)  # Reshape input: [batch_size, 1, input_size]
        out = F.relu(self.initial_bn(self.initial_conv(x)))

        # Residual connections through the TCN blocks
        for block in self.tcn_blocks:
            identity = out
            out = block(out)
            if out.shape[1] != identity.shape[1]:
                identity = nn.Conv1d(identity.shape[1], out.shape[1], kernel_size=1, stride=1, padding=0)(identity)
            out += identity  # Add the residual connection
            out = F.relu(out)

        # Global average pooling to reduce the output to match the linear layer input
        out = F.adaptive_avg_pool1d(out, 1).view(out.size(0), -1)

        # Final linear layer for classification
        out = self.linear(out)
        return out

# --- Dilated CNN ---
class DilatedCNNBase(nn.Module):

    def __init__(self, input_size, output_size, channels=64, kernel_size=3, dilation=2):
        super(DilatedCNNBase, self).__init__()
        
        # Define the dilated convolutional layers
        self.conv1 = nn.Conv1d(1, channels, kernel_size, dilation=dilation, padding=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=dilation)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=dilation)
        
        # Adaptive pooling to ensure consistent output size for the linear layer
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Linear layer for classification
        self.linear = nn.Linear(channels, output_size)
        print("Initialized DilatedCNNBase model with {} parameters".format(self.count_params()))

    def count_params(self):
        # Count the number of parameters in the model
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        # Forward pass through the dilated CNN
        x = x.view(-1, 1, x.shape[-1])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Apply global average pooling to reduce the sequence length to 1
        x = self.global_pool(x).view(x.size(0), -1)
        
        return self.linear(x)

