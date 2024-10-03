# The MNIST-1D dataset | 2020
# Sam Greydanus

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self,
                 input_size,
                 output_size,
                 num_channels,
                 kernel_size=2,
                 dropout=0.2):
        super(TCNBase, self).__init__()
        # Temporal convolutional network
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size,
                                   dropout)
        # Linear layer for output
        self.linear = nn.Linear(num_channels[-1], output_size)
        print("Initialized TCNBase model with {} parameters".format(
            self.count_params()))

    def count_params(self):
        # Count the number of parameters in the model
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        # Forward pass through the TCN
        out = self.tcn(x)
        out = out[:, :, -1]  # Take the last time step
        out = self.linear(out)
        return out


class TemporalConvNet(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            # Define dilation size for each layer
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(in_channels,
                              out_channels,
                              kernel_size,
                              stride=1,
                              dilation=dilation_size,
                              padding=(kernel_size - 1) * dilation_size,
                              dropout=dropout)
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the network
        return self.network(x)


class TemporalBlock(nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 kernel_size,
                 stride,
                 dilation,
                 padding,
                 dropout=0.2):
        super(TemporalBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv1d(n_inputs,
                               n_outputs,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)

        # Second convolutional layer
        self.conv2 = nn.Conv1d(n_outputs,
                               n_outputs,
                               kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        # Sequential network of layers
        self.net = nn.Sequential(self.conv1, self.bn1, nn.ReLU(),
                                 self.dropout1, self.conv2, self.bn2,
                                 nn.ReLU(), self.dropout2)
        # Shortcut connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs,
                                    1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass through the block
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# --- Dilated CNN ---
class DilatedCNNBase(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 channels=64,
                 kernel_size=3,
                 dilation=2):
        super(DilatedCNNBase, self).__init__()
        # Define the dilated convolutional layers
        self.conv1 = nn.Conv1d(1,
                               channels,
                               kernel_size,
                               dilation=dilation,
                               padding=dilation)
        self.conv2 = nn.Conv1d(channels,
                               channels,
                               kernel_size,
                               dilation=dilation,
                               padding=dilation)
        self.conv3 = nn.Conv1d(channels,
                               channels,
                               kernel_size,
                               dilation=dilation,
                               padding=dilation)

        # Linear layer for classification
        self.linear = nn.Linear(channels * (input_size // (dilation * 2)),
                                output_size)
        print("Initialized DilatedCNNBase model with {} parameters".format(
            self.count_params()))

    def count_params(self):
        # Count the number of parameters in the model
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        # Forward pass through the dilated CNN
        x = x.view(-1, 1,
                   x.shape[-1])  # Reshape input to [batch, channels, length]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the features
        return self.linear(x)
