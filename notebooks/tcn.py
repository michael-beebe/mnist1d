import time, copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from mnist1d.utils import ObjectView


def accuracy(model, inputs, targets):
    model.eval()
    with torch.no_grad():
        preds = model(inputs).argmax(-1).cpu().numpy()
        targets = targets.cpu().numpy().astype(np.float32)
        acc = 100 * sum(preds == targets) / len(targets)
    model.train()
    return acc


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


def train_TCN(data, model, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

    # Reshape input data if training a TCN model
    if isinstance(model, TCNBase):  # Check if the model is an instance of TCNBase
        data['x'] = data['x'].reshape((4000, 1, 40))  # [batch_size, input_channels, sequence_length]
        data['x_test'] = data['x_test'].reshape((1000, 1, 40))

    x_train, x_test = torch.Tensor(data['x']), torch.Tensor(data['x_test'])
    y_train, y_test = torch.LongTensor(data['y']), torch.LongTensor(data['y_test'])

    model = model.to(args.device)
    x_train, x_test, y_train, y_test = [v.to(args.device) for v in [x_train, x_test, y_train, y_test]]

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    results = {'checkpoints': [], 'train_losses': [], 'test_losses': [], 'train_acc': [], 'test_acc': []}
    t0 = time.time()

    for step in range(args.total_steps + 1):
        bix = (step * args.batch_size) % len(x_train)  # Batch index
        x_batch, y_batch = x_train[bix:bix + args.batch_size], y_train[bix:bix + args.batch_size]

        optimizer.zero_grad()  # Zero the gradients
        loss = criterion(model(x_batch), y_batch)
        results['train_losses'].append(loss.item())
        loss.backward()
        optimizer.step()

        # Evaluate the model at specified intervals
        if args.eval_every > 0 and step % args.eval_every == 0:
            test_loss = criterion(model(x_test), y_test)
            results['test_losses'].append(test_loss.item())
            results['train_acc'].append(accuracy(model, x_train, y_train))
            results['test_acc'].append(accuracy(model, x_test, y_test))

        if step > 0 and step % args.print_every == 0:  # Print training progress
            t1 = time.time()
            print("step {}, dt {:.2f}s, train_loss {:.3e}, test_loss {:.3e}, train_acc {:.1f}, test_acc {:.1f}"
                  .format(step, t1 - t0, loss.item(), results['test_losses'][-1],
                          results['train_acc'][-1], results['test_acc'][-1]))
            t0 = t1

        if args.checkpoint_every > 0 and step % args.checkpoint_every == 0:  # Save checkpoints
            model.step = step
            results['checkpoints'].append(copy.deepcopy(model))

    return results
