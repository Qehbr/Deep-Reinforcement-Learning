import torch
from torch import nn


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim):
        super(PolicyNetwork, self).__init__()
        layers = []
        prev_size = input_dim
        for idx, hs in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hs))
            layers.append(nn.ReLU())
            prev_size = hs
        layers.append(nn.Linear(prev_size, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.model(x)
        return torch.softmax(logits, dim=-1) + 1e-8


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super(ValueNetwork, self).__init__()
        layers = []
        prev_size = input_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev_size, hs))
            layers.append(nn.ReLU())
            prev_size = hs
        layers.append(nn.Linear(prev_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
