import torch
from torch import nn


class MyAbstractNetwork(nn.Module):
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class PolicyNetwork(MyAbstractNetwork):
    def __init__(self, input_dim, hidden_sizes, output_dim, dropout_layers=None, dropout_p=0.7):
        super(PolicyNetwork, self).__init__()
        layers = []
        prev_size = input_dim
        self.dropout_layers = set(dropout_layers or [])
        self.dropout_p = dropout_p
        for idx, hs in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hs))
            layers.append(nn.ReLU())
            if idx in self.dropout_layers:
                layers.append(nn.Dropout(p=self.dropout_p))
            prev_size = hs
        layers.append(nn.Linear(prev_size, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.model(x)
        return torch.softmax(logits, dim=-1) + 1e-8


class ValueNetwork(MyAbstractNetwork):
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
