import numpy as np
import torch
from torch import nn


class UnifiedPolicyNetwork(nn.Module):
    """
    A unified network that outputs both the mean and log_std (for a Gaussian policy)
    given the (padded) state.

    Input dimension is set to 6 (maximum across tasks),
    and output dimension is set to 3 (maximum across tasks).
    We'll only use the first action dimension for MountainCarContinuous.
    """

    def __init__(self, input_dim, hidden_sizes, output_dim):
        super(UnifiedPolicyNetwork, self).__init__()
        layers = []
        prev_size = input_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev_size, hs))
            layers.append(nn.ReLU())
            prev_size = hs

        # Separate heads: one for mean, one for log_std
        self.mean_layer = nn.Linear(prev_size, output_dim)
        self.log_std_layer = nn.Linear(prev_size, output_dim)

        # Initialize log_std_layer weights and biases
        nn.init.constant_(self.log_std_layer.weight, 0.0)
        nn.init.constant_(self.log_std_layer.bias, -0.5)  # log_std ~ -0.5 => std ~ 0.6

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Returns:
            mean (Tensor): shape (batch_size, output_dim)
            std  (Tensor): shape (batch_size, output_dim)
        """
        x = self.model(x)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # Clamp log_std to prevent numerical issues
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        return mean, std


class ValueNetwork(nn.Module):
    """
    A simple value network that outputs state-value (V(s)).
    Also uses input dimension = 6.
    """

    def __init__(self, input_dim, hidden_sizes):
        super(ValueNetwork, self).__init__()
        layers = []
        prev_size = input_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev_size, hs))
            layers.append(nn.ReLU())
            prev_size = hs

        layers.append(nn.Linear(prev_size, 1))  # single scalar value
        self.model = nn.Sequential(*layers)

        # Initialize the final layer's weights
        nn.init.kaiming_uniform_(layers[-1].weight, a=np.sqrt(5))
        nn.init.constant_(layers[-1].bias, 0.0)

    def forward(self, x):
        return self.model(x)