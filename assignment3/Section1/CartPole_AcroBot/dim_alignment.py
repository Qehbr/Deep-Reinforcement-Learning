import numpy as np
import torch

ENV_OBS_DIM = {
    "CartPole-v1": 4,
    "Acrobot-v1": 6,
    "MountainCarContinuous-v0": 2
}

ENV_ACT_DIM = {
    "CartPole-v1": 2,
    "Acrobot-v1": 3,
    "MountainCarContinuous-v0": 1
}

max_input_dim = 6  # e.g., max of (4, 6, 2)
max_output_dim = 3  # e.g., max of (2, 3, 1)


def pad_state(state, max_dim):
    """
    Pads a 1D state array to length `max_dim`.
    """
    padded = np.zeros(max_dim, dtype=np.float32)
    actual_len = len(state)
    padded[:actual_len] = state
    return padded


def sample_valid_action(action_probs, valid_action_dim):
    """
    Samples an action from the first `valid_action_dim` entries of `action_probs`.
    If an "empty" action (>= valid_action_dim) is chosen, resample.

    """
    if torch.isnan(action_probs).any() or not torch.isfinite(action_probs).all():
        raise ValueError(f"Invalid action probabilities: {action_probs}")
    dist = torch.distributions.Categorical(action_probs[0, :valid_action_dim])
    return dist.sample()
