import numpy as np
import torch
from assignment3.dim_alignment import sample_valid_action


class ActionSelector:
    def select_action(self, action_probs, valid_action_dim):
        # Sample valid action from first 'actual_act_dim' entries
        action = sample_valid_action(action_probs, valid_action_dim=valid_action_dim)
        log_prob_action = torch.log(action_probs[0, action])
        return action.item(), log_prob_action

    def decay_epsilon(self):
        pass


class ContinuousActionSelector(ActionSelector):
    """
    Used for MountainCarContinuous-v0 with some noise and epsilon-greedy exploration.
    """

    def __init__(self, epsilon, epsilon_decay, min_noise_std, max_noise_std):
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std
        super().__init__()

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def select_action(self, action_probs, valid_action_dim):
        # Epsilon-greedy action for continuous action space
        action = self._epsilon_greedy_noisy_action(action_probs)
        log_prob_action = torch.log(action_probs[0, 0])
        return np.array([action.detach().numpy()]), log_prob_action

    def _epsilon_greedy_noisy_action(self, action_probs):
        # 1dim mountain car continuous action space, action_probs is a scalar, scale to [-1, 1], add noise
        action = torch.tanh(action_probs[0, 0])
        if torch.rand(1) < self.epsilon:
            # std needs to scale according to epsilon e.g. between 0.1 and 0.5, so we will explore a lot in the beginning, but then reduce
            scale = self.min_noise_std + (self.max_noise_std - self.min_noise_std) * self.epsilon
            noise = np.random.normal(0, scale)
            return torch.clamp(action + noise, -1, 1)
        return action
