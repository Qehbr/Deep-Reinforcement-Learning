import torch

from assignment3.Section1.CartPole_AcroBot.dim_alignment import sample_valid_action


class ActionSelector:
    def select_action(self, action_probs, valid_action_dim):
        # Sample valid action from first 'actual_act_dim' entries
        action = sample_valid_action(action_probs, valid_action_dim=valid_action_dim)
        log_prob_action = torch.log(action_probs[0, action])
        return action.item(), log_prob_action