import torch
from dim_alignment import pad_state, max_input_dim, max_output_dim
from models import PolicyNetwork
from device import get_device
import numpy as np
import gymnasium as gym


def test_policy(env_name, policy_path, episodes=100):
    """
    Test a trained policy network on the given environment for a specified number of episodes.

    Args:
        env_name (str): Name of the gym environment.
        policy_path (str): Path to the saved policy network.
        episodes (int): Number of test episodes.

    Returns:
        float: Average reward over all test episodes.
    """
    device = get_device()
    env = gym.make(env_name)
    num_actions = env.action_space.n  # Number of valid actions (e.g., 2 for CartPole-v1)

    # Initialize the policy network with the correct output dimension
    policy_network = PolicyNetwork(max_input_dim, [32, 64, 32], max_output_dim).to(device)
    policy_network.load_state_dict(torch.load(policy_path, map_location=device))
    policy_network.eval()

    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Pad state to match input dimensions
            padded_state = pad_state(state, max_input_dim)
            state_tensor = torch.tensor(padded_state, dtype=torch.float32, device=device).unsqueeze(0)

            # Get action probabilities from the policy network
            with torch.no_grad():
                action_probs = policy_network(state_tensor)

            # Restrict action selection to valid actions
            valid_action_probs = action_probs[:, :num_actions]  # Consider only valid actions
            action = torch.argmax(valid_action_probs, dim=-1).item()

            # Take a step in the environment
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)
        print(f'Episode {episode + 1}, Reward: {total_reward:.2f}')

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {episodes} episodes: {avg_reward:.2f}")
    env.close()
    return avg_reward


# Testing the trained model
test_policy(
    env_name="Acrobot-v1",
    # policy_path="models/CartPole-v1/trial_2/policy.pth",
    policy_path="models/Acrobot-v1/trial_1/policy.pth",
    episodes=100
)
