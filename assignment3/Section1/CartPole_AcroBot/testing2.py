import torch
import json
from dim_alignment import pad_state, max_input_dim, max_output_dim
from models import PolicyNetwork
from device import get_device
import numpy as np
import gymnasium as gym
import os
import random

def test_policy(
    env_name,
    policy_path,
    episodes=1,  # Default to 1 for testing
    render=True,  # Enable rendering
    seed=42       # For reproducibility
):
    """
    Test a trained policy network on the given environment for a specified number of episodes with rendering.

    Args:
        env_name (str): Name of the gym environment.
        policy_path (str): Path to the saved policy network.
        episodes (int): Number of test episodes.
        render (bool): Whether to render the environment.
        seed (int): Random seed for reproducibility.

    Returns:
        float: Average reward over all test episodes.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = get_device()
    env = gym.make(env_name, render_mode='human') if render else gym.make(env_name)
    env.reset(seed=seed)

    # Instantiate the PolicyNetwork with the correct architecture
    policy_network = PolicyNetwork(max_input_dim, [16,32,16], max_output_dim).to(device)

    # Load the trained weights
    try:
        policy_network.load_state_dict(torch.load(policy_path, map_location=device))
    except Exception as e:
        raise ValueError(f"Error loading policy network: {e}")

    policy_network.eval()

    total_rewards = []

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        while not (done):
            if render:
                env.render()

            # Pad state to match input dimensions
            padded_state = pad_state(state, max_input_dim)
            state_tensor = torch.tensor(padded_state, dtype=torch.float32, device=device).unsqueeze(0)

            # Get action probabilities and select action
            with torch.no_grad():
                action_probs = policy_network(state_tensor)
            action = torch.argmax(action_probs, dim=-1).item()

            # Take a step in the environment
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)
        print(f'Episode {episode}, Reward: {total_reward:.2f}')

    avg_reward = np.mean(total_rewards)
    print(f"Average reward over {episodes} episode(s): {avg_reward:.2f}")
    env.close()
    return avg_reward

# Example usage:
if __name__ == "__main__":
    test_policy(
        env_name="CartPole-v1",
        policy_path="pretrained_models/CartPole-v1_policy.pth",
        episodes=1,   # Single test episode with rendering
        render=True,
        seed=42
    )
