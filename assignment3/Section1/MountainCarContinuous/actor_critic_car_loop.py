import os
import time

import gymnasium
import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from assignment3.Section1.MountainCarContinuous.models import UnifiedPolicyNetwork, ValueNetwork


def actor_critic(
        env_name="MountainCarContinuous-v0",
        # Force the padded observation dimension to 6
        input_dim=6,
        # Force the padded action dimension to 3
        output_dim=3,
        hidden_sizes_theta=[64, 64],
        hidden_sizes_w=[64, 64],
        alpha_theta=0.001,
        alpha_w=0.001,
        episodes=500,
        gamma=0.99,
        entropy_coeff=0.01,
        start_noise_std=0.2,
        end_noise_std=0.05,
        noise_decay=0.99,
        log_dir="runs/actor_critic",
        model_save_path="models"
):
    """
    Actor-Critic training function that zero-pads states up to input_dim=6
    and outputs 3-dimensional actions, but only the first dimension is used
    in MountainCarContinuous.

    Args:
        env_name (str): Gym environment name (default: "MountainCarContinuous-v0").
        input_dim (int): Padded observation dimension (fixed = 6).
        output_dim (int): Padded action dimension (fixed = 3).
        hidden_sizes_theta (list[int]): Hidden layer sizes for policy network.
        hidden_sizes_w (list[int]): Hidden layer sizes for value network.
        alpha_theta (float): Learning rate for policy network.
        alpha_w (float): Learning rate for value network.
        episodes (int): Max number of training episodes.
        gamma (float): Discount factor.
        entropy_coeff (float): Coefficient for entropy bonus in policy loss.
        start_noise_std (float): Initial std for added Gaussian noise on actions.
        end_noise_std (float): Final std for action noise after decay.
        noise_decay (float): Decay rate per episode for action noise std.
        log_dir (str): Directory for TensorBoard logging.
        model_save_path (str): Directory to save the best model.

    Returns:
        policy_network (nn.Module)
        value_network (nn.Module)
        rewards_per_episode (list[float])
        train_time (float)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gymnasium.make(env_name)
    writer = SummaryWriter(log_dir=f"{log_dir}_{env_name}")

    # Create networks
    policy_network = UnifiedPolicyNetwork(
        input_dim, hidden_sizes_theta, output_dim
    ).to(device)
    value_network_ = ValueNetwork(input_dim, hidden_sizes_w).to(device)

    # Optimizers
    policy_optimizer = optim.Adam(policy_network.parameters(), lr=alpha_theta)
    value_optimizer = optim.Adam(value_network_.parameters(), lr=alpha_w)

    rewards_per_episode = []
    start_time = time.time()

    # Track best avg reward over last 50 episodes
    best_avg_reward_50 = float('-inf')

    # Training loop
    for episode in range(episodes):
        # Reset environment
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        # Zero-pad the initial state to dimension=6
        padded_state = np.zeros(input_dim, dtype=np.float32)
        padded_state[: len(state)] = state

        # For decaying noise
        current_noise_std = max(end_noise_std, start_noise_std * (noise_decay ** episode))

        # For policy gradient discount
        I = 1.0

        while not (done or truncated):
            # Convert padded_state to tensor of shape (1, input_dim)
            state_tensor = torch.tensor(
                padded_state, dtype=torch.float32, device=device
            ).reshape(1, -1)

            # Get mean and std from the policy network
            mean, std = policy_network(state_tensor)
            action_distribution = torch.distributions.Normal(mean, std)

            # Sample action from the distribution
            action = action_distribution.sample()
            log_prob_action = action_distribution.log_prob(action).sum(dim=-1)

            # Add Gaussian noise for exploration
            noise = torch.randn_like(action) * current_noise_std
            noisy_action = action + noise

            # We only use the first dimension for the actual environment step
            # Because MountainCarContinuous expects a 1D action
            # shape: (1, 3) -> pick the first entry -> shape (1,)
            # clamp to env.action_space bounds
            clipped_dim0 = noisy_action[0, 0].clamp(
                env.action_space.low[0], env.action_space.high[0]
            )
            final_action = clipped_dim0.cpu().numpy().reshape(-1)

            # Step in the environment
            next_state, reward, done, truncated, _info = env.step(final_action)
            total_reward += reward

            # Zero-pad the next_state to dimension=6
            padded_next_state = np.zeros(input_dim, dtype=np.float32)
            padded_next_state[: len(next_state)] = next_state

            # Convert next_state to tensor
            next_state_tensor = torch.tensor(
                padded_next_state, dtype=torch.float32, device=device
            ).reshape(1, -1)

            # Compute value estimates
            value = value_network_(state_tensor)
            with torch.no_grad():
                if not (done or truncated):
                    next_value = value_network_(next_state_tensor)
                else:
                    next_value = torch.zeros_like(value)

            # TD-error (delta)
            delta = reward + gamma * next_value - value

            # Update value network
            value_loss = delta.pow(2).mean()
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

            # Compute entropy for the entropy bonus
            entropy = action_distribution.entropy().sum(dim=-1).mean()

            # Update policy network
            #  - The gradient is with respect to log_prob_action * delta * I
            policy_loss = - (log_prob_action * delta.detach() * I).mean()
            #  - Add entropy bonus
            policy_loss -= (entropy_coeff * entropy)

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            # Discount factor for policy updates
            I *= gamma

            # Move to next padded state
            padded_state = padded_next_state

        # Logging
        rewards_per_episode.append(total_reward)
        writer.add_scalar("Episode Reward", total_reward, episode)
        writer.add_scalar("Value Loss", value_loss.item(), episode)
        writer.add_scalar("Noise STD", current_noise_std, episode)

        avg_reward_100 = np.mean(rewards_per_episode[-100:])
        avg_reward_50 = np.mean(rewards_per_episode[-50:])
        print(
            f"Episode {episode + 1}: "
            f"Reward={total_reward:.2f}, "
            f"Avg(100)={avg_reward_100:.2f}, "
            f"Avg(50)={avg_reward_50:.2f}, "
            f"Noise STD={current_noise_std:.4f}"
        )

        # Save the model if avg_reward_50 improves
        if avg_reward_50 > best_avg_reward_50 and avg_reward_50 > 0:
            best_avg_reward_50 = avg_reward_50
            if episode > 49:
                os.makedirs(model_save_path, exist_ok=True)
                torch.save(policy_network.state_dict(),
                           os.path.join(model_save_path,
                                        f"best_policy_network_{alpha_theta}_{gamma}_{entropy_coeff}_{start_noise_std}_{end_noise_std}_{noise_decay}.pth"))
                torch.save(value_network_.state_dict(),
                           os.path.join(model_save_path,
                                        f"best_value_network_{alpha_theta}_{gamma}_{entropy_coeff}_{start_noise_std}_{end_noise_std}_{noise_decay}.pth"))
                print(f"New best model saved with Avg(50)={best_avg_reward_50:.2f}")

        # Optional solve condition
        if episode > 49 and avg_reward_50 > 10:
            print(f"Solved {env_name} in {episode + 1} episodes!")
            break

    train_time = time.time() - start_time
    writer.close()
    env.close()

    return policy_network, value_network_, rewards_per_episode, train_time, best_avg_reward_50