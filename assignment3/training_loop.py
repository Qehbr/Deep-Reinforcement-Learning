import time
from time import sleep

from torch import Tensor
from tqdm import tqdm
import numpy as np
import torch

from assignment3.device import get_device
from assignment3.dim_alignment import pad_state, sample_valid_action


def training_loop(
        input_dim,
        actual_act_dim,
        policy_network,
        value_network,
        policy_optimizer,
        value_optimizer,
        env,
        env_name,
        episodes,
        gamma,
        writer,
        rewards_per_episode
):
    device = get_device()
    # Start timing
    start_time = time.time()
    epsilon = 0.999
    epsilon_decay = 0.99995
    with tqdm(total=episodes, desc="Training", unit="episode") as pbar:
        for episode in range(episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0.0
            I = 1.0  # discount factor for policy updates

            while not (done or truncated):
                # Pad the state
                padded_state = pad_state(state, input_dim)
                state_tensor = torch.tensor(padded_state, dtype=torch.float32, device=device).unsqueeze(0)

                # Get action probabilities
                action_probs = policy_network(state_tensor)

                # Step in the environment
                if env_name == "MountainCarContinuous-v0":
                    # Epsilon-greedy action for continuous action space
                    action = epsilon_greedy_action(action_probs, epsilon=epsilon)
                    log_prob_action = torch.log(action_probs[0, 0])
                    next_state, reward, done, truncated, _info = env.step(np.array([action.item()]))
                else:
                    # Sample valid action from first 'actual_act_dim' entries
                    action = sample_valid_action(action_probs, valid_action_dim=actual_act_dim)
                    log_prob_action = torch.log(action_probs[0, action])
                    next_state, reward, done, truncated, _info = env.step(action.item())
                total_reward += reward

                # Pad next state
                padded_next_state = pad_state(next_state, input_dim)
                next_state_tensor = torch.tensor(padded_next_state, dtype=torch.float32, device=device).unsqueeze(0)

                # Current value
                value = value_network(state_tensor)

                # Next value = 0 if done/truncated
                with torch.no_grad():
                    next_value = value_network(next_state_tensor) if not (done or truncated) else torch.tensor([[0.0]],
                                                                                                               device=device)

                # TD error
                delta = reward + gamma * next_value - value

                # Value update
                value_loss = -value * delta.detach() * I
                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()

                # Policy update
                policy_loss = -log_prob_action * delta.detach() * I
                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                # Update the factor I
                I *= gamma

                # Move on
                state = next_state

            # Logging
            rewards_per_episode.append(total_reward)
            writer.add_scalar("Episode Reward", total_reward, episode)
            writer.add_scalar("Value Loss", value_loss.item(), episode)

            # Print some info
            avg_reward = np.mean(rewards_per_episode[-100:])
            if (episode + 1) % 100 == 0:
                pbar.set_postfix({'Avg Reward(100)': f"{avg_reward:.2f}"})

            if check_solved(env_name, avg_reward, episode):
                break

            epsilon *= epsilon_decay
            pbar.update(1)
    # End timing
    train_time = time.time() - start_time

    return train_time

def check_solved(env_name, avg_reward, episode):
    if env_name == "CartPole-v1" and avg_reward >= 475.0:
        print(f"Solved {env_name} in {episode + 1} episodes!")
        return True
    if env_name == "Acrobot-v1" and avg_reward >= -100.0:
        print(f"Solved {env_name} in {episode + 1} episodes!")
        return True
    if env_name == "MountainCarContinuous-v0" and avg_reward >= 90.0:
        print(f"Solved {env_name} in {episode + 1} episodes!")
        return True
    return False

def epsilon_greedy_action(action_probs, epsilon):
    # 1dim mountain car continuous action space, action_probs is a scalar, scale to [-1, 1], add noise
    action = torch.tanh(action_probs[0, 0])
    if torch.rand(1) < epsilon:
        # std needs to scale according to epsilon between 0.1 and 0.5, so we will explore a lot in the beginning, but then reduce
        scale = 0.1 + (0.5 - 0.1) * epsilon
        noise = np.random.normal(0, scale)
        return torch.clamp(action + noise, -1, 1)
    return action
