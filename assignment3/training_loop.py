import time

import numpy as np
import torch

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
        device,
        writer,
        rewards_per_episode
):
    # Start timing
    start_time = time.time()

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

            # Sample valid action from first 'actual_act_dim' entries
            action = sample_valid_action(action_probs, valid_action_dim=actual_act_dim)
            log_prob_action = torch.log(action_probs[0, action])

            # Step in the environment
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
            print(f"Episode {episode + 1}: Reward={total_reward:.2f}, Avg(100)={avg_reward:.2f}")

        if env_name == "CartPole-v1" and avg_reward >= 475.0:
            print(f"Solved {env_name} in {episode + 1} episodes!")
            break
        if env_name == "Acrobot-v1" and avg_reward >= -100.0:
            print(f"Solved {env_name} in {episode + 1} episodes!")
            break
        if env_name == "MountainCarContinuous-v0" and avg_reward >= 90.0:
            print(f"Solved {env_name} in {episode + 1} episodes!")
            break

    # End timing
    train_time = time.time() - start_time

    return train_time
