import gymnasium
import torch
import torch.nn as nn
import numpy as np

# Define the UnifiedPolicyNetwork class
class UnifiedPolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim):
        super(UnifiedPolicyNetwork, self).__init__()
        layers = []
        prev_size = input_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev_size, hs))
            layers.append(nn.ReLU())
            prev_size = hs

        self.mean_layer = nn.Linear(prev_size, output_dim)
        self.log_std_layer = nn.Linear(prev_size, output_dim)

        nn.init.constant_(self.log_std_layer.weight, 0.0)
        nn.init.constant_(self.log_std_layer.bias, -0.5)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        return mean, std

# Load the model
policy_model_path = "models/best_policy_network_0.000432860862913505_0.9571389501004702_0.0075942800574955_0.12655319079849925_0.2507795275354682_0.9952850823459259.pth"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input and output dimensions
input_dim = 6
output_dim = 3
hidden_sizes = (256,)  # Match the trained model's architecture

# Initialize the policy network
policy_network = UnifiedPolicyNetwork(input_dim=input_dim, hidden_sizes=hidden_sizes, output_dim=output_dim).to(device)
policy_network.load_state_dict(torch.load(policy_model_path, map_location=device))
policy_network.eval()  # Set the model to evaluation mode

# Test the model in the environment with rendering
env_name = "MountainCarContinuous-v0"
env = gymnasium.make(env_name, render_mode="human")
state, _ = env.reset()

# Zero-pad the state to match input dimension
padded_state = np.zeros(input_dim, dtype=np.float32)
padded_state[: len(state)] = state

done = False
total_reward = 0.0

while not done:
    # Render the environment
    env.render()

    # Convert state to tensor
    state_tensor = torch.tensor(padded_state, dtype=torch.float32, device=device).reshape(1, -1)

    # Get the action from the policy network
    with torch.no_grad():
        mean, std = policy_network(state_tensor)
        action_distribution = torch.distributions.Normal(mean, std)
        action = action_distribution.sample()

    # Use only the first action dimension for MountainCarContinuous
    action_dim0 = action[0, 0].clamp(env.action_space.low[0], env.action_space.high[0])
    final_action = action_dim0.cpu().numpy().reshape(-1)

    # Step the environment
    next_state, reward, done, truncated, _info = env.step(final_action)
    total_reward += reward

    # Zero-pad the next state
    padded_state = np.zeros(input_dim, dtype=np.float32)
    padded_state[: len(next_state)] = next_state

print(f"Total reward in this test run: {total_reward:.2f}")

env.close()
