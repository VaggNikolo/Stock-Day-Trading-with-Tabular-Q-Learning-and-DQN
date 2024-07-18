import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Load the stock parameters from the CSV file
file_path = 'stock_params.csv'
stock_params = pd.read_csv(file_path)

# Display the loaded stock parameters
print(stock_params)

class StockMarketEnv:
    def __init__(self, stock_params, transaction_fee, gamma):
        self.stock_params = stock_params
        self.transaction_fee = transaction_fee
        self.gamma = gamma
        self.num_stocks = stock_params.shape[0]
        self.state = None
        self.reset()
        
    def reset(self):
        # Initialize the state with random yields
        self.state = np.random.rand(self.num_stocks)
        return torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
    
    def step(self, action):
        current_stock = action
        next_state = []
        reward = 0
        
        for i in range(self.num_stocks):
            if random.random() < self.stock_params.iloc[i]['p_HH'] if self.state[i] > 0.5 else self.stock_params.iloc[i]['p_LL']:
                next_state.append(self.stock_params.iloc[i]['r_H'] if self.state[i] > 0.5 else self.stock_params.iloc[i]['r_L'])
            else:
                next_state.append(self.stock_params.iloc[i]['r_L'] if self.state[i] > 0.5 else self.stock_params.iloc[i]['r_H'])
        
        reward = next_state[current_stock]
        
        if action != current_stock:
            reward -= self.transaction_fee
        
        self.state = next_state
        done = False  # In this environment, the episode does not terminate
        return torch.tensor(next_state, dtype=torch.float32).unsqueeze(0), reward, done, {}

# Initialize the environment
env = StockMarketEnv(stock_params, transaction_fee=0.1, gamma=0.99)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the DQN
input_dim = env.num_stocks
output_dim = env.num_stocks
dqn = DQN(input_dim, output_dim)

def train_dqn(env, dqn, num_episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, lr):
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    memory = []
    epsilon = epsilon_start
    all_rewards = []
    all_value_functions = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_rewards = 0

        for t in range(100):  # Limit the number of steps per episode
            if random.random() < epsilon:
                action = random.choice(range(env.num_stocks))
            else:
                with torch.no_grad():
                    action = dqn(state).argmax().item()

            next_state, reward, done, _ = env.step(action)
            episode_rewards += reward

            memory.append((state, action, reward, next_state, done))

            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)

                states = torch.cat(states)
                next_states = torch.cat(next_states)

                q_values = dqn(states)
                next_q_values = dqn(next_states)

                q_target = torch.tensor(rewards, dtype=torch.float32) + gamma * next_q_values.max(1)[0] * (1 - torch.tensor(dones, dtype=torch.float32))

                q_values = q_values.gather(1, torch.tensor(actions, dtype=torch.int64).unsqueeze(1)).squeeze()

                loss = F.mse_loss(q_values, q_target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            if done:
                break

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        all_rewards.append(episode_rewards)
        all_value_functions.append(dqn(state).max().item())

        if episode % 10 == 0:
            print(f'Episode {episode}, Reward: {episode_rewards}, Epsilon: {epsilon}')

    return all_rewards, all_value_functions

# Train the DQN agent
num_episodes = 1000
batch_size = 32
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = 0.995
lr = 1e-3

rewards, value_functions = train_dqn(env, dqn, num_episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay, lr)

# Plot the average reward per episode
plt.figure(figsize=(12, 6))
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Average Reward per Episode')
plt.show()

# Plot the average value function per episode
plt.figure(figsize=(12, 6))
plt.plot(value_functions)
plt.xlabel('Episode')
plt.ylabel('Average Value Function')
plt.title('Average Value Function per Episode')
plt.show()
