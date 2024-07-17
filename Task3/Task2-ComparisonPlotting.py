import numpy as np
import random
import itertools
import csv
import matplotlib.pyplot as plt

# Parameters
N = 8  # Adjust based on your actual problem
gamma = 0.99  # Discount factor
alpha = 0.1  # Learning rate
epsilon = 0.1  # Exploration rate
num_episodes = 15000  # Number of episodes for training

# Initialize transition probabilities and rewards from CSV file
def load_params_from_csv(file_path):
    p_HH, p_LH, p_LL, p_HL, r_H, r_L = [], [], [], [], [], []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            p_HH.append(float(row['p_HH']))
            p_LH.append(float(row['p_LH']))
            p_LL.append(float(row['p_LL']))
            p_HL.append(float(row['p_HL']))
            r_H.append(float(row['r_H']))
            r_L.append(float(row['r_L']))
    return np.array(p_HH), np.array(p_LH), np.array(p_LL), np.array(p_HL), np.array(r_H), np.array(r_L)

p_HH, p_LH, p_LL, p_HL, r_H, r_L = load_params_from_csv('stock_params.csv')

# Generate all possible states
def generate_states(N):
    states = []
    for i in range(1, N + 1):
        for stock_states in itertools.product(['H', 'L'], repeat=N):
            states.append((i, *stock_states))
    return states

states = generate_states(N)
actions = list(range(1, N + 1))

# Initialize Q-table
Q = {state: {action: 0 for action in actions} for state in states}

# Define the reward function using loaded parameters
def get_reward(state, action, c=0.1):
    i, *stock_states = state
    reward = r_H[action - 1] if stock_states[action - 1] == 'H' else r_L[action - 1]
    if action != i:
        reward -= c  # Apply transaction cost
    return reward

# Define the transition function using loaded parameters
def get_next_state(state, action):
    i, *stock_states = state
    next_stock_states = []
    for j, stock_state in enumerate(stock_states):
        if stock_state == 'H':
            next_stock_states.append('H' if random.random() < p_HH[j] else 'L')
        else:
            next_stock_states.append('H' if random.random() < p_LH[j] else 'L')
    next_state = (action, *next_stock_states)
    return next_state

# Lists to track metrics
average_rewards_per_episode = []
average_value_functions_per_episode = []

# Q-learning algorithm
for episode in range(num_episodes):
    # Initialize the state
    state = random.choice(states)
    total_reward = 0
    
    for t in range(10000):  # Assume a finite horizon for each episode
        if random.random() < epsilon:
            action = random.choice(actions)  # Explore
        else:
            action = max(Q[state], key=Q[state].get)  # Exploit
        
        # Get the reward and next state
        reward = get_reward(state, action)
        next_state = get_next_state(state, action)
        
        # Update Q-value
        best_next_action = max(Q[next_state], key=Q[next_state].get)
        Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])
        
        # Move to the next state
        state = next_state
        
        # Accumulate reward
        total_reward += reward

    # Compute the average reward for this episode
    average_rewards_per_episode.append(total_reward / 100)
    
    # Compute the average value function for this episode
    average_value_function = np.mean([max(Q[s].values()) for s in states])
    average_value_functions_per_episode.append(average_value_function)

plt.figure(figsize=(12, 5))

# Plot the average reward per episode
plt.subplot(1, 2, 1)
plt.plot(average_rewards_per_episode, label='Average Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Average Reward per Episode')

# Plot the average value function per episode
plt.subplot(1, 2, 2)
plt.plot(average_value_functions_per_episode, label='Average Value Function per Episode')
plt.xlabel('Episode')
plt.ylabel('Average Value Function')
plt.title('Average Value Function per Episode')

plt.tight_layout()
plt.show()

# Extract the optimal policy and state values
optimal_policy = {}
state_values = {}
for state in states:
    best_action = max(Q[state], key=Q[state].get)
    if state[0] == best_action:
        optimal_policy[state] = 'keep'
    else:
        optimal_policy[state] = f'switch to stock {best_action}'
    state_values[state] = max(Q[state].values())

# Print the optimal policy and state values
print("Optimal Policy:")
for state, action in optimal_policy.items():
    print(f"State {state}: {action}")

print("State Values:")
for state, value in state_values.items():
    print(f"State {state}: {value:.4f}")
