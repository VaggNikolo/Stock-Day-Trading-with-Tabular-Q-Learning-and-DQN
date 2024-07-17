import numpy as np
import random
import itertools

# Parameters
N = 2  # Number of stocks
gamma = 0  # Discount factor
alpha = 0.1  # Learning rate
epsilon = 0.1  # Exploration rate
num_episodes = 10000  # Number of episodes for training
transaction_cost = 0.12  # Flat transaction fee

# Reward parameters
rewards = {
    1: {'H': 0.1, 'L': -0.02},
    2: {'H': 0.05, 'L': -0.01}
}

# Transition probabilities
transition_probs = {
    1: {'H': {'H': 0.9, 'L': 0.1}, 'L': {'L': 0.9, 'H': 0.1}},
    2: {'H': {'H': 0.9, 'L': 0.1}, 'L': {'L': 0.9, 'H': 0.1}}
}

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

# Define the reward function based on the provided parameters
def get_reward(state, action):
    current_stock, *stock_states = state
    current_state = stock_states[current_stock - 1]
    reward = rewards[action][current_state]
    if action != current_stock:
        reward -= transaction_cost  # Apply transaction cost
    return reward

# Define the transition function based on the provided parameters
def get_next_state(state, action):
    current_stock, *stock_states = state
    next_stock_states = []
    for i in range(N):
        current_state = stock_states[i]
        next_state = 'H' if random.random() < transition_probs[i + 1][current_state]['H'] else 'L'
        next_stock_states.append(next_state)
    next_state = (action, *next_stock_states)
    return next_state

# Q-learning algorithm
for episode in range(num_episodes):
    # Initialize the state
    state = random.choice(states)
    
    for t in range(100):  # Assume a finite horizon for each episode
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
