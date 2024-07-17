import numpy as np
import csv
import itertools

# Environment parameters
N = 8  # Number of stocks, you can change this to any value
c = 0.1  # Transaction fee
gamma = 0.99  # Discount factor

# Reward and transition probabilities
#np.random.seed(0)  # For reproducibility
r_H = np.random.uniform(-0.02, 0.1, N)
r_L = np.random.uniform(-0.02, 0.1, N)
p_HL = np.array([0.1 if i < N // 2 else 0.5 for i in range(N)])
p_LH = np.array([0.1 if i < N // 2 else 0.5 for i in range(N)])
p_HH = 1 - p_HL
p_LL = 1 - p_LH

# State space
states = list(itertools.product(range(N), itertools.product(['H', 'L'], repeat=N)))

# Action space
actions = list(range(N))

# Helper function to get the state index
def get_state_index(state):
    return states.index(state)

# Transition probabilities and rewards
def get_transition_prob_reward(state, action):
    i, stock_states = state
    next_states = []
    stock_states = list(stock_states)
    if action == i:  # Stay with the current stock
        if stock_states[action] == 'H':
            next_states.append(((i, tuple(stock_states)), p_HH[action], r_H[action]))
            stock_states[action] = 'L'
            next_states.append(((i, tuple(stock_states)), p_HL[action], r_H[action]))
        else:
            next_states.append(((i, tuple(stock_states)), p_LH[action], r_L[action]))
            stock_states[action] = 'H'
            next_states.append(((i, tuple(stock_states)), p_LL[action], r_L[action]))
    else:  # Switch to a different stock
        if stock_states[action] == 'H':
            next_states.append(((action, tuple(stock_states)), p_HH[action], r_H[action] - c))
            stock_states[action] = 'L'
            next_states.append(((action, tuple(stock_states)), p_HL[action], r_H[action] - c))
        else:
            next_states.append(((action, tuple(stock_states)), p_LH[action], r_L[action] - c))
            stock_states[action] = 'H'
            next_states.append(((action, tuple(stock_states)), p_LL[action], r_L[action] - c))
    return next_states

# Policy Iteration
def policy_evaluation(policy, V, theta=1e-6):
    while True:
        delta = 0
        for state in states:
            v = V[get_state_index(state)]
            action = policy[get_state_index(state)]
            new_v = 0
            for next_state, prob, reward in get_transition_prob_reward(state, action):
                new_v += prob * (reward + gamma * V[get_state_index(next_state)])
            V[get_state_index(state)] = new_v
            delta = max(delta, abs(v - new_v))
        if delta < theta:
            break
    return V

def policy_improvement(V, policy):
    policy_stable = True
    new_policy = policy.copy()
    for state in states:
        old_action = policy[get_state_index(state)]
        action_values = np.zeros(len(actions))
        for action in actions:
            for next_state, prob, reward in get_transition_prob_reward(state, action):
                action_values[action] += prob * (reward + gamma * V[get_state_index(next_state)])
        new_action = np.argmax(action_values)
        new_policy[get_state_index(state)] = new_action
        if new_action != old_action:
            policy_stable = False
    return new_policy, policy_stable

def policy_iteration():
    V = np.zeros(len(states))
    policy = np.random.choice(actions, len(states))

    while True:
        V = policy_evaluation(policy, V)
        policy, policy_stable = policy_improvement(V, policy)
        if policy_stable:
            break
    return policy, V

optimal_policy, V = policy_iteration()

# Display the optimal policy
print("Optimal Policy:")
for state in states:
    print(f"State {state}: Invest in stock {optimal_policy[get_state_index(state)]}")

print("State Values:")
for state in states:
    print(f"State {state}: {V[get_state_index(state)]:.4f}")

# Output optimal policy and state values to CSV
output_file_policy = 'optimal_policy.csv'
with open(output_file_policy, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["State", "Optimal Action", "State Value"])
    for state in states:
        writer.writerow([state, optimal_policy[get_state_index(state)], V[get_state_index(state)]])

# Output transition probabilities and rewards to CSV
output_file_params = 'stock_params.csv'
with open(output_file_params, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Stock", "p_HH", "p_LH", "p_LL", "p_HL", "r_H", "r_L"])
    for i in range(N):
        writer.writerow([i, p_HH[i], p_LH[i], p_LL[i], p_HL[i], r_H[i], r_L[i]])

print(f"Results have been written to {output_file_policy} and {output_file_params}")
