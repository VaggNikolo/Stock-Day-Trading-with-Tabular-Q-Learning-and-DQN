import EnvSetup
import numpy as np
import DQN_Agent
import pandas as pd
import tensorflow as tf

file_path = 'stock_params.csv'
stock_params_df = pd.read_csv(file_path)
transaction_fee = 0.01
gamma = 0.99

env = EnvSetup.StockMarketEnv(stock_params_df, transaction_fee, gamma)
state_size = len(stock_params_df)
action_size = len(stock_params_df)
agent = DQN_Agent.DQNAgent(state_size, action_size)

# Training loop
episodes = 1000
scores = []

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            agent.update_target_model()
            break
        agent.replay()
    
    scores.append(total_reward)
    print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward:.4f}, Epsilon: {agent.epsilon:.4f}")

print("Training finished.")