import numpy as np

class StockMarketEnv:
    def __init__(self, stock_params_df, transaction_fee, gamma):
        self.N = len(stock_params_df)  # Number of stocks
        self.transaction_fee = transaction_fee  # Transaction fee for switching stocks
        self.gamma = gamma  # Discount factor
        
        # Parse transition probabilities and rewards from the dataframe
        self.transition_probabilities = []
        self.rewards = []
        for _, row in stock_params_df.iterrows():
            self.transition_probabilities.append({
                'HH': row['p_HH'],
                'HL': row['p_HL'],
                'LL': row['p_LL'],
                'LH': row['p_LH']
            })
            self.rewards.append({
                'H': row['r_H'],
                'L': row['r_L']
            })
        
        self.state = np.random.choice(['H', 'L'], self.N)  # Initial state
        self.current_stock = np.random.randint(self.N)  # Initial stock choice
    
    def step(self, action):
        if action == self.current_stock:
            reward = self.rewards[self.current_stock][self.state[self.current_stock]]
        else:
            reward = self.rewards[action][self.state[action]] - self.transaction_fee
        
        for i in range(self.N):
            if self.state[i] == 'H':
                self.state[i] = 'H' if np.random.rand() < self.transition_probabilities[i]['HH'] else 'L'
            else:
                self.state[i] = 'L' if np.random.rand() < self.transition_probabilities[i]['LL'] else 'H'
        
        self.current_stock = action
        return self.get_state(), reward, False
    
    def get_state(self):
        return np.array([self.rewards[i][self.state[i]] for i in range(self.N)])
    
    def reset(self):
        self.state = np.random.choice(['H', 'L'], self.N)
        self.current_stock = np.random.randint(self.N)
        return self.get_state()