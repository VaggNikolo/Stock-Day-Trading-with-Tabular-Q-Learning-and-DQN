import numpy as np
import gym
from gym import spaces

class InvestmentEnv(gym.Env):
    def __init__(self, N, c, r, p):
        super(InvestmentEnv, self).__init__()
        self.N = N  # Number of stocks
        self.c = c  # Transaction fee
        self.r = r  # Expected gains matrix
        self.p = p  # Transition probabilities matrix
        self.state = self._get_initial_state()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(N)  # Choose one of N stocks
        self.observation_space = spaces.Box(low=0, high=1, shape=(N,), dtype=np.float32)

    def _get_initial_state(self):
        # Randomly initialize the state of each stock (0 for Low, 1 for High)
        return np.random.choice([0, 1], size=(self.N,))

    def step(self, action):
        # Calculate the reward based on the chosen action
        reward = self.r[action][self.state[action]]
        
        # Apply transaction fee if switching stocks
        if action != self.current_stock:
            reward -= self.c
        
        self.current_stock = action
        
        # Transition to next state based on Markov chain
        new_state = []
        for i in range(self.N):
            if self.state[i] == 1:
                new_state.append(np.random.choice([0, 1], p=[self.p[i][1][0], self.p[i][1][1]]))
            else:
                new_state.append(np.random.choice([0, 1], p=[self.p[i][0][0], self.p[i][0][1]]))
        
        self.state = np.array(new_state)
        
        return self.state, reward, False, {}

    def reset(self):
        self.state = self._get_initial_state()
        self.current_stock = np.random.choice(self.N)
        return self.state

    def render(self, mode='human', close=False):
        pass
