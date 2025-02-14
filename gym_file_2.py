import gym
import numpy as np
import pandas as pd

class AdCampaignEnv(gym.Env):
    def __init__(self, data_path):
        super(AdCampaignEnv, self).__init__()
        
        # Load dataset
        self.data = pd.read_csv(data_path)
        
        # Normalize numerical columns
        self.data['budget'] = self.data['budget'] / self.data['budget'].max()
        self.data['impressions'] = self.data['impressions'] / self.data['impressions'].max()
        self.data['clicks'] = self.data['clicks'] / self.data['clicks'].max()
        self.data['conversions'] = self.data['conversions'] / self.data['conversions'].max()
        
        # State space: budget, impressions, clicks, conversions
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        # Action space: budget allocation to each channel
        self.action_space = gym.spaces.Discrete(len(self.data['channel'].unique()))
        
        # Available channels
        self.channels = self.data['channel'].unique()
        
        # Initial state
        self.current_index = 0
        self.state = self._get_state()
        
    def _get_state(self):
        row = self.data.iloc[self.current_index]
        return np.array([row['budget'], row['impressions'], row['clicks'], row['conversions']], dtype=np.float32)
    
    def step(self, action):
        # Simulating ad campaign performance based on action
        row = self.data.iloc[self.current_index]
        reward = row['roi']  # Using actual ROI as reward
        self.current_index = (self.current_index + 1) % len(self.data)
        self.state = self._get_state()
        done = self.current_index == 0  # End episode when dataset loops
        return self.state, reward, done, {}
    
    def reset(self):
        self.current_index = 0
        self.state = self._get_state()
        return self.state
    
    def render(self, mode='human'):
        print(f"Current State: {self.state}")

# Environment test
if __name__ == "__main__":
    env = AdCampaignEnv("ad_campaigns.csv")  # Replace with actual CSV path
    state = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward:.2f}, Next State: {next_state}")
        if done:
            break
