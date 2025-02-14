import gym
import numpy as np

class AdCampaignEnv(gym.Env):
    def __init__(self):
        super(AdCampaignEnv, self).__init__()
        
        # State space: budget, impressions, clicks, conversions
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        # Action space: budget allocation to each channel (0-4)
        self.action_space = gym.spaces.Discrete(5)
        
        # Available channels
        self.channels = ['Google Ads', 'Facebook Ads', 'Instagram Ads', 'Twitter Ads', 'LinkedIn Ads']
        
        # Initial state
        self.state = np.random.rand(4)
        
    def step(self, action):
        # Simulating ad campaign performance
        reward = np.random.rand() * 10  # Simulated ROI
        self.state = np.random.rand(4)  # New campaign state
        done = np.random.rand() > 0.95  # End condition probability
        return self.state, reward, done, {}
    
    def reset(self):
        self.state = np.random.rand(4)
        return self.state
    
    def render(self, mode='human'):
        print(f"State: {self.state}")

# Environment test
if __name__ == "__main__":
    env = AdCampaignEnv()
    state = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward:.2f}, Next State: {next_state}")
        if done:
            break
