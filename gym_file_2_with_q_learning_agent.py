import gym
import numpy as np
import pandas as pd
import random

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
        self.data['cost_per_click'] = self.data['cost_per_click'] / self.data['cost_per_click'].max()
        
        # State space: budget, impressions, clicks, conversions, cost_per_click
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        
        # Action space: budget allocation to each channel
        self.action_space = gym.spaces.Discrete(len(self.data['channel'].unique()))
        
        # Available channels
        self.channels = self.data['channel'].unique()
        
        # Initial state
        self.current_index = 0
        self.state = self._get_state()
        
    def _get_state(self):
        row = self.data.iloc[self.current_index]
        return np.array([row['budget'], row['impressions'], row['clicks'], row['conversions'], row['cost_per_click']], dtype=np.float32)
    
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

# Q-learning agent
class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((state_size, action_size))
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_size))  # Explore
        return np.argmax(self.q_table[state])  # Exploit
    
    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + self.alpha * (reward + self.gamma * self.q_table[next_state, best_next_action])
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Train agent
if __name__ == "__main__":
    env = AdCampaignEnv("ad_campaigns.csv")  # Replace with actual CSV path
    agent = QLearningAgent(state_size=5, action_size=len(env.channels))
    
    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
"""
How it Works:
State Representation:

The environment represents ad campaigns using normalized features (budget, impressions, clicks, conversions, cost_per_click).
Each row in the dataset corresponds to a campaign.
Actions:

The agent selects a channel (Google Ads, Twitter, etc.) to allocate budget.
The action space is discrete, meaning the agent picks one of the available channels.
Reward Function:

The reward is ROI (Return on Investment) from the selected channel.
Higher ROI means the agent is learning to allocate budget effectively.
Q-learning Updates:

The agent chooses actions based on an exploration-exploitation tradeoff.
It updates the Q-table using past rewards to improve future decisions.
Over time, the agent learns the best budget allocation per channel for maximizing ROI.
What’s Next?
Hyperparameter Tuning: Adjust learning rate (alpha), discount factor (gamma), and exploration (epsilon).
More Advanced Models: Try Deep Q-Networks (DQN) for continuous learning.
More Features: Include time-based trends, ad formats, audience targeting in state representation.

"""

"""
Next Steps:
Now that we have the Q-learning agent interacting with the simulated ad campaign environment, we can move on to the next phase:

1️⃣ Debug & Validate Training
✅ Verify Q-table Updates

Print the Q-table after training to ensure it’s learning.
Check if certain channels consistently get higher values.
✅ Analyze Training Progress

Plot total rewards per episode to see if the agent is improving.
Look for convergence (steady rewards over episodes).
2️⃣ Hyperparameter Tuning
Adjust alpha (learning rate), gamma (discount factor), and epsilon (exploration-exploitation).
Test different epsilon_decay values to balance exploration and exploitation.
3️⃣ Model Evaluation
Compare Q-learning performance against random allocation or equal budget distribution.
Run multiple tests to ensure consistency.
4️⃣ MLOps Integration (Next Phase)
Track Experiments with MLflow (log hyperparameters & results).
Automate Training using Airflow (schedule & monitor training jobs).
🔥 Let’s start by debugging & validating training. Do you want a function to visualize Q-table and training progress?
"""