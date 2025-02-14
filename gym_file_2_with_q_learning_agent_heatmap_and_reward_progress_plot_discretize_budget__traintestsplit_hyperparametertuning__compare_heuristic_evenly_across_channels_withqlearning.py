import gym
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from itertools import product

class AdCampaignEnv(gym.Env):
    def __init__(self, data_path, bins=10, train=True):
        super(AdCampaignEnv, self).__init__()
        
        # Load dataset
        data = pd.read_csv(data_path)
        
        # Split into train and test sets
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        self.data = train_data if train else test_data
        
        # Normalize numerical columns
        self.columns_to_discretize = ['budget', 'impressions', 'clicks', 'conversions', 'cost_per_click']
        self.bins = bins
        self.bin_edges = {}
        
        for col in self.columns_to_discretize:
            self.bin_edges[col] = np.linspace(self.data[col].min(), self.data[col].max(), bins)
            self.data[col] = np.digitize(self.data[col], self.bin_edges[col]) - 1
        
        # State space: Discretized budget, impressions, clicks, conversions, cost_per_click
        self.observation_space = gym.spaces.MultiDiscrete([bins] * len(self.columns_to_discretize))
        
        # Action space: budget allocation to each channel
        self.action_space = gym.spaces.Discrete(len(self.data['channel'].unique()))
        
        # Available channels
        self.channels = self.data['channel'].unique()
        
        # Initial state
        self.current_index = 0
        self.state = self._get_state()
        
    def _get_state(self):
        row = self.data.iloc[self.current_index]
        return tuple(row[col] for col in self.columns_to_discretize)
    
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
    def __init__(self, state_shape, action_size, alpha, gamma, epsilon, epsilon_decay, epsilon_min=0.01):
        self.state_shape = state_shape
        self.action_size = action_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros(state_shape + (action_size,))
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice(range(self.action_size))  # Explore
        return np.argmax(self.q_table[state])  # Exploit
    
    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state + (action,)] = (1 - self.alpha) * self.q_table[state + (action,)] + self.alpha * (reward + self.gamma * self.q_table[next_state + (best_next_action,)])
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Heuristic allocation: Budget proportional to past conversions
def heuristic_budget_allocation(data):
    conversions_per_channel = data.groupby('channel')['conversions'].sum()
    total_conversions = conversions_per_channel.sum()
    budget_allocation = conversions_per_channel / total_conversions
    return budget_allocation

# Evenly split budget across all channels
def even_budget_allocation(data):
    num_channels = len(data['channel'].unique())
    budget_allocation = {channel: 1 / num_channels for channel in data['channel'].unique()}
    return budget_allocation

# Hyperparameter tuning grid search
alpha_values = [0.1, 0.2, 0.5]
gamma_values = [0.8, 0.9, 0.99]
epsilon_decay_values = [0.995, 0.99, 0.98]

best_params = None
best_reward = -np.inf

for alpha, gamma, epsilon_decay in product(alpha_values, gamma_values, epsilon_decay_values):
    env = AdCampaignEnv("ad_campaigns.csv", train=True)  # Train environment
    agent = QLearningAgent(state_shape=env.observation_space.nvec, action_size=len(env.channels), alpha=alpha, gamma=gamma, epsilon=1.0, epsilon_decay=epsilon_decay)
    
    episodes = 500
    total_rewards = []
    
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
        total_rewards.append(total_reward)
    
    avg_reward = np.mean(total_rewards)
    print(f"Alpha: {alpha}, Gamma: {gamma}, Epsilon Decay: {epsilon_decay} => Avg Reward: {avg_reward}")
    
    if avg_reward > best_reward:
        best_reward = avg_reward
        best_params = (alpha, gamma, epsilon_decay)

print(f"Best Hyperparameters: Alpha={best_params[0]}, Gamma={best_params[1]}, Epsilon Decay={best_params[2]}")

# Compute average ROI per channel
def compute_avg_roi(data):
    avg_roi = data.groupby('channel')['roi'].mean()
    print("Average ROI per Channel:")
    print(avg_roi)
    return avg_roi

env_test = AdCampaignEnv("ad_campaigns.csv", train=False)
compute_avg_roi(env_test.data)

# Visualization of ROI trends
def visualize_roi_trends(data):
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=data, x='budget', y='roi', hue='channel', marker='o')
    plt.xlabel("Budget")
    plt.ylabel("ROI")
    plt.title("ROI Trends per Channel")
    plt.legend(title="Channel")
    plt.show()

visualize_roi_trends(env_test.data)



"""
The next step is to analyze the performance of the Q-learning agent against heuristic and even budget allocation strategies.

Steps:
Compute Key Metrics:

Average ROI per strategy (Q-learning, heuristic, even split).
Total profit generated by each strategy.
Performance comparison of each channel.
Statistical Analysis:

Use t-tests or ANOVA to determine if Q-learning significantly outperforms other strategies.
Analyze variance in performance across different budget levels.
Optimize Further:

If Q-learning is not significantly better, adjust the reward function or exploration strategy.
Test different binning strategies in the environment.
"""