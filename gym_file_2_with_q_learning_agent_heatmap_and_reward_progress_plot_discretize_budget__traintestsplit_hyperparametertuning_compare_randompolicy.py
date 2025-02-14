import gym
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

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
    def __init__(self, state_shape, action_size, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
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

# Random policy agent
class RandomAgent:
    def __init__(self, action_size):
        self.action_size = action_size
    
    def choose_action(self, _state):
        return random.choice(range(self.action_size))

# Train Q-learning agent
if __name__ == "__main__":
    env = AdCampaignEnv("ad_campaigns.csv", train=True)  # Train environment
    agent = QLearningAgent(state_shape=env.observation_space.nvec, action_size=len(env.channels))
    
    episodes = 1000
    rewards_history = []
    
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
        rewards_history.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
    # Evaluate Q-learning agent
    test_env = AdCampaignEnv("ad_campaigns.csv", train=False)
    test_rewards = []
    for _ in range(100):  # Run 100 test episodes
        state = test_env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state)
            state, reward, done, _ = test_env.step(action)
            total_reward += reward
        test_rewards.append(total_reward)
    
    print(f"Q-learning Average Test Reward: {np.mean(test_rewards)}")
    
    # Evaluate Random Agent
    random_agent = RandomAgent(action_size=len(env.channels))
    random_rewards = []
    for _ in range(100):  # Run 100 test episodes
        state = test_env.reset()
        total_reward = 0
        done = False
        while not done:
            action = random_agent.choose_action(state)
            state, reward, done, _ = test_env.step(action)
            total_reward += reward
        random_rewards.append(total_reward)
    
    print(f"Random Policy Average Test Reward: {np.mean(random_rewards)}")
    
    # Plot reward progression
    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, label='Total Reward per Episode')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Q-learning Training Progress')
    plt.legend()
    plt.show()
    
    # Plot Q-table heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(agent.q_table.mean(axis=-1), cmap='coolwarm', annot=False)
    plt.xlabel("Actions (Channels)")
    plt.ylabel("States")
    plt.title("Q-Table Heatmap")
    plt.show()




"""
The next step is hyperparameter tuning and policy evaluation. We'll experiment with different values of learning rate (alpha), discount factor (gamma), and exploration rate (epsilon). Additionally, we can compare the Q-learning policy against a random policy or a heuristic allocation strategy (e.g., evenly splitting budget across all channels).

Would you like to:

Tune hyperparameters (grid search different values)?
Compare against baseline policies (random or heuristic allocation)?
Visualize performance trends (ROI per channel over time)?

"""
"""
Compare against baseline policies (random or heuristic allocation)?

Random policy

compare the Q-learning agent against baseline policies.

Next Steps:
Random Policy Benchmark:

Implement a simple agent that randomly allocates budget to channels.
Run it on the same environment and compare its performance.
Heuristic-Based Policy:

Create a rule-based strategy (e.g., allocating budget proportionally to past ROI).
Compare it against Q-learning.
Performance Analysis:

Plot and compare the total reward (ROI) from Q-learning, random policy, and heuristic-based allocation.
"""

"""
Now, the script evaluates both the Q-learning agent and a random agent over 100 test episodes and prints their average rewards. 
It also plots the training progression and a heatmap of the Q-table. """
