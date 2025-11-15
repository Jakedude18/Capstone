from collections import defaultdict
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class cartPoleAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        state_dim = 4,
        action_dim = 2
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-network
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Target network
        self.target_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.discount_factor = 0.99
        
        # For tracking errors (optional)
        self.training_error = []

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, obs) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (left) or 1 (right)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            # Convert obs to a tensor
            state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # shape: (1, state_dim)

            # Forward pass through Q-network
            q_values = self.q_net(state_tensor)  # shape: (1, num_actions)

            # Pick action with max Q-value
            action = int(torch.argmax(q_values[0]).item()) #last edit
            print(action)
            return action

    def update(
        self,
        obs, 
        action: int,
        reward: float,
        terminated: bool,
        next_obs
    ):
        """Deep Q-Learning update using a single experience."""
    
        # Convert obs/next_obs to tensors
        state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)      # shape (1, state_dim)
        next_state = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)

        # Predict current Q-value for this action
        q_values = self.q_net(state)                 # shape (1, num_actions)
        current_q = q_values[0, action]             # scalar

        # Compute target Q-value (using target network)
        with torch.no_grad():
            next_q_values = self.target_net(next_state)  # shape (1, num_actions)
            max_next_q = next_q_values.max(1)[0]         # scalar
            target = reward + self.discount_factor * max_next_q * (not terminated)

        # Compute loss (temporal difference)
        loss = nn.MSELoss()(current_q, target)

        # Backpropagate
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Optional: track TD error for debugging
        td_error = (target - current_q).item()
        self.training_error.append(td_error)


    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)



   
