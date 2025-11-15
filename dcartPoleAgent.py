from collections import deque
import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn



# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)


class dcartPoleAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate=1e-3,
        initial_epsilon=1.0,
        epsilon_decay=0.995,
        final_epsilon=0.05,
        discount_factor=0.99,
        buffer_size=10000,
        batch_size=128,
        target_update_freq=500,
        state_dim=4,
        action_dim=2
    ):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim

        # --- Q networks ---
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        self.target_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # --- Optimizer ---
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        # --- Hyperparameters ---
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0

        # --- Replay buffer ---
        self.replay_buffer = deque(maxlen=buffer_size)

        # Track learning progress
        self.training_error = []

        # For Q-value convergence diagnostics
        self.q_tracking_states = np.array([env.observation_space.sample() for _ in range(10)])  # 10 representative states
        self.q_history = []  # store Q-values of those states over time

    # -------------------------------------------------------
    def get_action(self, obs) -> int:
        """Choose action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return int(torch.argmax(q_values[0]).item())

    # -------------------------------------------------------
    def store_experience(self, obs, action, reward, done, next_obs):
        self.replay_buffer.append((obs, action, reward, done, next_obs))

    # -------------------------------------------------------
    def sample_batch(self):
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs, actions, rewards, dones, next_obs = zip(*batch)

        obs = torch.tensor(obs, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)

        return obs, actions, rewards, dones, next_obs

    # -------------------------------------------------------
    def update(self):
        """Train the Q-network using a minibatch from replay buffer."""
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples yet

        obs, actions, rewards, dones, next_obs = self.sample_batch()

        # Compute current Q-values
        q_values = self.q_net(obs)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using target network
        with torch.no_grad():
            # 1. Use main network to choose best next action
            next_actions = self.q_net(next_obs).argmax(1)

            # 2. Use target network to evaluate those actions
            next_q_values = self.target_net(next_obs).gather(1, next_actions.unsqueeze(1)).squeeze(1)

            target_q = rewards + self.discount_factor * next_q_values * (1 - dones)

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0) 
        self.optimizer.step()

        self.learn_step_counter += 1

        # Update target network periodically
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Optional: track TD error for debugging
        td_error = (target_q - current_q).mean().item()
        self.training_error.append(td_error)

        self.track_q_values()


    # -------------------------------------------------------
    def decay_epsilon(self):
        """Multiplicative epsilon decay."""
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)


    def track_q_values(self):
            """Store Q-values for a fixed set of states."""
            states_tensor = torch.tensor(self.q_tracking_states, dtype=torch.float32)
            with torch.no_grad():
                q_vals = self.q_net(states_tensor).cpu().numpy()
            self.q_history.append(q_vals)

    
    def plot_q_convergence(self):
        """Plot norm of Q-value changes over time for diagnostic purposes."""
        if len(self.q_history) < 2:
            print("Not enough data to plot Q-value convergence.")
            return

        q_history = np.array(self.q_history)
        delta_q = np.linalg.norm(q_history[1:] - q_history[:-1], axis=(1,2))  # norm over states and actions

        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(delta_q)
        plt.xlabel('Tracking step')
        plt.ylabel('||Q(t) - Q(t-1)||')
        plt.title('Q-value Convergence over Training')
        plt.yscale('log')
        plt.grid(True)
        plt.show()