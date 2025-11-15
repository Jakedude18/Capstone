from dcartPoleAgent import dcartPoleAgent

from tqdm import tqdm  # Progress bar
from matplotlib import pyplot as plt
import numpy as np

import gymnasium as gym

# Training hyperparameters
learning_rate = 0.0001      # How fast to learn (higher = faster but less stable)
n_episodes = 10000            # Number of hands to practice
start_epsilon = 1          # Start with 90% random actions
epsilon_decay = .98        # start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1        # Always keep some exploration




# Create environment and agent
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = dcartPoleAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)




for episode in tqdm(range(n_episodes)):
    # Start a new hand
    obs, info = env.reset()
    done = False

    # Play one complete hand
    while not done:
        # Agent chooses action (initially random, gradually more intelligent)
        action = agent.get_action(obs)

        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Move to next state
        done = terminated or truncated

        # store transition, then attempt a minibatch update (if enough samples)
        agent.store_experience(obs, action, reward, done, next_obs)
        agent.update()
        obs = next_obs

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon()


showEnv = gym.make("CartPole-v1", render_mode="human")
showEnv = showEnv.env

done = True
SEED = 42
showEnv.reset(seed=SEED)
while not done:
    # Agent chooses action (initially random, gradually more intelligent)
    action = agent.get_action(obs)

    # Take action and observe result
    next_obs, reward, terminated, truncated, info = showEnv.step(action)
    done = terminated or truncated

    obs = next_obs


def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window


# Smooth over a 500-episode window
rolling_length = 100
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

# Episode rewards (win/loss performance)
axs[0].set_title("Episode rewards")
reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_ylabel("Average Reward")
axs[0].set_xlabel("Episode")

# Episode lengths (how many actions per hand)
axs[1].set_title("Episode lengths")
length_moving_average = get_moving_avgs(
    env.length_queue,
    rolling_length,
    "valid"
)
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_ylabel("Average Episode Length")
axs[1].set_xlabel("Episode")

# Training error (how much we're still learning)
axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(
    agent.training_error,
    rolling_length,
    "same"
)
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_ylabel("Temporal Difference Error")
axs[2].set_xlabel("Step")

plt.tight_layout()
plt.show()

agent.plot_q_convergence()


