import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from grid_environment import MazeEnv

# ----- Setup -----
env = MazeEnv()
total_timesteps = 20000   # Total number of training steps
increment = 500          # Train for these many steps before each evaluation
num_eval_episodes = 20    # Number of evaluation episodes per checkpoint
max_steps = 500           # Max steps per episode (prevents endless wandering)

metrics = []
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=42,
    n_steps=increment    # Align PPO buffer with logging increments
)

for step_count in range(increment, total_timesteps + increment, increment):
    # ---- Train for 'increment' steps ----
    model.learn(total_timesteps=increment, reset_num_timesteps=False)

    # ---- Evaluate agent after every 'increment' steps ----
    rewards, steps_list, successes = [], [], 0
    for _ in range(num_eval_episodes):
        obs, info = env.reset()
        done, steps, reward_sum = False, 0, 0
        reached_goal = False
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            steps += 1
            done = terminated or truncated
            if terminated and reward > 0:
                reached_goal = True
        rewards.append(reward_sum)
        steps_list.append(steps)
        if reached_goal:
            successes += 1

    metrics.append({
        "timesteps": step_count,
        "avg_reward": np.mean(rewards),
        "avg_steps": np.mean(steps_list),
        "success_rate": successes / num_eval_episodes * 100
    })

# ---- Save metrics ----
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("learning_progress.csv", index=False)

# ---- Plot learning curves ----
# Plot Average Reward
plt.figure(figsize=(8, 5))
plt.plot(metrics_df["timesteps"], metrics_df["avg_reward"], color='blue', label="Avg Reward")
plt.xlabel("Total Training Timesteps")
plt.ylabel("Average Reward")
plt.title("Average Reward over Time")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("avg_reward_curve.png")
plt.show()

# Plot Success Rate
plt.figure(figsize=(8, 5))
plt.plot(metrics_df["timesteps"], metrics_df["success_rate"], color='green', label="Success Rate (%)")
plt.xlabel("Total Training Timesteps")
plt.ylabel("Success Rate (%)")
plt.title("Success Rate over Time")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("success_rate_curve.png")
plt.show()
