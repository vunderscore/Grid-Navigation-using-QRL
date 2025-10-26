import numpy as np
import torch
import torch.optim as optim
from quantum_policy import QuantumPolicy
from encode_state import encode_state
from grid_environment import MazeEnv
import pandas as pd
import matplotlib.pyplot as plt

n_qubits = 6
n_layers = 4
n_actions = 4
preprocessing_dim = 64
episodes = 20000
max_steps = 500
learning_rate = 2e-4
gamma = 0.99
clip_ratio = 0.2
ppo_epochs = 8
batch_size = 1024
gae_lambda = 0.95
entropy_coef_start = 0.13  # Anneal as training progresses
entropy_coef_end = 0.03
value_coef = 1.0
grad_clip = 1.0
entropy_anneal_steps = int(episodes * 0.8)

env = MazeEnv()
policy_net = QuantumPolicy(n_qubits=n_qubits, n_actions=n_actions, n_layers=n_layers, preprocessing_dim=preprocessing_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=700, factor=0.9)

history = {
    "episode": [], "reward": [], "steps": [], "success": [],
    "mean_reward": [], "median_reward": [], "std_reward": [],
    "entropy": [], "policy_loss": [], "value_loss": [],
    "unique_states": [], "max_reward": []
}
transitions = []

def get_entropy_coef(ep):
    """Linearly anneal entropy coefficient from start to end over entropy_anneal_steps episodes."""
    return max(entropy_coef_end, entropy_coef_start - (entropy_coef_start-entropy_coef_end)*min(ep,entropy_anneal_steps)/entropy_anneal_steps)

def compute_gae(rewards, values, next_value, gamma=0.99, lam=0.95):
    values = values + [next_value]
    advantage = 0
    advantages = []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        advantage = delta + gamma * lam * advantage
        advantages.insert(0, advantage)
    return advantages

def push_and_maybe_learn(ep):
    global transitions
    if len(transitions) < batch_size:
        return
    obs_batch, action_batch, reward_batch, next_obs_batch, logprob_batch, value_batch, entropy_batch = zip(*transitions)
    transitions = []
    arr = np.array([encode_state(next_obs_batch[-1], grid_shape=(7,7), n_qubits=n_qubits)], dtype=np.float32)
    with torch.no_grad():
        _, next_value = policy_net(torch.from_numpy(arr))
    advantages = compute_gae(list(reward_batch), list(value_batch), next_value.item(), gamma, gae_lambda)
    returns = [adv + val for adv, val in zip(advantages, list(value_batch))]
    obs_tensor  = torch.stack([torch.from_numpy(np.array(encode_state(obs, (7,7), n_qubits), dtype=np.float32)) for obs in obs_batch])
    action_tensor = torch.tensor(action_batch, dtype=torch.int64)
    old_logprob_tensor = torch.stack(logprob_batch)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
    entropy_batch = torch.tensor(entropy_batch, dtype=torch.float32)

    entropy_coef = get_entropy_coef(ep)

    for _ in range(ppo_epochs):
        inds = np.arange(len(obs_tensor))
        np.random.shuffle(inds)
        for start in range(0, len(inds), 256):
            end = start + 256
            mb_idx = inds[start:end]
            if len(mb_idx) == 0: continue
            minibatch_obs = obs_tensor[mb_idx]
            minibatch_actions = action_tensor[mb_idx]
            minibatch_returns = returns_tensor[mb_idx]
            minibatch_advantages = advantages_tensor[mb_idx]
            minibatch_old_logprobs = old_logprob_tensor[mb_idx]
            action_probs, values = policy_net(minibatch_obs)
            dist = torch.distributions.Categorical(action_probs)
            entropy = dist.entropy().mean()
            new_logprobs = dist.log_prob(minibatch_actions)
            ratios = torch.exp(new_logprobs - minibatch_old_logprobs)
            surr1 = ratios * minibatch_advantages
            surr2 = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * minibatch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (minibatch_returns - values).pow(2).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), grad_clip)
            optimizer.step()
        history["entropy"].append(entropy.item())
        history["policy_loss"].append(policy_loss.item())
        history["value_loss"].append(value_loss.item())
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(policy_loss.item())
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr != old_lr:
        print(f"Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")

for episode in range(1, episodes + 1):
    obs, _ = env.reset()
    total_reward, steps = 0, 0
    done, success = False, False
    episode_rewards = []
    visited_states = set()
    for step in range(max_steps):
        arr = np.array([encode_state(obs, (7,7), n_qubits)], dtype=np.float32)
        obs_tensor = torch.from_numpy(arr)
        with torch.no_grad():
            action_probs, value = policy_net(obs_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample().item()
        log_prob = torch.log(action_probs.squeeze(0)[action] + 1e-8)
        entropy = dist.entropy()[0]
        obs2, reward, terminated, truncated, _ = env.step(action)
        transitions.append((obs, action, reward, obs2, log_prob, value.item(), entropy.item()))
        total_reward += reward
        obs = obs2
        steps += 1
        episode_rewards.append(reward)
        visited_states.add(tuple(obs))
        done = terminated or truncated
        if terminated and reward > 0:
            success = True
        if done: break
    push_and_maybe_learn(episode)
    history["episode"].append(episode)
    history["reward"].append(total_reward)
    history["steps"].append(steps)
    history["success"].append(int(success))
    history["mean_reward"].append(np.mean(episode_rewards))
    history["median_reward"].append(np.median(episode_rewards))
    history["std_reward"].append(np.std(episode_rewards))
    history["unique_states"].append(len(visited_states))
    history["max_reward"].append(np.max(episode_rewards))
    if episode % 20 == 0:
        avg_reward = np.mean(history["reward"][-20:])
        avg_success = np.mean(history["success"][-20:])*100
        avg_entropy = np.mean(history["entropy"][-20:]) if history["entropy"] else float('nan')
        print(f"Ep {episode}: avg_reward={avg_reward:.2f}, avg_success={avg_success:.1f}%, last_step={steps}, avg_entropy={avg_entropy:.3f}, unique_states={history['unique_states'][-1]}, LR={optimizer.param_groups[0]['lr']:.2e}")

df = pd.DataFrame(history)
df.to_csv("qppo_results.csv", index=False)

plt.figure(figsize=(8,4))
plt.plot(df["episode"], df["reward"], label="Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.title("Quantum PPO RL Reward per Episode")
plt.tight_layout()
plt.savefig("qppo_reward_curve.png")
plt.show()

plt.figure(figsize=(8,4))
plt.plot(df["episode"], pd.Series(df["success"]).rolling(50).mean()*100, label="Success Rate (50ep MA)")
plt.xlabel("Episode")
plt.ylabel("Success Rate (%)")
plt.legend()
plt.title("Quantum PPO RL Success Rate")
plt.tight_layout()
plt.savefig("qppo_success_curve.png")
plt.show()

plt.figure(figsize=(8,4))
plt.plot(df["episode"], df["entropy"], label="Policy Entropy")
plt.xlabel("Episode")
plt.ylabel("Entropy")
plt.legend()
plt.title("Policy Entropy Over Time")
plt.tight_layout()
plt.savefig("qppo_entropy_curve.png")
plt.show()

plt.figure(figsize=(8,4))
plt.plot(df["episode"], df["unique_states"], label="Unique States Visited")
plt.xlabel("Episode")
plt.ylabel("Unique States")
plt.legend()
plt.title("Unique States per Episode")
plt.tight_layout()
plt.savefig("qppo_unique_states.png")
plt.show()

plt.figure(figsize=(8,4))
plt.plot(df["episode"], df["policy_loss"], label="Policy Loss")
plt.plot(df["episode"], df["value_loss"], label="Value Loss")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.legend()
plt.title("PPO Loss Curves")
plt.tight_layout()
plt.savefig("qppo_loss_curve.png")
plt.show()
