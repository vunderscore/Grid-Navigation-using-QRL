# Quantum Reinforcement Learning on a 7×7 Maze using PPO

This repository implements a **custom Proximal Policy Optimization (PPO)** algorithm designed from scratch for Quantum Reinforcement Learning (QRL) experiments. The goal is to evaluate and compare the performance of hybrid quantum-classical agents against classical reinforcement learning models in grid-based navigation environments.

The project also serves as a **plug-and-play template** for future research on quantum or hybrid reinforcement learning. It is modular, adaptable, and fully reproducible across different grid sizes and architectures.

---

## Project Overview

This work explores quantum-enhanced reinforcement learning through a 7×7 maze navigation task.  
Agents are trained using:
- **Quantum PPO (QPPO)**: A hybrid model integrating parameterized quantum circuits (PQCs) with a classical PPO framework.
- **Classical PPO**: A baseline model built using Stable Baselines3 with identical environment and reward setup.

Each agent learns to navigate from a start position to a target goal while avoiding static obstacles.  
Reward functions and encoding schemes are designed to test scalability and generalization between classical and quantum agents.

---

## Key Features

- **Custom PPO Implementation**:  
  PPO algorithm fully implemented from scratch using PyTorch, integrating GAE (Generalized Advantage Estimation), entropy annealing, and gradient clipping.

- **Quantum Policy Network**:  
  A hybrid model combining classical preprocessing layers with a parameterized quantum circuit (PQC) built using **PennyLane**.

- **Plug-and-Play Template**:  
  The architecture and training framework can be easily reused and extended for new environments or alternative quantum circuit layouts.

- **Unified Environment Interface**:  
  Same `MazeEnv` instance can be used with either classical PPO or quantum PPO without modification.

- **Logging and Visualization**:  
  Detailed episode-wise metrics (reward, success rate, entropy, loss trends) automatically logged and plotted.

---

## Repository Structure

