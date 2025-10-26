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
grid_environment.py # Custom 7×7 maze environment
encode_state.py # State encoding for agent and goal positions
quantum_policy.py # Quantum policy model (hybrid PQC + classical heads)
train_qppo_q7.py # Custom PPO training implementation from scratch
classical_baseline.py # Classical PPO baseline using Stable Baselines3
results/ # Folder for generated plots and CSVs
qppo_results.csv # Quantum PPO training logs
learning_progress.csv # Classical PPO training logs


---

## 4. Environment and Reward Setup

**Environment**
- Grid size: 7×7  
- Agent starts at a fixed or randomly chosen position.  
- Goal is defined in a separate grid cell.  
- Obstacles occupy predefined or randomized cells.

**Reward Structure**
- +1 for reaching the goal  
- -1 for hitting an obstacle  
- -0.1 penalty per time step (encourages shorter paths)

This reward setup promotes efficient, goal-oriented navigation.

---

## 5. State Encoding

The environment state (agent and goal positions) is converted into sinusoidal features suitable for quantum parameterization.

Quantum state encoding for the 7×7 maze:
[sin(a_row), cos(a_row), sin(a_col), cos(a_col), sin(g_row), cos(g_col)]


This encoding ensures spatial continuity and aligns with the rotation-based operations of quantum gates.  
The classical PPO baseline uses raw coordinate representations for comparison.

---

## 6. Quantum Policy Network

The quantum policy architecture consists of:
1. A classical preprocessing neural network for input projection.  
2. A parameterized quantum circuit (PQC) consisting of 6 qubits and 4 variational layers.  
3. Classical policy and value heads for PPO optimization.

**Configuration for the 7×7 Maze:**
- Qubits: 6  
- Quantum Layers: 4  
- Preprocessing Units: 64  
- Entropy Coefficient: 0.08 → 0.01 (annealed)  
- Learning Rate: 2e-4  
- Batch Size: 1024  

---

## 7. Training Procedure

The agent is trained using the custom PPO algorithm:
- Episodes: 20,000  
- Max Steps per Episode: 100  
- Optimization uses minibatches and GAE for stable updates.  
- Gradients are clipped, and learning rate scheduling is applied when plateaus are detected.  
- Metrics logged: reward, success rate, entropy, losses, and unique states visited.

All data is automatically saved and visualized as learning curves in the results directory.

---

## 8. Classical PPO Baseline

A classical PPO agent is trained using Stable Baselines3 with an MLP policy.  
The same environment, reward structure, and episode limits are applied to ensure identical experimental conditions.

---

## 9. Results and Logs

During training, detailed logs and plots are automatically generated:
- Reward progression per episode  
- Success rate trends  
- Entropy and loss curves  
- Unique states visited per episode  

All plots are stored in the `results/` folder, and raw CSV data files (`qppo_results.csv` and `learning_progress.csv`) are available for analysis and replication.

---

## 10. Setup and Usage Guide

### Step 1: Clone the Repository
git clone https://github.com/vunderscore/Grid-Navigation-using-QRL.git
cd quantum-ppo-maze

### Step 2: Create a Virtual Environment
**Using venv:**
conda create -n qrl_env python=3.10
conda activate qrl_env

### Step 3: Install Dependencies
pip install -r requirements.txt

### Step 4: Run Quantum PPO Training
python train_ppo.py


After training completes, all result plots and CSV logs will be available in the `results/` folder.

---

## 11. Extending the Framework

This repository is designed as a **scalable and modular QRL template**.  
You can extend it easily by:
- Modifying maze sizes or layouts.  
- Testing new state encodings or quantum circuit topologies.  
- Experimenting with alternative hybrid or value-based RL strategies.

---

## 12. Dependencies

Refer to `requirements.txt` for all dependencies.  
Core libraries:
- PennyLane (for quantum circuit simulation)  
- PyTorch (for PPO implementation)  
- Gymnasium (for environment interaction)  
- Stable Baselines3 (for classical PPO baseline)

---

## 13. License

This project is released under the MIT License.  
You may use, modify, and distribute this code with proper attribution.

---

## 14. Citation

If you use this repository for research or reference, please cite:
Vishaak. "Grid Navigation using QRL", 2025
