import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["console"]}

    def __init__(self):
        super().__init__()
        self.grid = np.array([
            [0, 0, 1, 0, 0, 1, 0],
            [1, 0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 1, 1, 0],
            [2, 0, 0, 1, 0, 3, 0],
        ])
        self.n_rows, self.n_cols = self.grid.shape
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([self.n_rows, self.n_cols])
        self.reward_goal = 10
        self.reward_step = -0.1
        self.reward_obstacle = -1
        self.start_pos = tuple(np.argwhere(self.grid == 2)[0])
        self.goal_pos = tuple(np.argwhere(self.grid == 3)[0])
        self.agent_pos = self.start_pos

    def manhattan(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        return np.array(self.agent_pos, dtype=int), {}

    def step(self, action):
        r, c = self.agent_pos
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dr, dc = moves[action]
        nr, nc = r + dr, c + dc
        prev_dist = self.manhattan(self.agent_pos, self.goal_pos)
        terminated = False
        reward = self.reward_step

        if 0 <= nr < self.n_rows and 0 <= nc < self.n_cols:
            if self.grid[nr, nc] == 1:
                reward += self.reward_obstacle
            else:
                self.agent_pos = (nr, nc)
                if self.grid[nr, nc] == 3:
                    reward += self.reward_goal
                    terminated = True
        else:
            reward = self.reward_obstacle

        new_dist = self.manhattan(self.agent_pos, self.goal_pos)
        # Only a shaping bonus; never a penalty for detours
        if new_dist < prev_dist:
            reward += 0.05

        return (np.array(self.agent_pos, dtype=int), reward, terminated, False, {})

    def render(self):
        grid_show = self.grid.copy()
        r, c = self.agent_pos
        grid_show[r, c] = 8
        chars = {0: '.', 1: '#', 2: 'S', 3: 'E', 8: 'A'}
        print("\n".join(
            "".join(chars[cell] for cell in row)
            for row in grid_show
        ))

    def close(self):
        pass
