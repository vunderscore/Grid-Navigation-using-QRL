import numpy as np

def encode_state(obs, grid_shape=(7,7), n_qubits=6):
    """
    Encode the agent (row,col) and goal (row,col) as sin/cos for each coordinate
    for richer quantum input. 6 qubits: [sin_agent_row, cos_agent_row, sin_agent_col, cos_agent_col,
                                         sin_goal_row, cos_goal_row]
    """
    arr = np.zeros(n_qubits, dtype=np.float32)
    a_row, a_col = obs
    g_row, g_col = grid_shape[0]-2, grid_shape[1]-2  # Defaults in case not set
    # If your env exposes goal coord, use that
    try:
        from grid_environment import MazeEnv
        env = MazeEnv()
        g_row, g_col = env.goal_pos
    except Exception:
        pass
    arr[0] = np.sin(np.pi * a_row / (grid_shape[0] - 1))
    arr[1] = np.cos(np.pi * a_row / (grid_shape[0] - 1))
    arr[2] = np.sin(np.pi * a_col / (grid_shape[1] - 1))
    arr[3] = np.cos(np.pi * a_col / (grid_shape[1] - 1))
    arr[4] = np.sin(np.pi * g_row / (grid_shape[0] - 1))
    arr[5] = np.cos(np.pi * g_row / (grid_shape[0] - 1))
    return arr
