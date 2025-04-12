# q_learning_agent.py
import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        Initialize the Q-learning agent.
        :param action_space: A list of discrete actions.
        """
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # Q-table: maps a composite (discrete) state string to an array of Q-values for each action.
        self.Q_table = defaultdict(lambda: np.zeros(len(action_space)))
        print(f"Agent initialized with {len(action_space)} actions, α={learning_rate}, γ={discount_factor}, ε={epsilon}")

    def discretize_state(self, state: dict) -> str:
        """
        Convert a continuous state into a discrete composite state using our binning strategy.
        The state dictionary is expected to have the keys: 'spo2', 'resp_rate', 'fio2', and 'o2_flow'.
        """
        print(f"Discretizing state: {state}")
        
        # SpO₂ discretization: using clinical thresholds:
        spo2 = state['spo2']
        if spo2 < 88:
            spo2_bin = "<88"
        elif spo2 < 92:
            spo2_bin = "88-91"
        elif spo2 <= 96:
            spo2_bin = "92-96"
        else:
            spo2_bin = ">96"

        # Respiratory Rate discretization (using quantile-like thresholds):
        resp = state['resp_rate']
        if resp <= 18:
            resp_bin = "<=18"
        elif resp <= 22:
            resp_bin = "18-22"
        elif resp <= 27:
            resp_bin = "22-27"
        else:
            resp_bin = ">27"

        # FiO₂ discretization in 5% increments:
        fio2 = state['fio2']
        fio2_low = int(fio2 // 5 * 5)
        fio2_bin = f"{fio2_low}-{fio2_low+5}"

        # O₂ Flow discretization in 5 L/min increments:
        o2_flow = state['o2_flow']
        flow_low = int(o2_flow // 5 * 5)
        o2_flow_bin = f"{flow_low}-{flow_low+5}"

        composite_state = f"spo2:{spo2_bin}|resp:{resp_bin}|fio2:{fio2_bin}|flow:{o2_flow_bin}"
        print(f"Discretized to: {composite_state}")
        return composite_state

    def choose_action(self, state: dict) -> int:
        """
        Use an epsilon-greedy policy to choose an action.
        Returns the index of the chosen action.
        """
        discrete_state = self.discretize_state(state)
        
        # Check if we've seen this state before
        if discrete_state not in self.Q_table:
            print(f"New state encountered: {discrete_state}")
        
        if np.random.rand() < self.epsilon:
            action = np.random.choice(len(self.action_space))
            print(f"Exploration: Random action {action} chosen (ε-greedy)")
        else:
            action = int(np.argmax(self.Q_table[discrete_state]))
            print(f"Exploitation: Greedy action {action} chosen with Q-value {self.Q_table[discrete_state][action]:.4f}")
        
        print(f"Q-values for state: {[f'{q:.4f}' for q in self.Q_table[discrete_state]]}")
        return action

    def update(self, state: dict, action: int, reward: float, next_state: dict, done: bool):
        """
        Update the Q-table for a given state-action transition.
        """
        s = self.discretize_state(state)
        next_s = self.discretize_state(next_state) if next_state is not None else None

        current_q = self.Q_table[s][action]
        print(f"Updating Q for state: {s}, action: {action}")
        print(f"Current Q-value: {current_q:.4f}")
        
        if done or next_s is None:
            target = reward
            print(f"Terminal state or no next state. Target = reward: {reward:.4f}")
        else:
            target = reward + self.discount_factor * np.max(self.Q_table[next_s])
            print(f"Next state max Q: {np.max(self.Q_table[next_s]):.4f}")
            print(f"Target Q: reward + γ * max(Q') = {reward:.4f} + {self.discount_factor} * {np.max(self.Q_table[next_s]):.4f} = {target:.4f}")
        
        self.Q_table[s][action] += self.learning_rate * (target - current_q)
        print(f"Updated Q-value: {self.Q_table[s][action]:.4f} (Δ = {self.learning_rate * (target - current_q):.4f})")

    def save_q_table(self, path: str):
        """
        Save the Q-table to a JSON file.
        """
        import json
        print(f"Saving Q-table with {len(self.Q_table)} states to {path}")
        with open(path, 'w') as f:
            json.dump({k: v.tolist() for k, v in self.Q_table.items()}, f)
        print(f"Q-table saved successfully")

    def load_q_table(self, path: str):
        """
        Load the Q-table from a JSON file.
        """
        import json
        print(f"Loading Q-table from {path}")
        with open(path, 'r') as f:
            loaded = json.load(f)
        self.Q_table = {k: np.array(v) for k, v in loaded.items()}
        print(f"Q-table loaded successfully with {len(self.Q_table)} states")