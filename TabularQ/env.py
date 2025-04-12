# env.py
import pandas as pd

class HFNCEnvironment:
    def __init__(self, data_path: str):
        """
        Initialize the HFNC environment with the preprocessed data.
        :param data_path: Path to the preprocessed CSV file.
        """
        self.df = pd.read_csv(data_path)
        # Ensure data is sorted by patient identifier and time (hour_ts)
        self.df.sort_values(['stay_id', 'hour_ts'], inplace=True)
        # Group data by 'stay_id' so that each group represents one patient trajectory (episode)
        self.episodes = [group.reset_index(drop=True) for _, group in self.df.groupby('stay_id')]
        self.current_episode = None
        self.current_step = 0

    def reset(self):
        """
        Start a new episode by selecting the next patient trajectory.
        Returns the initial state for the episode.
        """
        if not self.episodes:
            raise Exception("No more episodes available. Reset the environment or reload data.")
        
        # For now, simply pop the first episode; later you might randomize or cycle.
        self.current_episode = self.episodes.pop(0)
        self.current_step = 0
        return self._get_state(self.current_episode, self.current_step)

    def _get_state(self, episode, step):
        """
        Extract the state at the specified step for the given episode.
        This function should return the observation as a dictionary. Adjust based on the features you want.
        For our purposes, we will include SpO₂, respiratory rate, FiO₂, and O₂ flow.
        """
        row = episode.iloc[step]
        state = {
            "spo2": row['spo2'],
            "resp_rate": row['resp_rate'],
            "fio2": row['fio2'],
            "o2_flow": row['o2_flow']
        }
        return state

    def step(self, action):
        """
        Advance the environment by one step given an action.
        :param action: The action chosen by the agent (currently a placeholder value).
        :return: next_state, reward, done flag.
        
        For now, the state transition is simulated using the actual next observation from historical data.
        The reward is computed simply based on the next state's SpO₂.
        """
        # Increment step counter
        self.current_step += 1
        done = (self.current_step >= len(self.current_episode) - 1)
        
        # If not done, get the next state; otherwise, return None for state
        next_state = self._get_state(self.current_episode, self.current_step) if not done else None
        
        # A simple placeholder reward:
        # +1 if SpO₂ is within the optimal range (92-96), -1 otherwise.
        current_spo2 = self.current_episode.iloc[self.current_step]['spo2']
        if 92 <= current_spo2 <= 96:
            reward = 1
        else:
            reward = -1

        return next_state, reward, done

# Example usage (for testing purposes only)
if __name__ == "__main__":
    # Replace with your actual file path
    env = HFNCEnvironment('data/preprocessed_data_reduced_3.csv')
    state = env.reset()
    print("Initial state:", state)
    
    done = False
    while not done:
        # For now, action is just a placeholder (e.g., 0)
        action = 0
        next_state, reward, done = env.step(action)
        print("Next state:", next_state, "Reward:", reward, "Done:", done)