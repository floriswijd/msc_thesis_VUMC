#!/usr/bin/env python3
# -----------------------------------------------------------
# dataset.py  --  MDP Dataset handling for HFNC CQL
# 
# This module handles the creation and management of MDP datasets:
# - Creating MDPDataset objects from processed arrays
# - Splitting datasets into train/validation/test sets
# - Counting transitions for dataset analysis
# 
# The MDPDataset is a critical component that d3rlpy uses to
# represent reinforcement learning trajectories as episodes.
# -----------------------------------------------------------
from d3rlpy.dataset import MDPDataset, Episode # Ensure Episode is imported
from sklearn.model_selection import train_test_split
import numpy as np

def create_mdp_dataset(observations_arr, actions_arr, rewards_arr, dones_arr):
    """
    Creates an MDPDataset from numpy arrays, respecting episode boundaries.

    Args:
        observations_arr (np.ndarray): Array of observations.
        actions_arr (np.ndarray): Array of actions.
        rewards_arr (np.ndarray): Array of rewards.
        dones_arr (np.ndarray): Array of done flags (boolean), marking episode ends.

    Returns:
        MDPDataset: A d3rlpy MDPDataset object containing the episodes.
    """
    episodes_list = []
    current_episode_start_idx = 0
    for i in range(len(dones_arr)):
        if dones_arr[i]:  # If this is the last step of an episode
            # Extract data for the current episode
            # Episode data includes the terminal state, action, and reward
            episode_observations = observations_arr[current_episode_start_idx : i + 1]
            episode_actions = actions_arr[current_episode_start_idx : i + 1]
            episode_rewards = rewards_arr[current_episode_start_idx : i + 1]

            if len(episode_observations) > 0: # Ensure episode is not empty
                # d3rlpy.dataset.Episode expects observations, actions, rewards.
                # For discrete actions, actions should be shaped (N,) or (N, 1).
                # If actions_arr is 1D, episode_actions will be 1D, which is fine.
                ep = Episode(
                    observations=episode_observations,
                    actions=episode_actions, # Ensure this is (episode_length,) or (episode_length, 1)
                    rewards=episode_rewards
                    # terminals are implicitly handled by d3rlpy based on episode structure
                )
                episodes_list.append(ep)
            current_episode_start_idx = i + 1  # Next episode starts at the next index

    # Handle the case where the data doesn't end with a 'done' flag for the last segment
    if current_episode_start_idx < len(dones_arr):
        print(f"Warning: Data might not end with a terminal state. Processing {len(dones_arr) - current_episode_start_idx} remaining transitions as a partial episode.")
        episode_observations = observations_arr[current_episode_start_idx:]
        episode_actions = actions_arr[current_episode_start_idx:]
        episode_rewards = rewards_arr[current_episode_start_idx:]
        if len(episode_observations) > 0: # Only add if there's actual data
            ep = Episode(
                observations=episode_observations,
                actions=episode_actions,
                rewards=episode_rewards,
            )
            episodes_list.append(ep)
    
    print(f"Created {len(episodes_list)} episodes from the provided data.")
    return MDPDataset(episodes=episodes_list)


def split_dataset(mdp_dataset_full, test_size=0.3, val_size=0.5, random_state=42):
    """
    Splits the MDPDataset into training, validation, and test sets at the episode level.

    Args:
        mdp_dataset_full (MDPDataset): The full dataset containing all episodes.
        test_size (float): Proportion of episodes to include in the test split.
        val_size (float): Proportion of the remaining (non-test) episodes to include in the validation split.
        random_state (int): Seed for random number generator for reproducibility.

    Returns:
        tuple: (train_episodes, val_episodes, test_episodes)
               Each element is a list of d3rlpy.dataset.Episode objects.
    """
    all_episodes = mdp_dataset_full.episodes
    
    if not all_episodes:
        print("Warning: No episodes found in the dataset to split.")
        return [], [], []

    # Split into training+validation and test sets
    # Ensure there are enough episodes to split
    if len(all_episodes) < 2 : # Need at least 2 episodes to make a split meaningful
        print("Warning: Not enough episodes to perform a meaningful train/test split. Returning all episodes as training.")
        return all_episodes, [], []

    train_val_episodes, test_episodes = train_test_split(
        all_episodes,
        test_size=test_size,
        random_state=random_state,
        shuffle=True  # Shuffle episodes before splitting
    )

    # Split training+validation into training and validation sets
    # Ensure there are enough episodes in train_val_episodes for a further split
    if len(train_val_episodes) < 2 or val_size == 0.0:
        # If not enough to split or val_size is 0, all remaining go to train
        train_episodes = train_val_episodes
        val_episodes = []
    else:
        train_episodes, val_episodes = train_test_split(
            train_val_episodes,
            test_size=val_size,  # val_size is proportion of train_val_episodes to become val_episodes
            random_state=random_state,  # Consistent shuffling if desired
            shuffle=True
        )
    
    print(f"Dataset split: {len(train_episodes)} train, {len(val_episodes)} val, {len(test_episodes)} test episodes.")
    return train_episodes, val_episodes, test_episodes

def count_transitions(episodes_list):
    """
    Counts the total number of transitions in a list of episodes.

    Args:
        episodes_list (list): A list of d3rlpy.dataset.Episode objects.

    Returns:
        int: The total number of transitions.
    """
    if not episodes_list:
        return 0
    return sum(episode.size() for episode in episodes_list)