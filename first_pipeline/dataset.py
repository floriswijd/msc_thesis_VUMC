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
from d3rlpy.dataset import MDPDataset # MDPDataset itself handles Episode creation internally here
from sklearn.model_selection import train_test_split
import numpy as np # Ensure numpy is imported if not already

def create_mdp_dataset(states, actions, rewards, dones):
    """
    Create an MDP dataset from preprocessed arrays.

    Args:
        states (np.ndarray): Array of state observations.
        actions (np.ndarray): Array of actions taken.
        rewards (np.ndarray): Array of rewards received.
        dones (np.ndarray): Array of done flags (boolean), marking episode ends.

    Returns:
        MDPDataset: A d3rlpy MDPDataset object containing the episodes.

    Note:
        The MDPDataset constructor automatically segments the data into episodes
        based on the 'terminals' array (previously 'dones'), creating a list of Episode objects internally.
    """
    try:
        # Explicitly pass 'dones' as the 'terminals' argument.
        # Also, it's good practice to name all arguments for clarity and
        # robustness to changes in argument order in future d3rlpy versions.
        dataset = MDPDataset(
            observations=states,
            actions=actions,
            rewards=rewards,
            terminals=dones  # This is the key change
        )
        print(f"✅  MDP dataset created with {len(dataset.episodes)} episodes.")
        return dataset
    except Exception as e:
        print(f"⚠️  Error creating MDP dataset: {e}")
        # Try to diagnose the issue by checking shapes and types
        print(f"States shape: {states.shape if hasattr(states, 'shape') else 'N/A'}, dtype: {states.dtype if hasattr(states, 'dtype') else 'N/A'}")
        print(f"Actions shape: {actions.shape if hasattr(actions, 'shape') else 'N/A'}, dtype: {actions.dtype if hasattr(actions, 'dtype') else 'N/A'}")
        print(f"Rewards shape: {rewards.shape if hasattr(rewards, 'shape') else 'N/A'}, dtype: {rewards.dtype if hasattr(rewards, 'dtype') else 'N/A'}")
        print(f"Dones (terminals) shape: {dones.shape if hasattr(dones, 'shape') else 'N/A'}, dtype: {dones.dtype if hasattr(dones, 'dtype') else 'N/A'}")
        
        # Check for NaN or inf values in the inputs (basic check)
        if isinstance(states, np.ndarray):
            print(f"NaN in states: {np.isnan(states).any()}")
            print(f"Inf in states: {np.isinf(states).any()}")
        raise

def split_dataset(dataset, test_size=0.3, val_size=0.5, random_state=42):
    """
    Split dataset into train, validation and test sets.

    Args:
        dataset (MDPDataset): The full dataset containing all episodes.
        test_size (float): Proportion of episodes to include in the test split.
        val_size (float): Proportion of the remaining (non-test) episodes to include in the validation split.
        random_state (int): Seed for random number generator for reproducibility.

    Returns:
        tuple: (train_episodes, val_episodes, test_episodes)
               Each element is a list of d3rlpy.dataset.Episode objects.

    Note:
        The episodes are split randomly but deterministically based on random_state.
        This ensures reproducibility while maintaining independence between sets.
    """
    # Ensure dataset has episodes
    if not dataset.episodes:
        print("⚠️ Warning: No episodes in the dataset to split. Returning empty lists.")
        return [], [], []
    
    # Ensure there are enough episodes for all splits
    if len(dataset.episodes) < 2: # Need at least 2 to split meaningfully
        print("⚠️ Warning: Not enough episodes for a full split. Returning all as training.")
        return dataset.episodes, [], []

    # Split into train and temporary sets
    # With default parameters: 70% train, 30% temp
    train_eps, temp_eps = train_test_split(
        dataset.episodes,
        test_size=test_size, # This proportion of dataset.episodes goes to temp_eps
        random_state=random_state,
        shuffle=True # Good practice to shuffle before splitting
    )
    
    # Split temporary set into validation and test sets
    # Ensure temp_eps is not empty and has enough for another split if val_size > 0
    if not temp_eps:
        val_eps, test_eps = [], []
    elif len(temp_eps) < 2 and val_size > 0 and val_size < 1.0 : # Not enough to split temp_eps further
        print("⚠️ Warning: Not enough episodes in temp_eps to split into validation and test. Assigning all to test.")
        val_eps = []
        test_eps = temp_eps
    elif val_size == 0.0: # No validation set requested from temp
        val_eps = []
        test_eps = temp_eps
    elif val_size == 1.0: # All of temp goes to validation
        val_eps = temp_eps
        test_eps = []
    else:
        val_eps, test_eps = train_test_split(
            temp_eps,
            test_size=val_size,  # This is relative to temp_eps; e.g. if val_size=0.5, temp_eps is split 50/50
            random_state=random_state,
            shuffle=True
        )
    
    print(f"📊 Episodes  train={len(train_eps)}, val={len(val_eps)}, test={len(test_eps)}")
    
    return train_eps, val_eps, test_eps

def count_transitions(episodes):
    """
    Count the total number of transitions (steps) in a list of episodes.

    Args:
        episodes (list): A list of d3rlpy.dataset.Episode objects.

    Returns:
        int: The total number of transitions.
    """
    if not episodes: # Handle empty list
        return 0
    return sum(len(episode.observations) for episode in episodes)