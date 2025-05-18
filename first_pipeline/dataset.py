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
from d3rlpy.dataset import MDPDataset
from sklearn.model_selection import train_test_split

def create_mdp_dataset(states, actions, rewards, dones):
    """
    Create an MDP dataset from preprocessed arrays.
    
    The MDPDataset is the core data structure in d3rlpy that represents
    reinforcement learning trajectories. It organizes data into episodes
    based on the 'dones' array which marks episode boundaries.
    
    Args:
        states (np.ndarray): Array of state observations with shape (n_steps, state_dim)
        actions (np.ndarray): Array of actions with shape (n_steps,)
        rewards (np.ndarray): Array of rewards with shape (n_steps,)
        dones (np.ndarray): Boolean array indicating episode terminations with shape (n_steps,)
        
    Returns:
        d3rlpy.dataset.MDPDataset: Dataset containing organized episodes
        
    Raises:
        Exception: If dataset creation fails, provides detailed diagnostic information
        
    Note:
        The MDPDataset constructor automatically segments the data into episodes
        based on the 'dones' array, creating a list of Episode objects internally.
    """
    try:
        dataset = MDPDataset(states, actions, rewards, dones)
        return dataset
    except Exception as e:
        print(f"⚠️  Error creating MDP dataset: {e}")
        # Try to diagnose the issue by checking shapes and types
        print(f"States shape: {states.shape}, dtype: {states.dtype}")
        print(f"Actions shape: {actions.shape}, dtype: {actions.dtype}")
        print(f"Rewards shape: {rewards.shape}, dtype: {rewards.dtype}")
        print(f"Dones shape: {dones.shape}, dtype: {dones.dtype}")
        
        # Check for NaN or inf values in the inputs
        print(f"NaN in states: {any(any(x) for x in states)}")
        print(f"Inf in states: {any(any(x) for x in states)}")
        raise

def split_dataset(dataset, test_size=0.3, val_size=0.5, random_state=42):
    """
    Split dataset into train, validation and test sets.
    
    This function creates a strategically split dataset to enable:
    1. Training on the training set
    2. Hyperparameter tuning using the validation set
    3. Final evaluation on the held-out test set
    
    The splits are done at the episode level, so entire patient episodes
    will be kept together in the same split.
    
    Args:
        dataset (d3rlpy.dataset.MDPDataset): The complete dataset to split
        test_size (float): Proportion of data to allocate to test+val sets (0.0-1.0)
                          Default 0.3 means 70% train, 30% for test+val combined
        val_size (float): Proportion of the test+val data to allocate to validation
                         Default 0.5 means test and val sets are equal sized
                         (with test_size=0.3 and val_size=0.5, you get:
                          train=70%, val=15%, test=15%)
        random_state (int): Random seed for reproducible splits
        
    Returns:
        tuple: (train_episodes, val_episodes, test_episodes)
            Each element is a list of d3rlpy.dataset.Episode objects
            
    Note:
        The episodes are split randomly but deterministically based on random_state.
        This ensures reproducibility while maintaining independence between sets.
    """
    # Split into train and temporary sets
    # With default parameters: 70% train, 30% temp
    train_eps, temp_eps = train_test_split(
        dataset.episodes,
        test_size=test_size,
        random_state=random_state
    )
    
    # Split temporary set into validation and test sets
    # With default parameters: 15% val, 15% test (each half of the temp set)
    val_eps, test_eps = train_test_split(
        temp_eps,
        test_size=val_size,  # This is relative to temp_eps
        random_state=random_state
    )
    
    print(f"📊 Episodes  train={len(train_eps)}, val={len(val_eps)}, test={len(test_eps)}")
    
    return train_eps, val_eps, test_eps

def count_transitions(episodes):
    """
    Count the total number of transitions (steps) in a list of episodes.
    
    This is useful for estimating computational requirements and
    reporting dataset statistics. In reinforcement learning, the
    number of transitions is often more relevant than the number of episodes,
    especially for algorithms that learn from individual transitions.
    
    Args:
        episodes (list): List of d3rlpy.dataset.Episode objects
        
    Returns:
        int: Total number of transitions across all episodes
        
    Note:
        Each episode.observations corresponds to one state-action-reward-next_state tuple,
        which represents one transition in the MDP.
    """
    return sum(len(episode.observations) for episode in episodes)