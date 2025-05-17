#!/usr/bin/env python3
# -----------------------------------------------------------
# dataset.py  --  MDP Dataset handling for HFNC CQL
# -----------------------------------------------------------
from d3rlpy.dataset import MDPDataset
from sklearn.model_selection import train_test_split

def create_mdp_dataset(states, actions, rewards, dones):
    """Create an MDP dataset from arrays"""
    try:
        dataset = MDPDataset(states, actions, rewards, dones)
        return dataset
    except Exception as e:
        print(f"⚠️  Error creating MDP dataset: {e}")
        # Try to diagnose the issue
        print(f"States shape: {states.shape}, dtype: {states.dtype}")
        print(f"Actions shape: {actions.shape}, dtype: {actions.dtype}")
        print(f"Rewards shape: {rewards.shape}, dtype: {rewards.dtype}")
        print(f"Dones shape: {dones.shape}, dtype: {dones.dtype}")
        
        # Check for NaN or inf values in the inputs
        print(f"NaN in states: {any(any(x) for x in states)}")
        print(f"Inf in states: {any(any(x) for x in states)}")
        raise

def split_dataset(dataset, test_size=0.3, val_size=0.5, random_state=42):
    """Split dataset into train, validation and test sets"""
    # Split into train and temporary sets
    train_eps, temp_eps = train_test_split(
        dataset.episodes,
        test_size=test_size,
        random_state=random_state
    )
    
    # Split temporary set into validation and test sets
    val_eps, test_eps = train_test_split(
        temp_eps,
        test_size=val_size,  # This is relative to temp_eps
        random_state=random_state
    )
    
    print(f"📊 Episodes  train={len(train_eps)}, val={len(val_eps)}, test={len(test_eps)}")
    
    return train_eps, val_eps, test_eps

def count_transitions(episodes):
    """Count the total number of transitions in a list of episodes"""
    return sum(len(episode.observations) for episode in episodes)