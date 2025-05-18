#!/usr/bin/env python3
from d3rlpy.dataset import MDPDataset
from sklearn.model_selection import train_test_split
import numpy as np

def create_mdp_dataset(states, actions, rewards, dones):
    try:
        dataset = MDPDataset(
            observations=states,
            actions=actions,
            rewards=rewards,
            terminals=dones
        )
        print(f"✅  MDP dataset created with {len(dataset.episodes)} episodes.")
        return dataset
    except Exception as e:
        print(f"⚠️  Error creating MDP dataset: {e}")
        print(f"States shape: {states.shape if hasattr(states, 'shape') else 'N/A'}, dtype: {states.dtype if hasattr(states, 'dtype') else 'N/A'}")
        print(f"Actions shape: {actions.shape if hasattr(actions, 'shape') else 'N/A'}, dtype: {actions.dtype if hasattr(actions, 'dtype') else 'N/A'}")
        print(f"Rewards shape: {rewards.shape if hasattr(rewards, 'shape') else 'N/A'}, dtype: {rewards.dtype if hasattr(rewards, 'dtype') else 'N/A'}")
        print(f"Dones (terminals) shape: {dones.shape if hasattr(dones, 'shape') else 'N/A'}, dtype: {dones.dtype if hasattr(dones, 'dtype') else 'N/A'}")
        if isinstance(states, np.ndarray):
            print(f"NaN in states: {np.isnan(states).any()}")
            print(f"Inf in states: {np.isinf(states).any()}")
        raise

def split_dataset(dataset, test_size=0.3, val_size=0.5, random_state=42):
    if not dataset.episodes:
        print("⚠️ Warning: No episodes in the dataset to split. Returning empty lists.")
        return [], [], []
    if len(dataset.episodes) < 2:
        print("⚠️ Warning: Not enough episodes for a full split. Returning all as training.")
        return dataset.episodes, [], []
    train_eps, temp_eps = train_test_split(
        dataset.episodes,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    if not temp_eps:
        val_eps, test_eps = [], []
    elif len(temp_eps) < 2 and val_size > 0 and val_size < 1.0 :
        print("⚠️ Warning: Not enough episodes in temp_eps to split into validation and test. Assigning all to test.")
        val_eps = []
        test_eps = temp_eps
    elif val_size == 0.0:
        val_eps = []
        test_eps = temp_eps
    elif val_size == 1.0:
        val_eps = temp_eps
        test_eps = []
    else:
        val_eps, test_eps = train_test_split(
            temp_eps,
            test_size=val_size,
            random_state=random_state,
            shuffle=True
        )
    print(f"📊 Episodes  train={len(train_eps)}, val={len(val_eps)}, test={len(test_eps)}")
    return train_eps, val_eps, test_eps

def count_transitions(episodes):
    if not episodes:
        return 0
    return sum(len(episode.observations) for episode in episodes)