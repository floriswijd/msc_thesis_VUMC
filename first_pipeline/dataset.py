#!/usr/bin/env python3
from d3rlpy.dataset import MDPDataset, Episode
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


def create_mdp_dataset_with_metadata(states, actions, rewards, dones, stay_ids, subject_ids, episode_ids):
    """Create standard MDP dataset and add stay metadata mapping"""
    # Use the existing, working dataset creation
    dataset = create_mdp_dataset(states, actions, rewards, dones)
    
    # Create metadata mapping based on episode boundaries
    episode_ends = np.where(dones)[0]
    episode_metadata = {}
    start_idx = 0
    
    for episode_idx, end_idx in enumerate(episode_ends):
        episode_metadata[episode_idx] = {
            'stay_id': stay_ids[start_idx],
            'subject_id': subject_ids[start_idx], 
            'hfnc_episode': episode_ids[start_idx],
            'episode_length': end_idx - start_idx + 1
        }
        start_idx = end_idx + 1
    
    # Attach metadata to the dataset
    dataset.episode_metadata = episode_metadata
    print(f"✅  MDP dataset created with metadata for {len(dataset.episodes)} episodes.")
    return dataset

def extract_stay_id(episode):
    """Extract stay_id from episode metadata"""
    return getattr(episode, 'stay_id', None)

def extract_subject_id(episode):
    """Extract subject_id from episode metadata"""
    return getattr(episode, 'subject_id', None)

def extract_episode_id(episode):
    """Extract HFNC episode ID from episode metadata"""
    return getattr(episode, 'hfnc_episode', None)

def split_dataset_by_stay(dataset, test_size=0.3, val_size=0.5, random_state=42):
    """Split dataset by stay_id using dataset-level metadata"""
    if not dataset.episodes:
        print("⚠️ Warning: No episodes in the dataset to split. Returning empty lists.")
        return [], [], []
    
    if not hasattr(dataset, 'episode_metadata'):
        raise ValueError("Dataset missing episode metadata. Use create_mdp_dataset_with_metadata()")
    
    # Group episodes by stay_id using dataset metadata
    stay_to_episodes = {}
    
    for episode_idx, episode in enumerate(dataset.episodes):
        if episode_idx not in dataset.episode_metadata:
            raise ValueError(f"Episode {episode_idx} missing from metadata mapping")
        
        stay_id = dataset.episode_metadata[episode_idx]['stay_id']
        if stay_id not in stay_to_episodes:
            stay_to_episodes[stay_id] = []
        stay_to_episodes[stay_id].append(episode)
    
    # Get unique stay_ids
    stay_ids = list(stay_to_episodes.keys())
    
    if len(stay_ids) < 2:
        print("⚠️ Warning: Not enough stays for splitting. Returning all as training.")
        return dataset.episodes, [], []
    
    # Split by stays
    train_stays, temp_stays = train_test_split(
        stay_ids, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    # Handle validation/test split
    if not temp_stays or val_size == 0.0:
        val_stays, test_stays = [], temp_stays
    elif val_size == 1.0:
        val_stays, test_stays = temp_stays, []
    elif len(temp_stays) < 2:
        print("⚠️ Warning: Not enough stays in temp for val/test split. Assigning all to test.")
        val_stays, test_stays = [], temp_stays
    else:
        val_stays, test_stays = train_test_split(
            temp_stays,
            test_size=val_size,
            random_state=random_state,
            shuffle=True
        )
    
    # Convert stays back to episodes
    train_eps = [ep for stay in train_stays for ep in stay_to_episodes[stay]]
    val_eps = [ep for stay in val_stays for ep in stay_to_episodes[stay]] 
    test_eps = [ep for stay in test_stays for ep in stay_to_episodes[stay]]
    
    print(f"📊 Stay-level split:")
    print(f"   Training:   {len(train_stays):,} stays, {len(train_eps):,} episodes")
    print(f"   Validation: {len(val_stays):,} stays, {len(val_eps):,} episodes") 
    print(f"   Test:       {len(test_stays):,} stays, {len(test_eps):,} episodes")
    print(f"   Total:      {len(stay_ids):,} unique stays")
    
    return train_eps, val_eps, test_eps