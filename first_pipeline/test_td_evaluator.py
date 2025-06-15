#!/usr/bin/env python3
"""
Test script to explore TDErrorEvaluator and DiscreteCQL methods
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add the current directory to Python path
sys.path.append('.')

# Import d3rlpy components
import d3rlpy
from d3rlpy.metrics import TDErrorEvaluator
from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig
from d3rlpy.dataset import MDPDataset, Episode
from d3rlpy.preprocessing import StandardObservationScaler

print(f"d3rlpy version: {d3rlpy.__version__}")

def explore_td_evaluator():
    """Explore TDErrorEvaluator methods and initialization"""
    print("\n" + "="*60)
    print("EXPLORING TDErrorEvaluator")
    print("="*60)
    
    # Create a simple dummy dataset first
    n_samples = 20
    observations = np.random.randn(n_samples, 5).astype(np.float32)
    actions = np.random.randint(0, 3, size=n_samples).astype(np.int64)
    rewards = np.random.randn(n_samples).astype(np.float32)
    terminals = np.zeros(n_samples, dtype=bool)
    terminals[::5] = True  # Every 5th step is terminal
    
    # Create dataset and get episodes
    dataset = MDPDataset(observations, actions, rewards, terminals)
    episodes = dataset.episodes[:2]  # Use first 2 episodes
    
    print(f"Created {len(episodes)} episodes from dataset")
    
    print("1. TDErrorEvaluator initialization methods:")
    
    # Method 1: Initialize with episodes
    try:
        td_eval_1 = TDErrorEvaluator(episodes=episodes)
        print("   ✅ TDErrorEvaluator(episodes=episodes) - SUCCESS")
        print(f"   Available methods: {[method for method in dir(td_eval_1) if not method.startswith('_')]}")
    except Exception as e:
        print(f"   ❌ TDErrorEvaluator(episodes=episodes) - FAILED: {e}")
    
    # Method 2: Initialize without parameters
    try:
        td_eval_2 = TDErrorEvaluator()
        print("   ✅ TDErrorEvaluator() - SUCCESS")
        print(f"   Available methods: {[method for method in dir(td_eval_2) if not method.startswith('_')]}")
    except Exception as e:
        print(f"   ❌ TDErrorEvaluator() - FAILED: {e}")
    
    return episodes

def explore_cql_methods():
    """Explore DiscreteCQL methods without full initialization"""
    print("\n" + "="*60)
    print("EXPLORING DiscreteCQL METHODS")
    print("="*60)
    
    print("1. DiscreteCQL class inspection:")
    
    # Check class methods without instantiation
    cql_class_methods = [attr for attr in dir(DiscreteCQL) if not attr.startswith('_')]
    print(f"   Available class methods: {cql_class_methods}")
    
    # Look for Q-value related methods
    q_methods = [method for method in cql_class_methods if 'q' in method.lower() or 'value' in method.lower()]
    print(f"   Q-value related methods: {q_methods}")
    
    # Look for prediction methods
    predict_methods = [method for method in cql_class_methods if 'predict' in method.lower()]
    print(f"   Prediction methods: {predict_methods}")

def test_td_evaluator_with_dummy_model():
    """Test TDErrorEvaluator with a minimal setup"""
    print("\n" + "="*60)
    print("TESTING TDErrorEvaluator WITH DUMMY MODEL")
    print("="*60)
    
    # Create dummy data
    n_samples = 100
    n_features = 5
    n_actions = 3
    
    observations = np.random.randn(n_samples, n_features).astype(np.float32)
    actions = np.random.randint(0, n_actions, size=n_samples).astype(np.int64)
    rewards = np.random.randn(n_samples).astype(np.float32)
    terminals = np.zeros(n_samples, dtype=bool)
    terminals[::20] = True  # Every 20th step is terminal
    
    # Create dataset
    dataset = MDPDataset(observations, actions, rewards, terminals)
    episodes = dataset.episodes[:5]  # Use first 5 episodes
    
    print(f"Created {len(episodes)} episodes for testing")
    
    try:
        # Create a minimal CQL config
        config = DiscreteCQLConfig()
        
        # Try to create and build a CQL model
        cql = DiscreteCQL(
            config=config,
            device="cpu",
            enable_ddp=False
        )
        
        # Build with dataset
        cql.build_with_dataset(dataset)
        
        print("   ✅ Successfully created and built DiscreteCQL model")
        
        # Test TDErrorEvaluator
        print("\n2. Testing TDErrorEvaluator:")
        
        # Method 1: Initialize with episodes and call with model + dataset
        td_evaluator = TDErrorEvaluator(episodes=episodes)
        td_loss = td_evaluator(cql, dataset)
        print(f"   ✅ TD Loss (with episodes init): {td_loss:.6f}")
        
        # Method 2: Initialize without episodes and call with model + dataset
        try:
            td_evaluator_2 = TDErrorEvaluator()
            td_loss_2 = td_evaluator_2(cql, dataset)
            print(f"   ✅ TD Loss (without episodes init): {td_loss_2:.6f}")
        except Exception as e:
            print(f"   ❌ TD Loss without episodes failed: {e}")
        
        # Method 3: Try calling with just the model (this should fail)
        try:
            td_evaluator_3 = TDErrorEvaluator(episodes=episodes)
            td_loss_3 = td_evaluator_3(cql)  # This should fail
            print(f"   ✅ TD Loss (model only): {td_loss_3:.6f}")
        except Exception as e:
            print(f"   ❌ TD Loss (model only) failed as expected: {e}")
        
        # Test other evaluation methods if available
        print("\n3. Testing other evaluation capabilities:")
        
        # Check if model has predict_value method
        if hasattr(cql, 'predict_value'):
            try:
                q_values = cql.predict_value(observations[:5], actions[:5].reshape(-1, 1))
                print(f"   ✅ predict_value works, shape: {q_values.shape}")
            except Exception as e:
                print(f"   ❌ predict_value failed: {e}")
        
        # Check if model has predict method
        if hasattr(cql, 'predict'):
            try:
                predicted_actions = cql.predict(observations[:5])
                print(f"   ✅ predict works, shape: {predicted_actions.shape}")
            except Exception as e:
                print(f"   ❌ predict failed: {e}")
                
        return cql, episodes
        
    except Exception as e:
        print(f"   ❌ Failed to create CQL model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_conservative_loss_calculation():
    """Test the conservative loss calculation method"""
    print("\n" + "="*60)
    print("TESTING CONSERVATIVE LOSS CALCULATION")
    print("="*60)
    
    # Import the evaluator class
    from evaluator import CQLEvaluator
    
    # Create dummy data
    n_samples = 50
    n_features = 5
    n_actions = 3
    
    observations = np.random.randn(n_samples, n_features).astype(np.float32)
    actions = np.random.randint(0, n_actions, size=n_samples).astype(np.int64)
    rewards = np.random.randn(n_samples).astype(np.float32)
    terminals = np.zeros(n_samples, dtype=bool)
    terminals[::10] = True  # Every 10th step is terminal
    
    # Create dataset and episodes
    dataset = MDPDataset(observations, actions, rewards, terminals)
    episodes = dataset.episodes[:3]  # Use first 3 episodes
    
    try:
        # Create and build CQL model
        config = DiscreteCQLConfig()
        cql = DiscreteCQL(config=config, device="cpu", enable_ddp=False)
        cql.build_with_dataset(dataset)
        
        print("   ✅ Created CQL model for conservative loss testing")
        
        # Test the static conservative loss method
        conservative_loss = CQLEvaluator.conservative_loss_discrete(
            algo=cql,
            episodes=episodes,
            alpha=1.0,  # Specify alpha explicitly
            batch_size=32
        )
        
        print(f"   ✅ Conservative loss calculated: {conservative_loss:.6f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Conservative loss calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_episodes_to_dataset_conversion():
    """Test converting episodes back to MDPDataset - the working approach"""
    print("\n" + "="*60)
    print("TESTING EPISODES TO DATASET CONVERSION")
    print("="*60)
    
    # Create dummy data
    n_samples = 20
    observations = np.random.randn(n_samples, 5).astype(np.float32)
    actions = np.random.randint(0, 3, size=n_samples).astype(np.int64)
    rewards = np.random.randn(n_samples).astype(np.float32)
    terminals = np.zeros(n_samples, dtype=bool)
    terminals[::5] = True  # Every 5th step is terminal
    
    # Create original dataset and get episodes
    original_dataset = MDPDataset(observations, actions, rewards, terminals)
    episodes = original_dataset.episodes
    
    print(f"Original dataset has {len(episodes)} episodes")
    
    # Now convert episodes back to flattened arrays
    print('\nConverting episodes back to flattened arrays...')
    all_obs = []
    all_acts = []
    all_rews = []
    all_terms = []

    for ep in episodes:
        all_obs.append(ep.observations)
        all_acts.append(ep.actions)
        all_rews.append(ep.rewards)

        # Create terminals array for this episode
        ep_terminals = np.zeros(ep.size(), dtype=bool)
        ep_terminals[-1] = True  # Last step is terminal
        all_terms.append(ep_terminals)

    # Flatten everything
    flat_obs = np.vstack(all_obs)
    flat_acts = np.concatenate(all_acts)
    flat_rews = np.concatenate(all_rews)
    flat_terms = np.concatenate(all_terms)

    print(f'Flattened shapes:')
    print(f'  observations: {flat_obs.shape}')
    print(f'  actions: {flat_acts.shape}')
    print(f'  rewards: {flat_rews.shape}')
    print(f'  terminals: {flat_terms.shape}')

    # Try to create new dataset
    try:
        # Need to flatten actions if they have extra dimension
        if flat_acts.ndim > 1:
            flat_acts = flat_acts.flatten()
        if flat_rews.ndim > 1:
            flat_rews = flat_rews.flatten()
            
        new_dataset = MDPDataset(flat_obs, flat_acts, flat_rews, flat_terms.astype(np.float32))
        print(f'\n✅ Successfully created MDPDataset from episodes!')
        print(f'New dataset has {len(new_dataset.episodes)} episodes')
        return new_dataset, episodes
    except Exception as e:
        print(f'\n❌ Failed to create MDPDataset: {e}')
        import traceback
        traceback.print_exc()
        return None, episodes

if __name__ == "__main__":
    print("Starting TDErrorEvaluator and CQL exploration...")
    
    # 1. Explore TDErrorEvaluator
    episodes = explore_td_evaluator()
    
    # 2. Explore CQL methods
    explore_cql_methods()
    
    # 3. Test episodes to dataset conversion
    reconstructed_dataset, test_episodes = test_episodes_to_dataset_conversion()
    
    # 4. Test with actual models
    cql_model, test_episodes = test_td_evaluator_with_dummy_model()
    
    # 5. Test conservative loss
    test_conservative_loss_calculation()
    
    print("\n" + "="*60)
    print("SUMMARY - CORRECT USAGE PATTERNS")
    print("="*60)
    print("Key findings for using TDErrorEvaluator:")
    print("1. CORRECT Initialize: td_evaluator = TDErrorEvaluator(episodes=episodes)")
    print("2. CORRECT Calculate:  td_loss = td_evaluator(model, dataset)  # BOTH model AND dataset required!")
    print("3. Alternative: td_evaluator = TDErrorEvaluator()")
    print("                td_loss = td_evaluator(model, dataset)")
    print("4. IMPORTANT: Even if initialized with episodes, you still need to pass dataset when calling!")
    print("\nKey findings for DiscreteCQL:")
    print("1. Must build with dataset before use: model.build_with_dataset(dataset)")
    print("2. Main methods: predict(), predict_value()")
    print("3. predict_value(obs, actions) returns Q-values for specific state-action pairs")
    print("4. predict(obs) returns the best action for given observations")
    print("5. For loss calculation, use the static methods in CQLEvaluator")
    print("\nFIX NEEDED IN YOUR EVALUATOR.PY:")
    print("Change this line:")
    print("   td_loss = td_evaluator(self.model)")
    print("To this:")
    print("   td_loss = td_evaluator(self.model, val_dataset)")
    print("Where val_dataset = MDPDataset.from_episodes(val_episodes)")