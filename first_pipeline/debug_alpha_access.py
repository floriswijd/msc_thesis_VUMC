#!/usr/bin/env python3
"""
Diagnostic script to explore what's accessible from the CQL algorithm object
to find the alpha parameter and other configuration details.
"""

import sys
import numpy as np
from pathlib import Path

# Add the current directory to Python path
sys.path.append('.')

# Import d3rlpy components
import d3rlpy
from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig
from d3rlpy.dataset import MDPDataset

print(f"d3rlpy version: {d3rlpy.__version__}")

def explore_cql_config_and_alpha():
    """Comprehensive exploration of CQL algorithm configuration access"""
    print("\n" + "="*70)
    print("EXPLORING CQL ALGORITHM CONFIGURATION ACCESS")
    print("="*70)
    
    # Create a simple dataset for building the model
    n_samples = 50
    n_features = 5
    n_actions = 3
    
    observations = np.random.randn(n_samples, n_features).astype(np.float32)
    actions = np.random.randint(0, n_actions, size=n_samples).astype(np.int64)
    rewards = np.random.randn(n_samples).astype(np.float32)
    terminals = np.zeros(n_samples, dtype=bool)
    terminals[::10] = True
    
    dataset = MDPDataset(observations, actions, rewards, terminals)
    
    # Test different alpha values
    test_alphas = [1.0, 2.5, 5.0, 10.0]
    
    for alpha_val in test_alphas:
        print(f"\n{'='*50}")
        print(f"TESTING WITH ALPHA = {alpha_val}")
        print(f"{'='*50}")
        
        try:
            # Create CQL config with specific alpha
            config = DiscreteCQLConfig(
                alpha=alpha_val,  # Set alpha explicitly
                batch_size=32,
                learning_rate=3e-4,
                gamma=0.99
            )
            
            print(f"1. DiscreteCQLConfig attributes after creation:")
            config_attrs = [attr for attr in dir(config) if not attr.startswith('_')]
            for attr in config_attrs:
                try:
                    value = getattr(config, attr)
                    if not callable(value):
                        print(f"   config.{attr}: {value}")
                except:
                    print(f"   config.{attr}: <could not access>")
            
            # Create CQL algorithm
            cql = DiscreteCQL(
                config=config,
                device="cpu",
                enable_ddp=False
            )
            
            print(f"\n2. DiscreteCQL algorithm attributes before building:")
            algo_attrs = [attr for attr in dir(cql) if not attr.startswith('_')]
            relevant_attrs = [attr for attr in algo_attrs if any(keyword in attr.lower() 
                            for keyword in ['alpha', 'config', 'param', 'hyperparameter'])]
            
            print(f"   Relevant attributes found: {relevant_attrs}")
            
            for attr in relevant_attrs:
                try:
                    value = getattr(cql, attr)
                    if not callable(value):
                        print(f"   cql.{attr}: {value}")
                    else:
                        print(f"   cql.{attr}: <method/function>")
                except:
                    print(f"   cql.{attr}: <could not access>")
            
            # Check if we can access config through the algorithm
            print(f"\n3. Accessing config through algorithm:")
            if hasattr(cql, 'config'):
                print(f"   ✅ cql.config exists: {type(cql.config)}")
                if hasattr(cql.config, 'alpha'):
                    print(f"   ✅ cql.config.alpha: {cql.config.alpha}")
                else:
                    print(f"   ❌ cql.config has no 'alpha' attribute")
                    config_attrs = [attr for attr in dir(cql.config) if not attr.startswith('_')]
                    alpha_related = [attr for attr in config_attrs if 'alpha' in attr.lower()]
                    print(f"   Alpha-related attributes in config: {alpha_related}")
            
            if hasattr(cql, '_config'):
                print(f"   ✅ cql._config exists: {type(cql._config)}")
                if hasattr(cql._config, 'alpha'):
                    print(f"   ✅ cql._config.alpha: {cql._config.alpha}")
                elif hasattr(cql._config, 'initial_alpha'):
                    print(f"   ✅ cql._config.initial_alpha: {cql._config.initial_alpha}")
                else:
                    print(f"   ❌ cql._config has no 'alpha' or 'initial_alpha' attribute")
                    config_attrs = [attr for attr in dir(cql._config) if not attr.startswith('_')]
                    alpha_related = [attr for attr in config_attrs if 'alpha' in attr.lower()]
                    print(f"   Alpha-related attributes in _config: {alpha_related}")
            
            # Build the algorithm
            print(f"\n4. Building algorithm with dataset...")
            cql.build_with_dataset(dataset)
            
            print(f"\n5. DiscreteCQL algorithm attributes AFTER building:")
            
            # Check config access after building
            if hasattr(cql, 'config'):
                print(f"   ✅ cql.config after build: {type(cql.config)}")
                if hasattr(cql.config, 'alpha'):
                    print(f"   ✅ cql.config.alpha: {cql.config.alpha}")
            
            if hasattr(cql, '_config'):
                print(f"   ✅ cql._config after build: {type(cql._config)}")
                if hasattr(cql._config, 'alpha'):
                    print(f"   ✅ cql._config.alpha: {cql._config.alpha}")
                elif hasattr(cql._config, 'initial_alpha'):
                    print(f"   ✅ cql._config.initial_alpha: {cql._config.initial_alpha}")
            
            # Check implementation object
            if hasattr(cql, '_impl'):
                print(f"   ✅ cql._impl exists: {type(cql._impl)}")
                if hasattr(cql._impl, 'alpha'):
                    print(f"   ✅ cql._impl.alpha: {cql._impl.alpha}")
                elif hasattr(cql._impl, '_alpha'):
                    print(f"   ✅ cql._impl._alpha: {cql._impl._alpha}")
                else:
                    impl_attrs = [attr for attr in dir(cql._impl) if not attr.startswith('_')]
                    alpha_related = [attr for attr in impl_attrs if 'alpha' in attr.lower()]
                    print(f"   Alpha-related attributes in _impl: {alpha_related}")
            
            # Try to access through conservative_loss_discrete function
            print(f"\n6. Testing conservative_loss_discrete function alpha access:")
            try:
                from evaluator import CQLEvaluator
                
                # Test episodes for the function
                test_episodes = dataset.episodes[:2]
                
                # Call the static method and see what alpha it extracts
                conservative_loss = CQLEvaluator.conservative_loss_discrete(
                    algo=cql,
                    episodes=test_episodes,
                    alpha=None  # Let it extract from algo
                )
                print(f"   ✅ Conservative loss calculated: {conservative_loss:.6f}")
                print(f"   ✅ Function successfully extracted alpha from algo")
                
                # Try to extract alpha the same way the function does
                extracted_alpha = getattr(getattr(cql, "_config", None), "initial_alpha", 1.0)
                print(f"   ✅ Extracted alpha value: {extracted_alpha}")
                
            except Exception as e:
                print(f"   ❌ Error testing conservative_loss_discrete: {e}")
            
            print(f"\n{'='*50}")
            
        except Exception as e:
            print(f"❌ Error creating/testing CQL with alpha {alpha_val}: {e}")
            import traceback
            traceback.print_exc()

def test_alpha_extraction_function():
    """Test a function that tries to extract alpha from various possible locations"""
    print("\n" + "="*70)
    print("TESTING ALPHA EXTRACTION FUNCTION")
    print("="*70)
    
    def extract_alpha_from_algo(algo, default_alpha=1.0):
        """Try to extract alpha from algorithm object in multiple ways"""
        print(f"\n🔍 Attempting to extract alpha from algorithm...")
        
        # Method 1: config.alpha
        try:
            if hasattr(algo, 'config') and hasattr(algo.config, 'alpha'):
                alpha = algo.config.alpha
                print(f"   ✅ Found alpha via config.alpha: {alpha}")
                return alpha
        except:
            pass
        
        # Method 2: _config.alpha
        try:
            if hasattr(algo, '_config') and hasattr(algo._config, 'alpha'):
                alpha = algo._config.alpha
                print(f"   ✅ Found alpha via _config.alpha: {alpha}")
                return alpha
        except:
            pass
        
        # Method 3: _config.initial_alpha
        try:
            if hasattr(algo, '_config') and hasattr(algo._config, 'initial_alpha'):
                alpha = algo._config.initial_alpha
                print(f"   ✅ Found alpha via _config.initial_alpha: {alpha}")
                return alpha
        except:
            pass
        
        # Method 4: _impl.alpha
        try:
            if hasattr(algo, '_impl') and hasattr(algo._impl, 'alpha'):
                alpha = algo._impl.alpha
                print(f"   ✅ Found alpha via _impl.alpha: {alpha}")
                return alpha
        except:
            pass
        
        # Method 5: _impl._alpha
        try:
            if hasattr(algo, '_impl') and hasattr(algo._impl, '_alpha'):
                alpha = algo._impl._alpha
                print(f"   ✅ Found alpha via _impl._alpha: {alpha}")
                return alpha
        except:
            pass
        
        print(f"   ⚠️  Could not extract alpha, using default: {default_alpha}")
        return default_alpha
    
    # Test with different alpha values
    for test_alpha in [1.0, 2.5, 5.0]:
        print(f"\n--- Testing with configured alpha = {test_alpha} ---")
        
        # Create simple dataset
        observations = np.random.randn(20, 5).astype(np.float32)
        actions = np.random.randint(0, 3, size=20).astype(np.int64)
        rewards = np.random.randn(20).astype(np.float32)
        terminals = np.zeros(20, dtype=bool)
        terminals[-1] = True
        
        dataset = MDPDataset(observations, actions, rewards, terminals)
        
        # Create and build CQL
        config = DiscreteCQLConfig(alpha=test_alpha)
        cql = DiscreteCQL(config=config, device="cpu", enable_ddp=False)
        cql.build_with_dataset(dataset)
        
        # Test extraction
        extracted_alpha = extract_alpha_from_algo(cql)
        
        success = abs(extracted_alpha - test_alpha) < 1e-6
        print(f"   📊 Expected: {test_alpha}, Extracted: {extracted_alpha}, Success: {success}")

if __name__ == "__main__":
    print("Starting CQL Algorithm Configuration Exploration...")
    
    # 1. Comprehensive configuration exploration
    explore_cql_config_and_alpha()
    
    # 2. Test extraction function
    test_alpha_extraction_function()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("Key findings for accessing alpha from CQL algorithm:")
    print("1. Check config.alpha")
    print("2. Check _config.alpha") 
    print("3. Check _config.initial_alpha")
    print("4. Check _impl attributes")
    print("5. Always have a fallback default value")
    print("\nRun this script to see exactly what's accessible in your d3rlpy version!")