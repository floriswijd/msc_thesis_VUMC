#!/usr/bin/env python3
# -----------------------------------------------------------
# trainer.py  --  Training module for HFNC CQL
# -----------------------------------------------------------
import numpy as np
from dataset import count_transitions

def train_model(model, dataset, n_epochs, experiment_name=None):
    """Train the CQL model with proper error handling and logging"""
    print("🚀  Starting training DiscreteCQL...")
    
    # Calculate total transitions for better step estimation
    n_transitions = sum(len(episode.observations) for episode in dataset.episodes)
    print(f"Using dataset with {n_transitions} transitions for training")
    
    # Estimate n_steps based on epochs (adjust multiplier as needed)
    n_steps = n_epochs * 10000  # Adjust this multiplier based on your dataset size
    print(f"Training for estimated {n_steps} steps...")
    
    try:
        # Try standard training approach
        result = model.fit(
            dataset=dataset,
            n_steps=n_steps,
            experiment_name=experiment_name,
            save_interval=n_steps // 10,  # Save model every 10%
            show_progress=True,
        )
        print("✅  Training completed successfully!")
        return result, None
    except Exception as first_error:
        print(f"⚠️  Error during training: {first_error}")
        print("Trying alternative training approach...")
        
        try:
            # Fallback to a simpler approach
            n_steps_reduced = n_epochs * 1000  # Use fewer steps as fallback
            print(f"Training with simplified approach for {n_steps_reduced} steps...")
            
            result = model.fit(
                dataset=dataset,
                n_steps=n_steps_reduced,
                experiment_name=experiment_name,
            )
            print("✅  Training completed with fallback approach!")
            return result, first_error
        except Exception as second_error:
            print(f"❌  Training failed: {second_error}")
            print("Please check d3rlpy documentation for version 2.8.1.")
            return None, [first_error, second_error]
            
def check_training_logs(log_dir):
    """Analyze training logs to detect issues like NaN values"""
    try:
        import pandas as pd
        import os
        import glob
        
        # Find all CSV log files
        log_files = glob.glob(os.path.join(log_dir, "*.csv"))
        print(f"Found {len(log_files)} log files to analyze")
        
        issues = {}
        
        # Check each log file for NaN or extreme values
        for log_file in log_files:
            try:
                log_name = os.path.basename(log_file)
                df = pd.read_csv(log_file)
                
                # Check for NaN values
                nan_count = df.isna().sum().sum()
                if nan_count > 0:
                    issues[log_name] = f"Contains {nan_count} NaN values"
                    print(f"⚠️  WARNING: {log_name} contains {nan_count} NaN values")
                    
                    # Find which columns have NaNs
                    nan_cols = df.columns[df.isna().any()].tolist()
                    print(f"  NaN values found in columns: {nan_cols}")
                    
                # Check for extremely large values that might indicate divergence
                max_vals = df.max().max()
                if max_vals > 1e6:
                    if log_name not in issues:
                        issues[log_name] = f"Contains extreme values ({max_vals})"
                    else:
                        issues[log_name] += f", contains extreme values ({max_vals})"
                    print(f"⚠️  WARNING: {log_name} contains extreme values: {max_vals}")
            
            except Exception as e:
                print(f"Error analyzing log file {log_file}: {e}")
        
        return issues
    except Exception as e:
        print(f"Error checking training logs: {e}")
        return None