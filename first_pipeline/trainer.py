#!/usr/bin/env python3
# -----------------------------------------------------------
# trainer.py  --  Training module for HFNC CQL
# 
# This module handles the training of the CQL model:
# - Executing the training loop with robust error handling
# - Determining appropriate training parameters
# - Analyzing training logs for potential issues
# 
# The training procedure is designed with failsafes to handle
# potential issues in the d3rlpy training process.
# -----------------------------------------------------------
import numpy as np
from dataset import count_transitions

def train_model(model, dataset, n_epochs, experiment_name=None):
    """
    Train the CQL model with proper error handling and logging.
    
    This function orchestrates the model training process, handling potential
    errors and providing informative feedback. It attempts to use multiple
    approaches if the initial training fails.
    
    Args:
        model (DiscreteCQL): The CQL model to train
        dataset (MDPDataset): The complete dataset for training 
        n_epochs (int): Number of training epochs
        experiment_name (str, optional): Name for logging directory
                                        Default is None (uses default naming)
                                        
    Returns:
        tuple: (result, errors)
            - result: Training result from the model.fit method (if successful)
            - errors: List of encountered errors or None if training succeeded
            
    Note:
        The number of actual training steps is estimated based on the epochs
        parameter, but the relationship depends on d3rlpy's internal batching.
        The function includes a fallback to reduced steps if the first attempt fails.
    """
    print("🚀  Starting training DiscreteCQL...")
    
    # Calculate total transitions for better step estimation
    # This helps determine an appropriate number of training steps
    n_transitions = sum(len(episode.observations) for episode in dataset.episodes)
    print(f"Using dataset with {n_transitions} transitions for training")
    
    # Estimate n_steps based on epochs (adjust multiplier as needed)
    # The multiplier should be chosen based on the dataset size and complexity
    n_steps = n_epochs * 10000  # Adjust this multiplier based on your dataset size
    print(f"Training for estimated {n_steps} steps...")
    
    try:
        # Try standard training approach
        # This uses d3rlpy's fit method with progress tracking and model saving
        result = model.fit(
            dataset=dataset,
            n_steps=n_steps,
            experiment_name=experiment_name,
            save_interval=n_steps // 10,  # Save model every 10% of training
            # verbose=True,
            show_progress=True,
        )
        print("✅  Training completed successfully!")
        return result, None
    except Exception as first_error:
        # First training attempt failed - log the error and try an alternative approach
        print(f"⚠️  Error during training: {first_error}")
        print("Trying alternative training approach...")
        
        try:
            # Fallback to a simpler approach with fewer steps
            # This can help if the original error was related to resource constraints
            n_steps_reduced = n_epochs * 1000  # Use fewer steps as fallback
            print(f"Training with simplified approach for {n_steps_reduced} steps...")
            
            result = model.fit(
                dataset=dataset,
                n_steps=n_steps_reduced,
                experiment_name=experiment_name,
                # verbose=True,
            )
            print("✅  Training completed with fallback approach!")
            return result, first_error
        except Exception as second_error:
            # Both training attempts failed - return errors for diagnosis
            print(f"❌  Training failed: {second_error}")
            print("Please check d3rlpy documentation for version 2.8.1.")
            return None, [first_error, second_error]
            
def check_training_logs(log_dir):
    """
    Analyze training logs to detect issues like NaN values or extreme gradients.
    
    This function examines the CSV log files generated during training to identify
    potential issues that might explain training failures or poor performance,
    particularly focusing on NaN values and extreme values that indicate instability.
    
    Args:
        log_dir (str): Directory containing training logs
        
    Returns:
        dict: Dictionary mapping log file names to identified issues,
              or None if analysis could not be performed
              
    Note:
        This analysis is particularly important for diagnosing NaN issues
        in the training process, which are a common cause of training failures
        in deep reinforcement learning.
    """
    try:
        import pandas as pd
        import os
        import glob
        
        # Find all CSV log files from the training process
        log_files = glob.glob(os.path.join(log_dir, "*.csv"))
        print(f"Found {len(log_files)} log files to analyze")
        
        issues = {}
        
        # Check each log file for NaN or extreme values
        for log_file in log_files:
            try:
                log_name = os.path.basename(log_file)
                df = pd.read_csv(log_file)
                
                # Check for NaN values - these indicate numerical instability
                nan_count = df.isna().sum().sum()
                if nan_count > 0:
                    issues[log_name] = f"Contains {nan_count} NaN values"
                    print(f"⚠️  WARNING: {log_name} contains {nan_count} NaN values")
                    
                    # Find which columns have NaNs to help diagnose the source
                    nan_cols = df.columns[df.isna().any()].tolist()
                    print(f"  NaN values found in columns: {nan_cols}")
                    
                # Check for extremely large values that might indicate divergence
                # Extremely large values often precede NaNs in the training process
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