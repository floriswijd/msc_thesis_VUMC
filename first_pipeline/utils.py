#!/usr/bin/env python3
# -----------------------------------------------------------
# utils.py  --  Utility functions for HFNC CQL
# 
# This module provides utility functions for the HFNC pipeline:
# - Debugging tools for NaN and infinite value detection
# - Visualization functions for training curves
# - Gradient analysis for detecting training instabilities
# - Time formatting utilities
#
# These utilities help diagnose issues in the training process
# and provide insights into model behavior.
# -----------------------------------------------------------
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def timestamp_string():
    """
    Generate a timestamp string for logging and file naming.
    
    Returns a formatted timestamp string that can be used to create
    unique run identifiers or log file names.
    
    Returns:
        str: Formatted timestamp string in 'YYYYMMDDHHMMSS' format
        
    Note:
        Using timestamps in file names ensures unique identifiers
        for each training run, preventing accidental overwriting.
    """
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def debug_nan_values(array, name="array"):
    """
    Debug NaN values in an array by providing detailed statistics.
    
    NaN values are a common cause of training failures in deep learning.
    This function helps identify which features or samples contain NaNs
    and how prevalent they are.
    
    Args:
        array (np.ndarray): The array to check for NaN values
        name (str): Name of the array for reporting purposes
                    Default is "array"
        
    Returns:
        bool: True if NaN values were found, False otherwise
        
    Side effects:
        Prints detailed information about NaN values to the console
    """
    nan_mask = np.isnan(array)
    nan_count = np.sum(nan_mask)
    
    if nan_count > 0:
        print(f"⚠️  WARNING: {nan_count} NaN values found in {name}")
        if array.ndim > 1:
            # For 2D arrays, report NaN counts per column
            # This helps identify problematic features
            nan_cols = np.sum(nan_mask, axis=0)
            for i, count in enumerate(nan_cols):
                if count > 0:
                    print(f"  - Column {i}: {count} NaNs ({100 * count / array.shape[0]:.2f}%)")
        
        # For any array, report percentage of NaNs
        # High percentages may indicate systematic data quality issues
        print(f"  - Overall: {nan_count}/{array.size} ({100 * nan_count / array.size:.2f}%)")
        return True
    return False

def debug_inf_values(array, name="array"):
    """
    Debug infinite values in an array by providing detailed statistics.
    
    Infinite values often indicate numerical instabilities, division by zero,
    or exploding gradients during training.
    
    Args:
        array (np.ndarray): The array to check for infinite values
        name (str): Name of the array for reporting purposes
                    Default is "array"
        
    Returns:
        bool: True if infinite values were found, False otherwise
        
    Side effects:
        Prints detailed information about infinite values to the console
        
    Note:
        Infinite values often precede NaN errors in the training process
        and can help diagnose issues before they cause complete failure.
    """
    inf_mask = np.isinf(array)
    inf_count = np.sum(inf_mask)
    
    if inf_count > 0:
        print(f"⚠️  WARNING: {inf_count} infinite values found in {name}")
        if array.ndim > 1:
            # For 2D arrays, report infinite counts per column
            # This helps identify problematic features
            inf_cols = np.sum(inf_mask, axis=0)
            for i, count in enumerate(inf_cols):
                if count > 0:
                    print(f"  - Column {i}: {count} infs ({100 * count / array.shape[0]:.2f}%)")
        
        # For any array, report percentage of infinite values
        print(f"  - Overall: {inf_count}/{array.size} ({100 * inf_count / array.size:.2f}%)")
        return True
    return False

def plot_training_curves(log_dir):
    """
    Plot training curves from CSV log files to visualize learning progress.
    
    Creates plots for each CSV log file in the specified directory,
    which helps visualize training dynamics and identify potential issues
    like unstable learning or convergence problems.
    
    Args:
        log_dir (str): Directory containing CSV log files from training
        
    Side effects:
        - Creates and saves plot images in the same directory as the logs
        - Prints status messages about created plots
        
    Note:
        The plots provide visual indicators of training stability and convergence.
        Sudden spikes or oscillations may indicate potential instability.
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import glob
        import os
        
        # Find all CSV log files created during training
        log_files = glob.glob(os.path.join(log_dir, "*.csv"))
        for log_file in log_files:
            try:
                log_name = os.path.basename(log_file).replace('.csv', '')
                df = pd.read_csv(log_file)
                
                # Skip files with no data
                if df.empty:
                    continue
                
                # Create figure
                plt.figure(figsize=(10, 6))
                
                # If there's a 'step' column, use that for x-axis
                # This provides better alignment with training progress
                if 'step' in df.columns:
                    x = df['step']
                    x_label = 'Step'
                else:
                    x = df.index
                    x_label = 'Iteration'
                
                # Plot all numeric columns except 'step'
                # This visualizes metrics like loss values, Q-values, etc.
                for col in df.select_dtypes(include=[np.number]).columns:
                    if col != 'step':
                        plt.plot(x, df[col], label=col)
                
                plt.title(f'{log_name} Training Curve')
                plt.xlabel(x_label)
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                
                # Save figure for later analysis
                save_path = os.path.join(log_dir, f'{log_name}_plot.png')
                plt.savefig(save_path)
                plt.close()
                print(f"Plot saved to {save_path}")
                
            except Exception as e:
                print(f"Error plotting {log_file}: {e}")
                
    except Exception as e:
        print(f"Error plotting training curves: {e}")

def check_gradient_values(log_dir):
    """
    Check gradient values in log files for extreme values or NaNs.
    
    Analyzing gradient statistics can reveal training instabilities:
    - NaN gradients indicate numerical errors
    - Very large gradients suggest exploding gradients
    - Very small gradients suggest vanishing gradients
    
    Args:
        log_dir (str): Directory containing gradient log files
        
    Side effects:
        Prints warnings about potential gradient issues
        
    Note:
        Gradient issues are often the root cause of training instability
        and NaN errors. This function helps detect them early.
    """
    try:
        import pandas as pd
        import glob
        import os
        
        # Look for gradient log files (created by d3rlpy)
        grad_files = glob.glob(os.path.join(log_dir, "*grad.csv"))
        
        for grad_file in grad_files:
            try:
                grad_name = os.path.basename(grad_file)
                df = pd.read_csv(grad_file)
                
                # Check for NaNs in gradients - these indicate serious problems
                nan_count = df.isna().sum().sum()
                if nan_count > 0:
                    print(f"⚠️  WARNING: {grad_name} has {nan_count} NaN gradient values")
                
                # Check for extremely large gradients - potential exploding gradients
                # These can lead to NaN values in subsequent steps
                if df.abs().max().max() > 100:
                    max_val = df.abs().max().max()
                    print(f"⚠️  WARNING: {grad_name} has very large gradient values: {max_val}")
                    
                # Check for extremely small (vanishing) gradients
                # These can lead to stalled training
                non_zero = df[df != 0]
                if non_zero.size > 0 and non_zero.abs().min().min() < 1e-8:
                    min_val = non_zero.abs().min().min()
                    print(f"⚠️  WARNING: {grad_name} has very small gradient values: {min_val}")
                    
            except Exception as e:
                print(f"Error checking gradient file {grad_file}: {e}")
                
    except Exception as e:
        print(f"Error checking gradient values: {e}")