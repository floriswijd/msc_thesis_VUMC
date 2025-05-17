#!/usr/bin/env python3
# -----------------------------------------------------------
# utils.py  --  Utility functions for HFNC CQL
# -----------------------------------------------------------
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def timestamp_string():
    """Generate a timestamp string for logging"""
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def debug_nan_values(array, name="array"):
    """Debug NaN values in an array"""
    nan_mask = np.isnan(array)
    nan_count = np.sum(nan_mask)
    
    if nan_count > 0:
        print(f"⚠️  WARNING: {nan_count} NaN values found in {name}")
        if array.ndim > 1:
            # For 2D arrays, report NaN counts per column
            nan_cols = np.sum(nan_mask, axis=0)
            for i, count in enumerate(nan_cols):
                if count > 0:
                    print(f"  - Column {i}: {count} NaNs ({100 * count / array.shape[0]:.2f}%)")
        
        # For any array, report percentage
        print(f"  - Overall: {nan_count}/{array.size} ({100 * nan_count / array.size:.2f}%)")
        return True
    return False

def debug_inf_values(array, name="array"):
    """Debug infinite values in an array"""
    inf_mask = np.isinf(array)
    inf_count = np.sum(inf_mask)
    
    if inf_count > 0:
        print(f"⚠️  WARNING: {inf_count} infinite values found in {name}")
        if array.ndim > 1:
            # For 2D arrays, report infinite counts per column
            inf_cols = np.sum(inf_mask, axis=0)
            for i, count in enumerate(inf_cols):
                if count > 0:
                    print(f"  - Column {i}: {count} infs ({100 * count / array.shape[0]:.2f}%)")
        
        # For any array, report percentage
        print(f"  - Overall: {inf_count}/{array.size} ({100 * inf_count / array.size:.2f}%)")
        return True
    return False

def plot_training_curves(log_dir):
    """Plot training curves from CSV files"""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import glob
        import os
        
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
                if 'step' in df.columns:
                    x = df['step']
                    x_label = 'Step'
                else:
                    x = df.index
                    x_label = 'Iteration'
                
                # Plot all numeric columns except 'step'
                for col in df.select_dtypes(include=[np.number]).columns:
                    if col != 'step':
                        plt.plot(x, df[col], label=col)
                
                plt.title(f'{log_name} Training Curve')
                plt.xlabel(x_label)
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                
                # Save figure
                save_path = os.path.join(log_dir, f'{log_name}_plot.png')
                plt.savefig(save_path)
                plt.close()
                print(f"Plot saved to {save_path}")
                
            except Exception as e:
                print(f"Error plotting {log_file}: {e}")
                
    except Exception as e:
        print(f"Error plotting training curves: {e}")

def check_gradient_values(log_dir):
    """Check gradient values in log files for extreme values or NaNs"""
    try:
        import pandas as pd
        import glob
        import os
        
        # Look for gradient log files
        grad_files = glob.glob(os.path.join(log_dir, "*grad.csv"))
        
        for grad_file in grad_files:
            try:
                grad_name = os.path.basename(grad_file)
                df = pd.read_csv(grad_file)
                
                # Check for NaNs
                nan_count = df.isna().sum().sum()
                if nan_count > 0:
                    print(f"⚠️  WARNING: {grad_name} has {nan_count} NaN gradient values")
                
                # Check for extremely large gradients
                if df.abs().max().max() > 100:
                    max_val = df.abs().max().max()
                    print(f"⚠️  WARNING: {grad_name} has very large gradient values: {max_val}")
                    
                # Check for extremely small (vanishing) gradients
                non_zero = df[df != 0]
                if non_zero.size > 0 and non_zero.abs().min().min() < 1e-8:
                    min_val = non_zero.abs().min().min()
                    print(f"⚠️  WARNING: {grad_name} has very small gradient values: {min_val}")
                    
            except Exception as e:
                print(f"Error checking gradient file {grad_file}: {e}")
                
    except Exception as e:
        print(f"Error checking gradient values: {e}")