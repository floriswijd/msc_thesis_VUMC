#!/usr/bin/env python3
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def timestamp_string():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def debug_nan_values(array, name="array"):
    nan_mask = np.isnan(array)
    nan_count = np.sum(nan_mask)
    if nan_count > 0:
        print(f"⚠️  WARNING: {nan_count} NaN values found in {name}")
        if array.ndim > 1:
            nan_cols = np.sum(nan_mask, axis=0)
            for i, count in enumerate(nan_cols):
                if count > 0:
                    print(f"  - Column {i}: {count} NaNs ({100 * count / array.shape[0]:.2f}%)")
        print(f"  - Overall: {nan_count}/{array.size} ({100 * nan_count / array.size:.2f}%)")
        return True
    return False

def debug_inf_values(array, name="array"):
    inf_mask = np.isinf(array)
    inf_count = np.sum(inf_mask)
    if inf_count > 0:
        print(f"⚠️  WARNING: {inf_count} infinite values found in {name}")
        if array.ndim > 1:
            inf_cols = np.sum(inf_mask, axis=0)
            for i, count in enumerate(inf_cols):
                if count > 0:
                    print(f"  - Column {i}: {count} infs ({100 * count / array.shape[0]:.2f}%)")
        print(f"  - Overall: {inf_count}/{array.size} ({100 * inf_count / array.size:.2f}%)")
        return True
    return False

def plot_training_curves(log_dir):
    print(f"Plotting training curves from {log_dir}...")
    if not os.path.exists(log_dir):
        print(f"⚠️  WARNING: Log directory {log_dir} does not exist.")
        return
    try:
        log_files = glob.glob(os.path.join(log_dir, "*.csv"))
        if not log_files:
            print(f"No CSV files found in {log_dir}")
            return

        # Define default column names for headerless CSVs from d3rlpy
        # Column 0: epoch (or an equivalent index)
        # Column 1: step
        # Column 2: metric value
        default_col_names = ['epoch_idx', 'step', 'value']

        for log_file in log_files:
            # log_name is derived from the filename, e.g., "time_step", "conservative_loss"
            log_name = os.path.basename(log_file).replace('.csv', '')
            try:
                # Read CSV without a header row and assign default column names
                df = pd.read_csv(log_file, header=None, names=default_col_names)
                
                if df.empty:
                    print(f"Skipping empty log file: {log_file}")
                    continue
                
                # Ensure the necessary columns 'step' and 'value' are present after assignment
                if 'step' not in df.columns or 'value' not in df.columns:
                    print(f"Skipping {log_file}: Default columns 'step' or 'value' not found after assignment. DataFrame columns: {df.columns.tolist()}")
                    continue

                plt.figure(figsize=(10, 6))
                # Plot 'step' vs 'value'
                plt.plot(df['step'], df['value'], label=log_name)
                
                plt.xlabel("Step") # X-axis is always 'Step'
                # Y-axis label uses the filename-derived metric name
                plt.ylabel(log_name.replace('_', ' ').capitalize()) 
                # Title also uses the filename-derived metric name
                plt.title(f"{log_name.replace('_', ' ').capitalize()} Training Curve") 
                
                plt.legend() # Legend will show the filename-based log_name
                plt.grid(True)
                
                plot_save_path = os.path.join(log_dir, f"{log_name}_curve.png")
                plt.savefig(plot_save_path)
                plt.close() # Close the figure to free memory
                print(f"  📈 Saved plot: {plot_save_path}")
            except Exception as e:
                print(f"Error processing or plotting {log_file}: {e}")
                if 'plt' in locals() and plt.gcf().get_axes(): # Attempt to close figure on error
                     plt.close()
    except Exception as e:
        print(f"Error accessing log directory {log_dir} for plotting: {e}")

def check_gradient_values(log_dir):
    try:
        import pandas as pd
        import glob
        import os
        grad_files = glob.glob(os.path.join(log_dir, "*grad.csv"))
        for grad_file in grad_files:
            try:
                grad_name = os.path.basename(grad_file)
                df = pd.read_csv(grad_file)
                nan_count = df.isna().sum().sum()
                if nan_count > 0:
                    print(f"⚠️  WARNING: {grad_name} has {nan_count} NaN gradient values")
                if df.abs().max().max() > 100:
                    max_val = df.abs().max().max()
                    print(f"⚠️  WARNING: {grad_name} has very large gradient values: {max_val}")
                non_zero = df[df != 0]
                if non_zero.size > 0 and non_zero.abs().min().min() < 1e-8:
                    min_val = non_zero.abs().min().min()
                    print(f"⚠️  WARNING: {grad_name} has very small gradient values: {min_val}")
            except Exception as e:
                print(f"Error checking gradient file {grad_file}: {e}")
    except Exception as e:
        print(f"Error checking gradient values: {e}")