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

        # Define default column names for headerless CSVs
        default_col_names = ['epoch_idx', 'step', 'value']

        for log_file in log_files:
            log_name = os.path.basename(log_file).replace('.csv', '')
            try:
                df = None
                is_qfunc_log = log_name.startswith("q_funcs")

                if is_qfunc_log:
                    # q_funcs logs have headers with extended statistics
                    df = pd.read_csv(log_file, header=0)
                else:
                    # Other logs are headerless
                    df = pd.read_csv(log_file, header=None, names=default_col_names)
                
                if df.empty:
                    print(f"Skipping empty log file: {log_file}")
                    continue
                
                x_axis_col_name = 'step'
                
                if 'step' not in df.columns:
                    print(f"Skipping {log_file}: 'step' column not found. Available columns: {df.columns.tolist()}")
                    continue

                plt.figure(figsize=(10, 6))

                if is_qfunc_log and 'min' in df.columns and 'max' in df.columns and 'mean' in df.columns and 'std' in df.columns:
                    # Enhanced visualization for q_funcs files with statistical columns
                    
                    # Plot min-max range as light shaded area
                    plt.fill_between(df[x_axis_col_name], df['min'], df['max'], 
                                    alpha=0.2, color='blue', label='Min-Max Range')
                    
                    # Plot mean±std as a darker shaded area
                    plt.fill_between(df[x_axis_col_name], df['mean'] - df['std'], df['mean'] + df['std'], 
                                    alpha=0.4, color='blue', label='Mean ± Std')
                    
                    # Plot mean as a solid line
                    plt.plot(df[x_axis_col_name], df['mean'], 
                            color='blue', linewidth=2, label='Mean')
                    
                    # Customize for gradient plots
                    metric_name = log_name.replace('q_funcs.', '').replace('_grad', ' gradient')
                    plt.title(f"{metric_name.replace('_', ' ').capitalize()}")
                    plt.ylabel("Gradient Value")
                    
                else:
                    # Simple line plot for other files (like time_step.csv)
                    if is_qfunc_log:
                        y_axis_col_name = df.columns[2] if len(df.columns) >= 3 else None
                    else:
                        y_axis_col_name = 'value' if 'value' in df.columns else None
                    
                    if y_axis_col_name is None:
                        print(f"Skipping {log_file}: Unable to determine y-axis column")
                        plt.close()
                        continue
                    
                    plt.plot(df[x_axis_col_name], df[y_axis_col_name], 
                             linewidth=2, color='blue', label=log_name)
                    
                    # Use metric name from file for title and y-axis
                    clean_name = log_name.replace('_', ' ').capitalize()
                    plt.title(f"{clean_name} Training Curve")
                    plt.ylabel(y_axis_col_name if y_axis_col_name != 'value' else clean_name)
                
                plt.xlabel("Step")
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Add visual elements showing parameter thresholds for gradient plots
                if is_qfunc_log and 'max' in df.columns:
                    # Add horizontal lines at important threshold values
                    if df['max'].abs().max() > 1.0:
                        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
                        plt.axhline(y=-1.0, color='red', linestyle='--', alpha=0.5)
                
                # Ensure y-axis is symmetric around zero for gradient plots
                if is_qfunc_log:
                    y_max = df[['min', 'max']].abs().max().max() * 1.1  # Add 10% margin
                    plt.ylim(-y_max, y_max)
                
                plot_save_path = os.path.join(log_dir, f"{log_name}_curve.png")
                plt.tight_layout()
                plt.savefig(plot_save_path, dpi=120)
                plt.close()
                print(f"  📈 Saved plot: {plot_save_path}")
                
            except Exception as e:
                print(f"Error processing or plotting {log_file}: {e}")
                if 'plt' in locals() and plt.gcf().get_axes():
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