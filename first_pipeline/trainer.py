#!/usr/bin/env python3
import numpy as np
from dataset import count_transitions
from d3rlpy.dataset import FIFOBuffer, ReplayBuffer

def train_model(model, train_episodes, n_epochs, experiment_name=None, val_episodes=None):
    print("🚀  Starting training DiscreteCQL...")
    n_transitions = sum(len(ep.observations) for ep in train_episodes)
    print(f"Using dataset with {n_transitions} transitions for training")

    steps_per_epoch = 10000
    n_steps        = n_epochs * steps_per_epoch
    
    buffer_limit = n_transitions
    replay_buffer = ReplayBuffer(
        buffer=FIFOBuffer(limit=buffer_limit),
        episodes=train_episodes
    )
    print(f"Training for estimated {n_steps} steps...")
    try:
        result = model.fit(
            replay_buffer,               # first positional arg
            n_steps=n_steps,
            n_steps_per_epoch=steps_per_epoch,
            experiment_name=experiment_name,
            show_progress=True,
            save_interval=max(n_steps // 10, 1),
           evaluators=None if val_episodes is None else val_episodes,
        )
        print("✅  Training completed successfully!")
        return result, None
    except Exception as first_error:
        print(f"⚠️  Error during training: {first_error}")
        print("Trying alternative training approach...")
        quit()
        try:
            n_steps_reduced = n_epochs * 1000
            print(f"Training with simplified approach for {n_steps_reduced} steps...")
            result = model.fit(
                train_episodes,
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
    try:
        import pandas as pd
        import os
        import glob
        log_files = glob.glob(os.path.join(log_dir, "*.csv"))
        print(f"Found {len(log_files)} log files to analyze")
        issues = {}
        for log_file in log_files:
            try:
                log_name = os.path.basename(log_file)
                df = pd.read_csv(log_file)
                nan_count = df.isna().sum().sum()
                if nan_count > 0:
                    issues[log_name] = f"Contains {nan_count} NaN values"
                    print(f"⚠️  WARNING: {log_name} contains {nan_count} NaN values")
                    nan_cols = df.columns[df.isna().any()].tolist()
                    print(f"  NaN values found in columns: {nan_cols}")
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