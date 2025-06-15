#!/usr/bin/env python3
import numpy as np
import math
from dataset import count_transitions
from d3rlpy.dataset import FIFOBuffer, ReplayBuffer
from d3rlpy.metrics import TDErrorEvaluator, DiscreteActionMatchEvaluator 

def train_model(model, train_episodes,val_episodes, n_epochs, batch_size, experiment_name=None):
    print("🚀  Starting training DiscreteCQL...")
    n_transitions = sum(len(ep.observations) for ep in train_episodes)
    print(f"Using dataset with {n_transitions} transitions for training")

    # Calculate steps_per_epoch based on dataset size and batch size
    steps_per_epoch = math.ceil(n_transitions / batch_size)
    
    print(f"📊 Training configuration:")
    print(f"   Dataset size: {n_transitions} transitions")
    print(f"   Batch size: {batch_size}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Data passes per epoch: 1.0x (one complete pass)")
    
    n_steps = n_epochs * steps_per_epoch
    
    buffer_limit = n_transitions
    replay_buffer = ReplayBuffer(
        buffer=FIFOBuffer(limit=buffer_limit),
        episodes=train_episodes
    )
    print(f"Training for {n_epochs} epochs, with {steps_per_epoch} steps per epoch, totaling {n_steps} steps...")

    # --- THIS IS THE KEY CHANGE ---
    # Create the evaluators dictionary to pass to the .fit() method
    evaluators = {}
    if val_episodes:
        print(f"🔬 Validation set provided with {len(val_episodes)} episodes. Setting up TDErrorEvaluator.")
        # The TDErrorEvaluator computes the Bellman error on the validation set
        evaluators = {
        'validation_td_error': TDErrorEvaluator(episodes=val_episodes),
        'validation_action_match': DiscreteActionMatchEvaluator(episodes=val_episodes)
    }
        # td_error_evaluator = TDErrorEvaluator(episodes=val_episodes)
        # evaluators['validation'] = td_error_evaluator
    # --------------------------------

    result = model.fit(
        replay_buffer,
        n_steps=n_steps,
        n_steps_per_epoch=steps_per_epoch,
        experiment_name=experiment_name,
        show_progress=True,
        save_interval=max(n_steps // 10, 1),
        evaluators=evaluators,  # <--- PASS THE DICTIONARY HERE
    )
    print("✅  Training completed successfully!")
    return result, None






    #########OLD
    # print(f"Training for {n_epochs} epochs, with {steps_per_epoch} steps per epoch, totaling {n_steps} steps...")
    
    # result = model.fit(
    #     replay_buffer,
    #     n_steps=n_steps,
    #     n_steps_per_epoch=steps_per_epoch,
    #     experiment_name=experiment_name,
    #     show_progress=True,
    #     save_interval=max(n_steps // 10, 1),
    #     evaluators=None if val_episodes is None else val_episodes,
    # )
    # print("✅  Training completed successfully!")
    # return result, None

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
                
                # Check for extreme values, but exclude non-data columns like epoch and step
                if 'min' in df.columns and 'max' in df.columns:
                    # For gradient files with min/max columns, check actual gradient values
                    max_vals = max(df['max'].abs().max(), df['min'].abs().max())
                else:
                    # For other files, exclude epoch and step columns from extreme value check
                    data_cols = [col for col in df.columns if col not in ['epoch', 'step']]
                    if data_cols:
                        max_vals = df[data_cols].max().max()
                    else:
                        max_vals = 0
                
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