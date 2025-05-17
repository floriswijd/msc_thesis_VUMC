#!/usr/bin/env python3
# -----------------------------------------------------------
# data_loader.py  --  Data loading and preprocessing for HFNC
# -----------------------------------------------------------
import pandas as pd
import numpy as np

def load_data(data_path):
    """Load data from parquet file"""
    print("⏳  Loading Parquet …")
    return pd.read_parquet(data_path)

def preprocess_data(df):
    """Preprocess data for CQL training"""
    # Exclude non-numeric columns and other columns that shouldn't be part of the state
    columns_to_exclude = [
        "action", "reward", "done",
        "subject_id", "stay_id", "hfnc_episode",
        "outcome_label", "hour_ts", "ep_start_ts",
        "o2_delivery_device_1", "humidifier_water_changed",  # String columns
        # Add other non-numeric columns here if needed
    ]

    # Select only numeric columns for the state
    state_cols = []
    for col in df.columns:
        if col not in columns_to_exclude:
            try:
                # Test if column can be converted to float
                df[col].astype("float32")
                state_cols.append(col)
            except (ValueError, TypeError):
                print(f"Excluding non-numeric column from state: {col}")

    # Extract components
    states = df[state_cols].values.astype("float32")
    actions = df["action"].values.astype("int64")
    rewards = df["reward"].values.astype("float32")
    dones = df["done"].values.astype("bool")

    # Check for NaN values in states
    nan_mask = np.isnan(states)
    if np.any(nan_mask):
        nan_count = np.sum(nan_mask)
        nan_features = [(i, col) for i, col in enumerate(state_cols) if np.any(np.isnan(states[:, i]))]
        print(f"⚠️  WARNING: Found {nan_count} NaN values in states")
        print(f"NaN values found in features: {nan_features}")
        
        # Print statistics about NaN features
        for idx, feature_name in nan_features:
            nan_feature_count = np.sum(np.isnan(states[:, idx]))
            nan_percentage = 100 * nan_feature_count / len(states)
            print(f"  - {feature_name}: {nan_feature_count} NaNs ({nan_percentage:.2f}%)")
        
        # Basic NaN handling (fill with mean)
        print("Filling NaN values with feature means...")
        for i in range(states.shape[1]):
            if np.any(np.isnan(states[:, i])):
                col_mean = np.nanmean(states[:, i])
                states[:, i] = np.nan_to_num(states[:, i], nan=col_mean)

    # Check for NaN values in rewards
    if np.any(np.isnan(rewards)):
        nan_count = np.sum(np.isnan(rewards))
        print(f"⚠️  WARNING: Found {nan_count} NaN values in rewards")
        # Fill NaN rewards with zeros
        rewards = np.nan_to_num(rewards, nan=0.0)

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "state_columns": state_cols
    }