#!/usr/bin/env python3
# -----------------------------------------------------------
# data_loader.py  --  Data loading and preprocessing for HFNC
# 
# This module handles data loading and preprocessing for the HFNC pipeline:
# - Loading the parquet data file containing HFNC episodes
# - Filtering and preprocessing features for the state representation
# - Extracting actions, rewards, and done flags
# - Detecting and handling NaN values in the dataset
# -----------------------------------------------------------
import pandas as pd
import numpy as np

def load_data(data_path):
    """
    Load data from parquet file containing HFNC episodes.
    
    The parquet file should contain structured data from High-Flow Nasal Cannula
    treatment episodes, including patient states, actions taken (flow/FiO2 settings),
    and outcomes.
    
    Args:
        data_path (str): Path to the parquet file
        
    Returns:
        pandas.DataFrame: DataFrame containing the loaded HFNC episode data
        
    Note:
        The expected columns in the dataframe include:
        - Clinical features (physiological parameters)
        - "action" column (discrete action ID)
        - "reward" column (computed reward values)
        - "done" column (episode termination flags)
    """
    print("⏳  Loading Parquet …")
    df = pd.read_parquet(data_path)
    csv_path = data_path.replace(".parquet", ".csv")
    print(f"💾  Saving data to CSV: {csv_path} …")
    df.to_csv(csv_path, index=False)
    print("✅  CSV saved successfully.")
    return df

def preprocess_data(df):
    """
    Preprocess data for CQL training by extracting states, actions, rewards, and dones.
    
    This function:
    1. Identifies and filters non-numeric columns from state representation
    2. Extracts state features, actions, rewards, and done flags
    3. Detects and reports NaN values in the data
    4. Handles NaN values using mean imputation
    
    Args:
        df (pandas.DataFrame): DataFrame containing raw HFNC episode data
        
    Returns:
        dict: Dictionary containing processed components:
            - states (np.ndarray): State features array (float32)
            - actions (np.ndarray): Action IDs array (int64)
            - rewards (np.ndarray): Reward values array (float32)
            - dones (np.ndarray): Episode termination flags (bool)
            - state_columns (list): Names of the state feature columns
            
    Important:
        NaN values in the states are imputed with the feature mean values.
        This is critical for preventing training failures in the CQL algorithm.
    """
    # Exclude non-numeric columns and other columns that shouldn't be part of the state
    # These include MDP components (action, reward, done), identifiers, and non-numeric features
    columns_to_exclude = [
        "action", "reward", "done",  # MDP components
        "subject_id", "stay_id", "hfnc_episode",  # Patient/episode identifiers
        "outcome_label", "hour_ts", "ep_start_ts",  # Temporal and outcome indicators
        "o2_delivery_device_1", "humidifier_water_changed",  # String columns
        # Add other non-numeric columns here if needed
    ]

    # Select only numeric columns for the state representation
    # This dynamically tests each column to ensure it can be part of the state
    state_cols = []
    for col in df.columns:
        if col not in columns_to_exclude:
            try:
                # Test if column can be converted to float (required for RL algorithm)
                df[col].astype("float32")
                state_cols.append(col)
            except (ValueError, TypeError):
                print(f"Excluding non-numeric column from state: {col}")

    # Extract MDP components needed for reinforcement learning
    states = df[state_cols].values.astype("float32")  # State features
    actions = df["action"].values.astype("int64")     # Discrete actions
    rewards = df["reward"].values.astype("float32")   # Reward signals
    dones = df["done"].values.astype("bool")          # Episode termination flags

    # Check for NaN values in states - these can cause training to fail with NaN errors
    nan_mask = np.isnan(states)
    if np.any(nan_mask):
        nan_count = np.sum(nan_mask)
        nan_features = [(i, col) for i, col in enumerate(state_cols) if np.any(np.isnan(states[:, i]))]
        print(f"⚠️  WARNING: Found {nan_count} NaN values in states")
        print(f"NaN values found in features: {nan_features}")
        
        # Print detailed statistics about NaN features to help diagnose data quality issues
        for idx, feature_name in nan_features:
            nan_feature_count = np.sum(np.isnan(states[:, idx]))
            nan_percentage = 100 * nan_feature_count / len(states)
            print(f"  - {feature_name}: {nan_feature_count} NaNs ({nan_percentage:.2f}%)")
        
        # Mean imputation for NaN values in each feature column
        # This prevents NaN propagation during model training
        print("Filling NaN values with feature means...")
        for i in range(states.shape[1]):
            if np.any(np.isnan(states[:, i])):
                col_mean = np.nanmean(states[:, i])
                states[:, i] = np.nan_to_num(states[:, i], nan=col_mean)

    # Check for NaN values in rewards (also critical for training stability)
    if np.any(np.isnan(rewards)):
        nan_count = np.sum(np.isnan(rewards))
        print(f"⚠️  WARNING: Found {nan_count} NaN values in rewards")
# Print statistics about NaN rewards
        nan_rewards_count = np.sum(np.isnan(rewards))
        nan_rewards_percentage = 100 * nan_rewards_count / len(rewards)
        print(f"  - rewards: {nan_rewards_count} NaNs ({nan_rewards_percentage:.2f}%)")
        #print head of Nan rewards
        with pd.option_context('display.max_columns', None):
            print("Rows with NaN rewards (showing other columns):")
            nan_reward_indices = np.where(np.isnan(rewards))[0]
            print(df.iloc[nan_reward_indices].head(100))
            # Filter out excluded columns before printing
            columns_to_show = [col for col in df.columns if col not in columns_to_exclude]
            print(df.iloc[nan_reward_indices][columns_to_show].head(100))
            

        # Fill NaN rewards with zeros
        rewards = np.nan_to_num(rewards, nan=0.0)

    return {
        "states": states,         # State features (observations)
        "actions": actions,       # Discrete actions taken
        "rewards": rewards,       # Reward values
        "dones": dones,           # Episode termination flags
        "state_columns": state_cols  # Names of state features (for interpretation)
    }

