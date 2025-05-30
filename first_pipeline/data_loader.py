#!/usr/bin/env python3

import pandas as pd
import numpy as np


def load_data(data_path):
    print("⏳  Loading Parquet …")
    df = pd.read_parquet(data_path)
    csv_path = data_path.replace(".parquet", ".csv")
    print(f"💾  Saving data to CSV: {csv_path} …")
    df.to_csv(csv_path, index=False)
    print("✅  CSV saved successfully.")
    return df


def preprocess_data(df):
    columns_to_exclude = [
        "action",
        "reward",
        "done",
        "subject_id",
        "stay_id",
        "hfnc_episode",
        "outcome_label",
        "hour_ts",
        "ep_start_ts",
        "o2_delivery_device_1",
        "humidifier_water_changed",
        "o2_flow",           # Current flow rate - this is what we're trying to choose!
        "fio2",              # Current FiO2 - this is what we're trying to choose!
        "flow_bin",          # Binned current flow rate
        "fio2_bin",          # Binned current FiO2  
        "fio2_fraction",   
        "fio2_frac",
        "inspired_gas_temp", # Current temperature setting
        "humidification_Active",
        "humidification_HME",
        "humidification_nan",
        "episode_len",
        
        # 🚨 CRITICAL: Exclude future information (temporal leakage)
        "spo2_next",         # Next timestep's SpO2 - this is in the future!        
        # 🚨 CRITICAL: Exclude reward components (target leakage)
        "r_A", "r_B", "r_C", "r_D",  # Individual reward components
    ]
    
    state_cols = []
    for col in df.columns:
        if col not in columns_to_exclude:
            try:
                df[col].astype("float32")
                state_cols.append(col)
            except (ValueError, TypeError):
                print(f"Excluding non-numeric column from state: {col}")
    states = df[state_cols].values.astype("float32")
    actions = df["action"].values.astype("int64")
    rewards = df["reward"].values.astype("float32")
    dones = df["done"].values.astype("bool")
    nan_mask = np.isnan(states)
    if np.any(nan_mask):
        nan_count = np.sum(nan_mask)
        nan_features = [
            (i, col)
            for i, col in enumerate(state_cols)
            if np.any(np.isnan(states[:, i]))
        ]
        print(f"⚠️  WARNING: Found {nan_count} NaN values in states")
        print(f"NaN values found in features: {nan_features}")
        for idx, feature_name in nan_features:
            nan_feature_count = np.sum(np.isnan(states[:, idx]))
            nan_percentage = 100 * nan_feature_count / len(states)
            print(f"  - {feature_name}: {nan_feature_count} NaNs ({nan_percentage:.2f}%)")
        print("Filling NaN values with feature means...")
        for i in range(states.shape[1]):
            if np.any(np.isnan(states[:, i])):
                col_mean = np.nanmean(states[:, i])
                states[:, i] = np.nan_to_num(states[:, i], nan=col_mean)
    if np.any(np.isnan(rewards)):
        nan_count = np.sum(np.isnan(rewards))
        print(f"⚠️  WARNING: Found {nan_count} NaN values in rewards")
        nan_rewards_count = np.sum(np.isnan(rewards))
        nan_rewards_percentage = 100 * nan_rewards_count / len(rewards)
        print(f"  - rewards: {nan_rewards_count} NaNs ({nan_rewards_percentage:.2f}%)")
        with pd.option_context("display.max_columns", None):
            print("Rows with NaN rewards (showing other columns):")
            nan_reward_indices = np.where(np.isnan(rewards))[0]
            print(df.iloc[nan_reward_indices].head(100))
            columns_to_show = [col for col in df.columns if col not in columns_to_exclude]
            print(df.iloc[nan_reward_indices][columns_to_show].head(100))
        rewards = np.nan_to_num(rewards, nan=0.0)
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "state_columns": state_cols,
    }

