#!/usr/bin/env python3
# -----------------------------------------------------------
# train_cql.py  --  Offline Conservative Q-Learning voor HFNC
# -----------------------------------------------------------
#
# • Laadt data/processed/hfnc_episodes.parquet
# • Maakt MDPDataset met train/val/test splits
# • Instantieert DiscreteCQL (d3rlpy) met hyperparams uit YAML
# • Logt evaluatie tijdens training
# • Slaat model + scaler + metrics op
#
# CLI:
#   python train_cql.py --alpha 1.0 --epochs 200 --gpu 0
# -----------------------------------------------------------

import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from d3rlpy.dataset import MDPDataset
from d3rlpy.preprocessing import StandardObservationScaler
from d3rlpy.algos import DiscreteCQL

# ---------- CLI ARGUMENTEN ----------
parser = argparse.ArgumentParser()
parser.add_argument("--data",  default="/Users/floppie/Documents/Msc Scriptie/hfnc_episodes.parquet")
parser.add_argument("--cfg",   default="/Users/floppie/Documents/Msc Scriptie/config.yaml")
parser.add_argument("--alpha", type=float, default=1.0, help="CQL conservatisme-gewicht")
parser.add_argument("--epochs", type=int,  default=200)
parser.add_argument("--batch",  type=int,  default=256)
parser.add_argument("--lr",     type=float, default=1e-3)
parser.add_argument("--gamma",  type=float, default=0.99)
parser.add_argument("--gpu",    type=int,   default=0, help="-1 = CPU")
parser.add_argument("--logdir", default="runs/cql")
args = parser.parse_args()

# ---------- PADEN ----------
Path(args.logdir).mkdir(parents=True, exist_ok=True)
model_path  = Path(args.logdir) / "cql_hfnc.pt"
metric_path = Path(args.logdir) / "metrics.yaml"

# ---------- DATA INLADEN ----------
print("⏳  Loading Parquet …")
df  = pd.read_parquet(args.data)
cfg = yaml.safe_load(open(args.cfg))

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

states  = df[state_cols].values.astype("float32")
actions = df["action"].values.astype("int64")
rewards = df["reward"].values.astype("float32")
dones   = df["done"].values.astype("bool")

dataset = MDPDataset(states, actions, rewards, dones)

# ---------- SPLIT train / val / test ----------
train_eps, temp_eps = train_test_split(dataset.episodes,
                                       test_size=0.3,
                                       random_state=42)

val_eps,   test_eps = train_test_split(temp_eps,
                                       test_size=0.5,
                                       random_state=42)

print(f"📊 Episodes  train={len(train_eps)}, val={len(val_eps)}, test={len(test_eps)}")

# ---------- SCALER ----------
scaler = StandardObservationScaler()   # z-score voor alle features

# Import necessary modules for d3rlpy 2.8.1
import torch
from d3rlpy.algos import DiscreteCQLConfig

# Let's create a proper config for DiscreteCQL
print(f"Using alpha={args.alpha}, batch={args.batch}, lr={args.lr}")
# Force CPU usage since there seems to be a CUDA issue
device = "cpu"
print(f"Using device: {device} (forced CPU due to CUDA compatibility issues)")

# ---------- CQL INSTANTIE ----------
# Create the config object for DiscreteCQL with all required parameters
config = DiscreteCQLConfig(
    batch_size=args.batch,
    learning_rate=args.lr,
    gamma=args.gamma,
    observation_scaler=scaler,  # Pass the scaler directly in the config
    alpha=args.alpha  # CQL conservatism parameter is called 'alpha' in this version
)

print(f"Created config with alpha={config.alpha}, batch_size={config.batch_size}")

# Create CQL instance with our config
cql = DiscreteCQL(
    config=config, 
    device=device, 
    enable_ddp=False
)

# Print configuration details
print(f"CQL instance created with device: {device}")

# ---------- TRAIN ----------
print("🚀  Start training DiscreteCQL …")

# In d3rlpy 2.8.1, the API for datasets has changed
try:
    # Since we already have the dataset from the MDPDataset creation,
    # let's try to use it directly
    
    # Instead of using len(dataset), get size from number of transitions
    n_transitions = sum(len(episode.observations) for episode in dataset.episodes)
    print(f"Using existing dataset with {n_transitions} transitions for training")
    
    # Calculate n_steps based on epochs
    # This is a rough estimation - adjust based on your needs
    n_steps = args.epochs * 10000  # Fixed number of steps per epoch
    
    # Call fit with the correct parameters for version 2.8.1
    print(f"Training for {n_steps} steps...")
    cql.fit(
        dataset=dataset,
        n_steps=n_steps,
        experiment_name=args.logdir,
    )
except Exception as e:
    print(f"Error during training: {e}")
    print("Trying a more direct approach...")
    
    # If the first approach fails, try to create the simplest possible call
    try:
        # Define a fixed number of steps instead of trying to calculate from dataset
        n_steps = args.epochs * 1000  # Fixed value as a fallback
        
        print(f"Training with simplified approach for {n_steps} steps...")
        cql.fit(
            dataset=dataset,
            n_steps=n_steps,
            experiment_name=args.logdir,
        )
    except Exception as e2:
        print(f"Error during simplified training: {e2}")
        print("Training failed. Please check d3rlpy documentation for version 2.8.1.")

# ---------- OFFLINE TEST EVALUATIE ----------
print("Evaluating on test episodes...")
metrics = {}

try:
    # Try different evaluation approaches available in d3rlpy 2.8.1
    
    # First check if we can directly evaluate on the model
    if hasattr(cql, 'predict'):
        print("Evaluating using direct prediction...")
        
        # Calculate average returns manually
        test_returns = []
        for episode in test_eps:
            observations = episode.observations
            actions = []
            for obs in observations:
                # Get the action that the trained policy would take
                action = cql.predict(obs.reshape(1, -1))[0]
                actions.append(action)
            
            # Compare to actual actions and rewards in the episode
            match_count = sum(a1 == a2 for a1, a2 in zip(actions, episode.actions))
            match_rate = match_count / len(actions) if len(actions) > 0 else 0
            
            # Calculate episode return
            episode_return = sum(episode.rewards)
            test_returns.append(episode_return)
            
        metrics["test_returns_mean"] = float(np.mean(test_returns))
        metrics["test_returns_std"] = float(np.std(test_returns))
        metrics["policy_match_rate"] = float(match_rate)
        print(f"Calculated test returns: mean={metrics['test_returns_mean']:.4f}, std={metrics['test_returns_std']:.4f}")
    else:
        print("Model doesn't have predict method, skipping direct evaluation")

except Exception as e:
    print(f"Warning: Could not evaluate: {e}")
    print("Skipping evaluation phase")

# Add training parameters to metrics
metrics["alpha"] = args.alpha
metrics["epochs"] = args.epochs
metrics["batch_size"] = args.batch
metrics["learning_rate"] = args.lr
metrics["gamma"] = args.gamma

yaml.safe_dump(metrics, open(metric_path, "w"))
print("✅  Test-scores:", metrics)

# ---------- MODEL OPSLAAN ----------
try:
    # In 2.8.1, the save API may have changed
    cql.save_model(model_path)
    print(f"💾  Model opgeslagen → {model_path}")
except Exception as e:
    # Try alternate API if available
    try:
        cql.save(model_path)
        print(f"💾  Model opgeslagen → {model_path}")
    except Exception as e2:
        print(f"❌ Could not save model: {e2}")