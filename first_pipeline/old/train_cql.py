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
#   python train_cql.py --alpha 1.0 --epochs 200
# -----------------------------------------------------------

import argparse
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time

# Import d3rlpy components with compatibility for v2.8.1
from d3rlpy.dataset import MDPDataset
from d3rlpy.preprocessing import StandardObservationScaler
from d3rlpy.algos import DiscreteCQL
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.models.encoders import VectorEncoderFactory
# For d3rlpy 2.8.1, optimizer factories are accessed differently
import torch.optim as torch_optim  # Use PyTorch optimizers instead

# Helper function to create visualizations
def create_training_plots(metrics, log_dir):
    """Create and save training plots based on metrics collected during training."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot training loss curve
    if 'loss_history' in metrics and len(metrics['loss_history']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['loss_history'])
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.yscale('log')  # Use log scale to better visualize loss changes
        loss_path = os.path.join(log_dir, f'training_loss_{timestamp}.png')
        plt.savefig(loss_path)
        print(f"📊 Training loss plot saved to {loss_path}")
        plt.close()
    
    # Plot action match rate if available
    if 'action_match_history' in metrics and len(metrics['action_match_history']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['action_match_history'])
        plt.title('Action Match Rate')
        plt.xlabel('Epochs')
        plt.ylabel('Match Rate')
        plt.ylim(0, 1.0)  # Match rate should be between 0 and 1
        plt.grid(True)
        match_path = os.path.join(log_dir, f'action_match_{timestamp}.png')
        plt.savefig(match_path)
        print(f"📊 Action match rate plot saved to {match_path}")
        plt.close()
        
    # Plot number of NaN losses if available
    if 'nan_counts' in metrics and sum(metrics['nan_counts']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['nan_counts'])
        plt.title('NaN Loss Count Per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Count')
        plt.grid(True)
        nan_path = os.path.join(log_dir, f'nan_losses_{timestamp}.png')
        plt.savefig(nan_path)
        print(f"📊 NaN loss count plot saved to {nan_path}")
        plt.close()

# Gradient clipping helper function
def apply_gradient_clipping(model, max_norm=1.0):
    """Apply gradient clipping to model parameters to prevent exploding gradients."""
    if hasattr(model, 'modules'):
        for module in model.modules():
            if hasattr(module, 'parameters'):
                torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm)
    elif hasattr(model, 'parameters'):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# ---------- CLI ARGUMENTEN ----------
parser = argparse.ArgumentParser()
parser.add_argument("--data",  default="/Users/floppie/Documents/Msc Scriptie/hfnc_episodes.parquet")
parser.add_argument("--cfg",   default="/Users/floppie/Documents/Msc Scriptie/config.yaml")
parser.add_argument("--alpha", type=float, default=1.0, help="CQL conservatisme-gewicht")
parser.add_argument("--epochs", type=int,  default=200)
parser.add_argument("--batch",  type=int,  default=256)
parser.add_argument("--lr",     type=float, default=1e-4)  # Reduced learning rate for better stability
parser.add_argument("--gamma",  type=float, default=0.99)
parser.add_argument("--gpu",    type=int,   default=-1, help="-1 = CPU")
parser.add_argument("--logdir", default="runs/cql_v2")
parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
parser.add_argument("--early_stop_patience", type=int, default=10, help="Patience for early stopping")
args = parser.parse_args()

# ---------- PADEN ----------
log_dir = Path(args.logdir)
log_dir.mkdir(parents=True, exist_ok=True)
model_path = log_dir / "cql_hfnc_model.pt"
metric_path = log_dir / "metrics.yaml"

# ---------- DATA INLADEN ----------
print("⏳ Loading Parquet data...")
df = pd.read_parquet(args.data)
cfg = yaml.safe_load(open(args.cfg))

# Exclude non-numeric columns and other columns that shouldn't be part of the state
columns_to_exclude = [
    "action", "reward", "done",
    "subject_id", "stay_id", "hfnc_episode",
    "outcome_label", "hour_ts", "ep_start_ts",
    "o2_delivery_device_1", "humidifier_water_changed",  # String columns
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

print(f"Using {len(state_cols)} features for state representation")

# Check for and handle any NaN values in the state columns
nan_cols = df[state_cols].columns[df[state_cols].isna().any()].tolist()
if nan_cols:
    print(f"⚠️ Warning: NaN values found in columns: {nan_cols}")
    print("Filling NaN values with column means")
    df[state_cols] = df[state_cols].fillna(df[state_cols].mean())

# Convert to arrays for MDPDataset
states = df[state_cols].values.astype("float32")
actions = df["action"].values.astype("int64")
rewards = df["reward"].values.astype("float32")
dones = df["done"].values.astype("bool")

# Create dataset - this format works with d3rlpy 2.8.1
dataset = MDPDataset(
    observations=states,
    actions=actions,
    rewards=rewards,
    terminals=dones
)

# Count episodes and transitions
n_episodes = len(dataset.episodes)
n_transitions = sum(len(ep.observations) for ep in dataset.episodes)
print(f"Dataset contains {n_episodes} episodes with {n_transitions} total transitions")

# ---------- SPLIT train / val / test ----------
train_eps, temp_eps = train_test_split(
    dataset.episodes,
    test_size=0.3,
    random_state=42
)

val_eps, test_eps = train_test_split(
    temp_eps,
    test_size=0.5,
    random_state=42
)

print(f"📊 Episodes split: train={len(train_eps)}, val={len(val_eps)}, test={len(test_eps)}")

# ---------- MODEL CONFIGURATION ----------
# Force CPU usage to avoid CUDA issues
device = "cpu" if args.gpu < 0 else f"cuda:{args.gpu}"
print(f"Using device: {device}")

# Create the CQL configuration with more stable hyperparameters to prevent NaN losses
config = DiscreteCQLConfig(
    batch_size=args.batch,
    learning_rate=args.lr,  # Using smaller learning rate for better stability
    gamma=args.gamma,  # Discount factor
    observation_scaler=StandardObservationScaler(),
    alpha=args.alpha,  # CQL alpha parameter (conservative penalty)
    n_critics=2  # Using 2 critics for more stable learning
)

# Determine total training steps based on epochs
# A reasonable number of steps per epoch
steps_per_epoch = min(5000, n_transitions // args.batch)
total_steps = args.epochs * steps_per_epoch

print(f"Creating CQL with config: alpha={config.alpha}, batch_size={config.batch_size}, lr={config.learning_rate}")
print(f"Training for {args.epochs} epochs ({total_steps} total steps)")

# Create the CQL instance with minimal custom parameters
cql = DiscreteCQL(
    config=config,
    device=device,
    enable_ddp=False  # Required parameter in d3rlpy 2.8.1
)

# ---------- TRAIN ----------
# Main training loop with proper error handling
print("🚀 Starting DiscreteCQL training...")
try:
    print("Building model with dataset...")
    cql.build_with_dataset(train_eps)
    
    print(f"Training for {args.epochs} epochs ({total_steps} steps)...")
    # Modified training loop to sample batch properly
    for epoch in range(args.epochs):
        for step in range(steps_per_epoch):
            # Sample a batch from dataset
            batch = train_eps.sample_transition_batch(args.batch)
            # Update with the batch
            loss = cql.update(batch)
            
            total_step = epoch * steps_per_epoch + step
            if total_step % args.log_interval == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Step {step+1}/{steps_per_epoch}, Loss: {loss}")
        
        # Save model every epoch
        if (epoch + 1) % args.save_interval == 0:
            print(f"Saving checkpoint at epoch {epoch+1}...")
            output_path = os.path.join(log_dir, f"model_{epoch+1}.pt")
            cql.save_model(output_path)
except Exception as e:
    print(f"⚠️ Error during training: {e}")
    import traceback
    traceback.print_exc()

# ---------- EVALUATION ----------
print("Evaluating on test episodes...")
metrics = metrics if 'metrics' in locals() else {}

try:
    # Calculate manual evaluation metrics
    test_returns = []
    action_matches = []
    
    # Collect evaluation metrics for each test episode
    for episode in test_eps:
        observations = episode.observations
        episode_actions = episode.actions
        episode_rewards = episode.rewards
        
        # Get predicted actions for each observation in the episode
        predicted_actions = []
        for obs in observations:
            action = cql.predict(obs.reshape(1, -1))[0]
            predicted_actions.append(action)
        
        # Calculate action match rate
        matches = sum(p == a for p, a in zip(predicted_actions, episode_actions))
        match_rate = matches / len(episode_actions) if len(episode_actions) > 0 else 0
        action_matches.append(match_rate)
        
        # Calculate return (sum of rewards)
        episode_return = sum(episode_rewards)
        test_returns.append(episode_return)
    
    # Store metrics
    metrics["test_returns_mean"] = float(np.mean(test_returns))
    metrics["test_returns_std"] = float(np.std(test_returns))
    metrics["action_match_rate_mean"] = float(np.mean(action_matches))
    metrics["action_match_rate_min"] = float(np.min(action_matches))
    metrics["action_match_rate_max"] = float(np.max(action_matches))
    
    print(f"Test metrics:")
    print(f"  - Average return: {metrics['test_returns_mean']:.4f} ± {metrics['test_returns_std']:.4f}")
    print(f"  - Action match rate: {metrics['action_match_rate_mean']:.4f} (min: {metrics['action_match_rate_min']:.4f}, max: {metrics['action_match_rate_max']:.4f})")

    # Create action match rate distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(action_matches, bins=20, alpha=0.7)
    plt.title('Action Match Rate Distribution')
    plt.xlabel('Match Rate')
    plt.ylabel('Count')
    plt.grid(True)
    dist_path = os.path.join(str(log_dir), f'action_match_distribution.png')
    plt.savefig(dist_path)
    print(f"📊 Action match distribution saved to {dist_path}")
    plt.close()

except Exception as e:
    print(f"⚠️ Warning: Could not evaluate: {e}")
    import traceback
    traceback.print_exc()

# Add training configuration to metrics
metrics["alpha"] = args.alpha
metrics["epochs"] = args.epochs
metrics["batch_size"] = args.batch
metrics["learning_rate"] = args.lr
metrics["gamma"] = args.gamma
metrics["n_features"] = len(state_cols)
metrics["n_episodes"] = n_episodes
metrics["n_transitions"] = n_transitions
metrics["early_stopped"] = early_stop
metrics["completed_epochs"] = epoch + 1 if 'epoch' in locals() else 0
metrics["nan_count_total"] = sum(training_metrics['nan_counts']) if 'training_metrics' in locals() and 'nan_counts' in training_metrics else 0

# Save metrics
yaml.safe_dump(metrics, open(metric_path, "w"))
print(f"✅ Test scores saved to {metric_path}")

# ---------- SAVE MODEL ----------
try:
    # Save model using the save_model method
    cql.save_model(model_path)
    print(f"✅ Model saved to {model_path}")
except Exception as e:
    print(f"⚠️ Error saving model: {e}")
    import traceback
    traceback.print_exc()

print("🎉 Done! You can now use the trained CQL model for inference.")
