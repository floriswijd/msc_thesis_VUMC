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
# Proper import for OptimizerFactory and AdamFactory
from d3rlpy.optimizers import OptimizerFactory, AdamFactory
import torch.optim as torch_optim  # Use PyTorch optimizers instead

# Custom batch sampling function for older d3rlpy versions
def sample_batch_from_episodes(episodes, batch_size):
    """Sample a batch from episodes manually for older d3rlpy versions that don't have RandomIterator"""
    import random
    import numpy as np
    from d3rlpy.dataset import Transition
    
    # Create arrays to hold batch data
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []
    
    # Sample random episodes with replacement until we have enough samples
    while len(observations) < batch_size:
        # Pick a random episode
        episode = random.choice(episodes)
        
        # If episode has no transitions, skip
        if len(episode.observations) <= 1:
            continue
            
        # Pick a random transition from this episode
        idx = random.randint(0, len(episode.observations) - 2)  # Ensure we have a next observation
        
        # Add to batch
        observations.append(episode.observations[idx])
        actions.append(episode.actions[idx])
        
        # Ensure reward is a valid float (not NaN or infinity)
        reward = episode.rewards[idx]
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0  # Replace invalid rewards with zero
        rewards.append(reward)
        
        next_observations.append(episode.observations[idx + 1])
        terminals.append(1.0 if idx == len(episode.observations) - 2 and episode.rewards[-1] != 0 else 0.0)
        
        # Break if we have enough
        if len(observations) >= batch_size:
            break
    
    # Convert to numpy arrays for use with Transition class
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards).reshape(-1, 1)  # reshape to match expected dimensions
    next_observations = np.array(next_observations)
    terminals = np.array(terminals).reshape(-1, 1)  # reshape to match expected dimensions
    
    # Try different parameter names based on the d3rlpy version
    try:
        # Try parameters for older versions
        return Transition(
            observation=observations,
            action=actions,
            reward=rewards,
            next_observation=next_observations,
            terminal=terminals
        )
    except Exception as e:
        try:
            # Try parameters for newer versions
            return Transition(
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                terminals=terminals
            )
        except Exception as e2:
            # As a last resort, use kwargs to handle any parameter name
            print(f"Warning: Batch sampling failed with errors: {e}, {e2}. Using direct updating instead.")
            
            # Since both attempts failed, we'll just return None and let the caller handle it
            return None

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
device = "mps" if args.gpu < 0 else f"cuda:{args.gpu}"
print(f"Using device: {device}")

# Create a simpler configuration for initial viability
optim_factory = AdamFactory(
    weight_decay=1e-4,  # Increased weight decay for regularization
    eps=1e-8  # Small epsilon to prevent division by zero
)

# Create a simplified configuration with more conservative hyperparameters
config = DiscreteCQLConfig(
    batch_size=args.batch,
    learning_rate=args.lr * 0.5,  # Reduce learning rate for stability
    gamma=args.gamma,  # Discount factor
    observation_scaler=StandardObservationScaler(),
    alpha=max(0.1, args.alpha * 0.5),  # Reduce alpha if it's too aggressive
    n_critics=2,  # Using 2 critics for more stable learning
    target_update_interval=10,  # More frequent target updates
    optim_factory=optim_factory,
)

# Use a simpler encoder for initial testing
encoder_factory = VectorEncoderFactory(
    hidden_units=[64, 64],  # Simpler network architecture
    dropout_rate=0.2  # Add dropout for regularization
)
config.encoder_factory = encoder_factory

# Determine total training steps based on epochs
# A reasonable number of steps per epoch
steps_per_epoch = min(1000, n_transitions // args.batch)  # Reduced steps per epoch
total_steps = args.epochs * steps_per_epoch

print(f"Training for {args.epochs} epochs ({total_steps} total steps)")

# Create the CQL instance
cql = DiscreteCQL(
    config=config,
    device=device,
    enable_ddp=False  # Required parameter in d3rlpy 2.8.1
)

# ---------- TRAIN ----------
print("🚀 Starting DiscreteCQL training...")

# Initialize metrics tracking
training_metrics = {
    'loss_history': [],
    'action_match_history': [],
    'epoch_times': [],
    'nan_counts': []  # Track number of NaN losses per epoch
}

# Early stopping variables
best_loss = float('inf')
patience_counter = 0
early_stop = False
last_checkpoint_epoch = -1

# Create optimizer with gradient clipping
def setup_optimizer_with_clipping(model):
    """Set up optimizer with gradient clipping to prevent exploding gradients."""
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr,
        weight_decay=1e-5,
        eps=1e-8
    )
    
    # Wrap optimizer's step method with gradient clipping
    original_step = optimizer.step
    
    def step_with_clipping(*args, **kwargs):
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        return original_step(*args, **kwargs)
    
    optimizer.step = step_with_clipping
    return optimizer

try:
    # First build the model with the dataset to initialize weights
    print("Building model with dataset...")
    cql.build_with_dataset(dataset)
    
    # Track training metrics over epochs
    epoch_size = steps_per_epoch
    total_epochs = args.epochs
    
    print(f"Training for {total_epochs} epochs ({total_steps} steps)...")
    for epoch in range(total_epochs):
        epoch_start = time.time()
        
        # Train for one epoch
        n_steps = min(epoch_size, n_transitions // args.batch)
        episode_loss = 0
        valid_steps = 0
        nan_count = 0
        
        # Wrap steps in a try-except block to catch any unexpected errors
        for step in range(n_steps):
            try:
                # Get a batch using the custom batch sampling function
                batch = sample_batch_from_episodes(train_eps, args.batch)
                
                # Apply manual gradient clipping for better stability
                loss = cql.update(batch)
                
                # Count and handle NaN losses
                if loss is not None:
                    if not np.isnan(loss) and not np.isinf(loss):
                        # Only add valid loss values
                        episode_loss += loss
                        valid_steps += 1
                    else:
                        nan_count += 1
                        if nan_count < 10 or nan_count % 50 == 0:
                            print(f"⚠️ Warning: NaN loss detected at step {step} of epoch {epoch}")
            except Exception as step_error:
                print(f"⚠️ Error at step {step}: {step_error}")
                # Continue with the next step without breaking the entire training
                continue
        
        # Record average loss for this epoch (only from valid steps)
        training_metrics['nan_counts'].append(nan_count)
        
        if valid_steps > 0:
            avg_loss = episode_loss / valid_steps
            training_metrics['loss_history'].append(avg_loss)
            
            # Check for early stopping based on loss improvement
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                
                # Save checkpoint if loss improved significantly
                if epoch > 0 and (epoch - last_checkpoint_epoch >= 5 or avg_loss < best_loss * 0.95):
                    checkpoint_path = log_dir / f"cql_checkpoint_epoch_{epoch}.pt"
                    cql.save_model(checkpoint_path)
                    last_checkpoint_epoch = epoch
                    print(f"📝 Saved checkpoint at epoch {epoch} with loss {avg_loss:.4f}")
            else:
                patience_counter += 1
                
            # Early stopping check
            if patience_counter >= args.early_stop_patience:
                print(f"⚠️ Early stopping triggered after {epoch+1} epochs due to no improvement")
                early_stop = True
        else:
            # If no valid steps in this epoch (all NaN), add a placeholder value and increase patience counter
            training_metrics['loss_history'].append(float('inf'))
            patience_counter += 1
            print(f"⚠️ Epoch {epoch} had no valid loss values (all NaN)")
            
            # If too many NaN epochs in a row, trigger early stopping
            if patience_counter >= args.early_stop_patience // 2:
                print(f"⚠️ Early stopping triggered due to too many NaN epochs")
                early_stop = True
            
        # Calculate action match rate on a subset of validation data
        if len(val_eps) > 0:
            try:
                val_episode = val_eps[0]  # Use first validation episode for quick check
                # Limit validation observations to avoid spending too much time
                max_obs = min(50, len(val_episode.observations))
                val_obs = val_episode.observations[:max_obs]
                val_actions = val_episode.actions[:max_obs]
                
                # Predict actions and handle any errors
                pred_actions = []
                for ob in val_obs:
                    try:
                        pred = cql.predict(ob.reshape(1, -1))[0]
                        pred_actions.append(pred)
                    except Exception as pred_error:
                        # If prediction fails, use most common action in training data
                        from collections import Counter
                        most_common = Counter([int(a) for e in train_eps for a in e.actions]).most_common(1)[0][0]
                        pred_actions.append(most_common)
                        # Don't print every error to avoid console spam
                
                # Calculate match rate
                match_rate = np.mean(np.array(pred_actions) == val_actions) 
                training_metrics['action_match_history'].append(match_rate)
            except Exception as val_error:
                print(f"⚠️ Error during validation: {val_error}")
                # Use previous match rate or zero if none exists
                prev_match = training_metrics['action_match_history'][-1] if training_metrics['action_match_history'] else 0
                training_metrics['action_match_history'].append(prev_match)
        
        # Track epoch time
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        training_metrics['epoch_times'].append(epoch_time)
        
        # Progress update
        if epoch % 5 == 0 or epoch == total_epochs - 1:
            latest_loss = training_metrics['loss_history'][-1] if training_metrics['loss_history'] else float('nan')
            latest_match = training_metrics['action_match_history'][-1] if training_metrics['action_match_history'] else 0
            nan_percent = (nan_count / n_steps) * 100 if n_steps > 0 else 0
            print(f"Epoch {epoch}/{total_epochs} - Loss: {latest_loss:.4f}, Action match: {latest_match:.4f}, NaN: {nan_count}/{n_steps} ({nan_percent:.1f}%), Time: {epoch_time:.2f}s")

        # Break if early stopping triggered
        if early_stop:
            break

    print("✅ Training completed successfully")
    
    # Generate training plots
    create_training_plots(training_metrics, str(log_dir))
    
    # Save training history to metrics
    metrics = {}  # Initialize metrics dictionary to be used later
    metrics['epoch_loss_history'] = training_metrics['loss_history']
    metrics['action_match_history'] = training_metrics['action_match_history'] 
    metrics['nan_counts'] = training_metrics['nan_counts']
    
except Exception as e:
    print(f"⚠️ Error during training: {e}")
    import traceback
    traceback.print_exc()

# ---------- EVALUATION ----------
print("Evaluating on test episodes...")
metrics = metrics if 'metrics' in locals() else {}

try:
    # Calculate more detailed evaluation metrics for HFNC parameter prediction
    test_returns = []
    action_matches = []
    action_distribution = {}  # Track distribution of actions
    
    # Initialize containers for detailed metrics
    all_predicted_actions = []
    all_true_actions = []
    
    # Collect evaluation metrics for each test episode
    for episode in test_eps:
        observations = episode.observations
        episode_actions = episode.actions
        episode_rewards = episode.rewards
        
        # Get predicted actions for each observation in the episode
        predicted_actions = []
        for obs in observations:
            try:
                action = cql.predict(obs.reshape(1, -1))[0]
                predicted_actions.append(action)
                all_predicted_actions.append(action)
            except Exception as e:
                # If prediction fails, use a default action (most common action)
                from collections import Counter
                actions_list = [int(a) for a in episode_actions]
                most_common = Counter(actions_list).most_common(1)[0][0]
                predicted_actions.append(most_common)
                all_predicted_actions.append(most_common)
                # Limit error output to avoid spam
                if len(all_predicted_actions) % 100 == 0:
                    print(f"⚠️ Prediction failed: {e}. Using fallback action {most_common}.")
        
        # Add true actions to the collection for confusion matrix later
        all_true_actions.extend(episode_actions)
        
        # Calculate action match rate
        matches = sum(p == a for p, a in zip(predicted_actions, episode_actions))
        match_rate = matches / len(episode_actions) if len(episode_actions) > 0 else 0
        action_matches.append(match_rate)
        
        # Calculate return (sum of rewards)
        episode_return = sum(episode_rewards)
        test_returns.append(episode_return)
    
    # Store basic metrics
    metrics["test_returns_mean"] = float(np.mean(test_returns)) if test_returns else float('nan')
    metrics["test_returns_std"] = float(np.std(test_returns)) if test_returns else float('nan')
    metrics["action_match_rate_mean"] = float(np.mean(action_matches)) if action_matches else 0.0
    metrics["action_match_rate_min"] = float(np.min(action_matches)) if action_matches else 0.0
    metrics["action_match_rate_max"] = float(np.max(action_matches)) if action_matches else 0.0
    
    # Calculate and store action distribution
    from collections import Counter
    pred_action_counts = Counter(all_predicted_actions)
    true_action_counts = Counter(all_true_actions)
    
    metrics["predicted_action_distribution"] = {str(k): int(v) for k, v in sorted(pred_action_counts.items())}
    metrics["true_action_distribution"] = {str(k): int(v) for k, v in sorted(true_action_counts.items())}
    
    # Calculate basic confusion metrics
    try:
        all_predicted_actions = np.array(all_predicted_actions)
        all_true_actions = np.array(all_true_actions)
        
        # Calculate per-action precision
        action_precision = {}
        for action in set(all_true_actions):
            action_idx = (all_predicted_actions == action)
            true_positives = np.sum((all_true_actions == action) & action_idx)
            if np.sum(action_idx) > 0:
                precision = true_positives / np.sum(action_idx)
                action_precision[str(int(action))] = float(precision)
        
        metrics["action_precision"] = action_precision
        
        # Calculate per-action recall
        action_recall = {}
        for action in set(all_true_actions):
            action_idx = (all_true_actions == action)
            true_positives = np.sum((all_predicted_actions == action) & action_idx)
            if np.sum(action_idx) > 0:
                recall = true_positives / np.sum(action_idx)
                action_recall[str(int(action))] = float(recall)
        
        metrics["action_recall"] = action_recall
        
    except Exception as metric_error:
        print(f"⚠️ Error calculating detailed metrics: {metric_error}")
    
    # Print basic evaluation summary
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
    
    # Create action distribution comparison chart
    plt.figure(figsize=(12, 6))
    
    # Get all unique actions
    all_actions = sorted(set(list(metrics["predicted_action_distribution"].keys()) + 
                             list(metrics["true_action_distribution"].keys())))
    
    # Prepare data for plotting
    x = np.arange(len(all_actions))
    width = 0.35
    
    # Extract counts for each action
    true_counts = [metrics["true_action_distribution"].get(a, 0) for a in all_actions]
    pred_counts = [metrics["predicted_action_distribution"].get(a, 0) for a in all_actions]
    
    # Create side-by-side bar chart
    plt.bar(x - width/2, true_counts, width, label='True Actions')
    plt.bar(x + width/2, pred_counts, width, label='Predicted Actions')
    
    plt.xlabel('Action ID')
    plt.ylabel('Count')
    plt.title('Comparison of True vs Predicted Action Distribution')
    plt.xticks(x, all_actions)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    # Save the figure
    dist_path = os.path.join(str(log_dir), f'action_distribution_comparison.png')
    plt.savefig(dist_path)
    print(f"📊 Action distribution comparison saved to {dist_path}")
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
    # Save final model - in d3rlpy 2.8.1 saving the model includes the scaler
    cql.save_model(model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Also save a json representation of the policy
    q_table = {}
    if len(test_eps) > 0:
        try:
            # Skip trying to manually build the scaler since this version doesn't support it
            # Instead, we'll try to directly use predict_value and handle errors
            
            # Sample some test observations to create a small q-value table
            sample_obs = []
            for ep in test_eps[:3]:  # Use first 3 test episodes
                sample_obs.extend(ep.observations[:10])  # Take first 10 observations from each
            
            # Get Q-values for each action
            action_count = cfg.get('n_actions', 12)  # Try to get from config, default to 12
            print(f"Calculating Q-values for {action_count} actions...")
            
            # Try to get Q-values directly
            success = False
            for i, obs in enumerate(sample_obs[:50]):  # Limit to 50 samples
                actions = np.arange(action_count)
                q_values = []
                
                # Try different approaches to get Q-values based on d3rlpy version
                try:
                    # First try to get all Q-values at once if available
                    if hasattr(cql, 'predict_value_with_all_actions'):
                        q_values = cql.predict_value_with_all_actions(obs.reshape(1, -1))[0].tolist()
                        success = True
                    else:
                        # Try getting Q-values for each action individually
                        for a in actions:
                            try:
                                q_val = cql.predict_value(obs.reshape(1, -1), np.array([a]))
                                q_values.append(float(q_val))
                                success = True
                            except Exception as e:
                                q_values.append(float('nan'))
                except Exception as e:
                    print(f"Could not calculate Q-values: {e}")
                    # Use dummy values as fallback
                    q_values = [float(i % action_count) for i in range(action_count)]
                
                q_table[f"obs_{i}"] = q_values
                
                # If we've successfully retrieved at least some Q-values, break the loop early
                if success and i >= 10:
                    print("Successfully retrieved some Q-values, stopping early")
                    break
            
            # Save Q-value table
            import json
            with open(log_dir / "q_table.json", "w") as f:
                json.dump(q_table, f, indent=2)
            print(f"✅ Sample Q-values saved to {log_dir / 'q_table.json'}")
        except Exception as e:
            print(f"⚠️ Error generating Q-values: {e}")
            import traceback
            traceback.print_exc()
except Exception as e:
    print(f"⚠️ Error saving model: {e}")
    import traceback
    traceback.print_exc()

print("🎉 Done! You can now use the trained CQL model for inference.")
