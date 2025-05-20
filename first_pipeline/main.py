#!/usr/bin/env python3
# -----------------------------------------------------------
# main.py  --  Main script for HFNC CQL training
# -----------------------------------------------------------
#
# This script orchestrates the entire HFNC CQL training pipeline by:
# • Loading data from hfnc_episodes.parquet
# • Creating and splitting MDPDataset objects for reinforcement learning
# • Configuring and instantiating the DiscreteCQL algorithm
# • Managing the training process with robust error handling
# • Evaluating the trained model and analyzing predictions
# • Saving the model, scaler, and metrics for future use
#
# The pipeline is designed to:
# 1. Be robust to errors and data quality issues
# 2. Provide detailed diagnostics for NaN values and training instabilities
# 3. Produce comprehensive evaluation metrics and visualizations
#
# CLI Usage:
#   python main.py --alpha 1.0 --epochs 200 --gpu 0
# -----------------------------------------------------------

# Standard library imports
import os
import sys
from pathlib import Path

# Custom module imports from the reorganized pipeline
import config      # Configuration and argument parsing
import data_loader # Data loading and preprocessing
import dataset     # MDP dataset creation and splitting
import model       # CQL model configuration
import trainer     # Training orchestration
import evaluator   # Model evaluation
import utils       # Utility functions for debugging and visualization

def main():
    args = config.parse_args()
    paths = config.setup_paths(args)
    device = "cpu"
    if args.gpu >= 0:
        try:
            import torch
            if torch.cuda.is_available():
                device = f"cuda:{args.gpu}"
                print(f"Using CUDA device: {device}")
            else:
                print("CUDA not available, falling back to CPU")
        except ImportError:
            print("PyTorch CUDA support not available, using CPU")
    print("\n=== Loading and preprocessing data ===")
    df = data_loader.load_data(args.data)
    # cfg = config.load_config(args.cfg)
    
    data_dict = data_loader.preprocess_data(df)
    print("\n=== Creating dataset ===")
    try:
        mdp_dataset_full = dataset.create_mdp_dataset( # Renamed for clarity
            data_dict["states"],    # State features (observations)
            data_dict["actions"],   # Actions taken by clinicians
            data_dict["rewards"],   # Rewards (clinical outcomes)
            data_dict["dones"]      # Episode termination flags
        )
        
        utils.debug_nan_values(data_dict["states"], "states")
        utils.debug_nan_values(data_dict["rewards"], "rewards") 
        utils.debug_inf_values(data_dict["states"], "states")
        utils.debug_inf_values(data_dict["rewards"], "rewards")
        
        train_eps, val_eps, test_eps = dataset.split_dataset(mdp_dataset_full) # Split the full dataset

        if not train_eps:
            print("\n❌ Error: No training episodes after split. Exiting.")
            sys.exit(1)
        # train_dataset = dataset.MDPDataset(episodes=train_eps) <- #can't do this gives an error

    except Exception as e:
        print(f"\n❌ Error creating dataset: {e}")
        sys.exit(1)
    print("\n=== Creating model ===")
    scaler = model.create_scaler()
    cql_config = model.create_cql_config(
        batch_size=args.batch,    # Batch size for training updates
        learning_rate=args.lr,    # Step size for optimizer
        gamma=args.gamma,         # Discount factor for future rewards
        alpha=args.alpha,         # CQL conservatism parameter
        scaler=scaler             # Observation normalizer
    )
    
    try:
        cql = model.create_cql_model(
            config=cql_config,
            device=device,         # CPU or specific GPU
            enable_ddp=False       # No distributed training
        )
    except Exception as e:
        print(f"\n❌ Error creating model: {e}")
        sys.exit(1)
    
    # Construct the correct base path for d3rlpy logs
    # d3rlpy saves logs to d3rlpy_logs/<experiment_name>/
    # In this setup, experiment_name passed to model.fit() is args.logdir
    actual_d3rlpy_log_dir = Path("d3rlpy_logs") / args.logdir

    print("\n=== Training model ===")
    result, errors = trainer.train_model(
        model=cql,                  # CQL model to train
        train_episodes=train_eps,      # Pass the training-only dataset
        n_epochs=args.epochs,       # Number of training epochs
        experiment_name=args.logdir # Log directory name
    )

    if errors:
        print("\n⚠️ Training encountered errors, checking logs for diagnosis...")
        # Pass the corrected path to check_training_logs
        trainer.check_training_logs(actual_d3rlpy_log_dir)

    print("\n=== Evaluating model ===")
    metrics = evaluator.evaluate_model(cql, test_eps) 
    metrics = evaluator.add_training_params_to_metrics(metrics, args)
    print("\n=== Analyzing predictions ===")
    evaluator.analyze_predictions(cql, test_eps, top_n=3)

    print("\\n=== Analyzing training logs (from d3rlpy output) ===")
    
    # --- Find the latest d3rlpy log directory for the current run pattern ---
    base_d3rlpy_runs_dir = Path("d3rlpy_logs") / "runs"
    latest_log_dir = None
    
    if base_d3rlpy_runs_dir.exists() and base_d3rlpy_runs_dir.is_dir():
        # Assuming args.logdir is like "runs/cql", we get "cql"
        run_prefix = Path(args.logdir).name # e.g., "cql"
        
        # Find all subdirectories in base_d3rlpy_runs_dir that start with run_prefix + "_"
        # e.g., cql_20250519162207
        potential_dirs = sorted([
            d for d in base_d3rlpy_runs_dir.iterdir() 
            if d.is_dir() and d.name.startswith(f"{run_prefix}_")
        ])
        
        if potential_dirs:
            latest_log_dir = potential_dirs[-1] # Get the last one (latest timestamp)
            print(f"ℹ️  Found latest d3rlpy log directory for plotting: {latest_log_dir}")
        else:
            print(f"⚠️  Warning: No timestamped log directories found matching prefix '{run_prefix}_' in {base_d3rlpy_runs_dir}")
    else:
        print(f"⚠️  Warning: Base d3rlpy runs directory not found at {base_d3rlpy_runs_dir}")

    # Ensure the directory exists before trying to analyze logs
    if latest_log_dir and latest_log_dir.exists() and latest_log_dir.is_dir():
        utils.check_gradient_values(latest_log_dir)
        utils.plot_training_curves(latest_log_dir)
    else:
        print(f"⚠️  Warning: d3rlpy log directory for plotting not found or not valid, skipping plot generation.")

    print("\\n=== Saving results ===")
    config.save_metrics(metrics, paths["metric_path"])
    model.save_model(cql, paths["model_path"])
    print("\n=== Training complete ===")
    
if __name__ == "__main__":
    main()