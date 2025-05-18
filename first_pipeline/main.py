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
    cfg = config.load_config(args.cfg)
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
        train_dataset = dataset.MDPDataset(episodes=train_eps)

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
    print("\n=== Training model ===")
    result, errors = trainer.train_model(
        model=cql,                  # CQL model to train
        dataset=train_dataset,      # Pass the training-only dataset
        n_epochs=args.epochs,       # Number of training epochs
        experiment_name=args.logdir # Log directory name
    )
    
    if errors:
        print("\n⚠️ Training encountered errors, checking logs for diagnosis...")
        trainer.check_training_logs(args.logdir)
    print("\n=== Evaluating model ===")
    metrics = evaluator.evaluate_model(cql, test_eps) 
    metrics = evaluator.add_training_params_to_metrics(metrics, args)
    print("\n=== Analyzing predictions ===")
    evaluator.analyze_predictions(cql, test_eps, top_n=3)
    print("\n=== Analyzing training logs ===")
    utils.check_gradient_values(args.logdir)
    utils.plot_training_curves(args.logdir)
    print("\n=== Saving results ===")
    config.save_metrics(metrics, paths["metric_path"])
    model.save_model(cql, paths["model_path"])
    print("\n=== Training complete ===")
    
if __name__ == "__main__":
    main()