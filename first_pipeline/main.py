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
    """
    Main entry point for HFNC CQL training pipeline.
    
    This function orchestrates the entire training workflow:
    1. Parse command-line arguments
    2. Set up directories and device configuration
    3. Load and preprocess HFNC episode data
    4. Create and split MDP dataset
    5. Configure and instantiate the CQL model
    6. Train the model with robust error handling
    7. Evaluate the model and analyze predictions
    8. Save model and metrics for future use
    
    The function includes extensive error handling and diagnostics
    to help identify and resolve issues in the pipeline.
    """
    # ---------------------------------------------------------------------------
    # 1. Parse command-line arguments and set up paths
    # ---------------------------------------------------------------------------
    args = config.parse_args()  # Get command-line arguments
    paths = config.setup_paths(args)  # Create necessary directories
    
    # ---------------------------------------------------------------------------
    # 2. Set up computing device (CPU/GPU)
    # ---------------------------------------------------------------------------
    # Determine whether to use CPU or GPU based on arguments
    # Using CPU (-1) can be more stable for debugging purposes
    device = "cpu"  # Default to CPU for stability
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
    
    # ---------------------------------------------------------------------------
    # 3. Load and preprocess data
    # ---------------------------------------------------------------------------
    print("\n=== Loading and preprocessing data ===")
    # Load parquet file containing HFNC episodes
    df = data_loader.load_data(args.data)
    # Load configuration file with hyperparameters
    cfg = config.load_config(args.cfg)
    # Preprocess data: extract states, actions, rewards, dones
    data_dict = data_loader.preprocess_data(df)
    
    # ---------------------------------------------------------------------------
    # 4. Create MDP dataset and split into train/val/test
    # ---------------------------------------------------------------------------
    print("\n=== Creating dataset ===")
    try:
        # Create MDPDataset from preprocessed arrays
        mdp_dataset = dataset.create_mdp_dataset(
            data_dict["states"],    # State features (observations)
            data_dict["actions"],   # Actions taken by clinicians
            data_dict["rewards"],   # Rewards (clinical outcomes)
            data_dict["dones"]      # Episode termination flags
        )
        
        # Check for data quality issues that might cause NaNs
        utils.debug_nan_values(data_dict["states"], "states")
        utils.debug_nan_values(data_dict["rewards"], "rewards") 
        utils.debug_inf_values(data_dict["states"], "states")
        utils.debug_inf_values(data_dict["rewards"], "rewards")
        
        # Split dataset into train, validation and test sets
        # By default: 70% train, 15% val, 15% test
        train_eps, val_eps, test_eps = dataset.split_dataset(mdp_dataset)
    except Exception as e:
        print(f"\n❌ Error creating dataset: {e}")
        sys.exit(1)
    
    # ---------------------------------------------------------------------------
    # 5. Create scaler and model
    # ---------------------------------------------------------------------------
    print("\n=== Creating model ===")
    # Create observation scaler for state normalization
    scaler = model.create_scaler()
    # Configure CQL algorithm with hyperparameters
    cql_config = model.create_cql_config(
        batch_size=args.batch,    # Batch size for training updates
        learning_rate=args.lr,    # Step size for optimizer
        gamma=args.gamma,         # Discount factor for future rewards
        alpha=args.alpha,         # CQL conservatism parameter
        scaler=scaler             # Observation normalizer
    )
    
    try:
        # Instantiate CQL model with configuration
        cql = model.create_cql_model(
            config=cql_config,
            device=device,         # CPU or specific GPU
            enable_ddp=False       # No distributed training
        )
    except Exception as e:
        print(f"\n❌ Error creating model: {e}")
        sys.exit(1)
    
    # ---------------------------------------------------------------------------
    # 6. Train the model
    # ---------------------------------------------------------------------------
    print("\n=== Training model ===")
    # Train model with robust error handling and fallback strategies
    result, errors = trainer.train_model(
        model=cql,                  # CQL model to train
        dataset=mdp_dataset,        # Full dataset for training
        n_epochs=args.epochs,       # Number of training epochs
        experiment_name=args.logdir # Log directory name
    )
    
    # If training encountered errors, analyze logs to diagnose issues
    if errors:
        print("\n⚠️ Training encountered errors, checking logs for diagnosis...")
        trainer.check_training_logs(args.logdir)
    
    # ---------------------------------------------------------------------------
    # 7. Evaluate the model
    # ---------------------------------------------------------------------------
    print("\n=== Evaluating model ===")
    # Calculate evaluation metrics on test episodes
    metrics = evaluator.evaluate_model(cql, test_eps)
    # Add training parameters to metrics for complete reporting
    metrics = evaluator.add_training_params_to_metrics(metrics, args)
    
    # Perform more detailed analysis of model predictions
    print("\n=== Analyzing predictions ===")
    evaluator.analyze_predictions(cql, test_eps, top_n=3)
    
    # ---------------------------------------------------------------------------
    # 8. Analyze training logs and visualize results
    # ---------------------------------------------------------------------------
    print("\n=== Analyzing training logs ===")
    # Check gradient files for issues like NaNs or extreme values
    utils.check_gradient_values(args.logdir)
    # Create plots from training curves
    utils.plot_training_curves(args.logdir)
    
    # ---------------------------------------------------------------------------
    # 9. Save results
    # ---------------------------------------------------------------------------
    print("\n=== Saving results ===")
    # Save evaluation metrics to YAML file
    config.save_metrics(metrics, paths["metric_path"])
    # Save trained model for future use or deployment
    model.save_model(cql, paths["model_path"])
    
    print("\n=== Training complete ===")
    
if __name__ == "__main__":
    main()