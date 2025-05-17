#!/usr/bin/env python3
# -----------------------------------------------------------
# main.py  --  Main script for HFNC CQL training
# -----------------------------------------------------------
#
# • Laadt data/processed/hfnc_episodes.parquet
# • Maakt MDPDataset met train/val/test splits
# • Instantieert DiscreteCQL (d3rlpy) met hyperparams uit YAML
# • Logt evaluatie tijdens training
# • Slaat model + scaler + metrics op
#
# CLI:
#   python main.py --alpha 1.0 --epochs 200 --gpu 0
# -----------------------------------------------------------

# Import modules
import os
import sys
from pathlib import Path

# Import custom modules
import config
import data_loader
import dataset
import model
import trainer
import evaluator
import utils

def main():
    """Main entry point for HFNC CQL training"""
    # Parse command line arguments
    args = config.parse_args()
    paths = config.setup_paths(args)
    
    # Determine device (CPU vs GPU)
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
    
    # Load and preprocess data
    print("\n=== Loading and preprocessing data ===")
    df = data_loader.load_data(args.data)
    cfg = config.load_config(args.cfg)
    data_dict = data_loader.preprocess_data(df)
    
    # Create MDP dataset and split into train/val/test
    print("\n=== Creating dataset ===")
    try:
        mdp_dataset = dataset.create_mdp_dataset(
            data_dict["states"],
            data_dict["actions"],
            data_dict["rewards"],
            data_dict["dones"]
        )
        
        # Check for NaN values in the dataset
        utils.debug_nan_values(data_dict["states"], "states")
        utils.debug_nan_values(data_dict["rewards"], "rewards")
        utils.debug_inf_values(data_dict["states"], "states")
        utils.debug_inf_values(data_dict["rewards"], "rewards")
        
        train_eps, val_eps, test_eps = dataset.split_dataset(mdp_dataset)
    except Exception as e:
        print(f"\n❌ Error creating dataset: {e}")
        sys.exit(1)
    
    # Create scaler and model
    print("\n=== Creating model ===")
    scaler = model.create_scaler()
    cql_config = model.create_cql_config(
        batch_size=args.batch,
        learning_rate=args.lr,
        gamma=args.gamma,
        alpha=args.alpha,
        scaler=scaler
    )
    
    try:
        cql = model.create_cql_model(
            config=cql_config,
            device=device,
            enable_ddp=False
        )
    except Exception as e:
        print(f"\n❌ Error creating model: {e}")
        sys.exit(1)
    
    # Train the model
    print("\n=== Training model ===")
    result, errors = trainer.train_model(
        model=cql,
        dataset=mdp_dataset,
        n_epochs=args.epochs,
        experiment_name=args.logdir
    )
    
    if errors:
        print("\n⚠️ Training encountered errors, checking logs for diagnosis...")
        trainer.check_training_logs(args.logdir)
    
    # Evaluate the model
    print("\n=== Evaluating model ===")
    metrics = evaluator.evaluate_model(cql, test_eps)
    metrics = evaluator.add_training_params_to_metrics(metrics, args)
    
    # Perform more detailed analysis on predictions
    print("\n=== Analyzing predictions ===")
    evaluator.analyze_predictions(cql, test_eps, top_n=3)
    
    # Analyze training logs and plot training curves
    print("\n=== Analyzing training logs ===")
    utils.check_gradient_values(args.logdir)
    utils.plot_training_curves(args.logdir)
    
    # Save metrics and model
    print("\n=== Saving results ===")
    config.save_metrics(metrics, paths["metric_path"])
    model.save_model(cql, paths["model_path"])
    
    print("\n=== Training complete ===")
    
if __name__ == "__main__":
    main()