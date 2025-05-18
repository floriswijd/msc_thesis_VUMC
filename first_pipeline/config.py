#!/usr/bin/env python3
# -----------------------------------------------------------
# config.py  --  Configuration for HFNC CQL training
# 
# This module handles all configuration aspects of the HFNC reinforcement learning pipeline:
# - Command-line argument parsing
# - Path setup for model and metrics saving
# - Configuration loading from YAML files
# - Metrics saving
# -----------------------------------------------------------
import argparse
from pathlib import Path
import yaml

def parse_args():
    """
    Parse command line arguments for the HFNC CQL training pipeline.
    
    Returns:
        argparse.Namespace: Parsed arguments with the following attributes:
            - data: Path to the parquet data file containing HFNC episodes
            - cfg: Path to configuration YAML file with model settings
            - alpha: CQL conservatism weight (higher values enforce more conservative behavior)
            - epochs: Number of training epochs (controls training duration)
            - batch: Batch size for training (impacts memory usage and training stability)
            - lr: Learning rate (controls optimization step size)
            - gamma: Discount factor (determines importance of future vs immediate rewards)
            - gpu: GPU device ID (-1 for CPU, >=0 selects a specific GPU)
            - logdir: Directory for saving training logs and model checkpoints
    """
    parser = argparse.ArgumentParser(description="HFNC CQL Training Configuration")
    
    # Data and configuration paths
    parser.add_argument("--data", default="/Users/floppie/Documents/Msc Scriptie/HFNC codebase/first_pipeline/data/hfnc_episodes.parquet",
                        help="Path to the parquet data file")
    parser.add_argument("--cfg", default="/Users/floppie/Documents/Msc Scriptie/config.yaml",
                        help="Path to configuration YAML file")
    
    # Training hyperparameters
    parser.add_argument("--alpha", type=float, default=1.0, 
                        help="CQL conservatisme-gewicht (higher values enforce more conservative behavior)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs (controls training duration)")
    parser.add_argument("--batch", type=int, default=256,
                        help="Batch size for training (impacts memory usage and training stability)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (controls optimization step size)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (determines importance of future vs immediate rewards)")
    
    # Device configuration
    parser.add_argument("--gpu", type=int, default=0,
                        help="-1 = CPU, >=0 = GPU device ID")
    
    # Logging and output
    parser.add_argument("--logdir", default="runs/cql",
                        help="Directory for logging and saving models")
    
    return parser.parse_args()

def setup_paths(args):
    """
    Set up necessary paths for logging and saving models.
    
    Creates the logging directory if it doesn't exist and defines paths for model
    and metrics files based on the provided arguments.
    
    Args:
        args (argparse.Namespace): Parsed command line arguments
        
    Returns:
        dict: Dictionary containing paths for model and metrics files:
            - model_path: Path for saving the trained CQL model
            - metric_path: Path for saving the evaluation metrics
    """
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    paths = {
        "model_path": Path(args.logdir) / "cql_hfnc.pt",
        "metric_path": Path(args.logdir) / "metrics.yaml"
    }
    return paths

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    The YAML file typically contains model configuration parameters, 
    dataset settings, and other hyperparameters.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary loaded from the YAML file
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_metrics(metrics, metric_path):
    """
    Save metrics to YAML file and print a summary.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics
        metric_path (Path): Path where metrics will be saved
        
    Side effects:
        - Writes metrics to the specified YAML file
        - Prints metrics to the console
    """
    with open(metric_path, 'w') as f:
        yaml.safe_dump(metrics, f)
        
    print("✅  Test-scores:", metrics)