#!/usr/bin/env python3
# -----------------------------------------------------------
# config.py  --  Configuration for HFNC CQL training
# -----------------------------------------------------------
import argparse
from pathlib import Path
import yaml

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="HFNC CQL Training Configuration")
    
    # Data and configuration paths
    parser.add_argument("--data", default="/Users/floppie/Documents/Msc Scriptie/hfnc_episodes.parquet",
                        help="Path to the parquet data file")
    parser.add_argument("--cfg", default="/Users/floppie/Documents/Msc Scriptie/config.yaml",
                        help="Path to configuration YAML file")
    
    # Training hyperparameters
    parser.add_argument("--alpha", type=float, default=1.0, 
                        help="CQL conservatisme-gewicht")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    
    # Device configuration
    parser.add_argument("--gpu", type=int, default=0,
                        help="-1 = CPU, >=0 = GPU device ID")
    
    # Logging and output
    parser.add_argument("--logdir", default="runs/cql",
                        help="Directory for logging and saving models")
    
    return parser.parse_args()

def setup_paths(args):
    """Set up necessary paths for logging and saving models"""
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    paths = {
        "model_path": Path(args.logdir) / "cql_hfnc.pt",
        "metric_path": Path(args.logdir) / "metrics.yaml"
    }
    return paths

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_metrics(metrics, metric_path):
    """Save metrics to YAML file"""
    with open(metric_path, 'w') as f:
        yaml.safe_dump(metrics, f)
        
    print("✅  Test-scores:", metrics)