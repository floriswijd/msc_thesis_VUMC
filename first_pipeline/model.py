#!/usr/bin/env python3

import torch
from d3rlpy.preprocessing import StandardObservationScaler
from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig
from d3rlpy.algos import QLearningAlgoBase
from pathlib import Path #
import d3rlpy


def create_scaler():
    return StandardObservationScaler()


def create_cql_config(batch_size, learning_rate, gamma, alpha, scaler=None):
    config = DiscreteCQLConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        alpha=alpha,
        n_critics=2,
    )
    if scaler is not None:
        config.observation_scaler = scaler
    else:
        print("No scaler provided, using default configuration.")
    print(
        f"Created CQL config with alpha={config.alpha}, batch_size={config.batch_size}"
    )
    return config


def create_cql_model(config, device="cpu", enable_ddp=False):
    print(f"Using device: {device}")
    try:
        cql = DiscreteCQL(
            config=config, device=device, enable_ddp=enable_ddp
        )
        print(f"CQL instance created with device: {device}")
        return cql
    except Exception as e:
        print(f"Error creating CQL model: {e}")
        raise


def save_model(model, model_path):
    try:
        try:
            model.save_model(model_path)
        except AttributeError:
            model.save(model_path)
        print(f"💾  Model saved → {model_path}")
        return True
    except Exception as e:
        print(f"❌ Could not save model: {e}")
        return False
    
def load_model(model_path: Path, algo_class: type[QLearningAlgoBase], device: str = "cpu"):
    """
    Loads a d3rlpy model from a saved .d3 file by finding its
    corresponding params.json in the same directory.
    """
    try:
        # The configuration for the run is stored in 'params.json'
        # in the same directory as the model weights file.
        json_path = model_path.parent / "params.json"

        if not model_path.exists():
            print(f"❌ Error: Model weights file not found at {model_path}")
            return None
        # if not json_path.exists():
        #     print(f"❌ Error: JSON config file not found at {json_path}")
        #     return None

        # # Step 1: Build the empty model shell from the blueprint (params.json)
        # loaded_model = algo_class.from_json(json_path, device=device)

        # Step 2: Fill the shell with the learned weights (e.g., model_100000.d3)
        loaded_model = d3rlpy.load_learnable(model_path)

        print(f"✅ Model successfully loaded from → {model_path}")
        return loaded_model
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        import traceback
        traceback.print_exc()
        return None