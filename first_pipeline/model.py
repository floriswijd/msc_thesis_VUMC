#!/usr/bin/env python3

import torch
from d3rlpy.preprocessing import StandardObservationScaler
from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig


def create_scaler():
    return StandardObservationScaler()


def create_cql_config(batch_size, learning_rate, gamma, alpha, scaler=None):
    config = DiscreteCQLConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        alpha=alpha,
    )
    if scaler is not None:
        config.observation_scaler = scaler
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