#!/usr/bin/env python3
# -----------------------------------------------------------
# model.py  --  CQL model configuration and setup for HFNC
# 
# This module handles the creation and configuration of the CQL model:
# - Creating observation scalers for state normalization
# - Configuring CQL hyperparameters
# - Instantiating the CQL model with proper device settings
# - Saving trained models
#
# Conservative Q-Learning (CQL) is specifically designed for offline RL,
# making it suitable for learning from fixed clinical datasets.
# -----------------------------------------------------------
import torch
from d3rlpy.preprocessing import StandardObservationScaler
from d3rlpy.algos import DiscreteCQL, DiscreteCQLConfig

def create_scaler():
    """
    Create a standard observation scaler for normalizing input states.
    
    The StandardObservationScaler normalizes each feature dimension 
    to have zero mean and unit variance, which is crucial for:
    1. Improving neural network training stability
    2. Ensuring features with different scales contribute equally
    3. Accelerating gradient descent convergence
    
    Returns:
        StandardObservationScaler: Scaler object that will automatically
                                  compute stats from the dataset during training
                                  
    Note:
        The scaler is trained automatically when the model is fitted, using
        statistics from the training dataset.
    """
    return StandardObservationScaler()

def create_cql_config(batch_size, learning_rate, gamma, alpha, scaler=None):
    """
    Create a CQL configuration with the specified parameters.
    
    Conservative Q-Learning (CQL) is an offline RL algorithm designed to address
    overestimation bias in Q-learning when training on fixed datasets. It adds
    a regularization term controlled by alpha that penalizes values of actions
    not seen in the dataset.
    
    Args:
        batch_size (int): Number of transitions used in each gradient step
                         Larger values provide more stable gradients but use more memory
        learning_rate (float): Step size for optimizer updates
                              Controls how quickly the model adapts to the training data
        gamma (float): Discount factor for future rewards (0.0-1.0)
                      Lower values focus on immediate rewards, higher values on long-term
        alpha (float): CQL regularization parameter that controls conservatism
                      Higher values enforce more conservative behavior estimation
        scaler (StandardObservationScaler, optional): State normalizer
                                                     Default is None
    
    Returns:
        DiscreteCQLConfig: Configuration object for the CQL algorithm
        
    Note:
        The alpha parameter is particularly important in offline RL as it controls
        the trade-off between staying close to the behavior policy (higher alpha)
        versus potentially finding better policies (lower alpha).
    """
    config = DiscreteCQLConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        alpha=alpha  # CQL conservatism parameter
    )
    
    # Add scaler if provided
    if scaler is not None:
        config.observation_scaler = scaler
    
    print(f"Created CQL config with alpha={config.alpha}, batch_size={config.batch_size}")
    return config

def create_cql_model(config, device="cpu", enable_ddp=False):
    """
    Create a CQL model instance configured for the specified device.
    
    Instantiates a DiscreteCQL model that can be trained to learn
    an optimal policy from the HFNC dataset.
    
    Args:
        config (DiscreteCQLConfig): Algorithm configuration parameters
        device (str): Device to use for tensor operations
                     "cpu" for CPU, "cuda:0" for first GPU, etc.
        enable_ddp (bool): Whether to enable distributed data parallel training
                          Useful for multi-GPU setups
                          
    Returns:
        DiscreteCQL: Instantiated model ready for training
        
    Raises:
        Exception: If model creation fails, with detailed error information
        
    Note:
        For stability during development, "cpu" is often safer than GPU-based training
        as it avoids CUDA-specific errors, though it's significantly slower.
    """
    # Force CPU if specified
    print(f"Using device: {device}")
    
    # Create CQL instance with our config
    try:
        cql = DiscreteCQL(
            config=config, 
            device=device, 
            enable_ddp=enable_ddp
        )
        
        print(f"CQL instance created with device: {device}")
        return cql
    except Exception as e:
        print(f"Error creating CQL model: {e}")
        raise

def save_model(model, model_path):
    """
    Save the trained model to disk for later use or evaluation.
    
    Attempts to save the model using the appropriate method based on
    the d3rlpy version (API may vary between versions).
    
    Args:
        model (DiscreteCQL): The trained CQL model
        model_path (Path): Path where the model should be saved
        
    Returns:
        bool: True if saving succeeded, False otherwise
        
    Note:
        The saved model includes the trained networks, the scaler (if any),
        and algorithm configuration, allowing for complete restoration
        of the trained policy.
    """
    try:
        # Try different save methods since the API might vary
        try:
            model.save_model(model_path)
        except AttributeError:
            model.save(model_path)
            
        print(f"💾  Model saved → {model_path}")
        return True
    except Exception as e:
        print(f"❌ Could not save model: {e}")
        return False