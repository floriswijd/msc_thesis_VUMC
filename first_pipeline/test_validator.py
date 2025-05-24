#!/usr/bin/env python3
"""Test script to debug validator issues"""

from pathlib import Path
from validator import CQLValidator
import numpy as np

def test_validator():
    print("=== Testing Validator ===")
    
    # Test config loading
    config_path = Path("../../config.yaml")
    print(f"Config path exists: {config_path.exists()}")
    
    validator = CQLValidator(config_path)
    print(f"Config loaded: {validator.config}")
    
    # Test with minimal dummy data
    dummy_data_dict = {
        "states": np.random.rand(100, 10),
        "actions": np.random.randint(0, 12, 100),
        "rewards": np.random.rand(100),
        "dones": np.random.choice([True, False], 100),
        "state_columns": [f"feature_{i}" for i in range(10)]
    }
    
    print("\n=== Testing Data Quality Validation ===")
    result = validator.validate_data_quality(dummy_data_dict)
    print(f"Result: {result}")
    
if __name__ == "__main__":
    test_validator()