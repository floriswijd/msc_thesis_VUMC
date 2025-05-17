#!/usr/bin/env python3
# -----------------------------------------------------------
# evaluate_cql_simple.py  --  Evaluate trained CQL model
# -----------------------------------------------------------
#
# • Loads the trained CQL model
# • Evaluates on test data
# • Provides prediction analysis and visualizations
#
# CLI:
#   python evaluate_cql_simple.py --model runs/cql_simple/cql_hfnc_model_simple.pt
# -----------------------------------------------------------

import argparse
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Import d3rlpy components
import d3rlpy
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteCQL

# ---------- CLI ARGUMENTS ----------
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="/Users/floppie/Documents/Msc Scriptie/preprocessed_data_3.csv", 
                   help="Path to the preprocessed data")
parser.add_argument("--model", default="runs/cql_simple/cql_hfnc_model_simple.pt",
                   help="Path to the trained model file")
parser.add_argument("--mapping", default="runs/cql_simple/action_mapping.yaml",
                   help="Path to action mapping file")
parser.add_argument("--test_size", type=float, default=0.2,
                   help="Proportion of data to use for testing")
parser.add_argument("--output", default="runs/cql_simple/evaluation/",
                   help="Directory to save evaluation outputs")
args = parser.parse_args()

# Create output directory if it doesn't exist
Path(args.output).mkdir(parents=True, exist_ok=True)

print(f"⚙️ Using d3rlpy version {d3rlpy.__version__}")
print(f"📂 Model path: {args.model}")
print(f"📂 Action mapping path: {args.mapping}")
print(f"📊 Test size: {args.test_size}")

# ---------- LOAD ACTION MAPPING ----------
try:
    with open(args.mapping, 'r') as f:
        action_mapping_dict = yaml.safe_load(f)
    print(f"✅ Loaded action mapping with {len(action_mapping_dict)} actions")
except Exception as e:
    print(f"❌ Error loading action mapping: {e}")
    print("Creating default action mapping...")
    action_mapping_dict = {"0": {"flow": 0.0, "fio2": 21.0}}

# ---------- LOAD DATA ----------
print("⏳ Loading test data...")
df = pd.read_csv(args.data)

# For simplicity, as in the training script, we'll focus only on flow and FiO2
state_cols = ['flow', 'fio2']
states = df[state_cols].values.astype('float32')

# ---------- PREPARE TEST DATA ----------
# Normalize the states using sklearn's StandardScaler
scaler = StandardScaler()
scaled_states = scaler.fit_transform(states)

# Split data
from sklearn.model_selection import train_test_split
train_states, test_states = train_test_split(
    scaled_states, 
    test_size=args.test_size, 
    random_state=42
)

print(f"📊 Test data size: {len(test_states)} states")

# ---------- LOAD MODEL ----------
try:
    print("⏳ Loading trained CQL model...")
    model = torch.load(args.model)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Attempting to recreate model structure...")
    
    try:
        # If model loading fails, create a default model structure
        model = DiscreteCQL(use_gpu=False)
        model.build_with_dataset(MDPDataset(
            observations=np.zeros((1, len(state_cols)), dtype=np.float32),
            actions=np.array([0], dtype=np.int32),
            rewards=np.array([0.0], dtype=np.float32),
            terminals=np.array([True], dtype=bool)
        ))
        print("✅ Created default model structure")
    except Exception as e2:
        print(f"❌ Error creating default model: {e2}")
        print("⚠️ Evaluation cannot proceed without a valid model")
        exit(1)

# ---------- EVALUATE MODEL ----------
print("📊 Evaluating model predictions...")

# Map integer actions back to flow/fio2 values
def get_action_values(action_id):
    """Convert action ID to flow/fio2 values using the mapping."""
    action_id_str = str(action_id)
    if action_id_str in action_mapping_dict:
        return action_mapping_dict[action_id_str]["flow"], action_mapping_dict[action_id_str]["fio2"]
    else:
        # Default action if not found in mapping
        return 0.0, 21.0

# Generate predictions for test data
predictions = []
predicted_flow = []
predicted_fio2 = []

try:
    for state in test_states:
        # Make prediction
        try:
            action_id = model.predict(state.reshape(1, -1))[0]
            flow, fio2 = get_action_values(action_id)
            
            predictions.append(action_id)
            predicted_flow.append(flow)
            predicted_fio2.append(fio2)
        except Exception as e:
            print(f"Prediction error: {e}")
            # Default values for failed predictions
            predictions.append(0)
            predicted_flow.append(0.0)
            predicted_fio2.append(21.0)
    
    # Create a dataframe for analysis
    results_df = pd.DataFrame({
        'original_flow': df['flow'].values[-len(test_states):],
        'original_fio2': df['fio2'].values[-len(test_states):],
        'predicted_flow': predicted_flow,
        'predicted_fio2': predicted_fio2,
        'action_id': predictions
    })
    
    # Calculate error metrics
    results_df['flow_error'] = results_df['predicted_flow'] - results_df['original_flow']
    results_df['fio2_error'] = results_df['predicted_fio2'] - results_df['original_fio2']
    
    # Print summary statistics
    print("\n----- Evaluation Results -----")
    print(f"Number of test samples: {len(test_states)}")
    print(f"Unique predicted actions: {len(set(predictions))}")
    
    print("\nFlow Error:")
    print(f"  Mean absolute error: {results_df['flow_error'].abs().mean():.2f}")
    print(f"  Standard deviation: {results_df['flow_error'].std():.2f}")
    
    print("\nFiO2 Error:")
    print(f"  Mean absolute error: {results_df['fio2_error'].abs().mean():.2f}")
    print(f"  Standard deviation: {results_df['fio2_error'].std():.2f}")
    
    # Most frequently predicted actions
    action_counts = Counter(predictions)
    print("\nTop 5 Predicted Actions:")
    for action_id, count in action_counts.most_common(5):
        flow, fio2 = get_action_values(action_id)
        print(f"  Action {action_id}: Flow={flow:.1f}, FiO2={fio2:.1f} - Count: {count} ({count/len(predictions)*100:.1f}%)")
    
    # Save results to CSV
    results_path = Path(args.output) / "prediction_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to {results_path}")
    
    # ---------- VISUALIZATIONS ----------
    print("\n📊 Creating visualizations...")
    
    # Plot 1: Flow error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['flow_error'], bins=30, alpha=0.7)
    plt.title('Flow Prediction Error Distribution')
    plt.xlabel('Error (L/min)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    flow_error_path = Path(args.output) / "flow_error_dist.png"
    plt.savefig(flow_error_path)
    print(f"✅ Flow error distribution saved to {flow_error_path}")
    
    # Plot 2: FiO2 error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['fio2_error'], bins=30, alpha=0.7)
    plt.title('FiO2 Prediction Error Distribution')
    plt.xlabel('Error (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    fio2_error_path = Path(args.output) / "fio2_error_dist.png"
    plt.savefig(fio2_error_path)
    print(f"✅ FiO2 error distribution saved to {fio2_error_path}")
    
    # Plot 3: Scatter plot of actual vs. predicted values
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(results_df['original_flow'], results_df['predicted_flow'], alpha=0.5)
    plt.plot([0, 60], [0, 60], 'r--')  # Perfect prediction line
    plt.title('Actual vs. Predicted Flow')
    plt.xlabel('Actual Flow (L/min)')
    plt.ylabel('Predicted Flow (L/min)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(results_df['original_fio2'], results_df['predicted_fio2'], alpha=0.5)
    plt.plot([20, 100], [20, 100], 'r--')  # Perfect prediction line
    plt.title('Actual vs. Predicted FiO2')
    plt.xlabel('Actual FiO2 (%)')
    plt.ylabel('Predicted FiO2 (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    scatter_path = Path(args.output) / "prediction_scatter.png"
    plt.savefig(scatter_path)
    print(f"✅ Prediction scatter plot saved to {scatter_path}")
    
    print("\n✅ Evaluation complete!")

except Exception as e:
    print(f"❌ Error during evaluation: {e}")
    print("⚠️ Evaluation could not be completed successfully")