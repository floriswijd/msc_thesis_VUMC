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
from validator import CQLValidator      
import numpy as np

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
    # cfg = config.load_config(args.cfg)
    
    data_dict = data_loader.preprocess_data(df)
    print("\n=== Creating dataset ===")
    try:
        mdp_dataset_full = dataset.create_mdp_dataset(
            data_dict["states"],
            data_dict["actions"], 
            data_dict["rewards"],
            data_dict["dones"]
        )
        
        utils.debug_nan_values(data_dict["states"], "states")
        utils.debug_nan_values(data_dict["rewards"], "rewards") 
        utils.debug_inf_values(data_dict["states"], "states")
        utils.debug_inf_values(data_dict["rewards"], "rewards")
        
        train_eps, val_eps, test_eps = dataset.split_dataset(mdp_dataset_full)

        if not train_eps:
            print("\n❌ Error: No training episodes after split. Exiting.")
            sys.exit(1)
        
        # ADD THIS: Count transitions in each split
        train_transitions = dataset.count_transitions(train_eps)
        val_transitions = dataset.count_transitions(val_eps)
        test_transitions = dataset.count_transitions(test_eps)
        total_transitions = train_transitions + val_transitions + test_transitions
        
        print(f"\n📊 Dataset Split Summary:")
        print(f"   Training:   {len(train_eps):,} episodes, {train_transitions:,} transitions ({100*train_transitions/total_transitions:.1f}%)")
        print(f"   Validation: {len(val_eps):,} episodes, {val_transitions:,} transitions ({100*val_transitions/total_transitions:.1f}%)")
        print(f"   Test:       {len(test_eps):,} episodes, {test_transitions:,} transitions ({100*test_transitions/total_transitions:.1f}%)")
        print(f"   Total:      {len(train_eps + val_eps + test_eps):,} episodes, {total_transitions:,} transitions")
        
        # Average episode lengths
        avg_train_len = train_transitions / len(train_eps) if train_eps else 0
        avg_val_len = val_transitions / len(val_eps) if val_eps else 0
        avg_test_len = test_transitions / len(test_eps) if test_eps else 0
        
        print(f"\n📏 Average Episode Lengths:")
        print(f"   Training:   {avg_train_len:.1f} transitions/episode")
        print(f"   Validation: {avg_val_len:.1f} transitions/episode")
        print(f"   Test:       {avg_test_len:.1f} transitions/episode")

        # ADD THIS: Determine n_actions from the data
        if 'actions' in data_dict and data_dict['actions'] is not None:
            n_actions = int(np.max(data_dict['actions'])) + 1
            print(f"\n📊 Inferred n_actions from data: {n_actions}")
        else:
            print("\n❌ Cannot determine n_actions from data.")
            sys.exit(1)

        # ADD THIS: Initialize Behavior Policy Estimator BEFORE training
        print("\n=== Initializing Enhanced Off-Policy Evaluation ===")
        from evaluator import BehaviorPolicyEstimator
        
        behavior_policy_estimator = BehaviorPolicyEstimator(n_actions=n_actions)
        print("🔧 Fitting behavior policy on training episodes...")
        behavior_policy_estimator.fit(train_eps)  # Fit on training data

    except Exception as e:
        print(f"\n❌ Error creating dataset: {e}")
        sys.exit(1)

    print("\n=== Creating model ===")
    scaler = model.create_scaler()
    cql_config = model.create_cql_config(
        batch_size=args.batch, learning_rate=args.lr, gamma=args.gamma,
        alpha=args.alpha, scaler=scaler
    )
    cql = model.create_cql_model(config=cql_config, device=device)

    print("\n=== Training model ===")
    result, errors = trainer.train_model(model=cql, train_episodes=train_eps, n_epochs=args.epochs, batch_size=args.batch, experiment_name=args.logdir)
    
    if errors:
        print("\\n⚠️ Training encountered errors, checking logs for diagnosis...")
        # Pass the corrected path to check_training_logs
        actual_d3rlpy_log_dir = Path("d3rlpy_logs") / args.logdir
        trainer.check_training_logs(actual_d3rlpy_log_dir)

    # --- Determine the latest d3rlpy log directory for saving evaluation outputs and analyzing training logs ---
    # This logic is moved from its original position later in the script.
    latest_log_dir_for_outputs = None
    # Original logic to find the specific timestamped run directory:
    # Assumes args.logdir might be like "runs/cql", where "runs" is a subdir in d3rlpy_logs
    # and "cql" is the prefix for the timestamped folder.
    logdir_path_obj = Path(args.logdir) # e.g., Path("runs/cql")
    # search_parent_dir should be where timestamped folders like "cql_xxxx" reside.
    # Based on user's example: /Users/floppie/Documents/Msc Scriptie/HFNC codebase/first_pipeline/d3rlpy_logs/runs/cql_20250524174450
    # This implies base_d3rlpy_runs_dir = Path("d3rlpy_logs") / "runs" if args.logdir is "runs/cql"
    # or more generally, Path("d3rlpy_logs") / logdir_path_obj.parent
    
    # The original script had: base_d3rlpy_runs_dir = Path("d3rlpy_logs") / "runs"
    # Let's assume logdir_path_obj.parent correctly gives "runs" or similar if args.logdir is "runs/cql"
    # If args.logdir is just "cql", then logdir_path_obj.parent is ".", so search_parent_dir becomes "d3rlpy_logs"
    # The original code explicitly used Path("d3rlpy_logs") / "runs". We'll stick to that for base_d3rlpy_runs_dir
    # as it matches the user's example path structure.
    base_d3rlpy_runs_dir_for_search = Path("d3rlpy_logs") / "runs"
    run_prefix_for_search = logdir_path_obj.name # e.g., "cql" if args.logdir is "runs/cql" or just "cql"

    if base_d3rlpy_runs_dir_for_search.exists() and base_d3rlpy_runs_dir_for_search.is_dir():
        potential_dirs = sorted([
            d for d in base_d3rlpy_runs_dir_for_search.iterdir()
            if d.is_dir() and d.name.startswith(f"{run_prefix_for_search}_")
        ])
        if potential_dirs:
            latest_log_dir_for_outputs = potential_dirs[-1]
            print(f"ℹ️  Using latest run log directory for outputs: {latest_log_dir_for_outputs}")
        else:
            print(f"⚠️  Warning: No timestamped log directory found matching prefix '{run_prefix_for_search}_' in {base_d3rlpy_runs_dir_for_search}.")
    else:
        print(f"⚠️  Warning: Base d3rlpy runs directory for search ({base_d3rlpy_runs_dir_for_search}) not found.")

    # Determine save directories
    if latest_log_dir_for_outputs:
        evaluation_results_save_dir = latest_log_dir_for_outputs
        clinical_validation_results_save_dir = latest_log_dir_for_outputs
        save_location_message_suffix = f"in '{latest_log_dir_for_outputs}'"
    else:
        evaluation_results_save_dir = Path("evaluation_results")
        clinical_validation_results_save_dir = Path("clinical_validation")
        save_location_message_suffix = "in their respective default directories ('evaluation_results/', 'clinical_validation/')"
        print(f"⚠️  Outputs will be saved {save_location_message_suffix} as the specific run directory was not identified.")

    print("\\n=== Evaluating model ===")
    # Use the new comprehensive evaluation framework
    from evaluator import CQLEvaluator
    
    cql_evaluator_obj = CQLEvaluator(cql,  n_actions=n_actions,  behavior_policy_estimator=behavior_policy_estimator) # Renamed instance
    comprehensive_results = cql_evaluator_obj.evaluate_comprehensive(test_eps, save_dir=evaluation_results_save_dir)
    
    # Also keep basic metrics for backward compatibility
    basic_metrics = cql_evaluator_obj._evaluate_basic_performance(test_eps)
    metrics = cql_evaluator_obj.add_training_params_to_metrics(basic_metrics, args)
    
    print("\\n=== Academic Performance Summary ===")
    if 'basic_performance' in comprehensive_results:
        bp = comprehensive_results['basic_performance']
        print(f"📊 Action Agreement with Clinicians: {bp['mean_action_agreement']:.3f} ± {bp['std_action_agreement']:.3f}")
        print(f"📊 Mean Episode Return: {bp['mean_return']:.3f} ± {bp['std_return']:.3f}")
        print(f"📊 Total Episodes Evaluated: {bp['total_episodes']}")
    
    if 'clinical_performance' in comprehensive_results:
        cp = comprehensive_results['clinical_performance']
        print(f"🏥 Predicted Outcome Mean: {cp['predicted_outcome_mean']:.3f}")
        print(f"🏥 Clinician Outcome Mean: {cp['clinician_outcome_mean']:.3f}")
        print(f"🏥 Outcome Improvement vs Clinicians: {cp['outcome_improvement']:.3f}")
        print(f"🏥 Total Actions Evaluated: {cp['total_actions_evaluated']}")
        print(f"")
    
    if 'statistical_analysis' in comprehensive_results:
        sa = comprehensive_results['statistical_analysis']
        print(f"📈 Effect Size (Cohen's d): {sa['cohens_d']:.3f} ({sa['effect_size_interpretation']})")
        if 'paired_t_test' in sa:
            print(f"📈 Statistical Significance (p-value): {sa['paired_t_test']['p_value']:.4f}")
    
    if 'policy_analysis' in comprehensive_results:
        pa = comprehensive_results['policy_analysis']
        print(f"🎯 Policy Entropy: {pa['policy_entropy']:.3f}")
        print(f"🎯 Action Distribution: {dict(list(pa['action_distribution'].items())[:5])}")  # Show top 5

    # In main.py, after comprehensive_results = cql_evaluator_obj.evaluate_comprehensive(...)
    if 'weighted_importance_sampling' in comprehensive_results:
        wis = comprehensive_results['weighted_importance_sampling']
        print(f"\n🎯 Weighted Importance Sampling Results:")
        print(f"   WIS Estimate: {wis['wis_estimate']:.4f}")
        print(f"   Effective Sample Size: {wis['ess']:.2f}")
        print(f"   Mean Trajectory Weight: {wis['mean_trajectory_weight']:.4f}")

    if 'doubly_robust' in comprehensive_results:
        dr = comprehensive_results['doubly_robust']
        print(f"\n🎯 Doubly Robust Results:")
        print(f"   DR Estimate: {dr['dr_estimate']:.4f}")

    if 'fitted_q_evaluation' in comprehensive_results:
        fqe = comprehensive_results['fitted_q_evaluation']
        print(f"\n🎯 Fitted Q Evaluation Results:")
        print(f"   FQE Estimate: {fqe['fqe_estimate']:.4f} ± {fqe['fqe_std']:.4f}")
    
    print(f"\\n📊 Academic visualizations and detailed report saved {save_location_message_suffix}")
    
    # Add clinical safety validation
    print("\\n=== Clinical Safety Validation ===")
    from clinical_validator import validate_clinical_safety
    
    clinical_results = validate_clinical_safety(cql, test_eps, save_dir=clinical_validation_results_save_dir)
    
    print("🏥 Clinical Safety Results:")
    if 'parameter_safety' in clinical_results:
        ps = clinical_results['parameter_safety']
        print(f"   Parameter Safety Score: {ps['safety_score']:.3f}")
        print(f"   Total Safety Violations: {ps['total_violations']}")
    
    if 'clinical_appropriateness' in clinical_results:
        ca = clinical_results['clinical_appropriateness']
        print(f"   Clinical Appropriateness: {ca['mean_appropriateness']:.3f}")
        print(f"   High Appropriateness Rate: {ca['high_appropriateness_rate']:.3f}")
    
    if 'adverse_events' in clinical_results:
        ae = clinical_results['adverse_events']
        print(f"   Adverse Event Risk: {ae['overall_adverse_event_risk']:.3f}")
        print(f"   High Risk Decisions: {ae['high_risk_decisions']}")
    
    print(f"🏥 Clinical validation report saved {save_location_message_suffix}")
    
    print("\\n=== Analyzing predictions ===")
    cql_evaluator_obj.analyze_predictions(cql, test_eps, top_n=3) # Use renamed instance

    print("\\n=== Analyzing training logs (from d3rlpy output) ===")
    
    # --- The logic to find the latest d3rlpy log directory has been moved up ---
    # --- and its result is stored in 'latest_log_dir_for_outputs' ---

    # Ensure the directory exists before trying to analyze logs
    if latest_log_dir_for_outputs and latest_log_dir_for_outputs.exists() and latest_log_dir_for_outputs.is_dir():
        utils.check_gradient_values(latest_log_dir_for_outputs)
        utils.plot_training_curves(latest_log_dir_for_outputs)
    else:
        print(f"⚠️  Warning: d3rlpy log directory ({latest_log_dir_for_outputs if latest_log_dir_for_outputs else 'not found'}) for plotting training curves not found or not valid, skipping plot generation.")

    print("\n=== Feature Importance Analysis ===")
    from feature_importance import analyze_feature_importance
    
    # Run feature importance analysis
    importance_results = analyze_feature_importance(
        cql, 
        test_eps, 
        data_dict["state_columns"], 
        save_dir=evaluation_results_save_dir / "feature_importance"
    )
    
    print("🔍 Feature Importance Results:")
    if 'shap' in importance_results and importance_results['shap'] is not None:
        shap_df = importance_results['shap']['feature_importance']
        print(f"📊 Top 5 features (SHAP):")
        for i, row in shap_df.head(5).iterrows():
            print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    if 'permutation' in importance_results and importance_results['permutation'] is not None:
        perm_df = importance_results['permutation']['feature_importance']
        print(f"🔄 Top 5 features (Permutation):")
        
        # Check which column name is available
        if 'action_flip_rate' in perm_df.columns:
            # New direct permutation method
            for i, row in perm_df.head(5).iterrows():
                print(f"   {i+1}. {row['feature']}: {row['action_flip_rate']:.4f}")
        elif 'importance_mean' in perm_df.columns:
            # Old surrogate method (fallback)
            for i, row in perm_df.head(5).iterrows():
                print(f"   {i+1}. {row['feature']}: {row['importance_mean']:.4f}")
        else:
            # Generic handling
            importance_col = perm_df.columns[1]  # Second column after 'feature'
            for i, row in perm_df.head(5).iterrows():
                print(f"   {i+1}. {row['feature']}: {row[importance_col]:.4f}")

    print("\\n=== Saving results ===")
    config.save_metrics(metrics, paths["metric_path"])
    model.save_model(cql, paths["model_path"])
    print("\n=== Training complete ===")

    print("\n=== Validating pipeline ===")

    # Fix: Use the correct path to config.yaml (in parent directory)
    config_path = Path("../../config.yaml")  # Go up two levels to find config.yaml
    validator = CQLValidator(config_path)

    try:
        # 1) data checks (pass the raw df for episode-length analysis)
        print("Running data quality validation...")
        data_result = validator.validate_data_quality(data_dict, df)
        if not data_result["passed"]:
            print(f"❌ Data quality issues found: {data_result['errors']}")
        if data_result["warnings"]:
            print(f"⚠️  Data warnings: {data_result['warnings']}")

        # 2) model decision analysis
        print("Running model behavior validation...")
        model_result = validator.validate_model_behavior(cql, test_eps)
        if not model_result["passed"]:
            print(f"❌ Model behavior issues found: {model_result['errors']}")
        if model_result["warnings"]:
            print(f"⚠️  Model warnings: {model_result['warnings']}")

        # 3) clinical safety rules
        print("Running clinical plausibility validation...")
        clinical_result = validator.validate_clinical_plausibility(cql, test_eps)
        if not clinical_result["passed"]:
            print(f"❌ Clinical plausibility issues found: {clinical_result['errors']}")
        if clinical_result["warnings"]:
            print(f"⚠️  Clinical warnings: {clinical_result['warnings']}")

        # 4) training-log sanity checks
        print("Running training stability validation...")
        if latest_log_dir_for_outputs and latest_log_dir_for_outputs.exists(): # Use the determined log dir
            stability_result = validator.validate_training_stability(latest_log_dir_for_outputs)
            if not stability_result["passed"]:
                print(f"❌ Training stability issues found: {stability_result['errors']}")
            if stability_result["warnings"]:
                print(f"⚠️  Training warnings: {stability_result['warnings']}")
        else:
            print("⚠️  Could not find log directory for training stability validation")
            
        print("✅ Validation completed successfully!")
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
    
if __name__ == "__main__":
    main()