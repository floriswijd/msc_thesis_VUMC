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
from evaluator import CQLEvaluator # Import the class directly
import utils       # Utility functions for debugging and visualization

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
    print("\\n=== Loading and preprocessing data ===")
    df = data_loader.load_data(args.data)
    # cfg = config.load_config(args.cfg) # This was commented out
    
    data_dict = data_loader.preprocess_data(df)
    print("\\n=== Creating dataset ===")
    try:
        mdp_dataset_full = dataset.create_mdp_dataset( # Renamed for clarity
            data_dict["states"],    # State features (observations)
            data_dict["actions"],   # Actions taken by clinicians
            data_dict["rewards"],   # Rewards (clinical outcomes)
            data_dict["dones"]      # Episode termination flags
        )
        
        utils.debug_nan_values(data_dict["states"], "states")
        utils.debug_nan_values(data_dict["rewards"], "rewards") 
        utils.debug_inf_values(data_dict["states"], "states")
        utils.debug_inf_values(data_dict["rewards"], "rewards")
        
        train_eps, val_eps, test_eps = dataset.split_dataset(mdp_dataset_full) # Split the full dataset

        if not train_eps:
            print("\\n❌ Error: No training episodes after split. Exiting.")
            sys.exit(1)
        # train_dataset = dataset.MDPDataset(episodes=train_eps) <- #can't do this gives an error

    except Exception as e:
        print(f"\\n❌ Error creating dataset: {e}")
        sys.exit(1)
    print("\\n=== Creating model ===")
    scaler = model.create_scaler()
    cql_config = model.create_cql_config(
        batch_size=args.batch,    # Batch size for training updates
        learning_rate=args.lr,    # Step size for optimizer
        gamma=args.gamma,         # Discount factor for future rewards
        alpha=args.alpha,         # CQL conservatism parameter
        scaler=scaler             # Observation normalizer
    )
    
    try:
        cql = model.create_cql_model(
            config=cql_config,
            device=device,         # CPU or specific GPU
            enable_ddp=False       # No distributed training
        )
    except Exception as e:
        print(f"\\n❌ Error creating model: {e}")
        sys.exit(1)
    
    # Define config path for evaluator (used for internal validators)
    config_path_for_validation = Path("../../config.yaml") # As used by the old CQLValidator

    # Instantiate CQLEvaluator
    evaluator_instance = CQLEvaluator(cql, config_path=config_path_for_validation)

    # Construct the correct base path for d3rlpy logs
    # d3rlpy saves logs to d3rlpy_logs/<experiment_name>/<timestamped_run_folder>
    # In this setup, experiment_name passed to model.fit() is args.logdir
    # actual_d3rlpy_log_dir = Path("d3rlpy_logs") / args.logdir # This is the experiment folder

    print("\\n=== Training model ===")
    result, errors = trainer.train_model(
        model=cql,                  # CQL model to train
        train_episodes=train_eps,      # Pass the training-only dataset
        n_epochs=args.epochs,       # Number of training epochs
        experiment_name=args.logdir # Log directory name (used by d3rlpy to create d3rlpy_logs/args.logdir_timestamp)
    )

    # --- Find the latest d3rlpy log directory for the current run pattern ---
    # This logic is moved up to be available for comprehensive evaluation plots
    latest_log_dir = None
    # d3rlpy typically creates logs in "d3rlpy_logs/<experiment_name_timestamp>"
    # If args.logdir is "my_experiment", d3rlpy might create "d3rlpy_logs/my_experiment_20230101.../"
    # The trainer.train_model uses args.logdir as experiment_name.
    # d3rlpy's fit method creates a directory like `d3rlpy_logs/experiment_name_timestamp`.
    # We need to find this directory.
    
    # The `experiment_name` in `trainer.train_model` is `args.logdir`.
    # d3rlpy creates `d3rlpy_logs/<experiment_name_timestamp>`.
    # So, we search for directories in `d3rlpy_logs` that start with `args.logdir`.
    
    base_d3rlpy_logs_dir = Path("d3rlpy_logs")
    if base_d3rlpy_logs_dir.exists() and base_d3rlpy_logs_dir.is_dir():
        potential_dirs = sorted([
            d for d in base_d3rlpy_logs_dir.iterdir()
            if d.is_dir() and d.name.startswith(str(args.logdir)) # Match experiment name prefix
        ])
        if potential_dirs:
            latest_log_dir = potential_dirs[-1] # Get the last one (latest timestamp)
            print(f"ℹ️  Found latest d3rlpy log directory for plots and stability checks: {latest_log_dir}")
        else:
            print(f"⚠️  Warning: No timestamped log directories found matching prefix '{args.logdir}' in {base_d3rlpy_logs_dir}. Plots and stability checks might use a default path or fail.")
    else:
        print(f"⚠️  Warning: Base d3rlpy logs directory not found at {base_d3rlpy_logs_dir}. Plots and stability checks might use a default path or fail.")


    if errors:
        print("\\n⚠️ Training encountered errors, checking logs for diagnosis...")
        # Pass the found latest_log_dir to check_training_logs if it exists
        if latest_log_dir and latest_log_dir.exists():
            trainer.check_training_logs(latest_log_dir)
        else:
            # Fallback if specific run directory isn't found, try the general experiment dir
            # This might not contain the CSVs if d3rlpy failed early.
            trainer.check_training_logs(Path("d3rlpy_logs") / args.logdir)


    print("\\n=== Evaluating model (Comprehensive) ===")
    # Use the new comprehensive evaluation framework from the instantiated evaluator
    # Plots will be saved in latest_log_dir if found, otherwise "evaluation_results"
    eval_save_dir = latest_log_dir if latest_log_dir and latest_log_dir.exists() else Path("evaluation_results")
    if not eval_save_dir.exists(): # Ensure the directory exists if it's the fallback
        eval_save_dir.mkdir(parents=True, exist_ok=True)

    comprehensive_results = evaluator_instance.evaluate_comprehensive(test_eps, save_dir=eval_save_dir)
    
    # Keep basic metrics for backward compatibility if needed by other parts, though comprehensive_results is richer
    # metrics = evaluator_instance.add_training_params_to_metrics(comprehensive_results.get('basic_performance', {}), args)
    # The add_training_params_to_metrics is a standalone function in evaluator.py now.
    # Let's assume comprehensive_results is sufficient. If metrics object is strictly needed:
    from evaluator import add_training_params_to_metrics # import the standalone function
    metrics = add_training_params_to_metrics(comprehensive_results.get('basic_performance', {}), args)

    
    print("\\n=== Academic Performance Summary ===")
    if 'basic_performance' in comprehensive_results:
        bp = comprehensive_results['basic_performance']
        print(f"📊 Action Agreement with Clinicians: {bp.get('mean_action_agreement', 'N/A'):.3f} ± {bp.get('std_action_agreement', 'N/A'):.3f}")
        print(f"📊 Mean Episode Return: {bp.get('mean_return', 'N/A'):.3f} ± {bp.get('std_return', 'N/A'):.3f}")
        print(f"📊 Total Episodes Evaluated: {bp.get('total_episodes', 'N/A')}")
    
    if 'clinical_performance' in comprehensive_results:
        cp = comprehensive_results['clinical_performance']
        print(f"🏥 Safety Violation Rate: {cp.get('safety_violation_rate', 'N/A'):.3f}")
        print(f"🏥 Parameter Appropriateness: {cp.get('parameter_appropriateness_score', 'N/A'):.3f}")
        # print(f"🏥 Outcome Improvement vs Clinicians: {cp.get('outcome_improvement', 'N/A'):.3f}") # Was N/A
    
    if 'statistical_analysis' in comprehensive_results:
        sa = comprehensive_results['statistical_analysis']
        print(f"📈 Effect Size (Cohen's d): {sa.get('cohens_d', 'N/A'):.3f} ({sa.get('effect_size_interpretation', 'N/A')})")
        if 'paired_t_test' in sa:
            print(f"📈 Statistical Significance (p-value): {sa['paired_t_test'].get('p_value', 'N/A'):.4f}")
    
    if 'policy_analysis' in comprehensive_results:
        pa = comprehensive_results['policy_analysis']
        print(f"🎯 Policy Entropy: {pa.get('policy_entropy', 'N/A'):.3f}")
        action_dist_summary = pa.get('action_distribution', {})
        if isinstance(action_dist_summary, dict):
             print(f"🎯 Action Distribution (Top 5): {dict(list(action_dist_summary.items())[:5])}")
        else:
             print(f"🎯 Action Distribution: {action_dist_summary}")


    print(f"\\n📊 Academic visualizations and detailed report saved to '{eval_save_dir}' directory")
    
    # The old "Clinical Safety Validation" section is removed. Its functionality is now part of
    # evaluator_instance.perform_clinical_safety_evaluation() called below.

    # The old "Analyzing predictions" section is removed. Its functionality is covered by
    # comprehensive_results['policy_analysis'] printed above.

    print("\\n=== Analyzing training logs (from d3rlpy output) ===")
    # This uses latest_log_dir which was determined earlier.
    if latest_log_dir and latest_log_dir.exists() and latest_log_dir.is_dir():
        utils.check_gradient_values(latest_log_dir)
        utils.plot_training_curves(latest_log_dir) # This plots d3rlpy's own loss/grad curves
    else:
        print(f"⚠️  Warning: d3rlpy log directory for plotting training curves not found or not valid ({latest_log_dir}), skipping plot generation.")

    print("\\n=== Saving results (metrics and model) ===")
    config.save_metrics(metrics, paths["metric_path"]) # metrics now derived from comprehensive_results
    model.save_model(cql, paths["model_path"])


    print("\\n=== Consolidated Pipeline Validation (using CQLEvaluator) ===")
    # This runs data_quality, model_behavior, clinical_plausibility, training_stability
    # It uses the config loaded by CQLEvaluator (via config_path_for_validation)
    # log_dir for stability check is latest_log_dir
    # df is the original dataframe for data quality check
    effective_log_dir_for_validation = latest_log_dir if latest_log_dir and latest_log_dir.exists() else Path("d3rlpy_logs") # Fallback

    validation_summary = evaluator_instance.perform_full_validation(
        data_dict,
        test_eps,
        effective_log_dir_for_validation,
        df=df # Pass the full dataframe
    )
    # Print detailed validation summary
    for stage, v_result in validation_summary.items():
        print(f"--- {stage.replace('_', ' ').title()} ---")
        if not v_result.get("passed", True):
            print(f"  ❌ Issues found: {v_result.get('errors', [])}")
        if v_result.get("warnings"):
            print(f"  ⚠️ Warnings: {v_result.get('warnings', [])}")
        if not v_result.get("errors") and not v_result.get("warnings") and v_result.get("passed", True):
             print("  ✅ All checks passed.")
        # Optionally print stats: print(f"  Stats: {v_result.get('stats', {})}")


    print("\\n=== Consolidated Clinical Safety Checks (Adverse Events, using CQLEvaluator) ===")
    # This runs adverse event prediction
    # It uses the config loaded by CQLEvaluator
    clinical_safety_summary = evaluator_instance.perform_clinical_safety_evaluation(test_eps)
    # Print clinical_safety_summary (adverse_events part)
    if 'adverse_events' in clinical_safety_summary:
        ae = clinical_safety_summary['adverse_events']
        print(f"  Adverse Event Risk: {ae.get('overall_adverse_event_risk', 'N/A'):.3f}")
        print(f"  High Risk Decisions: {ae.get('high_risk_decisions', 'N/A')}")
        if ae.get('risk_category_rates'):
            print("  Risk Category Rates:")
            for cat, rate_val in ae['risk_category_rates'].items(): # Renamed to avoid conflict
                print(f"    {cat}: {rate_val:.3f}")
    else:
        print("  No adverse event data in clinical safety summary.")
            
    print("\\n=== Training complete ===")
    
if __name__ == "__main__":
    main()