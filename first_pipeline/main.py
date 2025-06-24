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
    
    # Print alpha configuration
    print(f"\n🎛️  CQL Configuration:")
    print(f"   Alpha (conservatism weight): {args.alpha}")
    print(f"   Gamma (discount factor): {args.gamma}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Batch size: {args.batch}")
    print(f"   Epochs: {args.epochs}")
    
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



    # for i, col in enumerate(data_dict['state_columns'], 1):
    #     print(f"   {i:2d}. {col}")

    # print(data_dict['stay_ids'][:5])  # Print first 5 stay_ids for debugging
    
    # # Calculate and print number of null values in stay_ids
    # s_ids = np.asarray(data_dict['stay_ids']) # Ensure it's a numpy array
    # null_count_in_stay_ids = 0
    # if np.issubdtype(s_ids.dtype, np.floating): # Handles float arrays (e.g., [1.0, np.nan, 2.0])
    #     null_count_in_stay_ids = np.sum(np.isnan(s_ids))
    # elif s_ids.dtype == object: # Handles object arrays (e.g., [1, None, 'text', np.nan])
    #     # Iterate and check for None or float NaN
    #     # This list comprehension creates a boolean array, then np.sum counts True values.
    #     null_count_in_stay_ids = np.sum([item is None or (isinstance(item, float) and np.isnan(item)) for item in s_ids])
    # # For other dtypes (e.g., int, bool, non-object string arrays),
    # # null_count_in_stay_ids remains 0 by default, as np.nan or None are not standard null representations for them
    # # without being explicitly cast to float or object types.
    
    # print(f"   Number of null values in stay_ids: {null_count_in_stay_ids}")



    print("\n=== Creating dataset ===")
    try:
        # Create enhanced dataset with metadata
        mdp_dataset_full = dataset.create_mdp_dataset_with_metadata(
            data_dict["states"],
            data_dict["actions"], 
            data_dict["rewards"],
            data_dict["dones"],
            data_dict["stay_ids"],
            data_dict["subject_ids"],
            data_dict["episode_ids"]
        )
        
        utils.debug_nan_values(data_dict["states"], "states")
        utils.debug_nan_values(data_dict["rewards"], "rewards") 
        utils.debug_inf_values(data_dict["states"], "states")
        utils.debug_inf_values(data_dict["rewards"], "rewards")
        
        # Use stay-level split to prevent data leakage
        print("\n=== Splitting dataset by stay (preventing data leakage) ===")
        train_eps, val_eps, test_eps = dataset.split_dataset_by_stay(
            mdp_dataset_full,
            test_size=0.3,
            val_size=0.5,
            random_state=42
        )

        MIN_LEN = 4

        def filter_short(eps):
            return [ep for ep in eps if len(ep) >= MIN_LEN]

        orig_counts = (len(train_eps), len(val_eps), len(test_eps))

        train_eps = filter_short(train_eps)
        val_eps   = filter_short(val_eps)
        test_eps  = filter_short(test_eps)
        new_counts = (len(train_eps), len(val_eps), len(test_eps))
        print(f"\n📊 Dataset Split Counts (after filtering short episodes):"
              f"\n   Training:   {new_counts[0]:,} episodes (was {orig_counts[0]:,})"
                f"\n   Validation: {new_counts[1]:,} episodes (was {orig_counts[1]:,})"
                f"\n   Test:       {new_counts[2]:,} episodes (was {orig_counts[2]:,})")
        if not train_eps:
            print("\n❌ Error: No training episodes after split. Exiting.")
            sys.exit(1)

        # def inspect_episode(ep, idx=0):
        #     """Pretty-print the structure of one d3rlpy Episode object."""
        #     print(f"\nEpisode #{idx}")
        #     print("--------------------------------------------------")
        #     print("Available attributes:", [attr for attr in dir(ep) if not attr.startswith('_')])

        #     print(f"observations : shape {ep.observations.shape}, dtype {ep.observations.dtype}")
        #     print(f"actions      : shape {ep.actions.shape},       dtype {ep.actions.dtype}")
        #     print(f"rewards      : shape {ep.rewards.shape},       dtype {ep.rewards.dtype}")

        #     # Newer d3rlpy (≥2.0) uses `terminated`; older versions used `terminal` or `done`.
        #     term_flag = getattr(ep, "terminated", None)
        #     print("terminated   :", term_flag, "(scalar)")
            
        #     # If you want to see the done-vector we’ll feed to scope-rl:
        #     T = len(ep.actions)
        #     done_vec = np.zeros(T, dtype=bool)
        #     done_vec[-1] = bool(term_flag)
        #     print("done vector  :", done_vec.astype(int))   # 1 == terminal step

        # # ---- call it on the first few episodes ----------------------------
        # for i, ep in enumerate(train_eps[:3]):   # `episodes` is your list
        #     inspect_episode(ep, i)
        # Count transitions in each split
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

        # Determine n_actions from the data
        n_actions = int(np.max(data_dict['actions'])) + 1
        print(f"\n📊 Inferred n_actions from data: {n_actions}")

        # ADD THIS: Determine n_actions from the data
        if 'actions' in data_dict and data_dict['actions'] is not None:
            n_actions = int(np.max(data_dict['actions'])) + 1
            print(f"\n📊 Inferred n_actions from data: {n_actions}")
        else:
            print("\n❌ Cannot determine n_actions from data.")
            sys.exit(1)
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
    # result, errors = trainer.train_model(model=cql, train_episodes=train_eps, n_epochs=args.epochs, batch_size=args.batch, experiment_name=args.logdir)
    
    result, errors = trainer.train_model(
        model=cql,
        train_episodes=train_eps,
        val_episodes=val_eps,  # <--- PASS THE VALIDATION EPISODES
        n_epochs=args.epochs,
        batch_size=args.batch,
        experiment_name=args.logdir
    )


    if errors:
        print("\\n⚠️ Training encountered errors, checking logs for diagnosis...")
        # Pass the corrected path to check_training_logs
        actual_d3rlpy_log_dir = Path("d3rlpy_logs") / args.logdir
        trainer.check_training_logs(actual_d3rlpy_log_dir)

        # ADD THIS: Initialize Behavior Policy Estimator BEFORE training

    # print("\n=== Fitting Behavior Policy Model (Behavior Cloning) ===")
    # from d3rlpy.algos import DiscreteBC, DiscreteBCConfig
    # # ADD THIS IMPORT
    # from d3rlpy.dataset import create_infinite_replay_buffer

    # # Configure and create the BC model
    # bc_config = DiscreteBCConfig(learning_rate=1e-3)
    # bc_model = DiscreteBC(config=bc_config, device=device, enable_ddp=False)
    
    # # --- THIS IS THE KEY FIX ---
    # # Create a ReplayBuffer object for the training episodes.
    # # The .fit() method needs this object, not a raw list.
    # print(f"🔧 Creating replay buffer for BC model with {len(train_eps)} episodes...")
    # bc_replay_buffer = create_infinite_replay_buffer(train_eps + val_eps)  # Combine train and validation episodes for BC training
    # # ---------------------------

    # # Train the BC model using n_steps
    # bc_model.fit(
    #     bc_replay_buffer,  # <--- Pass the ReplayBuffer object here
    #     n_steps=100000,     # BC learns fast, 50k steps is often plenty
    #     n_steps_per_epoch=1000,
    #     show_progress=True
    # )

  # --- THIS IS THE UPDATED SECTION ---
    print("\n=== Fitting or Loading Behavior Policy Model (Behavior Cloning) ===")
    from d3rlpy.algos import DiscreteBC, DiscreteBCConfig
    from d3rlpy.dataset import create_infinite_replay_buffer
    import math

    if args.bc_model_path:
        # Load the pre-trained model
        print(f"💾 Loading pre-trained BC model from: {args.bc_model_path}")
        bc_model = model.load_model(Path(args.bc_model_path), algo_class=DiscreteBC, device=device)
        if bc_model is None:
            print("❌ Failed to load BC model. Exiting.")
            sys.exit(1)
    else:
        # Train a new model as before
        bc_config = DiscreteBCConfig(learning_rate=1e-3)
        bc_model = DiscreteBC(config=bc_config, device=device, enable_ddp=False)
        
        # --- THIS IS THE KEY FIX ---
        # Create a ReplayBuffer object for the training episodes.
        # The .fit() method needs this object, not a raw list.
        print(f"🔧 Creating replay buffer for BC model with {len(train_eps)} episodes...")
        bc_replay_buffer = create_infinite_replay_buffer(train_eps + val_eps)  # Combine train and validation episodes for BC training
        # ---------------------------

        # Train the BC model using n_steps
        bc_model.fit(
            bc_replay_buffer,  # <--- Pass the ReplayBuffer object here
            n_steps=100000,     # BC learns fast, 50k steps is often plenty
            n_steps_per_epoch=1000,
            show_progress=True
        )

        print("✅ Behavior Cloning model fitted successfully.")




    print("\n=== Initializing Enhanced Off-Policy Evaluation ===")
    from evaluator import BehaviorPolicyEstimator
    
    behavior_policy_estimator = BehaviorPolicyEstimator(n_actions=n_actions)
    print("🔧 Fitting behavior policy on training episodes...")
    behavior_policy_estimator.fit(train_eps)  # Fit on training data
    print("\n=== Behaviour-policy validation ===")
    # metrics_bp = BehaviorPolicyEstimator.evaluate_behaviour_model(
    #     behavior_policy_estimator,
    #     train_eps,                 # use a slice of training data
    #     n_actions,
    #     title="Train-split")


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
        # clinical_validation_results_save_dir = latest_log_dir_for_outputs
        save_location_message_suffix = f"in '{latest_log_dir_for_outputs}'"
    else:
        evaluation_results_save_dir = Path("evaluation_results")
        # clinical_validation_results_save_dir = Path("clinical_validation")
        save_location_message_suffix = "in their respective default directories ('evaluation_results/', 'clinical_validation/')"
        print(f"⚠️  Outputs will be saved {save_location_message_suffix} as the specific run directory was not identified.")


    print("📊 Creating combined training analysis...")
    from plot_training_result import plot_dual_axis_curves, plot_subplots_version
    
    plot_dual_axis_curves(str(latest_log_dir_for_outputs), 
                         save_path=str(latest_log_dir_for_outputs / "training_analysis_dual.png"))
    plot_subplots_version(str(latest_log_dir_for_outputs), 
                         save_path=str(latest_log_dir_for_outputs / "training_analysis_subplots.png"))

    print("\\n=== Evaluating model ===")
    # Use the new comprehensive evaluation framework
    from evaluator import CQLEvaluator

    from spo2_counterfactual import train_spo2_dynamics, rollout_cql_episode

    ##Ff spo2 plotten

    def stay_of(ep):
        idx = ep.episode_id if hasattr(ep, "episode_id") else None
        if idx is None:                  # if attribute absent, fall back to lookup
            idx = next(i for i, e in enumerate(mdp_dataset_full.episodes) if e is ep)
        return mdp_dataset_full.episode_metadata[idx]["stay_id"]

        # sets of stay IDs in each split
    train_stays = {stay_of(ep) for ep in train_eps}
    val_stays   = {stay_of(ep) for ep in val_eps}
    test_stays  = {stay_of(ep) for ep in test_eps}


    df_train = df[df["stay_id"].isin(train_stays)].copy()
    df_val   = df[df["stay_id"].isin(val_stays)].copy()
    df_test  = df[df["stay_id"].isin(test_stays)].copy()
    state_cols = data_dict["state_columns"]      # <- exists here
    print("number of features:", len(state_cols))
    for i, col in enumerate(state_cols[:20]):    # first 20 just to keep output short
        print(f"{i:2d}: {col}")

    spo2_idx = state_cols.index("spo2")          # correct column for Episode.observations
    print("SpO₂ sits at column", spo2_idx)
    rox_idx = state_cols.index("rox")          # correct column for Episode.observations
    print("ROX sits at column", rox_idx)
    print(f"{len(df_train):,} rows in train  |  "
      f"{len(df_val):,} rows in val  |  "
      f"{len(df_test):,} rows in test")
    
    from spo2_counterfactual import train_spo2_dynamics, rollout_cql_episode_spo2_only, id_to_midpoints, plot_actions_and_spo2, assemble_features


    



    n_actions = cql.action_size           # 12
    dyn_model = train_spo2_dynamics(
        train_eps=train_eps + val_eps,              # only training episodes
        n_actions=n_actions,
        spo2_idx=spo2_idx,
        rox_idx=rox_idx,  # Add ROX index for dynamics model
        obs_scaler=scaler,  # use the same scaler as CQL
    )

    # Prepare for counter-factual plotting
    eye     = np.eye(n_actions)
    out_dir = evaluation_results_save_dir / "counterfactual_plots"
    out_dir.mkdir(parents=True, exist_ok=True)


    for k, ep in enumerate(test_eps[:10], start=1):
        t = np.arange(len(ep))
        spo2_obs  = ep.observations[:, spo2_idx]
        clin_act  = ep.actions.reshape(-1).astype(int)

        # ---- clinician actions & parameters --------------------------
        clin_act = ep.actions.reshape(-1).astype(int)
        flow_c, fio2_c = id_to_midpoints(clin_act)

        # ---- CQL roll-out -------------------------------------------
        spo2_cf, cql_act = rollout_cql_episode_spo2_only(
                            ep, cql, dyn_model,
                            spo2_idx=spo2_idx,
                            n_actions=n_actions,
                            rox_idx=rox_idx)
        flow_q, fio2_q = id_to_midpoints(cql_act)

        # ---- clinician SpO₂ prediction for reference ----------------
        sa_clin = assemble_features(ep.observations, ep.actions,
                                    spo2_idx, rox_idx, np.eye(n_actions))
        spo2_pred_clin = np.r_[ep.observations[:2, spo2_idx],
                            dyn_model.predict(sa_clin)]

        # ---- plot SpO₂ + action IDs (existing helper) ---------------
        plot_actions_and_spo2(ep,
                            clin_act, cql_act,
                            ep.observations[:, spo2_idx],
                            spo2_pred_clin, spo2_cf,
                            n_actions=n_actions,
                            title=f"Val ep {k} — action IDs",
                            save=f"val_ep_{k:02d}_ids.png")
        
        # plot_actions_and_spo2(       t,
        # flow_c, fio2_c, flow_q, fio2_q,
        # spo2_obs, spo2_pred_clin, spo2_cf,
        # title = f"Validation Episode {k}",
        # save  = out_dir / f"val_ep{k:02d}_params_spo2.png")
    

        # ---- extra figure with physical parameters ------------------
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,3), sharex=True)
        ax1.step(t, flow_c, c='r', where='mid', label='Clinician')
        ax1.step(t, flow_q, c='b', where='mid', label='CQL')
        ax1.set_ylabel("Flow (L min⁻¹)"); ax1.legend(); ax1.grid(alpha=.3)

        ax2.step(t, fio2_c, c='r', where='mid', label='Clinician')
        ax2.step(t, fio2_q, c='b', where='mid', label='CQL')
        ax2.set_ylabel("FiO₂ (%)"); ax2.legend(); ax2.grid(alpha=.3)

        plt.suptitle(f"Val ep {k} — parameter mid-points")
        plt.tight_layout()
        plt.savefig(f"val_ep_{k:02d}_params.png", dpi=180)
        plt.close(fig)


        cql_evaluator_obj = CQLEvaluator(cql,  n_actions=n_actions,  behavior_policy_estimator=behavior_policy_estimator) # Renamed instance
    
    # scope_rl_metrics = cql_evaluator_obj.evaluate_ope_with_scope_rl(
    #     cql_model=cql,
    #     bc_model=bc_model,
    #     test_episodes=test_eps,
    #     gamma=args.gamma
    # )
    # print("DR =", CQLEvaluator.doubly_robust_value(test_eps, bc_model, cql))
    def filter_long_episodes(episodes, max_len=None, pct=75):
        """Return two lists: kept, dropped."""
        lengths = np.array([len(ep.actions) for ep in episodes])

        if max_len is None:
            max_len = int(np.percentile(lengths, pct))   # keep up to 95-th percentile

        kept    = [ep for ep, L in zip(episodes, lengths) if L <= max_len]
        dropped = [ep for ep, L in zip(episodes, lengths) if L >  max_len]
        print(f"[INFO] filtering episodes longer than {max_len} steps:"
            f"  kept {len(kept)}, dropped {len(dropped)}")
        return kept, dropped

    kept, dropped = filter_long_episodes(test_eps, max_len=None, pct=85)

    # scope_rl_metrics = cql_evaluator_obj.evaluate_ope_with_scope_rl(
    #     cql_model=cql,
    #     bc_model=bc_model,
    #     test_episodes=test_eps,
    #     gamma=args.gamma
    # )
    print("sndr =", CQLEvaluator.doubly_robust_value(kept, bc_model, cql))


    pt, lo, hi = CQLEvaluator.bootstrap_sndr_value(
        kept,
        behavior_algo = bc_model,
        eval_algo     = cql,
        gamma         = 0.99,
        seed          = 0,
        n_boot        = 1,
        alpha         = 0.05,
    )
    print(f"SN-DR = {pt:.4f}")
    print(f"95% CI = [{lo:.4f}, {hi:.4f}]")

    # doubly_robust_value_varlen = CQLEvaluator.doubly_robust_value_varlen(kept, bc_model, cql)
    # print(f"DR (Doubly Robust Value, variable length episodes) = {doubly_robust_value_varlen:.4f}")


    print("\n=== SCOPE-RL OPE Summary ===")
    # print(scope_rl_metrics)
    # val_losses = cql_evaluator_obj.evaluate_validation_losses(val_eps)
        # Vergelijk met training losses uit CSV
    # if latest_log_dir_for_outputs:
    #     loss_csv = latest_log_dir_for_outputs / "loss.csv"
    #     if loss_csv.exists():
    #         import pandas as pd
    #         train_losses = pd.read_csv(loss_csv)
    #         if not train_losses.empty:
    #             last_train_loss = train_losses.iloc[-1]
                
    #             print(f"\n📈 Training vs Validation Comparison:")
    #             print(f"   TD Loss:          Train {last_train_loss.get('td_loss', 0):.6f} | Val {val_losses['td_loss']:.6f}")
    #             print(f"   Conservative Loss: Train {last_train_loss.get('conservative_loss', 0):.6f} | Val {val_losses['conservative_loss']:.6f}")
    #             print(f"   Total Loss:       Train {last_train_loss.get('loss', 0):.6f} | Val {val_losses['total_loss']:.6f}")

    comprehensive_results = cql_evaluator_obj.evaluate_comprehensive(test_eps, save_dir=evaluation_results_save_dir)
    
    # Also keep basic metrics for backward compatibility
    cql_evaluator_obj.plot_trajectory_comparison(test_eps, 
    save_dir=Path("evaluation_results"), 
    max_episodes=3
)
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
        print(f"clinician_entropy = {pa['clinician_entropy']:.3f}")

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

    fqe_metrics2 = cql_evaluator_obj.evaluate_fqe2(train_episodes=train_eps, test_episodes=test_eps,
                                     n_steps = 150_000,
                                     n_boot  = 200)
    print("\n=== FQE summary2 ===")
    print(fqe_metrics2)

    fqe_metrics = cql_evaluator_obj.evaluate_fqe(episodes=train_eps + val_eps,
                                     n_steps = 150_000,
                                     n_boot  = 200)
    print("\n=== FQE summary ===")
    print(fqe_metrics)
    
    print(f"\\n📊 Academic visualizations and detailed report saved {save_location_message_suffix}")
    
    # Add clinical safety validation
    # print("\\n=== Clinical Safety Validation ===")
    # from clinical_validator import validate_clinical_safety
    
    # clinical_results = validate_clinical_safety(cql, test_eps, save_dir=clinical_validation_results_save_dir)
    
    # print("🏥 Clinical Safety Results:")
    # if 'parameter_safety' in clinical_results:
    #     ps = clinical_results['parameter_safety']
    #     print(f"   Parameter Safety Score: {ps['safety_score']:.3f}")
    #     print(f"   Total Safety Violations: {ps['total_violations']}")
    
    # if 'clinical_appropriateness' in clinical_results:
    #     ca = clinical_results['clinical_appropriateness']
    #     print(f"   Clinical Appropriateness: {ca['mean_appropriateness']:.3f}")
    #     print(f"   High Appropriateness Rate: {ca['high_appropriateness_rate']:.3f}")
    
    # if 'adverse_events' in clinical_results:
    #     ae = clinical_results['adverse_events']
    #     print(f"   Adverse Event Risk: {ae['overall_adverse_event_risk']:.3f}")
    #     print(f"   High Risk Decisions: {ae['high_risk_decisions']}")
    
    # print(f"🏥 Clinical validation report saved {save_location_message_suffix}")
    
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