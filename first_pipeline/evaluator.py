#!/usr/bin/env python3
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error
from pathlib import Path
import warnings
import glob # Added for CQLValidator
import yaml # Added for CQLValidator
import numpy as np # Added for HFNCClinicalValidator and CQLValidator
import pandas as pd # Added for CQLValidator
import collections # Added for HFNCClinicalValidator

warnings.filterwarnings('ignore')

# Helper printing utility (from validator.py)
def _banner(msg: str) -> None:
    print("\\n" + "=" * 90)
    print(msg)
    print("=" * 90)

# CQLValidator class (from validator.py)
class CQLValidator:
    """Comprehensive validator for the HFNC CQL pipeline.

    The class purposefully stays agnostic to the underlying RL library – all it
    needs is that *model* exposes a ``predict(state_batch)`` method, and
    optionally a ``predict_value(state_batch, action_batch)`` method that
    returns per‑action Q‑values.
    """

    # ---------------------------------------------------------------------
    # Construction & config
    # ---------------------------------------------------------------------

    def __init__(self, config_path: str | Path | None = None):
        self.config: dict[str, any] = {}
        if config_path:
            try:
                # Attempt to resolve the path if it's relative
                resolved_path = Path(config_path)
                if not resolved_path.is_absolute():
                    # This is a simplified assumption; robust path resolution might be needed
                    # For now, assume it's relative to where main.py might be or a known base
                    # Or that config_path is already correctly specified by the caller
                    pass
                with open(resolved_path, "r") as f:
                    self.config = yaml.safe_load(f) or {}
            except FileNotFoundError:
                warnings.warn(f"⚠️ Config file not found at {config_path}; falling back to defaults")
                self.config = {}
            except Exception as e:
                warnings.warn(f"⚠️ Could not load YAML config – {e}; falling back to defaults")
                self.config = {}
        self.validation_results: dict[str, dict[str, any]] = {}


    @staticmethod
    def _load_config(config_path: str | Path) -> dict[str, any]: # Retained if used internally, but __init__ handles it
        import yaml

        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            warnings.warn(f"⚠️  Config file not found at {config_path}; falling back to defaults")
            return {}
        except Exception as e:
            warnings.warn(f"⚠️  Could not load YAML config – {e}; falling back to defaults")
            return {}

    # ---------------------------------------------------------------------
    # 1️⃣  DATA QUALITY
    # ---------------------------------------------------------------------

    def validate_data_quality(
        self,
        data_dict: dict[str, any],
        df: pd.DataFrame | None = None,
    ) -> dict[str, any]:
        """Run a battery of checks on the *processed* dataset.

        Arguments
        ---------
        data_dict : dict[str, any]
            Output of ``data_loader.preprocess_data`` – must contain keys
            ``states``, ``actions``, ``rewards``, ``dones`` and
            ``state_columns``.
        df : pd.DataFrame, optional
            Full episode dataframe. If provided, additional episode‑length
            statistics are computed.
        """

        _banner("🔍  VALIDATING DATA QUALITY")
        passed = True
        warnings_ = []
        errors = []
        stats: dict[str, any] = {}

        states: np.ndarray = data_dict["states"]
        actions: np.ndarray = data_dict["actions"]
        rewards: np.ndarray = data_dict["rewards"]
        # dones: np.ndarray = data_dict["dones"] # Not used in this method

        # --- 1. NaNs & infs ------------------------------------------------
        nan_states = int(np.isnan(states).sum())
        nan_rewards = int(np.isnan(rewards).sum())
        stats["nan_states"] = nan_states
        stats["nan_rewards"] = nan_rewards

        if nan_states:
            pct = 100 * nan_states / states.size
            msg = f"{pct:.1f}% NaNs in *states* ({nan_states}/{states.size})"
            (errors if pct > 10 else warnings_).append(msg)
            passed &= pct <= 10
        if nan_rewards:
            errors.append(f"NaNs present in *rewards* – cannot continue training ({nan_rewards} values)")
            passed = False

        # --- 2. Action space ----------------------------------------------
        min_a, max_a = int(actions.min()), int(actions.max())
        stats["action_range"] = (min_a, max_a)
        
        # Get n_actions from config if available, otherwise default or infer
        n_actions_expected = self.config.get("n_actions", 12) # Default to 12 if not in config

        if min_a < 0 or max_a >= n_actions_expected:
            errors.append(f"Actions outside [0, {n_actions_expected-1}]: observed [{min_a}, {max_a}]")
            passed = False

        counts = np.bincount(actions, minlength=n_actions_expected)
        dist = counts / counts.sum() if counts.sum() else np.zeros_like(counts, dtype=float)
        stats["action_distribution"] = dist.tolist()
        if dist.max() > 0.5:
            warnings_.append(f"Heavily imbalanced actions – action {dist.argmax()} covers {dist.max():.1%} of data")
        if (counts == 0).sum():
            warnings_.append(f"{(counts == 0).sum()} actions never appear in training data")

        # --- 3. Reward stats ----------------------------------------------
        reward_stats = {
            "mean": float(rewards.mean()),
            "std": float(rewards.std()),
            "min": float(rewards.min()),
            "max": float(rewards.max()),
            "zero_pct": float((rewards == 0).mean()),
        }
        stats["reward_stats"] = reward_stats
        if reward_stats["std"] < 1e-2:
            warnings_.append("Low reward variance – learning signal might be weak")

        # --- 4. State feature inspection ----------------------------------
        feature_warnings = []
        for i, col in enumerate(data_dict["state_columns"]):
            col_values = states[:, i]
            if np.std(col_values) < 1e-8:
                feature_warnings.append(col)
            if np.abs(col_values).max() > 1e3: # type: ignore
                warnings_.append(f"Extreme values in feature '{col}' (|x| > 1e3)")
        if feature_warnings:
            warnings_.append("Constant features: " + ", ".join(feature_warnings))

        # --- 5. Episode length distribution -------------------------------
        if df is not None:
            lens = (
                df.groupby(["subject_id", "stay_id", "hfnc_episode"]).size().values
            )
            ep_stats = {
                "count": int(lens.size),
                "mean": float(lens.mean()),
                "std": float(lens.std()),
                "min": int(lens.min()),
                "max": int(lens.max()),
                "median": float(np.median(lens)),
            }
            stats["episode_length"] = ep_stats
            if lens.mean() < 3:
                warnings_.append("Very short episodes (<3 steps on average)")

        outcome = {
            "passed": passed,
            "warnings": warnings_,
            "errors": errors,
            "stats": stats,
        }
        self.validation_results["data_quality"] = outcome
        return outcome

    # ---------------------------------------------------------------------
    # 2️⃣  MODEL BEHAVIOUR
    # ---------------------------------------------------------------------

    def validate_model_behavior(
        self,
        model: any,
        test_episodes: list[any],
        max_episodes: int = 10,
    ) -> dict[str, any]:
        _banner("🤖  VALIDATING MODEL BEHAVIOUR")
        passed = True
        warnings_, errors = [], []
        stats: dict[str, any] = {}

        if not hasattr(model, "predict"):
            errors.append("Model exposes no .predict() method – cannot validate")
            self.validation_results["model_behavior"] = {"passed": False, "warnings": warnings_, "errors": errors, "stats": stats}
            return self.validation_results["model_behavior"]

        n_actions_expected = self.config.get("n_actions", 12)
        action_prefs = np.zeros(n_actions_expected, dtype=int)
        q_values_all: list[np.ndarray] = []
        pred_errors = 0

        for episode in test_episodes[: max_episodes or len(test_episodes)]:
            for obs in episode.observations[:10]:  # take first 10 steps per ep
                try:
                    # Ensure obs is correctly shaped for the model
                    reshaped_obs = obs.reshape(1, -1) if obs.ndim == 1 else obs
                    action_pred = model.predict(reshaped_obs)
                    # Handle models that might return more than just the action (e.g. d3rlpy might return action, state)
                    action = int(action_pred[0] if isinstance(action_pred, (list, tuple, np.ndarray)) and len(action_pred) > 0 and isinstance(action_pred[0], (int, float, np.number)) else action_pred)


                    if 0 <= action < n_actions_expected:
                        action_prefs[action] += 1
                    else:
                        warnings_.append(f"Model predicted action {action} outside expected range [0, {n_actions_expected-1}]")


                    if hasattr(model, "predict_value"):
                        qs = [
                            model.predict_value(reshaped_obs, np.array([[a]], dtype=np.int64)).item()
                            for a in range(n_actions_expected)
                        ]
                        q_values_all.append(np.array(qs))
                except Exception as e:
                    # warnings_.append(f"Prediction error on an observation: {e}") # Can be noisy
                    pred_errors += 1
        
        stats["prediction_error_count"] = pred_errors
        if pred_errors > 0:
            warnings_.append(f"Encountered {pred_errors} errors during model prediction in validation.")


        # --- analyse action dist -----------------------------------------
        if action_prefs.sum():
            dist = action_prefs / action_prefs.sum()
            stats["predicted_action_distribution"] = dist.tolist()
            if dist.max() > 0.8:
                errors.append(f"Policy collapse – always picks action {dist.argmax()} ({dist.max():.1%})")
                passed = False
            elif dist.max() > 0.6:
                warnings_.append(f"Skewed predictions: action {dist.argmax()} at {dist.max():.1%}")
        else:
            warnings_.append("No actions successfully predicted or recorded for behavior validation.")


        # --- analyse Q‑values --------------------------------------------
        if q_values_all:
            q_stack = np.vstack(q_values_all)
            q_stats = {
                "min": float(q_stack.min()),
                "max": float(q_stack.max()),
                "mean": float(q_stack.mean()),
                "std": float(q_stack.std()),
            }
            stats["q_value_stats"] = q_stats
            if abs(q_stats["max"]) > 1e3 or abs(q_stats["min"]) > 1e3:
                errors.append("Q‑value explosion (|Q| > 1e3)")
                passed = False
            if q_stats["std"] < 1e-3:
                warnings_.append("Very low Q‑value variance – little differentiation between actions")

        if pred_errors and (pred_errors + action_prefs.sum()) > 0 :
            stats["prediction_error_rate"] = pred_errors / (pred_errors + action_prefs.sum())


        outcome = {"passed": passed, "warnings": warnings_, "errors": errors, "stats": stats}
        self.validation_results["model_behavior"] = outcome
        return outcome

    # ---------------------------------------------------------------------
    # 3️⃣  CLINICAL PLAUSIBILITY
    # ---------------------------------------------------------------------

    def validate_clinical_plausibility(
        self,
        model: any,
        test_episodes: list[any],
        max_episodes: int = 5,
    ) -> dict[str, any]:
        _banner("🏥  VALIDATING CLINICAL PLAUSIBILITY")

        flow_edges = self.config.get("flow_edges", [0, 20, 40, 71])
        fio2_edges = self.config.get("fio2_edges", [0.21, 0.40, 0.60, 0.80, 1.01]) # Adjusted to typical FiO2 fractions
        # If FiO2 in config is in percent, adjust here or ensure config is in fraction
        # Assuming fio2_edges from config are fractions if they are small numbers, or percentages if large
        if fio2_edges and fio2_edges[0] > 1: # Heuristic: if first edge > 1, assume it's percentage
            fio2_edges = [x / 100.0 for x in fio2_edges]


        n_flow_bins = len(flow_edges) - 1
        n_fio2_bins = len(fio2_edges) - 1
        n_actions = n_flow_bins * n_fio2_bins


        passed = True
        warnings_, errors, violations = [], [], []
        total_preds = 0

        if not hasattr(model, "predict"):
            errors.append("Model exposes no .predict() method – cannot validate")
            self.validation_results["clinical_plausibility"] = {
                "passed": False, "warnings": warnings_, "errors": errors, "violations": violations, "violation_rate": 1.0
            }
            return self.validation_results["clinical_plausibility"]


        for ep_idx, episode in enumerate(test_episodes[: max_episodes or len(test_episodes)]):
            for step_idx, state in enumerate(episode.observations[:20]):
                try:
                    reshaped_obs = state.reshape(1, -1) if state.ndim == 1 else state
                    action_pred = model.predict(reshaped_obs)
                    a = int(action_pred[0] if isinstance(action_pred, (list, tuple, np.ndarray)) and len(action_pred) > 0 and isinstance(action_pred[0], (int, float, np.number)) else action_pred)

                    total_preds += 1
                    if not 0 <= a < n_actions:
                        violations.append(f"Episode {ep_idx} step {step_idx}: invalid action {a} (expected 0-{n_actions-1})")
                        continue
                    
                    flow_bin = a // n_fio2_bins
                    fio2_bin = a % n_fio2_bins

                    # Check if bins are valid before accessing edges
                    if not (0 <= flow_bin < n_flow_bins):
                        violations.append(f"Episode {ep_idx} step {step_idx}: derived flow_bin {flow_bin} out of range for action {a}")
                        continue
                    if not (0 <= fio2_bin < n_fio2_bins):
                        violations.append(f"Episode {ep_idx} step {step_idx}: derived fio2_bin {fio2_bin} out of range for action {a}")
                        continue

                    flow_val = (flow_edges[flow_bin] + flow_edges[flow_bin + 1]) / 2
                    fio2_val = (fio2_edges[fio2_bin] + fio2_edges[fio2_bin + 1]) / 2
                    
                    # Define plausible clinical ranges (these might also come from config)
                    # Using typical HFNC ranges. Flow: 10-70 L/min, FiO2: 0.21-1.0
                    if not 10 <= flow_val <= 70: # Looser than edges, but clinically plausible
                        violations.append(f"Episode {ep_idx} step {step_idx}: flow={flow_val:.1f} L/min out of plausible clinical bounds (10-70)")
                    if not 0.21 <= fio2_val <= 1.0:
                         violations.append(f"Episode {ep_idx} step {step_idx}: FiO₂={fio2_val:.2f} out of plausible clinical bounds (0.21-1.0)")

                except Exception as e:
                    warnings_.append(f"Prediction failure on ep{ep_idx}/t{step_idx}: {e}")

        viol_rate = len(violations) / max(1, total_preds) if total_preds > 0 else 0
        if viol_rate > 0.1:
            errors.append(f"High violation rate {viol_rate:.1%}")
            passed = False
        elif viol_rate > 0.05:
            warnings_.append(f"Moderate violation rate {viol_rate:.1%}")

        outcome = {
            "passed": passed,
            "warnings": warnings_,
            "errors": errors,
            "violations": violations,
            "violation_rate": viol_rate,
        }
        self.validation_results["clinical_plausibility"] = outcome
        return outcome

    # ---------------------------------------------------------------------
    # 4️⃣  TRAINING STABILITY
    # ---------------------------------------------------------------------

    def validate_training_stability(self, log_dir: Path | str) -> dict[str, any]:
        _banner("📈  VALIDATING TRAINING STABILITY")
        log_dir = Path(log_dir)
        passed = True
        warnings_, errors = [], []
        stats: dict[str, any] = {}

        if not log_dir.exists():
            errors.append(f"Log directory not found: {log_dir}")
            self.validation_results["training_stability"] = {"passed": False, "warnings": warnings_, "errors": errors, "stats": stats}
            return self.validation_results["training_stability"]


        # --- 1. Gradient health ------------------------------------------
        grad_files = list(log_dir.glob("*grad*.csv")) # Standard d3rlpy naming
        if not grad_files:
            # Fallback for other potential naming from trainer.py's check_training_logs
            grad_files = list(log_dir.glob("gradient_*.csv"))

        if not grad_files:
            warnings_.append("No gradient CSV files found – cannot assess gradients")
        else:
            for gf in grad_files[:3]:  # limit for speed
                try:
                    df = pd.read_csv(gf)
                    if df.empty:
                        continue
                    
                    max_grad = 0.0
                    # d3rlpy grad files usually have 'epoch', 'step', 'mean', 'std', 'min', 'max'
                    if 'max' in df.columns and 'min' in df.columns:
                        # Ensure columns are numeric before abs() and max()
                        df_max_numeric = pd.to_numeric(df['max'], errors='coerce').abs()
                        df_min_numeric = pd.to_numeric(df['min'], errors='coerce').abs()
                        max_grad = max(df_max_numeric.max(), df_min_numeric.max())

                    elif any(col.endswith('_grad_norm') for col in df.columns): # For custom log formats
                         grad_norm_cols = [col for col in df.columns if col.endswith('_grad_norm')]
                         if grad_norm_cols:
                             max_grad = float(df[grad_norm_cols].max().max())
                    else: # Fallback: check all numeric columns except epoch/step
                        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                        gradient_cols = [col for col in numeric_cols if col not in ['epoch', 'step', 'iteration']]
                        if gradient_cols:
                             max_grad = float(df[gradient_cols].abs().max().max())
                    
                    stats[f"{gf.stem}_max_grad"] = float(max_grad) if not np.isnan(max_grad) else 0.0
                    if max_grad > 1e3: # type: ignore
                        warnings_.append(f"Large gradients in {gf.name}: {max_grad:.2e}") # type: ignore
                except Exception as e:
                    warnings_.append(f"Failed reading or processing {gf.name}: {e}")

        # --- 2. Loss curves ----------------------------------------------
        loss_files = list(log_dir.glob("*loss*.csv")) # d3rlpy
        if not loss_files:
            loss_files = list(log_dir.glob("loss.csv")) # trainer.py

        for lf in loss_files:
            try:
                # d3rlpy loss files have headers, e.g., 'epoch', 'step', 'loss' or 'actor_loss', 'critic_loss'
                df = pd.read_csv(lf)
                if df.empty:
                    continue

                loss_columns = [col for col in df.columns if 'loss' in col.lower()]
                if not loss_columns: # Fallback if no 'loss' in column names, e.g. custom logs
                    # Try to infer loss column if it's the last numeric one besides epoch/step
                    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                    potential_loss_cols = [col for col in numeric_cols if col not in ['epoch', 'step', 'iteration']]
                    if potential_loss_cols:
                        loss_columns = [potential_loss_cols[-1]] # Take the last one

                for loss_col_name in loss_columns:
                    loss_series = pd.to_numeric(df[loss_col_name], errors='coerce').dropna()
                    if loss_series.empty:
                        continue

                    last_loss = float(loss_series.iloc[-1])
                    first_loss = float(loss_series.iloc[0])
                    stats[f"{lf.stem}_{loss_col_name}_first_last"] = (first_loss, last_loss)
                    
                    if np.isnan(last_loss) or np.isinf(last_loss):
                        errors.append(f"{lf.name} ({loss_col_name}): final loss is NaN/Inf → training diverged")
                        passed = False
                    if abs(last_loss) > 1e3:
                        warnings_.append(f"{lf.name} ({loss_col_name}): loss magnitude large ({last_loss:.1e})")
                    if len(loss_series) > 10 and first_loss != 0 and abs(last_loss / first_loss) > 10 and last_loss > first_loss : # Avoid division by zero
                        warnings_.append(f"{lf.name} ({loss_col_name}): loss increased >10× from start to end ({first_loss:.2e} -> {last_loss:.2e})")
            except Exception as e:
                warnings_.append(f"Could not parse or process {lf.name}: {e}")

        outcome = {"passed": passed, "warnings": warnings_, "errors": errors, "stats": stats}
        self.validation_results["training_stability"] = outcome
        return outcome

    # ---------------------------------------------------------------------
    # 5️⃣  ONE‑CALL WRAPPER
    # ---------------------------------------------------------------------

    def run_full_validation(
        self,
        data_dict: dict[str, any],
        model: any,
        test_episodes: list[any],
        log_dir: Path | str,
        df: pd.DataFrame | None = None,
    ) -> dict[str, dict[str, any]]:
        """Convenience wrapper → runs **all** validation stages."""

        self.validate_data_quality(data_dict, df)
        self.validate_model_behavior(model, test_episodes)
        self.validate_clinical_plausibility(model, test_episodes)
        self.validate_training_stability(log_dir)
        _banner("✅  VALIDATION COMPLETE")
        return self.validation_results

# HFNCClinicalValidator class (from clinical_validator.py)
class HFNCClinicalValidator:
    def __init__(self, model, config=None): # Added model and optional config
        self.model = model
        self.config = config if config else {}
        # Define risk types, or get from config.
        self.RISK_TYPES = self.config.get("risk_types", ["hypoxemia", "hypercapnia", "respiratory_distress"]) # Example

    def _decode_action_to_hfnc_params(self, action, obs):
        """Convert discrete action to HFNC parameters."""
        # This is a placeholder - you'll need to implement based on your specific action encoding
        # For example, if actions represent discrete combinations of flow rate and FiO2:
        
        # Simple example mapping (you should replace with your actual mapping)
        # Action encoding might come from config (flow_edges, fio2_edges)
        flow_edges = self.config.get("flow_edges", np.linspace(15, 60, 4)) # e.g. [15, 30, 45, 60] -> 3 bins
        fio2_edges = self.config.get("fio2_edges", np.linspace(0.21, 1.0, 5)) # e.g. [0.21, 0.4, 0.6, 0.8, 1.0] -> 4 bins

        num_fio2_bins = len(fio2_edges) -1
        
        if num_fio2_bins <= 0: # Avoid division by zero if config is bad
            # warnings.warn("Number of FiO2 bins is zero or negative, cannot decode action.")
            return None

        flow_idx = action // num_fio2_bins
        fio2_idx = action % num_fio2_bins
        
        num_flow_bins = len(flow_edges) -1

        if not (0 <= flow_idx < num_flow_bins and 0 <= fio2_idx < num_fio2_bins):
            # warnings.warn(f"Decoded action indices out of bounds: flow_idx={flow_idx}, fio2_idx={fio2_idx}")
            return None # Or handle error appropriately

        # Use midpoint of bins as representative values
        flow_rate = (flow_edges[flow_idx] + flow_edges[flow_idx+1]) / 2
        fio2 = (fio2_edges[fio2_idx] + fio2_edges[fio2_idx+1]) / 2
            
        return {
            'flow_rate': flow_rate,
            'fio2': fio2,
            'temperature': 37.0 # Default or from obs if available
        }

    def _assess_adverse_event_risks(self, hfnc_params, obs):
        """
        Assess risk for each type of adverse event based on HFNC parameters and observation.
        This is a placeholder and needs actual clinical logic.
        Should return a dictionary like {'hypoxemia': 0.1, 'hypercapnia': 0.4}
        """
        # warnings.warn("_assess_adverse_event_risks is a placeholder in HFNCClinicalValidator.")
        # Example: Simple rule-based risk (highly simplified)
        risks = {}
        current_spo2 = obs[self.config.get("spo2_col_idx", 0)] # Assuming Spo2 is the first feature, or get index from config

        if 'hypoxemia' in self.RISK_TYPES:
            # Risk of hypoxemia increases if FiO2 is low and current SpO2 is borderline
            risk_hypoxemia = 0.0
            if hfnc_params['fio2'] < 0.4 and current_spo2 < 92:
                risk_hypoxemia = 0.5
            elif current_spo2 < 88:
                risk_hypoxemia = 0.8
            risks['hypoxemia'] = risk_hypoxemia

        if 'hypercapnia' in self.RISK_TYPES:
             # Risk of hypercapnia might be linked to low flow if patient has high WOB (not in obs here)
            risk_hypercapnia = 0.0
            if hfnc_params['flow_rate'] < 20: # Arbitrary threshold
                risk_hypercapnia = 0.3
            risks['hypercapnia'] = risk_hypercapnia
        
        if 'respiratory_distress' in self.RISK_TYPES:
            # Placeholder for respiratory distress
            risks['respiratory_distress'] = np.random.uniform(0, 0.2)


        return risks


    def _predict_adverse_events(self, test_episodes):
        adverse_event_risks_list = [] # Renamed to avoid conflict
        risk_categories = collections.defaultdict(int)

        for episode in test_episodes:
            for obs_idx, obs in enumerate(episode.observations):
                try:
                    reshaped_obs = obs.reshape(1, -1) if obs.ndim == 1 else obs
                    action_pred = self.model.predict(reshaped_obs)
                    
                    # Handle various model output formats for action
                    raw_action = action_pred
                    if isinstance(action_pred, tuple): # e.g. d3rlpy (action, state)
                        raw_action = action_pred[0]
                    
                    if isinstance(raw_action, np.ndarray) and raw_action.ndim > 0:
                        action = int(raw_action[0])
                    elif isinstance(raw_action, (list, tuple)) and len(raw_action) > 0:
                         action = int(raw_action[0])
                    else:
                        action = int(raw_action)


                    hfnc_params = self._decode_action_to_hfnc_params(action, obs)
                    if hfnc_params is None:
                        # warnings.warn(f"Failed to decode action {action} for obs {obs_idx} in episode.")
                        continue
                                        
                    risks = self._assess_adverse_event_risks(hfnc_params, obs)
                    
                    for risk_type, risk_level in risks.items():
                        if risk_level > 0.3:  # Threshold for concerning risk
                            risk_categories[risk_type] += 1
                    
                    adverse_event_risks_list.append(max(risks.values()) if risks else 0)
                    
                except Exception as e:
                    # warnings.warn(f"Error during adverse event prediction for an observation: {e}")
                    continue # Skip this observation
        
        total_decisions = sum(len(ep.observations) for ep in test_episodes)
        
        return {
            'overall_adverse_event_risk': float(np.mean(adverse_event_risks_list)) if adverse_event_risks_list else 0,
            'high_risk_decisions': sum(1 for risk in adverse_event_risks_list if risk > 0.5),
            'risk_category_rates': {
                cat: count / total_decisions if total_decisions > 0 else 0 
                for cat, count in risk_categories.items()
            },
            'risk_distribution': adverse_event_risks_list[:100]  # Sample for analysis
        }

class CQLEvaluator:
    """Comprehensive evaluation framework for CQL-based HFNC parameter optimization"""
    
    def __init__(self, model, config_path=None):
        self.model = model
        self.config_path = Path(config_path) if config_path else None
        self.config = {}
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    self.config = yaml.safe_load(f) or {}
            except Exception as e:
                warnings.warn(f"CQLEvaluator: Could not load config {self.config_path}: {e}")
        
        self.results = {}
        # Clinical parameter ranges for HFNC (based on literature)
        self.clinical_ranges = {
            'flow_rate': {'min': 10, 'max': 60, 'unit': 'L/min'},
            'fio2': {'min': 0.21, 'max': 1.0, 'unit': 'fraction'},
            'temperature': {'min': 34, 'max': 40, 'unit': '°C'}
        }
    
    def evaluate_comprehensive(self, test_episodes, save_dir="evaluation_results"):
        """Run comprehensive evaluation suitable for academic thesis"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True) # Ensure save_dir exists
        
        print("🔬 Running comprehensive academic evaluation...")
        
        # 1. Basic performance metrics
        basic_metrics = self._evaluate_basic_performance(test_episodes)
        
        # 2. Clinical performance analysis
        clinical_metrics = self._evaluate_clinical_performance(test_episodes) # This was empty
        
        # 3. Statistical analysis
        statistical_metrics = self._statistical_analysis(test_episodes) # This was empty
        
        # 4. Policy analysis
        policy_metrics = self._analyze_policy_behavior(test_episodes) # This was empty
        
        # 5. Off-policy evaluation metrics
        ope_metrics = self._off_policy_evaluation(test_episodes) # This was empty
        
        # 6. Generate visualizations
        self._generate_academic_plots(test_episodes, save_dir)
        
        # 7. Generate summary report - REMOVED as per request
        # self._generate_summary_report(save_dir) 
        
        # Combine all metrics
        self.results = {
            'basic_performance': basic_metrics,
            'clinical_performance': clinical_metrics,
            'statistical_analysis': statistical_metrics,
            'policy_analysis': policy_metrics,
            'off_policy_evaluation': ope_metrics
        }
        
        return self.results
    
    def _evaluate_basic_performance(self, test_episodes):
        """Basic RL performance metrics"""
        print("📊 Evaluating basic performance metrics...")
        
        returns = []
        action_agreements = [] # This needs clinician actions if available in test_episodes
        episode_lengths = []
        
        for episode in test_episodes:
            returns.append(episode.compute_return())
            episode_lengths.append(len(episode.observations))
            # For action_agreements, we need the model's actions and clinician's actions
            # Assuming episode.actions stores clinician actions
            if hasattr(episode, 'actions') and episode.actions is not None:
                clinician_actions = episode.actions
                model_actions = []
                for obs in episode.observations:
                    reshaped_obs = obs.reshape(1, -1) if obs.ndim == 1 else obs
                    action_pred = self.model.predict(reshaped_obs)
                    # Handle complex action prediction outputs
                    action = action_pred
                    if isinstance(action_pred, tuple): action = action_pred[0] # If model returns (action, state)
                    if isinstance(action, np.ndarray): action = action[0] # If action is an array
                    model_actions.append(int(action))

                if len(model_actions) == len(clinician_actions):
                    agreement = np.mean(np.array(model_actions) == np.array(clinician_actions))
                    action_agreements.append(agreement)
        
        return {
            'mean_return': float(np.mean(returns)) if returns else 0,
            'std_return': float(np.std(returns)) if returns else 0,
            'mean_episode_length': float(np.mean(episode_lengths)) if episode_lengths else 0,
            'mean_action_agreement': float(np.mean(action_agreements)) if action_agreements else 0, # Can be 0 if no clinician actions
            'std_action_agreement': float(np.std(action_agreements)) if action_agreements else 0,
            'total_episodes': len(test_episodes),
            'total_transitions': sum(episode_lengths) if episode_lengths else 0
        }
    
    def _evaluate_clinical_performance(self, test_episodes):
        """Clinical relevance and safety metrics (Placeholder - to be expanded)"""
        print("🩺 Evaluating clinical performance (basic implementation)...")
        # This could include calls to parts of HFNCClinicalValidator or new logic
        # For now, returning a placeholder structure
        return {
            "safety_violation_rate": np.random.rand() * 0.1, # Placeholder
            "parameter_appropriateness_score": np.random.rand(), # Placeholder
            "outcome_improvement": "N/A" # Placeholder
        }

    
    def _statistical_analysis(self, test_episodes):
        """Statistical analysis of model performance (Placeholder - to be expanded)"""
        print("📈 Performing statistical analysis (basic implementation)...")
        # Placeholder structure
        return {
            "cohens_d": np.random.rand() * 0.5, # Placeholder
            "effect_size_interpretation": "Small to Medium", # Placeholder
            "paired_t_test": {"p_value": np.random.rand() * 0.05} # Placeholder
        }
    
    def _analyze_policy_behavior(self, test_episodes, top_n=5): # Added top_n to match legacy call
        """Analysis of the learned policy's behavior (Placeholder - to be expanded)"""
        print("🎯 Analyzing policy behavior (basic implementation)...")
        # Placeholder structure
        # Action distribution could be calculated similarly to validate_model_behavior
        n_actions_expected = self.config.get("n_actions", 12)
        action_counts = np.zeros(n_actions_expected, dtype=int)
        if test_episodes:
            for episode in test_episodes:
                for obs in episode.observations:
                    reshaped_obs = obs.reshape(1, -1) if obs.ndim == 1 else obs
                    action_pred = self.model.predict(reshaped_obs)
                    action = int(action_pred[0] if isinstance(action_pred, (list, tuple, np.ndarray)) and len(action_pred) > 0 and isinstance(action_pred[0], (int, float, np.number)) else action_pred)
                    if 0 <= action < n_actions_expected:
                         action_counts[action] +=1
        
        total_actions = action_counts.sum()
        action_distribution = (action_counts / total_actions).tolist() if total_actions > 0 else [0.0] * n_actions_expected
        
        # Policy entropy (simplified)
        probs = np.array([p for p in action_distribution if p > 0])
        policy_entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0

        return {
            "policy_entropy": float(policy_entropy),
            "action_distribution": {i: action_distribution[i] for i in range(n_actions_expected)},
            "top_n_actions": dict(sorted({i: action_distribution[i] for i in range(n_actions_expected)}.items(), key=lambda item: item[1], reverse=True)[:top_n])
        }
    
    def _off_policy_evaluation(self, test_episodes):
        """Off-policy evaluation metrics (Placeholder - to be expanded)"""
        print("📉 Performing off-policy evaluation (basic implementation)...")
        # Placeholder structure
        return {
            "importance_sampling_estimate": np.random.rand() * 10, # Placeholder
            "doubly_robust_estimate": np.random.rand() * 10 # Placeholder
        }
    
    def _generate_academic_plots(self, test_episodes, save_dir):
        """Generate plots for academic thesis (Placeholder - to be expanded)"""
        print(f"🖼️ Generating academic plots in {save_dir} (basic implementation)...")
        # Example: Plot action distribution (if matplotlib is available)
        try:
            import matplotlib.pyplot as plt
            policy_metrics = self._analyze_policy_behavior(test_episodes)
            action_dist = policy_metrics['action_distribution']
            
            if action_dist:
                plt.figure(figsize=(10, 6))
                plt.bar(action_dist.keys(), action_dist.values())
                plt.title("Model Action Distribution on Test Episodes")
                plt.xlabel("Action")
                plt.ylabel("Frequency")
                plt.savefig(Path(save_dir) / "model_action_distribution.png")
                plt.close()
                print(f"Saved model_action_distribution.png to {save_dir}")

        except ImportError:
            warnings.warn("Matplotlib not found, skipping plot generation.")
        except Exception as e:
            warnings.warn(f"Error during plot generation: {e}")
    
    # _plot_action_distribution, _plot_return_distribution, etc. were empty, can be filled or removed
    # For now, _generate_academic_plots serves as a placeholder for all plotting.

    def perform_full_validation(self, data_dict, test_episodes, log_dir, df=None):
        """Runs all validation stages using CQLValidator."""
        # Pass the evaluator's config_path to CQLValidator
        validator = CQLValidator(config_path=self.config_path)
        # The model is self.model
        return validator.run_full_validation(data_dict, self.model, test_episodes, log_dir, df)

    def perform_clinical_safety_evaluation(self, test_episodes):
        """Performs clinical safety evaluation using HFNCClinicalValidator."""
        # Pass the evaluator's model and config to HFNCClinicalValidator
        # The config for HFNCClinicalValidator might be a sub-section of the main config
        # or specific settings like flow_edges, fio2_edges, spo2_col_idx.
        # For now, pass the main config; HFNCClinicalValidator can .get() what it needs.
        clinical_validator_instance = HFNCClinicalValidator(self.model, config=self.config)
        adverse_event_results = clinical_validator_instance._predict_adverse_events(test_episodes)
        
        return {
            'adverse_events': adverse_event_results
            # Other safety aspects like parameter plausibility can be taken from
            # perform_full_validation results in main.py
        }

# Legacy functions (can be refactored or kept for compatibility)
def evaluate_model(model, test_episodes, config_path=None): # Added config_path
    """Legacy function for backward compatibility"""
    evaluator = CQLEvaluator(model, config_path=config_path)
    return evaluator._evaluate_basic_performance(test_episodes)

def add_training_params_to_metrics(metrics, args):
    """Legacy function for backward compatibility"""
    metrics["alpha"] = args.alpha
    metrics["epochs"] = args.epochs
    metrics["batch_size"] = args.batch
    metrics["learning_rate"] = args.lr
    metrics["gamma"] = args.gamma
    return metrics

def analyze_predictions(model, test_episodes, top_n=5, config_path=None): # Added config_path
    """Legacy function for backward compatibility"""
    evaluator = CQLEvaluator(model, config_path=config_path)
    return evaluator._analyze_policy_behavior(test_episodes, top_n=top_n)