#!/usr/bin/env python3
"""
validator.py  --  Comprehensive validation utilities for the HFNC CQL pipeline

Add this file to the **first_pipeline** package (alongside `model.py`, `trainer.py`, ...).
Import the class `CQLValidator` from this module inside `main.py` (or any other
script) to run a suite of sanity‑checks covering data, model behaviour,
clinical plausibility and training stability.

Example (short):
    from validator import CQLValidator
    validator = CQLValidator("config.yaml")
    validator.run_full_validation(data_dict, cql_model, test_eps, latest_log_dir, df)

The methods return detailed dictionaries and print human‑readable reports so
that you can decide whether the current experiment is trustworthy before moving
on to prospective evaluation or deployment.
"""

from __future__ import annotations

import glob
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
#  Helper printing utilities
# ──────────────────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    print("\n" + "=" * 90)
    print(msg)
    print("=" * 90)


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
        self.config: Dict[str, Any] = {}
        if config_path:
            self.config = self._load_config(config_path)
        self.validation_results: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _load_config(config_path: str | Path) -> Dict[str, Any]:
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
        data_dict: Dict[str, Any],
        df: pd.DataFrame | None = None,
    ) -> Dict[str, Any]:
        """Run a battery of checks on the *processed* dataset.

        Arguments
        ---------
        data_dict : Dict[str, Any]
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
        stats: Dict[str, Any] = {}

        states: np.ndarray = data_dict["states"]
        actions: np.ndarray = data_dict["actions"]
        rewards: np.ndarray = data_dict["rewards"]
        dones: np.ndarray = data_dict["dones"]

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
        if min_a < 0 or max_a >= 12:
            errors.append(f"Actions outside [0, 11]: observed [{min_a}, {max_a}]")
            passed = False

        counts = np.bincount(actions, minlength=12)
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
            if np.abs(col_values).max() > 1e3:
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
        model: Any,
        test_episodes: List[Any],
        max_episodes: int = 10,
    ) -> Dict[str, Any]:
        _banner("🤖  VALIDATING MODEL BEHAVIOUR")
        passed = True
        warnings_, errors = [], []
        stats: Dict[str, Any] = {}

        if not hasattr(model, "predict"):
            errors.append("Model exposes no .predict() method – cannot validate")
            return {"passed": False, "warnings": warnings_, "errors": errors, "stats": stats}

        action_prefs = np.zeros(12, dtype=int)
        q_values_all: List[np.ndarray] = []
        pred_errors = 0

        for episode in test_episodes[: max_episodes or len(test_episodes)]:
            for obs in episode.observations[:10]:  # take first 10 steps per ep
                try:
                    action = int(model.predict(obs.reshape(1, -1))[0])
                    if 0 <= action < 12:
                        action_prefs[action] += 1
                    if hasattr(model, "predict_value"):
                        qs = [
                            model.predict_value(obs.reshape(1, -1), np.array([[a]], dtype=np.int64)).item()
                            for a in range(12)
                        ]
                        q_values_all.append(np.array(qs))
                except Exception:
                    pred_errors += 1

        # --- analyse action dist -----------------------------------------
        if action_prefs.sum():
            dist = action_prefs / action_prefs.sum()
            stats["predicted_action_distribution"] = dist.tolist()
            if dist.max() > 0.8:
                errors.append(f"Policy collapse – always picks action {dist.argmax()} ({dist.max():.1%})")
                passed = False
            elif dist.max() > 0.6:
                warnings_.append(f"Skewed predictions: action {dist.argmax()} at {dist.max():.1%}")

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

        if pred_errors:
            warnings_.append(f"Prediction errors on {pred_errors} state(s)")
            stats["prediction_error_rate"] = pred_errors / (pred_errors + action_prefs.sum())

        outcome = {"passed": passed, "warnings": warnings_, "errors": errors, "stats": stats}
        self.validation_results["model_behavior"] = outcome
        return outcome

    # ---------------------------------------------------------------------
    # 3️⃣  CLINICAL PLAUSIBILITY
    # ---------------------------------------------------------------------

    def validate_clinical_plausibility(
        self,
        model: Any,
        test_episodes: List[Any],
        max_episodes: int = 5,
    ) -> Dict[str, Any]:
        _banner("🏥  VALIDATING CLINICAL PLAUSIBILITY")

        flow_edges = self.config.get("flow_edges", [0, 20, 40, 71])
        fio2_edges = self.config.get("fio2_edges", [21, 40, 60, 80, 101])
        n_actions = (len(flow_edges) - 1) * (len(fio2_edges) - 1)

        passed = True
        warnings_, errors, violations = [], [], []
        total_preds = 0

        if not hasattr(model, "predict"):
            errors.append("Model exposes no .predict() method – cannot validate")
            return {
                "passed": False,
                "warnings": warnings_,
                "errors": errors,
                "violations": violations,
            }

        for ep_idx, episode in enumerate(test_episodes[: max_episodes or len(test_episodes)]):
            for step_idx, state in enumerate(episode.observations[:20]):
                try:
                    a = int(model.predict(state.reshape(1, -1))[0])
                    total_preds += 1
                    if not 0 <= a < n_actions:
                        violations.append(f"Episode {ep_idx} step {step_idx}: invalid action {a}")
                        continue
                    flow_bin = a // (len(fio2_edges) - 1)
                    fio2_bin = a % (len(fio2_edges) - 1)
                    flow_val = (flow_edges[flow_bin] + flow_edges[flow_bin + 1]) / 2
                    fio2_val = (fio2_edges[fio2_bin] + fio2_edges[fio2_bin + 1]) / 2
                    if not 0 <= flow_val <= 70:
                        violations.append(f"Episode {ep_idx} step {step_idx}: flow={flow_val} L/min out of bounds")
                    if not 21 <= fio2_val <= 100:
                        violations.append(f"Episode {ep_idx} step {step_idx}: FiO₂={fio2_val}% out of bounds")
                except Exception as e:
                    warnings_.append(f"Prediction failure on ep{ep_idx}/t{step_idx}: {e}")

        viol_rate = len(violations) / max(1, total_preds)
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

    def validate_training_stability(self, log_dir: Path | str) -> Dict[str, Any]:
        _banner("📈  VALIDATING TRAINING STABILITY")
        log_dir = Path(log_dir)
        passed = True
        warnings_, errors = [], []
        stats: Dict[str, Any] = {}

        if not log_dir.exists():
            errors.append(f"Log directory not found: {log_dir}")
            return {"passed": False, "warnings": warnings_, "errors": errors, "stats": stats}

        # --- 1. Gradient health ------------------------------------------
        grad_files = list(log_dir.glob("*grad*.csv"))
        if not grad_files:
            warnings_.append("No gradient CSV files found – cannot assess gradients")
        else:
            for gf in grad_files[:3]:  # limit for speed
                try:
                    df = pd.read_csv(gf)
                    if df.empty:
                        continue
                    
                    # For gradient files with min/max columns, look at actual gradient values
                    if 'min' in df.columns and 'max' in df.columns:
                        # Get the maximum absolute gradient value from min/max columns
                        max_grad = max(float(df['max'].abs().max()), float(df['min'].abs().max()))
                    else:
                        # Fallback for other gradient file formats - exclude non-gradient columns
                        gradient_cols = [col for col in df.columns if col not in ['epoch', 'step']]
                        if gradient_cols:
                            max_grad = float(df[gradient_cols].abs().max().max())
                        else:
                            max_grad = 0.0
                    
                    stats[f"{gf.stem}_max_grad"] = max_grad
                    if max_grad > 1e3:
                        warnings_.append(f"Large gradients in {gf.name}: {max_grad:.2e}")
                except Exception as e:
                    warnings_.append(f"Failed reading {gf.name}: {e}")

        # --- 2. Loss curves ----------------------------------------------
        loss_files = list(log_dir.glob("*loss*.csv"))
        for lf in loss_files:
            try:
                df = pd.read_csv(lf, header=None, names=["epoch", "step", "loss"])
                if df.empty:
                    continue
                last_loss = float(df["loss"].iloc[-1])
                first_loss = float(df["loss"].iloc[0])
                stats[f"{lf.stem}_first_last"] = (first_loss, last_loss)
                if np.isnan(last_loss) or np.isinf(last_loss):
                    errors.append(f"{lf.name}: final loss is NaN/Inf → training diverged")
                    passed = False
                if abs(last_loss) > 1e3:
                    warnings_.append(f"{lf.name}: loss magnitude large ({last_loss:.1e})")
                # Check monotonic explosion
                if len(df) > 10 and last_loss > first_loss * 10:
                    warnings_.append(f"{lf.name}: loss increased >10× from start to end")
            except Exception as e:
                warnings_.append(f"Could not parse {lf.name}: {e}")

        outcome = {"passed": passed, "warnings": warnings_, "errors": errors, "stats": stats}
        self.validation_results["training_stability"] = outcome
        return outcome

    # ---------------------------------------------------------------------
    # 5️⃣  ONE‑CALL WRAPPER
    # ---------------------------------------------------------------------

    def run_full_validation(
        self,
        data_dict: Dict[str, Any],
        model: Any,
        test_episodes: List[Any],
        log_dir: Path | str,
        df: pd.DataFrame | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Convenience wrapper → runs **all** validation stages."""

        self.validate_data_quality(data_dict, df)
        self.validate_model_behavior(model, test_episodes)
        self.validate_clinical_plausibility(model, test_episodes)
        self.validate_training_stability(log_dir)
        _banner("✅  VALIDATION COMPLETE")
        return self.validation_results
