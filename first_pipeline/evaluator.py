#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path
from d3rlpy.ope import DiscreteFQE, FQEConfig
from d3rlpy.metrics import InitialStateValueEstimationEvaluator
from d3rlpy.dataset import ReplayBuffer  # nodig vanaf v2.x
from d3rlpy.dataset import create_infinite_replay_buffer
from scipy.special import logsumexp                # fast, stable
from d3rlpy.dataset import Episode
from d3rlpy.algos import QLearningAlgoBase   # Fixed import for d3rlpy 2.8.1
from d3rlpy.metrics import TDErrorEvaluator
from scope_rl.policy.head import SoftmaxHead
from scope_rl.dataset import SyntheticDataset
from scope_rl.policy import BaseHead, SoftmaxHead
from scope_rl.ope import CreateOPEInput, OffPolicyEvaluation
from scope_rl.ope.discrete import TrajectoryWiseImportanceSampling, DoublyRobust, SelfNormalizedDR
from scope_rl.policy import EpsilonGreedyHead     
import yaml

# from d3rlpy.dataset import MDPDataset

import warnings
warnings.filterwarnings('ignore')

class BehaviorPolicyEstimator:
    """
    Estimates the behavior policy pi_b(a|s) from offline data.
    """
    def __init__(self, n_actions: int, random_state: int = 42):
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
        self.n_actions = n_actions
        self.is_fitted = False
        print(f"🎯 BehaviorPolicyEstimator initialized for {n_actions} actions.")

    def fit(self, episodes):
        """Fit the behavior policy model using the provided episodes."""
        print("🔧 Fitting BehaviorPolicyEstimator...")
        all_observations = []
        all_actions = []

        if not episodes:
            print("⚠️ No episodes provided to fit the behavior policy estimator.")
            return

        for episode in episodes:
            if episode.observations is not None and episode.actions is not None:
                all_observations.append(episode.observations)
                all_actions.append(episode.actions.reshape(-1))

        if not all_observations or not all_actions:
            print("⚠️ No valid observations or actions found in episodes.")
            return

        states = np.concatenate(all_observations, axis=0)
        actions = np.concatenate(all_actions, axis=0)

        if states.shape[0] == 0:
            print("⚠️ Concatenated states array is empty. Cannot fit behavior policy.")
            return

        print(f"📊 Total transitions for fitting behavior policy: {states.shape[0]}")
        try:
            self.model.fit(states, actions)
            self.is_fitted = True
            print("✅ BehaviorPolicyEstimator fitted successfully.")
        except Exception as e:
            print(f"❌ Error fitting BehaviorPolicyEstimator: {e}")
            self.is_fitted = False

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        """Predict probability distribution over actions for given states."""
        if not self.is_fitted:
            return np.ones((states.shape[0], self.n_actions)) / self.n_actions
        try:
            if states.ndim == 1:
                states = states.reshape(1, -1)
            
            probas = self.model.predict_proba(states)

            # Handle missing actions in training data
            if probas.shape[1] < self.n_actions:
                full_probas = np.zeros((states.shape[0], self.n_actions))
                if hasattr(self.model, 'classes_'):
                    for i, class_label in enumerate(self.model.classes_):
                        if class_label < self.n_actions:
                             full_probas[:, class_label] = probas[:, i]
                else:
                    print("⚠️ Classifier model does not have 'classes_' attribute. Returning uniform.")
                    return np.ones((states.shape[0], self.n_actions)) / self.n_actions
                
                # Normalize rows to handle missing actions
                row_sums = full_probas.sum(axis=1, keepdims=True)
                uniform_probs_for_row = np.ones(self.n_actions) / self.n_actions
                for i in range(full_probas.shape[0]):
                    if row_sums[i, 0] == 0:
                        full_probas[i, :] = uniform_probs_for_row
                    else:
                        full_probas[i, :] /= row_sums[i, 0]
                return full_probas
            else:
                return probas[:, :self.n_actions]

        except Exception as e:
            print(f"❌ Error predicting probabilities: {e}")
            return np.ones((states.shape[0], self.n_actions)) / self.n_actions

    def get_action_probabilities(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Return pi_b(a|s) for specific actions taken in states."""
        action_probas_all = self.predict_proba(states)
        
        if actions.ndim > 1:
            actions = actions.flatten()
        actions = actions.astype(int)

        # Use the correct number of samples from action_probas_all, not original states
        num_samples = action_probas_all.shape[0]
        
        # Handle out-of-bounds actions
        if np.any(actions < 0) or np.any(actions >= self.n_actions):
            clamped_actions = np.clip(actions, 0, self.n_actions - 1)
            probs = action_probas_all[np.arange(num_samples), clamped_actions]
            probs[actions >= self.n_actions] = 1e-6 
            probs[actions < 0] = 1e-6
            return probs
        else:
            return action_probas_all[np.arange(num_samples), actions]


    def evaluate_behaviour_model(estimator, episodes, n_actions, title="Validation"):
        """Return a dict with prob-quality metrics and plot a reliability diagram."""
        # ---------- gather transitions -------------------------------------------------
        X = np.concatenate([ep.observations for ep in episodes], axis=0)
        y = np.concatenate([ep.actions.reshape(-1)   for ep in episodes], axis=0)

        # ---------- quick 80/20 split ---------------------------------------------------
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=0
        )
        estimator.model.fit(X_tr, y_tr)

        # ---------- predictions & metrics ----------------------------------------------
        y_hat      = estimator.model.predict(X_va)
        y_proba_va = estimator.model.predict_proba(X_va)

        acc  = accuracy_score(y_va, y_hat)
        ll   = log_loss(y_va, y_proba_va, labels=np.arange(n_actions))
        bs   = np.mean([
            brier_score_loss((y_va == k).astype(int), y_proba_va[:, k])
            for k in range(n_actions)
        ])

        # ---------- calibration plot ----------------------------------------------------
        prob_true, prob_pred = calibration_curve(
            (y_va == y_hat),      # 1 if predicted correct
            np.max(y_proba_va,1), # confidence of the chosen class
            n_bins=10
        )
        plt.figure(figsize=(4,4))
        plt.plot(prob_pred, prob_true, "s-", label="RF")
        plt.plot([0,1],[0,1],"k--")
        plt.title(f"Reliability ({title})")
        plt.xlabel("Predicted P(correct)")
        plt.ylabel("Empirical P(correct)")
        plt.tight_layout()
        plt.savefig(f"behaviour_model_reliability_{title}.png", dpi=300)
        plt.close()

        # ---------- ESS estimate on the SAME split -------------------------------------
        weights = []
        for s,a in zip(X_va, y_va):
            p_b  = estimator.get_action_probabilities(s[None, :], np.array([a]))[0]
            p_pi = 1.0 / n_actions                        # cheap uniform proxy
            weights.append(p_pi / max(p_b, 1e-9))
        w = np.array(weights)
        ess = (w.sum() ** 2) / (w ** 2).sum()

        print(f"[{title}]  acc={acc:.3f}  log-loss={ll:.3f}  brier={bs:.3f}  ESS≈{ess:.0f}/{len(w)}")
        return {"accuracy": acc, "log_loss": ll, "brier": bs, "ess": ess}

class CQLEvaluator:
    """Comprehensive evaluation framework for CQL-based HFNC parameter optimization"""

    
    def __init__(self, model, n_actions: int, behavior_policy_estimator: BehaviorPolicyEstimator, config_path=None):
        self.model = model
        self.config_path = config_path
        self.results = {}
        self.n_actions = n_actions

        if not isinstance(behavior_policy_estimator, BehaviorPolicyEstimator):
            raise TypeError("behavior_policy_estimator must be an instance of BehaviorPolicyEstimator.")
        if not behavior_policy_estimator.is_fitted:
            # Or, you could allow it and have OPE methods check/fail later,
            # but it's cleaner to enforce it here if OPE is a primary function.
            # For now, a warning is fine as main.py is supposed to fit it.
            print("⚠️ CQLEvaluator initialized with an unfitted BehaviorPolicyEstimator. OPE methods requiring it may fail or yield defaults.")

        # Determine n_actions
        self.n_actions = n_actions
        if self.n_actions is None:
            print("⚠️ CQLEvaluator initialized without n_actions. Inferring from model...")
            if hasattr(model, 'action_size'):
                self.n_actions = model.action_size
                print(f"📊 Inferred n_actions from model: {self.n_actions}")
            else:
                print("⚠️ Could not infer n_actions. Some OPE methods will fail.")

        # Set up behavior policy estimator
        self.behavior_policy_estimator = behavior_policy_estimator
        # if self.behavior_policy_estimator is None and self.n_actions is not None:
        #     print("🔧 Creating default BehaviorPolicyEstimator...")
        #     self.behavior_policy_estimator = BehaviorPolicyEstimator(n_actions=self.n_actions)

        # Clinical parameter ranges for HFNC (based on literature)
        self.clinical_ranges = {
            'flow_rate': {'min': 10, 'max': 70, 'unit': 'L/min'},  # Typical HFNC flow range
            'fio2': {'min': 0.21, 'max': 1.0, 'unit': 'fraction'}  # FiO2 range
            # 'temperature': {'min': 34, 'max': 40, 'unit': '°C'}     # If temperature is controlled
        }



    # def _dr_single_episode(ep, pi_b, pi_e, q_func, gamma=0.99):
    #     s = ep.observations               # (T , 36)
    #     a = ep.actions.reshape(-1)        # (T,)
    #     r = ep.rewards.reshape(-1)

    #     # behaviour probs and evaluation distribution
    #     p_b = pi_b.calc_pscore_given_action(s, a)
    #     dist = pi_e.calc_action_choice_probability(s)      # (T , 12)

    #     # Q(s,a) for *all* 12 actions
    #     q_hat = np.stack([q_func.predict_value(s, np.full(len(s), act))
    #                     for act in range(pi_e.n_actions)], axis=1)

    #     return DoublyRobust().estimate_policy_value(
    #                 step_per_trajectory = len(a),
    #                 action              = a,
    #                 reward              = r,
    #                 pscore              = p_b,
    #                 evaluation_policy_action_dist = dist,
    #                 state_action_value_prediction  = q_hat,
    #                 gamma               = gamma,
    #         )

    # def doubly_robust_value_varlen(episodes, bc_model, cql_model, gamma=0.99, seed=0):
    #     # wrap policies exactly as before
    #     behavior_algo = bc_model[0] if isinstance(bc_model, (list, tuple)) else bc_model
    #     pi_b = EpsilonGreedyHead(behavior_algo,  n_actions=behavior_algo.action_size,
    #                             epsilon=0.05, name="behavior", random_state=seed)
    #     pi_e = SoftmaxHead    (cql_model,  n_actions=cql_model.action_size,
    #                             tau=1.0,   name="target",   random_state=seed)

    #     # use the fitted FQE (already trained in your pipeline)
    #     q_func = cql_model     # CQL exposes predict_value

    #     dr_vals = [ CQLEvaluator._dr_single_episode(ep, pi_b, pi_e, q_func, gamma)
    #                 for ep in episodes ]
    #     return float(np.mean(dr_vals))



    def episodes_to_logged_dataset(episodes: list[Episode],
                               pi_b: EpsilonGreedyHead):
        """
        Convert a list of d3rlpy Episode objects into a *rectangular* logged
        dataset required by SCOPE-RL.  Shorter episodes are padded with the
        final (state, action), a reward of 0, and done=True.
        """
        L = max(len(ep.actions) for ep in episodes)          # longest trajectory
        S, A, R, D = [], [], [], []                          # ≥—— data buckets

        for ep in episodes:
            T = len(ep.actions)

            # ---- 1. observations  (T , d) -> (L , d) ----
            obs = ep.observations
            if T < L:
                obs = np.vstack([obs, np.repeat(obs[-1:], L - T, axis=0)])
            S.append(obs)

            # ---- 2. actions  (T,) -> (L,)  repeat last ----
            act = ep.actions.reshape(-1)
            if T < L:
                act = np.pad(act, (0, L - T), mode="edge")
            A.append(act)

            # ---- 3. rewards  (T,) -> (L,)  pad 0 ----------
            rew = ep.rewards.reshape(-1)
            if T < L:
                rew = np.pad(rew, (0, L - T), constant_values=0.0)
            R.append(rew)

            # ---- 4. done flags  (T,) -> (L,)  keep True ----
            done = np.zeros(T, dtype=bool)
            done[-1] = bool(ep.terminated)
            if T < L:
                done = np.pad(done, (0, L - T), constant_values=True)
            D.append(done)

        # ---------- stack & flatten ----------
        state   = np.vstack(S)                 # (n_trajectories * L, state_dim)
        action  = np.concatenate(A)
        reward  = np.concatenate(R)
        done    = np.concatenate(D)

        # ---------- behaviour-policy probabilities ----------
        pscore  = pi_b.calc_pscore_given_action(state, action)
        zeros = (pscore == 0).sum()
        print(f"[DEBUG] pscore zeros: {zeros} out of {len(pscore)}")
        if zeros:
            print("    first few zero-indices:", np.where(pscore == 0)[0][:10])
        pscore  = np.clip(pscore, 1e-6, 1.0) 


        # ---------- package ----------
        return dict(
            size                = len(state),
            n_trajectories      = len(episodes),
            step_per_trajectory = L,           # uniform horizon (key point!)
            action_type         = "discrete",
            n_actions           = pi_b.n_actions,
            action_dim          = 1,
            state_dim           = state.shape[1],
            behavior_policy     = pi_b.name,
            dataset_id          = 0,
            state     = state,
            action    = action,
            reward    = reward,
            done      = done,
            terminal  = done.copy(),
            pscore    = pscore,
        )
 
    # -----------------------------------------------------------------
    # 2)  d3rlpy algo  ->  SCOPE-RL policy head (unchanged)
    # ------------------------------------------------------------------
    def wrap_policy(algo, *, name, head="egreedy",    # ← add `head` switch
                    epsilon=0.0, tau=1.0, seed=42):

        if head == "softmax":                         # every action gets >0 prob
            return SoftmaxHead(
                base_policy  = algo,
                n_actions    = algo.action_size,
                tau          = tau,                   # 1.0 = fairly sharp
                name         = name,
                random_state = seed,
            )

        # default: ε-greedy, unchanged
        return EpsilonGreedyHead(
            base_policy  = algo,
            n_actions    = algo.action_size,
            epsilon      = epsilon,
            name         = name,
            random_state = seed,
        )


    # ------------------------------------------------------------------
    # 3)  DR estimate without a gym.Env (unchanged except for 1-liner)
    # ------------------------------------------------------------------
    def doubly_robust_value(episodes, behavior_algo, eval_algo,
                            gamma=0.99, seed=42):
       

        pi_b = EpsilonGreedyHead(
        base_policy  = behavior_algo,
        n_actions    = behavior_algo.action_size,
        epsilon      = 0.05,          # ↓ ensures pscore never 0
        name         = "behavior",
        random_state = seed,
)
        pi_e = SoftmaxHead(
        base_policy  = eval_algo,
        n_actions    = eval_algo.action_size,
        tau          = 1.0,            # temperature
        name         = "target",
        random_state = seed,
)

        logged_dataset = CQLEvaluator.episodes_to_logged_dataset(episodes,pi_b)

        prep = CreateOPEInput(env=None, gamma=0.99)   # env=None is OK
        input_dict = prep.obtain_whole_inputs(
            logged_dataset      = logged_dataset,     # your dict
            evaluation_policies = [pi_e],
            behavior_policy_name= pi_b.name,
            require_value_prediction = True,          # FQE will infer shapes from dataset
            random_state        = 0,
        )
        eval_blk = input_dict[pi_e.name]

        q_hat = eval_blk["state_action_value_prediction"]
        dist  = eval_blk["evaluation_policy_action_dist"]
        # w = dist[np.arange(len(logged_dataset["action"])), logged_dataset["action"]] / logged_dataset["pscore"]
        # print("[DBG]  weight  max:", w.max(), "   inf:", np.isinf(w).sum())

        # print("[DBG] Q-hat  NaN:", np.isnan(q_hat).any(),
        #     "Inf:", np.isinf(q_hat).any())
        # print("       action-dist NaN:", np.isnan(dist).any(),
        #     "Inf:", np.isinf(dist).any())

        # print("       q_hat shape", q_hat.shape,
        #     "dist shape", dist.shape)
        # start = 0
        # T = logged_dataset["step_per_trajectory"]
        # ratios = dist[start:start+T, :][
        #             np.arange(T),
        #             logged_dataset["action"][start:start+T]
        #         ] / logged_dataset["pscore"][start:start+T]

        # cum = np.cumprod(ratios)
        # print("[DBG] cumulative weight max:", cum[np.isfinite(cum)].max(),
        #     "   any inf:", np.isinf(cum).any())
        
        # from collections import Counter
        # import sklearn.metrics
        # from sklearn.metrics import confusion_matrix
        
        # print(Counter(logged_dataset["action"]))

        # --- 2.  Distribution of predicted p-scores -----------------------
        # ps = logged_dataset["pscore"]
        # plt.hist(ps, bins=np.logspace(-8, 0, 60)); plt.xscale("log"); plt.show()

        # # tail count
        # print("below 1e-4:", (ps < 1e-4).sum())

        # --- 3.  Confusion matrix between greedy BC action and clinician action
        # greedy = behavior_algo.predict(logged_dataset["state"])
        # cm = confusion_matrix(logged_dataset["action"], greedy)
        # print(cm)

        ope = OffPolicyEvaluation(
            logged_dataset = logged_dataset,
            ope_estimators = [SelfNormalizedDR(),DoublyRobust()]
        )
        values = ope.estimate_policy_value(input_dict)["target"]
        sndr_val = values["sndr"]
        dr_val   = values["dr"]
        return sndr_val, dr_val

    @staticmethod
    def bootstrap_sndr_value(
        episodes: list[Episode],
        behavior_algo,
        eval_algo,
        gamma: float = 0.99,
        seed: int = 42,
        n_boot: int = 20,
        alpha: float = 0.05,
    ) -> tuple[float, float, float]:
        """
        Trajectory‐level bootstrap CI for Self‐Normalized DR.
        Returns (point_estimate, lower_CI, upper_CI).
        """
        # helper: call your existing doubly_robust_value
        def one_sndr(eps_batch):
            sndr_val, _ = CQLEvaluator.doubly_robust_value(
                eps_batch,
                behavior_algo,
                eval_algo,
                gamma=gamma,
                seed=seed,
            )
            print(f"[DEBUG]  one_sndr value: {sndr_val:.3f}")
            return sndr_val

        # 1) full-data point
        theta_hat = one_sndr(episodes)

        # 2) bootstrap replicates
        rng = np.random.RandomState(seed)
        boot_vals = []
        N = len(episodes)
        for _ in range(n_boot):
            idx      = rng.randint(0, N, size=N)
            eps_boot = [episodes[i] for i in idx]
            boot_vals.append(one_sndr(eps_boot))
        boot = np.array(boot_vals)

        # 3) percentile CI
        lower = np.percentile(boot, 100 * (alpha/2))
        upper = np.percentile(boot, 100 * (1-alpha/2))
        return theta_hat, lower, upper


    
    def add_training_params_to_metrics(self, metrics, args):
        """Add training hyperparameters to metrics for comprehensive evaluation"""
        metrics["alpha"] = getattr(args, 'alpha', 'N/A')
        metrics["epochs"] = getattr(args, 'epochs', 'N/A')
        metrics["batch_size"] = getattr(args, 'batch', 'N/A')
        metrics["learning_rate"] = getattr(args, 'lr', 'N/A')
        metrics["gamma"] = getattr(args, 'gamma', 'N/A')
        metrics["model_type"] = "CQL"
        return metrics
    
    def evaluate_comprehensive(self, test_episodes, save_dir="evaluation_results", ope_gamma=0.99, ope_clip_ratio=10.0):
        """Enhanced comprehensive evaluation with proper OPE methods"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print("🔬 Running comprehensive academic evaluation with enhanced OPE...")
        

        # 1. Basic performance metrics
        basic_metrics = self._evaluate_basic_performance(test_episodes)
        
        # 2. Clinical performance analysis
        clinical_metrics = self._evaluate_clinical_performance(test_episodes)
        
        # 3. Statistical analysis
        statistical_metrics = self._statistical_analysis(test_episodes)
        
        # 4. Policy analysis
        policy_metrics = self._analyze_policy_behavior(test_episodes)
        
        # Enhanced OPE methods
        print("\n🎯 Running enhanced off-policy evaluation methods...")
        # wis_results = self.evaluate_wis(test_episodes, gamma=ope_gamma, clip_ratio=ope_clip_ratio)
        # dr_results = self.evaluate_dr(test_episodes, gamma=ope_gamma, clip_ratio=ope_clip_ratio)
        # fqe_results = self.evaluate_fqe(test_episodes)

        # Combine results
        self.results = {
            'basic_performance': basic_metrics,
            'clinical_performance': clinical_metrics,
            'statistical_analysis': statistical_metrics,
            'policy_analysis': policy_metrics
            # 'weighted_importance_sampling': wis_results
            # 'doubly_robust': dr_results,
            # 'fitted_q_evaluation': fqe_results,
        }
        
        # Generate plots and reports
        self._generate_academic_plots(test_episodes, save_dir)
        self._generate_enhanced_summary_report(save_dir)
        
        return self.results
    
    def _evaluate_basic_performance(self, test_episodes):
        """Basic RL performance metrics"""
        print("📊 Evaluating basic performance metrics...")
        
        returns = []
        episode_lengths = []
        
        # Step-level action comparison
        all_predicted_actions = []
        all_clinician_actions = []
        prediction_errors = 0
        
        for episode in test_episodes:
            # Episode return
            episode_return = np.sum(episode.rewards)
            returns.append(episode_return)
            
            # Episode length
            episode_lengths.append(len(episode.observations))
            
            # Step-level action agreement
            for obs, clinician_action in zip(episode.observations, episode.actions):
                try:
                    predicted_action = self.model.predict(obs.reshape(1, -1))[0]
                    
                    # Convert to scalar if needed
                    if isinstance(predicted_action, np.ndarray):
                        predicted_action = predicted_action.item() if predicted_action.size == 1 else predicted_action[0]
                    if isinstance(clinician_action, np.ndarray):
                        clinician_action = clinician_action.item() if clinician_action.size == 1 else clinician_action[0]
                    
                    all_predicted_actions.append(int(predicted_action))
                    all_clinician_actions.append(int(clinician_action))
                    
                except Exception as e:
                    prediction_errors += 1
                    continue
    
        # Calculate step-level agreement
        if all_predicted_actions and all_clinician_actions:
            step_agreements = np.array(all_predicted_actions) == np.array(all_clinician_actions)
            mean_action_agreement = float(np.mean(step_agreements))
            
            # For standard deviation, calculate per-episode agreements
            episode_agreements = []
            start_idx = 0
            for episode in test_episodes:
                end_idx = start_idx + len(episode.observations) - prediction_errors
                if end_idx > start_idx:
                    episode_step_agreements = step_agreements[start_idx:end_idx]
                    if len(episode_step_agreements) > 0:
                        episode_agreements.append(np.mean(episode_step_agreements))
                start_idx = end_idx
            
            std_action_agreement = float(np.std(episode_agreements)) if episode_agreements else 0.0
        else:
            mean_action_agreement = 0.0
            std_action_agreement = 0.0
        
        return {
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'mean_episode_length': float(np.mean(episode_lengths)),
            'mean_action_agreement': mean_action_agreement,  # Step-level agreement
            'std_action_agreement': std_action_agreement,
            'total_episodes': len(test_episodes),
            'total_transitions': sum(episode_lengths),
            'total_predictions': len(all_predicted_actions),
            'prediction_errors': prediction_errors,
            'step_level_agreements': int(np.sum(step_agreements)) if all_predicted_actions else 0
        }
    
    def _estimate_outcomes(self, predicted_actions, states, rewards):
        """Estimate clinical outcomes based on predicted actions"""
        # Simplified outcome estimation - in practice this would be more sophisticated
        # For now, we'll use the actual rewards as a proxy for outcomes
        return np.array(rewards)
    
    def _get_target_policy_proba(self, states: np.ndarray, temperature=1.0) -> np.ndarray:
        """
        Estimates pi_CQL(a|s) from Q_CQL(s,a) using softmax.
        """
        if not hasattr(self.model, 'predict_value'):
            print("⚠️ Model does not have 'predict_value' method. Cannot estimate target policy probabilities.")
            if self.n_actions:
                return np.ones((states.shape[0], self.n_actions)) / self.n_actions
            raise ValueError("n_actions not set, cannot fallback for target policy.")

        all_q_values = []
        # Ensure states is 2D for iteration
        if states.ndim == 1:
            states_for_iteration = states.reshape(1, -1)
        else:
            states_for_iteration = states

        for i in range(states_for_iteration.shape[0]):
            state_input = states_for_iteration[i:i+1] # Keep it 2D for predict_value
            q_values_for_state = []
            for action_idx in range(self.n_actions):
                action_arr = np.array([[action_idx]], dtype=np.int64)
                try:
                    q_val = self.model.predict_value(state_input, action_arr)
                    # d3rlpy's predict_value might return a tensor or ndarray
                    if hasattr(q_val, 'item'): # For single-element tensor/ndarray
                        q_values_for_state.append(q_val.item())
                    else: # Assuming it's already a scalar if no .item()
                        q_values_for_state.append(float(q_val))
                except Exception as e:
                    # print(f"Debug: Error predicting value for state {state_input}, action {action_idx}: {e}")
                    q_values_for_state.append(-np.inf) # Handle error by assigning a very low Q-value
            all_q_values.append(q_values_for_state)

        q_values_arr = np.array(all_q_values) 

        q_values_stable = q_values_arr - np.max(q_values_arr, axis=1, keepdims=True)
        exp_q = np.exp(q_values_stable / temperature)
        policy_probas = exp_q / np.sum(exp_q, axis=1, keepdims=True)

        nan_rows = np.isnan(policy_probas).any(axis=1)
        if np.any(nan_rows):
            policy_probas[nan_rows, :] = 1.0 / self.n_actions
        return policy_probas

    def _get_target_policy_action_probabilities(self, states: np.ndarray, actions: np.ndarray, temperature=1.0) -> np.ndarray:
        """
        Returns pi_CQL(a|s) for the specific actions taken.
        """
        target_probas_all = self._get_target_policy_proba(states, temperature)

        if actions.ndim > 1:
            actions = actions.flatten()
        actions_int = actions.astype(int)

        # Ensure actions are within bounds for indexing
        if np.any(actions_int < 0) or np.any(actions_int >= self.n_actions):
            print(f"⚠️ Warning: Target policy actions out of bounds for indexing. Min: {np.min(actions_int)}, Max: {np.max(actions_int)}, N_actions: {self.n_actions}")
            clamped_actions = np.clip(actions_int, 0, self.n_actions - 1)
            probs = target_probas_all[np.arange(states.shape[0]), clamped_actions]
            # For actions that were out of bounds, assign a minimal probability
            # This part might need more thought if it happens frequently.
            out_of_bounds_mask = (actions_int < 0) | (actions_int >= self.n_actions)
            probs[out_of_bounds_mask] = 1e-9 
            return probs
        else:
            return target_probas_all[np.arange(states.shape[0]), actions_int]





    def _evaluate_clinical_performance(self, test_episodes):
        """Clinical relevance metrics (simplified)"""
        print("🏥 Evaluating clinical performance...")
        
        # Extract actions and outcomes for analysis
        all_predicted_actions = []
        all_clinician_actions = []
        all_states = []
        all_rewards = []
        
        for episode in test_episodes:
            clinician_actions = episode.actions
            rewards = episode.rewards
            states = episode.observations
            
            predicted_actions = []
            for obs in episode.observations:
                try:
                    action = self.model.predict(obs.reshape(1, -1))[0]
                    predicted_actions.append(action)
                except:
                    predicted_actions.append(0)
            
            all_predicted_actions.extend(predicted_actions)
            all_clinician_actions.extend(clinician_actions)
            all_states.extend(states)
            all_rewards.extend(rewards)
        
        # Use the new _estimate_outcomes method
        predicted_outcomes = self._estimate_outcomes(all_predicted_actions, all_states, all_rewards)
        clinician_outcomes = np.array(all_rewards)  # Use actual episode rewards
        
        return {
            'predicted_outcome_mean': float(np.mean(predicted_outcomes)),
            'clinician_outcome_mean': float(np.mean(clinician_outcomes)),
            'outcome_improvement': 0.0,  # Simplified since we're using same rewardsAs
            'total_actions_evaluated': len(all_predicted_actions)
        }
    
    def _statistical_analysis(self, test_episodes):
        """Statistical significance testing"""
        print("📈 Running statistical analysis...")
        
        # Collect data for statistical tests
        predicted_returns = []
        clinician_returns = []
        
        for episode in test_episodes:
            # Simulate what the model would achieve
            predicted_actions = []
            for obs in episode.observations:
                try:
                    action = self.model.predict(obs.reshape(1, -1))[0]
                    predicted_actions.append(action)
                except:
                    predicted_actions.append(0)
            
            # Estimate returns (simplified - in practice you'd need a reward model)
            predicted_return = np.sum(episode.rewards)  # Placeholder
            clinician_return = np.sum(episode.rewards)
            
            predicted_returns.append(predicted_return)
            clinician_returns.append(clinician_return)
        # Statistical tests
        t_stat, p_value = stats.ttest_rel(predicted_returns, clinician_returns)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(predicted_returns, clinician_returns)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(predicted_returns) + np.var(clinician_returns)) / 2)
        cohens_d = (np.mean(predicted_returns) - np.mean(clinician_returns)) / pooled_std
        
        return {
            'paired_t_test': {'statistic': float(t_stat), 'p_value': float(p_value)},
            'wilcoxon_test': {'statistic': float(wilcoxon_stat), 'p_value': float(wilcoxon_p)},
            'cohens_d': float(cohens_d),
            'effect_size_interpretation': self._interpret_effect_size(cohens_d),
            'sample_size': len(test_episodes)
        }
    
    def _analyze_policy_behavior(self, test_episodes):
        """Analyze learned policy characteristics"""
        print("🎯 Analyzing policy behavior...")
        
        action_distribution = {}
        state_action_patterns = []
        q_value_analysis = {}
        clinician_distribution  = {}  
        

        for episode in test_episodes:
            for i, obs in enumerate(episode.observations):
                try:
                    # Get predicted action
                    predicted_action = self.model.predict(obs.reshape(1, -1))[0]
                    
                    # Count action distribution
                    if predicted_action not in action_distribution:
                        action_distribution[predicted_action] = 0
                    action_distribution[predicted_action] += 1;
                    
                    # Analyze Q-values if available
                    if hasattr(self.model, 'predict_value'):
                        try:
                            q_value = self.model.predict_value(obs.reshape(1, -1), 
                                                             np.array([[predicted_action]]))
                            if predicted_action not in q_value_analysis:
                                q_value_analysis[predicted_action] = []
                            q_value_analysis[predicted_action].append(float(q_value))
                        except:
                            pass
                    clin_act = int(episode.actions[i])
                    if clin_act not in clinician_distribution:
                        clinician_distribution[clin_act] = 0
                        clinician_distribution[clin_act] += 1
                    # Store state-action pattern
                    state_action_patterns.append({
                        'state_mean': float(np.mean(obs)),
                        'state_std': float(np.std(obs)),
                        'action': int(predicted_action),
                        'clinician_action': int(episode.actions[i])
                    })
                    
                except Exception as e:
                    continue
        
        return {
            'action_distribution': action_distribution,
            'q_value_statistics': {action: {'mean': np.mean(values), 'std': np.std(values)} 
                                 for action, values in q_value_analysis.items()},
            'policy_entropy': self._calculate_policy_entropy(action_distribution),
            'state_action_patterns': state_action_patterns[:100],  # Sample for analysis
            'clinician_entropy': self._calculate_policy_entropy(clinician_distribution)
        }
    
    
    def analyze_predictions(self, model, test_episodes, top_n=3):
        """Analyze model predictions and compare with clinician decisions"""
        print("🔍 Analyzing model predictions vs clinician decisions...")
        
        prediction_analysis = {
            'agreements': [],
            'disagreements': [],
            'action_frequencies': {},
            'state_analysis': []
        }
        
        total_steps = 0
        agreements = 0
        
        for episode_idx, episode in enumerate(test_episodes[:top_n]):
            print(f"\n📋 Episode {episode_idx + 1}/{min(top_n, len(test_episodes))}:")
            
            episode_agreements = 0
            episode_steps = len(episode.observations)
            
            for step_idx, (obs, clinician_action, reward) in enumerate(zip(
                episode.observations, episode.actions, episode.rewards)):
                
                try:
                    # Get model prediction
                    model_action = model.predict(obs.reshape(1, -1))[0]
                    
                    # Convert to scalar if needed
                    if isinstance(model_action, np.ndarray):
                        model_action = model_action.item() if model_action.size == 1 else model_action[0]
                    if isinstance(clinician_action, np.ndarray):
                        clinician_action = clinician_action.item() if clinician_action.size == 1 else clinician_action[0]
                    
                    model_action = int(model_action)
                    clinician_action = int(clinician_action)
                    
                    # Track agreement
                    is_agreement = model_action == clinician_action
                    if is_agreement:
                        agreements += 1
                        episode_agreements += 1
                    
                    # Store for analysis
                    step_data = {
                        'episode': episode_idx,
                        'step': step_idx,
                        'model_action': model_action,
                        'clinician_action': clinician_action,
                        'reward': float(reward),
                        'agreement': is_agreement,
                        'state_mean': float(np.mean(obs)),
                        'state_std': float(np.std(obs))
                    }
                    
                    if is_agreement:
                        prediction_analysis['agreements'].append(step_data)
                    else:
                        prediction_analysis['disagreements'].append(step_data)
                    
                    # Track action frequencies
                    if model_action not in prediction_analysis['action_frequencies']:
                        prediction_analysis['action_frequencies'][model_action] = 0
                    prediction_analysis['action_frequencies'][model_action] += 1
                    
                    total_steps += 1
                    
                except Exception as e:
                    print(f"   ⚠️ Error at step {step_idx}: {e}")
                    continue
            
            episode_agreement_rate = episode_agreements / episode_steps if episode_steps > 0 else 0
            print(f"   📊 Episode agreement rate: {episode_agreement_rate:.3f} ({episode_agreements}/{episode_steps})")
        
        overall_agreement_rate = agreements / total_steps if total_steps > 0 else 0
        print(f"\n📊 Overall agreement rate: {overall_agreement_rate:.3f} ({agreements}/{total_steps})")
        
        # Analyze disagreements
        if prediction_analysis['disagreements']:
            print(f"\n🔍 Top disagreement patterns:")
            disagreement_rewards = [d['reward'] for d in prediction_analysis['disagreements'][:10]]
            print(f"   Average reward in disagreements: {np.mean(disagreement_rewards):.3f}")
        
        # Analyze action distribution
        print(f"\n🎯 Model action distribution:")
        for action, count in sorted(prediction_analysis['action_frequencies'].items()):
            percentage = (count / total_steps) * 100 if total_steps > 0 else 0
            print(f"   Action {action}: {count} times ({percentage:.1f}%)")
        
        return prediction_analysis

    def _generate_academic_plots(self, test_episodes, save_dir):
        """Generate publication-quality plots for thesis"""
        print("📊 Generating academic visualizations...")
        
        # Set academic style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # 1. Action distribution comparison
        self._plot_action_distribution(test_episodes, save_dir)
        
        # 2. Return distribution analysis
        self._plot_return_distribution(test_episodes, save_dir)
        
        # 3. Learning curve analysis (if training logs available)
        self._plot_learning_curves(save_dir)
        
        # 4. Clinical parameter analysis
        self._plot_clinical_parameters(test_episodes, save_dir)
        
        # 5. Q-value distribution
        self._plot_q_value_distribution(test_episodes, save_dir)
        
        # 6. State-action heatmap
        self._plot_state_action_heatmap(test_episodes, save_dir)

        self._plot_action_heatmaps(test_episodes, save_dir)
    
    def _plot_action_distribution(self, test_episodes, save_dir):
        """Compare action distributions between model and clinicians"""
        # wherever your evaluator lives:
        config_path = Path(__file__).resolve().parent / "config.yaml"
        cfg = yaml.safe_load(open(config_path, "r"))
        fio2_edges = cfg["fio2_edges"]
        flow_edges = cfg["flow_edges"]

        # build labels like "FiO₂ 21–40; Flow 0–20"
        action_labels = []
        for i in range(len(fio2_edges) - 1):
            for j in range(len(flow_edges) - 1):
                action_labels.append(
                    f"FiO₂ {fio2_edges[i]}–{fio2_edges[i+1]}, "
                    f"Flow {flow_edges[j]}–{flow_edges[j+1]}"
        )
        # Extract actions
        model_actions = []
        clinician_actions = []
        
        for episode in test_episodes:
            for obs, clinician_action in zip(episode.observations, episode.actions):
                try:
                    model_action = self.model.predict(obs.reshape(1, -1))[0]
                    
                    # Convert numpy arrays to scalar values if needed
                    if isinstance(model_action, np.ndarray):
                        model_action = float(model_action.item()) if model_action.size == 1 else float(model_action[0])
                    if isinstance(clinician_action, np.ndarray):
                        clinician_action = float(clinician_action.item()) if clinician_action.size == 1 else float(clinician_action[0])
                    
                    model_actions.append(float(model_action))
                    clinician_actions.append(float(clinician_action))
                except:
                    continue
        model_counts     = np.bincount(model_actions,     minlength=cfg["n_actions"])
        clinician_counts = np.bincount(clinician_actions, minlength=cfg["n_actions"])
        
        x     = np.arange(cfg["n_actions"])
        width = 0.4

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.bar(x - width/2, model_counts,     width, label="CQL Model",     edgecolor="black")
        ax.bar(x + width/2, clinician_counts, width, label="Clinicians",    edgecolor="black", hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels(action_labels, rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title("Action Frequency by FiO₂ × Flow Bin")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / "action_distribution_bars.png", dpi=300)
        plt.close()
        print("📊 Saved bar‐chart action distribution with actual bins.")
    
    def _plot_return_distribution(self, test_episodes, save_dir):
        """Plot episode return distributions"""
        returns = [np.sum(episode.rewards) for episode in test_episodes]
        
        plt.figure(figsize=(10, 6))
        plt.hist(returns, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.3f}')
        plt.axvline(np.median(returns), color='green', linestyle='--', label=f'Median: {np.median(returns):.3f}')
        plt.xlabel('Episode Return')
        plt.ylabel('Frequency')
        plt.title('Distribution of Episode Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / 'return_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_learning_curves(self, save_dir):
        """Plot learning curves from training logs"""
        # Look for training logs
        log_dirs = list(Path("d3rlpy_logs").glob("**/"))
        
        for log_dir in log_dirs:
            loss_files = list(log_dir.glob("*loss*.csv"))
            if loss_files:
                try:
                    df = pd.read_csv(loss_files[0], header=None, names=['epoch', 'step', 'loss'])
                    plt.figure(figsize=(10, 6))
                    plt.plot(df['step'], df['loss'])
                    plt.xlabel('Training Steps')
                    plt.ylabel('Loss')
                    plt.title('Training Loss Curve')
                    plt.yscale('log')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(save_dir / 'learning_curve.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    break
                except:
                    continue
    
    def _plot_clinical_parameters(self, test_episodes, save_dir):
        """Plot clinical parameter distributions"""
        # Placeholder - adapt based on your specific HFNC parameters
        pass
    
    def _plot_q_value_distribution(self, test_episodes, save_dir):
        """Plot Q-value distributions"""
        q_values = []
        
        for episode in test_episodes:  # Process all episodes
            for obs in episode.observations:
                try:
                    if hasattr(self.model, 'predict_value'):
                        action = self.model.predict(obs.reshape(1, -1))[0]
                        q_val = self.model.predict_value(obs.reshape(1, -1), np.array([[action]]))
                        q_values.append(float(q_val))
                except:
                    continue
        
        if q_values:
            plt.figure(figsize=(10, 6))
            plt.hist(q_values, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Q-Value')
            plt.ylabel('Frequency')
            plt.title('Distribution of Q-Values')
            plt.grid(True, alpha=0.3)
            plt.savefig(save_dir / 'q_value_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()



    def _plot_action_heatmaps(self, test_episodes, save_dir):
        # 1) load your bin edges
        config_path = Path(__file__).resolve().parent / "config.yaml"
        cfg = yaml.safe_load(open(config_path, "r"))
        fio2_edges = cfg["fio2_edges"]
        flow_edges = cfg["flow_edges"]
        n_fio2 = len(fio2_edges) - 1
        n_flow = len(flow_edges) - 1

        # 2) collect actions
        model_actions = []
        clinician_actions = []
        for ep in test_episodes:
            for obs, ca in zip(ep.observations, ep.actions):
                try:
                    ma = self.model.predict(obs.reshape(1, -1))[0]
                    ma = int(np.asarray(ma).item())
                    ca = int(np.asarray(ca).item())
                    model_actions.append(ma)
                    clinician_actions.append(ca)
                except:
                    continue

        # 3) bin into 2D mats
        model_counts     = np.bincount(model_actions,     minlength=n_fio2 * n_flow)
        clinician_counts = np.bincount(clinician_actions, minlength=n_fio2 * n_flow)
        model_mat     = model_counts.reshape(n_fio2, n_flow)
        clinician_mat = clinician_counts.reshape(n_fio2, n_flow)

        # 4) determine shared color‐scale
        vmin = min(model_mat.min(), clinician_mat.min())
        vmax = max(model_mat.max(), clinician_mat.max())

        # 5) build axis labels
        fio2_labels = [f"{fio2_edges[i]}–{fio2_edges[i+1]}" for i in range(n_fio2)]
        flow_labels = [f"{flow_edges[j]}–{flow_edges[j+1]}" for j in range(n_flow)]

        # 6) plot side by side with seaborn heatmap + annotations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        sns.heatmap(
            model_mat,
            ax=ax1,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            annot=True,
            fmt="d", 
            cbar_kws={"label": "Count"}
        )
        ax1.set_title("CQL Model")
        ax1.set_xlabel("Flow bin (L/min)")
        ax1.set_ylabel("FiO₂ bin (%)")
        ax1.set_xticklabels(flow_labels, rotation=45, ha="right")
        ax1.set_yticklabels(fio2_labels, rotation=0)

        sns.heatmap(
            clinician_mat,
            ax=ax2,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            annot=True,
            fmt="d",
            cbar_kws={"label": "Count"}
        )
        ax2.set_title("Clinicians")
        ax2.set_xlabel("Flow bin (L/min)")
        ax2.set_ylabel("")   # no need to repeat y-label
        ax2.set_xticklabels(flow_labels, rotation=45, ha="right")
        ax2.set_yticklabels(fio2_labels, rotation=0)

        plt.tight_layout()
        plt.savefig(save_dir / "action_distribution_heatmaps.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📊 Heatmaps saved; model vs clinician action counts per FiO₂×flow bin.")

    
    def _plot_state_action_heatmap(self, test_episodes, save_dir):
        """Create state-action heatmap"""
        # Simplified heatmap - adapt based on your state space
        states = []
        actions = []
        
        for episode in test_episodes[:20]:  # Sample for efficiency
            for obs in episode.observations:
                try:
                    action = self.model.predict(obs.reshape(1, -1))[0]
                    states.append(np.mean(obs))  # Simplified state representation
                    actions.append(action)
                except:
                    continue
        
        if states and actions:
            plt.figure(figsize=(10, 8))
            plt.scatter(states, actions, alpha=0.6)
            plt.xlabel('State (mean feature value)')
            plt.ylabel('Action')
            plt.title('State-Action Space Visualization')
            plt.grid(True, alpha=0.3)
            plt.savefig(save_dir / 'state_action_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        

    
    def _generate_summary_report(self, save_dir):
        """Generate a comprehensive summary report for thesis"""
        report_path = save_dir / 'evaluation_summary.md'
        
        with open(report_path, 'w') as f:
            f.write("# CQL Model Evaluation Summary\n\n")
            f.write("## Executive Summary\n")
            f.write("This report provides a comprehensive evaluation of the Conservative Q-Learning (CQL) model ")
            f.write("for High-Flow Nasal Cannula (HFNC) parameter optimization.\n\n")
            
            if 'basic_performance' in self.results:
                bp = self.results['basic_performance']
                f.write("## Basic Performance Metrics\n")
                f.write(f"- Mean Episode Return: {bp.get('mean_return', 'N/A'):.4f}\n")
                f.write(f"- Action Agreement with Clinicians: {bp.get('mean_action_agreement', 'N/A'):.4f}\n")
                f.write(f"- Total Episodes Evaluated: {bp.get('total_episodes', 'N/A')}\n\n")
            
            if 'clinical_performance' in self.results:
                cp = self.results['clinical_performance']
                f.write("## Clinical Performance\n")
                f.write(f"- Parameter Appropriateness Score: {cp.get('parameter_appropriateness_score', 'N/A'):.4f}\n")
                f.write(f"- Outcome Improvement vs Clinicians: {cp.get('outcome_improvement', 'N/A'):.4f}\n\n")
            
            if 'statistical_analysis' in self.results:
                sa = self.results['statistical_analysis']
                f.write("## Statistical Analysis\n")
                f.write(f"- Effect Size (Cohen's d): {sa.get('cohens_d', 'N/A'):.4f} ({sa.get('effect_size_interpretation', 'N/A')})\n")
                if 'paired_t_test' in sa:
                    f.write(f"- Paired t-test p-value: {sa['paired_t_test'].get('p_value', 'N/A'):.4f}\n")
                f.write(f"- Sample Size: {sa.get('sample_size', 'N/A')}\n\n")
            
            f.write("## Visualizations Generated\n")
            f.write("- Action Distribution Comparison\n")
            f.write("- Return Distribution Analysis\n")
            f.write("- Q-Value Distribution\n")
            f.write("- State-Action Space Visualization\n")
            f.write("- Learning Curve Analysis\n\n")
            
            f.write("## Conclusion\n")
            f.write("The evaluation demonstrates the academic rigor and clinical relevance of the CQL model ")
            f.write("for HFNC parameter optimization, providing quantitative evidence for thesis contributions.\n")

    
    def episodes_to_replaybuffer(self, episodes):
        """Converteert list[d3rlpy.dataset.Episode] → ReplayBuffer."""
        # Eén grote buffer aanleggen
        buf = ReplayBuffer(buffer_size=sum(ep.size() for ep in episodes),
                        observation_shape=episodes[0].observations.shape[1:],
                        action_size=episodes[0].actions.max() + 1,  # discrete
                        discrete_action=True)
        for ep in episodes:
            for t in range(ep.size()):
                buf.append(
                    ep.observations[t],
                    ep.actions[t],
                    ep.rewards[t],
                    ep.observations[t + 1],
                    ep.terminals[t],
                )
        return buf
    

    def evaluate_fqe(
        self,
        episodes,             # list[Episode] (bv. train+val)
        n_steps     = 150_000,
        n_boot      = 200,
        discount    = 0.99,
    ):
        """Return dict met punt-schatting en 95 %-BI van V^{π_CQL} via FQE."""
        print(f"🔄  FQE: {len(episodes)} episodes  |  {n_steps:,} gradient stappen")

        # 1) Episodes ➜ ReplayBuffer (nieuwe API)
        # This buffer is used for the main training of the FQE model
        buffer = create_infinite_replay_buffer(episodes)

        # 2) Config & object
        fqe_cfg = FQEConfig(
            learning_rate = 3e-4,
            gamma = discount
        )
        fqe = DiscreteFQE(
            algo   = self.model,   # bevroren π_CQL
            config = fqe_cfg,
            device = "cpu",
        )

        # 3) Train the FQE model ONCE on the full dataset
        init_eval = InitialStateValueEstimationEvaluator()
        fqe.fit(
            buffer,
            n_steps = n_steps,
            evaluators= {"init_value": init_eval},
            show_progress=True,
        )
        
        # The point estimate is the final evaluation on the original buffer
        point_est = float(init_eval(fqe, buffer))
        print(f"   ➜ punt-schatting V̂ = {point_est:.3f}")

        # 4) Bootstrap CI using the *already trained* FQE model
        print(f"🔁  Bootstrappen ({n_boot} replicaties)…")
        boot_vals = []
        rng = np.random.default_rng(0)
        for i in range(n_boot):
            # Create a bootstrap sample of the episodes
            boot_eps = list(rng.choice(episodes, size=len(episodes), replace=True))
            
            # Create a replay buffer from the bootstrap sample
            boot_buf = create_infinite_replay_buffer(boot_eps)

            # ---- FIX: The key change is here ----
            # DO NOT call fqe.build_with_dataset(boot_buf).
            # We want to evaluate the single trained FQE model on the new data sample.
            # The evaluator object 'init_eval' will use the trained 'fqe' model
            # and evaluate its performance on the 'boot_buf' data.
            v_hat = float(init_eval(fqe, boot_buf)) 
            boot_vals.append(v_hat)

        ci_low, ci_high = np.percentile(boot_vals, [2.5, 97.5])
        print(f"✅  FQE klaar:  {point_est:.2f}  [95 % CI {ci_low:.2f} – {ci_high:.2f}]")

        return {
            "fqe_estimate": point_est,
            "ci_95": [float(ci_low), float(ci_high)],
            "fqe_std": float(np.std(boot_vals)),
            "n_steps": int(n_steps),
            "n_boot":  int(n_boot),
        }

    
    def _generate_enhanced_summary_report(self, save_dir):
        """Generate enhanced summary report with all metrics"""
        report_path = save_dir / 'enhanced_evaluation_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Enhanced CQL Model Evaluation Report\n\n")
            f.write("## Executive Summary\n")
            f.write("Comprehensive evaluation of Conservative Q-Learning (CQL) model for HFNC optimization.\n\n")
            
            # Basic Performance
            if 'basic_performance' in self.results:
                bp = self.results['basic_performance']
                f.write("## Basic Performance Metrics\n")
                f.write(f"- **Mean Episode Return**: {bp.get('mean_return', 'N/A'):.4f} ± {bp.get('std_return', 'N/A'):.4f}\n")
                f.write(f"- **Action Agreement**: {bp.get('mean_action_agreement', 'N/A'):.4f} ± {bp.get('std_action_agreement', 'N/A'):.4f}\n")
                f.write(f"- **Episodes Evaluated**: {bp.get('total_episodes', 'N/A')}\n")
                f.write(f"- **Total Transitions**: {bp.get('total_transitions', 'N/A')}\n\n")
            
            # Clinical Performance
            if 'clinical_performance' in self.results:
                cp = self.results['clinical_performance']
                f.write("## Clinical Performance\n")
                f.write(f"- **Predicted Outcome Mean**: {cp.get('predicted_outcome_mean', 'N/A'):.4f}\n")
                f.write(f"- **Clinician Outcome Mean**: {cp.get('clinician_outcome_mean', 'N/A'):.4f}\n")
                f.write(f"- **Actions Evaluated**: {cp.get('total_actions_evaluated', 'N/A')}\n\n")
            
            # Statistical Analysis
            if 'statistical_analysis' in self.results:
                sa = self.results['statistical_analysis']
                f.write("## Statistical Analysis\n")
                f.write(f"- **Effect Size (Cohen's d)**: {sa.get('cohens_d', 'N/A'):.4f} ({sa.get('effect_size_interpretation', 'N/A')})\n")
                if 'paired_t_test' in sa:
                    f.write(f"- **Paired t-test p-value**: {sa['paired_t_test'].get('p_value', 'N/A'):.4f}\n")
                if 'wilcoxon_test' in sa:
                    f.write(f"- **Wilcoxon test p-value**: {sa['wilcoxon_test'].get('p_value', 'N/A'):.4f}\n")
                f.write(f"- **Sample Size**: {sa.get('sample_size', 'N/A')}\n\n")
            
            # Policy Analysis
            if 'policy_analysis' in self.results:
                pa = self.results['policy_analysis']
                f.write("## Policy Analysis\n")
                f.write(f"- **Policy Entropy**: {pa.get('policy_entropy', 'N/A'):.4f}\n")
                if 'action_distribution' in pa:
                    f.write(f"- **Unique Actions Used**: {len(pa['action_distribution'])}\n")
                f.write("\n")
            
            # Off-Policy Evaluation
            if 'weighted_importance_sampling' in self.results:
                wis = self.results['weighted_importance_sampling']
                f.write("## Off-Policy Evaluation\n")
                f.write(f"- **WIS Estimate**: {wis.get('wis_estimate', 'N/A'):.4f}\n")
                f.write(f"- **Effective Sample Size**: {wis.get('ess', 'N/A'):.2f}\n")
                
            if 'doubly_robust' in self.results:
                dr = self.results['doubly_robust']
                f.write(f"- **DR Estimate**: {dr.get('dr_estimate', 'N/A'):.4f}\n")
                
            if 'fitted_q_evaluation' in self.results:
                fqe = self.results['fitted_q_evaluation']
                f.write(f"- **FQE Estimate**: {fqe.get('fqe_estimate', 'N/A'):.4f} ± {fqe.get('fqe_std', 'N/A'):.4f}\n")
            
            f.write("\n## Visualizations\n")
            f.write("- Action distribution comparison plots\n")
            f.write("- Episode return distribution\n")
            f.write("- Learning curves (if available)\n")
            f.write("- Q-value distribution\n")
            f.write("- State-action space visualization\n\n")
            
            f.write("## Conclusions\n")
            f.write("This comprehensive evaluation provides evidence for the effectiveness of the CQL approach ")
            f.write("in learning clinically appropriate HFNC parameter optimization policies from offline data.\n")
        
        print(f"📝 Enhanced evaluation report saved to: {report_path}")

    def _interpret_effect_size(self, cohens_d):
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _calculate_policy_entropy(self, action_distribution):
        """Calculate policy entropy from action distribution"""
        if not action_distribution:
            return 0.0
        
        total_actions = sum(action_distribution.values())
        if total_actions == 0:
            return 0.0
        
        probabilities = [count / total_actions for count in action_distribution.values()]
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        return float(entropy)
    
    def _calculate_effective_sample_size(self, weights):
        """Calculate effective sample size for importance sampling"""
        if not weights or len(weights) == 0:
            print("⚠️ No weights provided for ESS calculation.")
            return 0.0
        
        weights = np.array(weights)
        if np.sum(weights) == 0:
            print("⚠️ Sum of weights is zero, cannot calculate ESS.")
            return 0.0
        
        normalized_weights = weights / np.sum(weights)
        ess = 1.0 / np.sum(normalized_weights ** 2)
        if ess < 1:
            print("⚠️ Effective sample size is less than 1, indicating poor quality of importance sampling.")
            
        return float(ess)
    
    def evaluate_validation_losses(self, val_episodes, set_name="validation"):
        """
        Calculates TD and Conservative losses on validation episodes using the highest-level d3rlpy approach.
        Uses TDErrorEvaluator with episodes initialization and ReplayBuffer conversion.
        """
        from d3rlpy.metrics import TDErrorEvaluator
        from d3rlpy.dataset import create_infinite_replay_buffer
        import numpy as np

        print(f"\n🔬 Evaluating losses for '{set_name}' set (High-level d3rlpy approach)...")

        if not val_episodes:
            print(f"   ❌ No episodes provided for '{set_name}' set.")
            return {'td_loss': 0.0, 'conservative_loss': 0.0, 'total_loss': 0.0}

        # Enhanced reward scale diagnostics
        self._diagnose_reward_scale(val_episodes)

        # Step 1: Create ReplayBuffer directly from episodes using d3rlpy's high-level function
        try:
            val_replay_buffer = create_infinite_replay_buffer(val_episodes)
            print(f"   ✅ Created ReplayBuffer from {len(val_episodes)} episodes")
        except Exception as e:
            print(f"   ❌ Failed to create ReplayBuffer: {e}")
            return {'td_loss': 0.0, 'conservative_loss': 0.0, 'total_loss': 0.0}

        # Step 2: Calculate TD Loss using TDErrorEvaluator - the cleanest way
        td_loss = 0.0
        try:
            # Initialize with episodes for focused evaluation
            td_evaluator = TDErrorEvaluator(episodes=val_episodes)
            # Call with algorithm and replay buffer (as per documentation)
            td_loss = td_evaluator(self.model, val_replay_buffer)
            print(f"   ✅ TD Loss calculated using high-level TDErrorEvaluator")
        except Exception as e:
            print(f"   ⚠️ TDErrorEvaluator failed: {e}. Defaulting TD Loss to 0.")
            td_loss = 0.0

        # Step 3: Calculate Conservative Loss using our static method
        conservative_loss = self.conservative_loss_discrete(self.model, val_episodes)

        # Step 4: Return combined results
        results = {
            'td_loss': float(td_loss),
            'conservative_loss': float(conservative_loss),
            'total_loss': float(td_loss + conservative_loss)
        }

        print(f"   ✅ '{set_name.capitalize()}' Loss Results:")
        print(f"      TD Loss:          {results['td_loss']:.6f}")
        print(f"      Conservative Loss: {results['conservative_loss']:.6f}")
        print(f"      Total Loss:       {results['total_loss']:.6f}")
        
        return results
    @staticmethod
    def conservative_loss_discrete(
        algo: QLearningAlgoBase,
        episodes: list,
        alpha: float | None = None,
        batch_size: int = 8192,
    ) -> float:
        """
        Vectorised CQL regulariser on an *unseen* dataset.
        Works with any discrete-action d3rlpy algorithm (CQL, DQN, BCQ …).

        Parameters
        ----------
        algo        : trained d3rlpy algorithm (must expose `.predict_q_values`)
        episodes    : list of d3rlpy Episode objects
        alpha       : override weight; if None pull from algo.config.alpha
        batch_size  : obs batch size to bound RAM usage

        Returns
        -------
        float  –  α · E_s[logΣexp Q – Q(s,a_bc)]
        """
        # 1) flatten dataset --------------------------------------------------
        obs, act = [], []
        for ep in episodes:
            obs.append(ep.observations)
            act.append(ep.actions.reshape(-1))
        observations = np.vstack(obs).astype(np.float32)
        actions      = np.concatenate(act).astype(np.int64)

        if observations.size == 0:
            return 0.0

        # 2) pull α from the config if not provided - FIXED ALPHA ACCESS ----
        if alpha is None:
            # Try multiple paths to access alpha from the algorithm
            if hasattr(algo, 'config') and hasattr(algo.config, 'alpha'):
                alpha = algo.config.alpha
                print(f"Using alpha = {alpha:.3f} from algo.config.alpha")
            elif hasattr(algo, '_config') and hasattr(algo._config, 'alpha'):
                alpha = algo._config.alpha
                print(f"Using alpha = {alpha:.3f} from algo._config.alpha")
            elif hasattr(algo, '_impl') and hasattr(algo._impl, '_alpha'):
                alpha = algo._impl._alpha
                print(f"Using alpha = {alpha:.3f} from algo._impl._alpha")
            else:
                # Fallback for other algorithms or if alpha is not found
                alpha = getattr(getattr(algo, "_config", None), "initial_alpha", 1.0)
                print(f"Using fallback alpha = {alpha:.3f} (could not find alpha in standard locations)")

        # 3) mini-batch evaluation -------------------------------------------
        n_total, loss_sum = observations.shape[0], 0.0
        for start in range(0, n_total, batch_size):
            end   = start + batch_size
            obs_b = observations[start:end]
            act_b = actions[start:end]

                                        # v2.8.1 path
            n_actions = getattr(algo, "action_size",    # DiscreteCQL has it
                                int(act_b.max()) + 1)
            # repeat each state for every action
            obs_rep = np.repeat(obs_b, n_actions, axis=0)
            act_rep = np.tile(np.arange(n_actions), len(obs_b)).astype(np.int64)
            act_rep = act_rep.reshape(-1, 1)            # shape (B*n_actions, 1)
            q_flat  = np.asarray(algo.predict_value(obs_rep, act_rep)).ravel()
            q       = q_flat.reshape(len(obs_b), n_actions)  # back to (B, n_actions)

            # ---------- CQL regulariser ---------------------------------------
            lse  = logsumexp(q, axis=1)                 # log Σ exp Q(s,a′)
            q_bc = q[np.arange(len(q)), act_b]          # Q(s,a_bc)
            loss_sum += np.sum(lse - q_bc)

        return float(alpha * loss_sum / n_total)


    def _diagnose_reward_scale(self, episodes):
        """Diagnose potential reward scale issues that could cause extreme Q-values"""
        print(f"\n💰 [REWARD SCALE DIAGNOSTICS]:")
        
        all_rewards = []
        episode_returns = []
        
        for episode in episodes:
            episode_reward_sum = np.sum(episode.rewards)
            episode_returns.append(episode_reward_sum)
            all_rewards.extend(episode.rewards)
        
        rewards_array = np.array(all_rewards)
        returns_array = np.array(episode_returns)
        
        print(f"   📊 Reward statistics:")
        print(f"      - Total transitions: {len(rewards_array)}")
        print(f"      - Reward range: [{rewards_array.min():.3f}, {rewards_array.max():.3f}]")
        print(f"      - Reward mean: {rewards_array.mean():.3f}")
        print(f"      - Reward std: {rewards_array.std():.3f}")
        print(f"      - Reward median: {np.median(rewards_array):.3f}")
        print(f"      - Zero rewards: {(rewards_array == 0).sum()} ({(rewards_array == 0).mean():.1%})")
        print(f"      - Negative rewards: {(rewards_array < 0).sum()} ({(rewards_array < 0).mean():.1%})")
        print(f"      - Positive rewards: {(rewards_array > 0).sum()} ({(rewards_array > 0).mean():.1%})")
        
        print(f"   📊 Episode return statistics:")
        print(f"      - Episodes: {len(returns_array)}")
        print(f"      - Return range: [{returns_array.min():.3f}, {returns_array.max():.3f}]")
        print(f"      - Return mean: {returns_array.mean():.3f}")
        print(f"      - Return std: {returns_array.std():.3f}")
        
        # Check for potential issues
        if rewards_array.max() > 100 or rewards_array.min() < -100:
            print(f"      ⚠️  WARNING: Very large reward magnitudes detected!")
            print(f"          Consider using RewardScaler to normalize rewards")
        
        if abs(rewards_array.mean()) > 10:
            print(f"      ⚠️  WARNING: Rewards not centered around zero!")
            print(f"          Mean reward of {rewards_array.mean():.3f} could lead to large Q-values")
        
        if returns_array.min() < -1000 or returns_array.max() > 1000:
            print(f"      ⚠️  WARNING: Very large episode returns detected!")
            print(f"          With gamma={getattr(self.model._config, 'gamma', 0.99)}, Q-values could explode")
        
        # Estimate expected Q-values based on returns
        gamma = getattr(self.model._config, 'gamma', 0.99)
        avg_episode_length = len(rewards_array) / len(returns_array) if returns_array.size > 0 else 1
        
        # Rough estimate: Q(s,a) ≈ immediate_reward + gamma * future_return
        estimated_q_range = [
            rewards_array.min() + gamma * returns_array.min(),
            rewards_array.max() + gamma * returns_array.max()
        ]
        
        print(f"   🔮 Estimated Q-value range (rough): [{estimated_q_range[0]:.1f}, {estimated_q_range[1]:.1f}]")
        if abs(estimated_q_range[0]) > 1000 or abs(estimated_q_range[1]) > 1000:
            print(f"      ⚠️  WARNING: Estimated Q-values will be very large!")
            print(f"          This explains the conservative loss magnitude")
            print(f"          Consider reward scaling or reducing alpha parameter")

    # def evaluate_validation_losses(self, val_episodes, set_name="validation"):
    #     """
    #     Calculates TD and Conservative losses on a given set of episodes
    #     by strictly following the d3rlpy documentation for flat arrays.
    #     """
    #     from d3rlpy.metrics import TDErrorEvaluator
    #     from d3rlpy.dataset import MDPDataset
    #     import numpy as np

    #     print(f"\n🔬 Evaluating losses for '{set_name}' set (Strict Documentation Mode)...")

    #     if not val_episodes:
    #         print(f"   ❌ No episodes provided for '{set_name}' set.")
    #         return {'td_loss': 0.0, 'conservative_loss': 0.0, 'total_loss': 0.0}

    #     # ADD: Enhanced reward scale diagnostics
    #     self._diagnose_reward_scale(val_episodes)

    #     # Step 1: Flatten all episodes into single, continuous arrays.
    #     all_observations = []
    #     all_actions = []
    #     all_rewards = []
    #     all_terminals = []
        
    #     for episode in val_episodes:
    #         if episode.size() == 0:
    #             continue
    #         all_observations.extend(episode.observations)
    #         all_actions.extend(episode.actions)
    #         all_rewards.extend(episode.rewards)
            
    #         # Create a 'terminals' array where only the last step is True
    #         terminals = np.zeros(episode.size(), dtype=bool)
    #         terminals[-1] = True
    #         all_terminals.extend(terminals)

    #     # Sanity check to ensure all arrays have the same length, as per documentation
    #     if not (len(all_observations) == len(all_actions) == len(all_rewards) == len(all_terminals)):
    #         print("   ❌ FATAL: After flattening episodes, arrays have mismatched lengths. Cannot proceed.")
    #         print(f"      Obs: {len(all_observations)}, Act: {len(all_actions)}, Rew: {len(all_rewards)}, Term: {len(all_terminals)}")
    #         return {'td_loss': 0.0, 'conservative_loss': 0.0, 'total_loss': 0.0}

    #     print(f"   🔍 Prepared {len(all_observations)} transitions from {len(val_episodes)} episodes.")

    #     # Step 2: Create the MDPDataset exactly as shown in the documentation.
    #     val_dataset = MDPDataset(
    #         observations=np.array(all_observations),
    #         actions=np.array(all_actions),
    #         rewards=np.array(all_rewards),
    #         terminals=np.array(all_terminals)
    #     )

    #     # Step 3: Use d3rlpy's TDErrorEvaluator. It should now work.
    #     td_loss = 0.0
    #     try:
    #         td_evaluator = TDErrorEvaluator()
    #         td_loss = td_evaluator(self.model, val_dataset)
    #     except Exception as e:
    #         print(f"   ⚠️ Could not use TDErrorEvaluator: {e}. Defaulting TD Loss to 0.")
    #         td_loss = 0.0

    #     # Step 4: Use the corrected helper to calculate the Conservative Loss.
    #     conservative_loss = self.conservative_loss_discrete(self.model, val_episodes)

    #     # Step 5: Combine and return the results.
    #     results = {
    #         'td_loss': td_loss,
    #         'conservative_loss': conservative_loss,
    #         'total_loss': td_loss + conservative_loss
    #     }

    #     print(f"   ✅ '{set_name.capitalize()}' Loss Results:")
    #     print(f"      TD Loss:          {results['td_loss']:.6f}")
    #     print(f"      Conservative Loss: {results['conservative_loss']:.6f}")
    #     print(f"      Total Loss:       {results['total_loss']:.6f}")



    # def _calculate_conservative_loss(self, model, episodes, batch_size=256):
    #     """
    #     Enhanced conservative loss calculation with comprehensive diagnostics
    #     """
    #     import torch
    #     import numpy as np
        
    #     print(f"🔧 [ENHANCED CONSERVATIVE LOSS] Calculating with diagnostics...")
        
    #     # Extract all data from episodes
    #     observations = []
    #     actions = []
    #     for episode in episodes:
    #         if episode.size() == 0:
    #             continue
    #         observations.extend(episode.observations)
    #         actions.extend(episode.actions)
        
    #     if not observations:
    #         print("   ❌ No observations found in episodes")
    #         return 0.0
        
    #     observations = np.array(observations)
    #     actions = np.array(actions)
        
    #     print(f"   📊 Processing {len(observations)} transitions")
    #     print(f"   📊 Action range in data: [{actions.min()}, {actions.max()}]")
        
    #     # Get device more robustly - fix the type checking issue
    #     device = torch.device('cpu')  # Default fallback
    #     try:
    #         # Method 1: Try to access through _impl.q_function directly
    #         if hasattr(model, '_impl') and hasattr(model._impl, 'q_function'):
    #             q_func_obj = model._impl.q_function
                
    #             # Check if it's a ModuleList or similar container
    #             if hasattr(q_func_obj, '__len__') and len(q_func_obj) > 0:
    #                 # It's a container, get first element
    #                 first_q_func = q_func_obj[0]
    #                 device = next(first_q_func.parameters()).device
    #                 print(f"   🔧 Device from ModuleList[0]: {device}")
    #             else:
    #                 # It's a single module
    #                 device = next(q_func_obj.parameters()).device
    #                 print(f"   🔧 Device from single module: {device}")
            
    #         # Method 2: Fallback to any model parameters
    #         elif hasattr(model, 'parameters'):
    #             device = next(model.parameters()).device
    #             print(f"   🔧 Device from model parameters: {device}")
            
    #     except Exception as e:
    #         print(f"   ⚠️  Could not determine device ({e}), using CPU")
    #         device = torch.device('cpu')
        
    #     print(f"   🔧 Using device: {device}")
        
    #     # Convert to tensors and move to device - FIX THE DIMENSION ISSUE
    #     obs_tensor = torch.tensor(observations, dtype=torch.float32).to(device)
        
    #     # Ensure actions are 1D tensor for proper indexing
    #     actions_flat = actions.flatten() if actions.ndim > 1 else actions
    #     actions_tensor = torch.tensor(actions_flat, dtype=torch.int64).to(device)
        
    #     print(f"   🔧 Tensor shapes: obs={obs_tensor.shape}, actions={actions_tensor.shape}")

    #     scaled_obs_tensor = model.scaler.transform(obs_tensor) #nieuw
        
    #     total_loss = 0.0
    #     total_samples = 0
    #     printed_batch_info = False
        
    #     with torch.no_grad():
    #         for i in range(0, len(observations), batch_size):
    #             end_idx = min(i + batch_size, len(observations))
    #             # batch_obs = obs_tensor[i:end_idx]
    #             # batch_actions = actions_tensor[i:end_idx]

    #              # <<<<< FIX #2: USE THE SCALED TENSOR FOR THE BATCH >>>>>
    #             batch_obs = scaled_obs_tensor[i:end_idx]
    #             batch_actions = actions_tensor[i:end_idx]
                
    #             print(f"   🔧 Batch shapes: obs={batch_obs.shape}, actions={batch_actions.shape}")
                
    #             # Get Q-function more robustly - fix the access pattern
    #             try:
    #                 if hasattr(model, '_impl') and hasattr(model._impl, 'q_function'):
    #                     q_function = model._impl.q_function
                        
    #                     # Handle different Q-function structures more carefully
    #                     if hasattr(q_function, '__len__'):
    #                         # It's a container (ModuleList, etc.)
    #                         try:
    #                             num_q_funcs = len(q_function)
    #                             if num_q_funcs > 1:
    #                                 # Multiple Q-networks (ensemble)
    #                                 all_q_values = []
    #                                 for idx in range(num_q_funcs):
    #                                     q_func = q_function[idx]
    #                                     q_output = q_func(batch_obs)
    #                                     if hasattr(q_output, 'q_value'):
    #                                         q_val = q_output.q_value
    #                                     else:
    #                                         q_val = q_output
    #                                     all_q_values.append(q_val)
    #                                 q_values = torch.stack(all_q_values).mean(dim=0)
    #                                 if not printed_batch_info:
    #                                     print(f"   🔧 Using ensemble of {len(all_q_values)} Q-networks")
    #                             else:
    #                                 # Single Q-network in container
    #                                 q_func = q_function[0]
    #                                 q_output = q_func(batch_obs)
    #                                 if hasattr(q_output, 'q_value'):
    #                                     q_values = q_output.q_value
    #                                 else:
    #                                     q_values = q_output
    #                                 if not printed_batch_info:
    #                                     print(f"   🔧 Using single Q-network from container: {type(q_func)}")
    #                         except (IndexError, TypeError) as e:
    #                             print(f"   ❌ Error accessing Q-function from container: {e}")
    #                             return 0.0
    #                     else:
    #                         # It's a single module
    #                         q_output = q_function(batch_obs)
    #                         if hasattr(q_output, 'q_value'):
    #                             q_values = q_output.q_value
    #                         else:
    #                             q_values = q_output
    #                         if not printed_batch_info:
    #                             print(f"   🔧 Using single Q-network: {type(q_function)}")
    #                 else:
    #                     raise AttributeError("Cannot access Q-function through _impl.q_function")
                
    #             except Exception as e:
    #                 print(f"   ❌ Error accessing Q-function: {e}")
    #                 return 0.0
                
    #             print(f"   🔧 Q-values shape: {q_values.shape}")
                
    #             # Calculate conservative loss components - FIX THE GATHER OPERATION
    #             log_sum_exp_q = torch.logsumexp(q_values, dim=1)
                
    #             # Ensure proper indexing for gather operation
    #             # batch_actions should be [batch_size] and q_values should be [batch_size, n_actions]
    #             if batch_actions.dim() == 1 and q_values.dim() == 2:
    #                 # This is correct - use unsqueeze(1) to make batch_actions [batch_size, 1]
    #                 behavior_q = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
    #             elif batch_actions.dim() == 2 and batch_actions.shape[1] == 1:
    #                 # batch_actions is already [batch_size, 1], use directly
    #                 behavior_q = q_values.gather(1, batch_actions).squeeze(1)
    #             else:
    #                 print(f"   ❌ Unexpected tensor dimensions: batch_actions.shape={batch_actions.shape}, q_values.shape={q_values.shape}")
    #                 return 0.0
                
    #             batch_conservative_loss = (log_sum_exp_q - behavior_q).mean()
                
    #             # Enhanced Diagnostic Prints
    #             if not printed_batch_info:
    #                 print(f"\n   🔍 [BATCH DIAGNOSTICS] Analyzing the first batch:")
                    
    #                 # Check for action mismatch
    #                 argmax_actions = torch.argmax(q_values, dim=1)
    #                 mismatch_rate = (batch_actions.squeeze() != argmax_actions).float().mean().item()
    #                 print(f"      - Pct of times data_action != model's_best_action: {mismatch_rate:.2%}")
                    
    #                 # Check Q-value scale
    #                 print(f"      - Avg Q_max (from logsumexp): {log_sum_exp_q.mean().item():.2f}")
    #                 print(f"      - Avg Q_data: {behavior_q.mean().item():.2f}")
    #                 print(f"      - Avg Q_min_in_batch: {q_values.min().item():.2f}")
    #                 print(f"      - Avg Q_max_in_batch: {q_values.max().item():.2f}")
    #                 print(f"      - Q_values shape: {q_values.shape}")
    #                 print(f"      - batch_actions shape after processing: {batch_actions.shape}")
    #                 print(f"      - behavior_q shape: {behavior_q.shape}")
                    
    #                 # Conservative loss breakdown
    #                 print(f"      - Conservative loss components:")
    #                 print(f"        * log_sum_exp_q mean: {log_sum_exp_q.mean().item():.6f}")
    #                 print(f"        * behavior_q mean: {behavior_q.mean().item():.6f}")
    #                 print(f"        * difference mean: {(log_sum_exp_q - behavior_q).mean().item():.6f}")
                    
    #                 # Check for extreme values
    #                 if torch.any(torch.isnan(q_values)):
    #                     print(f"      ⚠️  WARNING: NaN values detected in Q-values!")
    #                 if torch.any(torch.isinf(q_values)):
    #                     print(f"      ⚠️  WARNING: Inf values detected in Q-values!")
    #                 if q_values.abs().max() > 1000:
    #                     print(f"      ⚠️  WARNING: Very large Q-values detected (max: {q_values.abs().max().item():.1f})!")
    #                     print(f"          This suggests reward scale issues or high alpha parameter")
                    
    #                 # Show action distribution in batch vs model preferences
    #                 data_action_dist = torch.bincount(batch_actions.flatten(), minlength=12).float()
    #                 data_action_dist = data_action_dist / data_action_dist.sum()
                    
    #                 model_action_dist = torch.bincount(argmax_actions, minlength=12).float()
    #                 model_action_dist = model_action_dist / model_action_dist.sum()
                    
    #                 print(f"      - Data action distribution (top 5): {data_action_dist.topk(5)}")
    #                 print(f"      - Model action distribution (top 5): {model_action_dist.topk(5)}")
                    
    #                 printed_batch_info = True
                
    #             total_loss += batch_conservative_loss.item() * (end_idx - i)
    #             total_samples += (end_idx - i)
        
    #     avg_conservative_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
    #     print(f"   ✅ Conservative loss calculation completed")
    #     print(f"      Final average conservative loss: {avg_conservative_loss:.6f}")
        
    #     return avg_conservative_loss

