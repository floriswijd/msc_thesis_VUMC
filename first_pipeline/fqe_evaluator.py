# file: fqe_evaluator.py

import numpy as np
import d3rlpy
from d3rlpy.ope import FQEConfig, DiscreteFQE
from d3rlpy.metrics import InitialStateValueEstimationEvaluator
from d3rlpy.algos import QLearningAlgoBase   # Fixed import for d3rlpy 2.
from d3rlpy.dataset import create_infinite_replay_buffer
from typing import List, Dict, Any

# from d3rlpy.ope import DiscreteFQE, FQEConfig
# from d3rlpy.metrics import InitialStateValueEstimationEvaluator
# from d3rlpy.dataset import ReplayBuffer  # nodig vanaf v2.x
# from d3rlpy.dataset import create_infinite_replay_buffer
# from scipy.special import logsumexp                # fast, stable
# from d3rlpy.dataset import Episode
# from d3rlpy.algos import QLearningAlgoBase   # Fixed import for d3rlpy 2.8.1
# from d3rlpy.metrics import TDErrorEvaluator


# Your original function, refactored to be standalone.
#werkt het beste met 15000 stappen
def evaluate_policy_with_fqe(
    policy_to_evaluate: d3rlpy.algos.QLearningAlgoBase,
    train_episodes: List[d3rlpy.dataset.Episode],
    test_episodes: List[d3rlpy.dataset.Episode],
    n_fqe_steps: int = 50000,
    n_bootstrap_samples: int = 200,
    gamma: float = 0.99,
    fqe_learning_rate: float = 3e-4,
    device: str = "cuda:0" # or "cpu"
) -> Dict[str, Any]:
    """
    Evaluates a trained policy using Fitted Q Evaluation (FQE).

    Args:
        policy_to_evaluate: The trained d3rlpy algorithm (e.g., CQL) to evaluate.
        train_episodes: The dataset used to train the FQE model.
        test_episodes: The dataset used to estimate the policy's value.
        n_fqe_steps: Number of training steps for the FQE model.
        n_bootstrap_samples: Number of bootstrap iterations for the confidence interval.
        gamma: Discount factor.
        fqe_learning_rate: Learning rate for the FQE optimizer.
        device: The device to use for FQE training.

    Returns:
        A dictionary containing the point estimate, confidence interval, and other metrics.
    """
    print(f"🔄 Starting FQE: Training on {len(train_episodes)} episodes, evaluating on {len(test_episodes)} test episodes.")

    # 1. Configure and initialize the FQE model
    # FQE is trained to estimate the value of our specific 'policy_to_evaluate'
    fqe_config = FQEConfig(learning_rate=fqe_learning_rate, gamma=gamma)
    fqe = DiscreteFQE(
        algo=policy_to_evaluate,
        config=fqe_config,
        device=device,
    )

    # 2. Train the FQE model using the training data
    # The replay buffer should be created from the episodes used to train the FQE model
    train_buffer = create_infinite_replay_buffer(train_episodes)
    
    print(f"💪 Training FQE model for {n_fqe_steps} steps...")
    fqe.fit(
        train_buffer,
        n_steps=n_fqe_steps,
        # No need for evaluators during the FQE training itself for this workflow
        show_progress=True,
    )
    print("✅ FQE model training complete.")

    # 3. Evaluate the trained policy using the FQE model on the clean test data
    value_estimator = InitialStateValueEstimationEvaluator()
    test_buffer = create_infinite_replay_buffer(test_episodes)
    point_estimate = float(value_estimator(algo=fqe, dataset=test_buffer))
  
    print(f"   ➜ Point estimate V̂ = {point_estimate:.3f}")

    # 4. Perform bootstrapping on the test set to get a 95% Confidence Interval
    if n_bootstrap_samples > 0:
        print(f"🔁 Bootstrapping CI with {n_bootstrap_samples} replications...")
        boot_values = []
        rng = np.random.default_rng(seed=0)
        for _ in range(n_bootstrap_samples):
            # Create a bootstrap sample from the TEST episodes
            boot_eps = list(rng.choice(test_episodes, size=len(test_episodes), replace=True))
            boot_buffer = create_infinite_replay_buffer(boot_eps)
            v_hat = float(value_estimator(algo=fqe, dataset=boot_buffer))
            boot_values.append(v_hat)

        ci_low, ci_high = np.percentile(boot_values, [2.5, 97.5])
        ci_std = np.std(boot_values)
        print(f"✅ FQE Evaluation Complete: {point_estimate:.3f} [95% CI: {ci_low:.3f} – {ci_high:.3f}]")
    else:
        ci_low, ci_high, ci_std = None, None, None
        print("✅ FQE Evaluation Complete (no bootstrapping).")
        
    return {
        "fqe_point_estimate": float(point_estimate),
        "fqe_ci_95": [float(ci_low), float(ci_high)] if ci_low is not None else None,
        "fqe_std_dev": float(ci_std) if ci_std is not None else None,
        "n_fqe_steps": int(n_fqe_steps),
        "n_bootstrap_samples": int(n_bootstrap_samples),
    }

def calculate_behavior_policy_value(
    episodes: List[d3rlpy.dataset.Episode],
    gamma: float
) -> Dict[str, float]:
    """
    Calculates the value of the behavior policy using Monte Carlo evaluation.

    This computes the average cumulative discounted reward achieved in the dataset,
    serving as a baseline for the clinicians' performance.

    Args:
        episodes: A list of d3rlpy episodes to evaluate.
        gamma: The discount factor, which must be the same as the one used
               for training the agent.

    Returns:
        A dictionary with the mean, standard deviation, and standard error
        of the discounted returns.
    """
    all_returns = []
    for episode in episodes:
        episode_return = 0.0
        # Loop through rewards and apply discount factor
        for i, reward in enumerate(episode.rewards):
            episode_return += (gamma ** i) * reward
        all_returns.append(episode_return)

    mean_return = float(np.mean(all_returns))
    std_dev = float(np.std(all_returns))
    # Standard Error of the Mean is useful for comparing against the FQE CI
    std_err = std_dev / np.sqrt(len(all_returns))

    return {
        "behavior_mean_return": mean_return,
        "behavior_std_dev": std_dev,
        "behavior_std_err": std_err,
    }
