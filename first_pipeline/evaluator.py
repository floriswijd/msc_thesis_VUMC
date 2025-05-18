#!/usr/bin/env python3

import numpy as np


def evaluate_model(model, test_episodes):
    print("Evaluating on test episodes...")
    metrics = {}
    try:
        if not hasattr(model, "predict"):
            print("Model doesn't have predict method, skipping direct evaluation")
            return metrics
        print("Evaluating using direct prediction...")
        test_returns = []
        match_rates = []
        for episode in test_episodes:
            observations = episode.observations
            actions = []
            for obs in observations:
                try:
                    action = model.predict(obs.reshape(1, -1))[0]
                    actions.append(action)
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    actions.append(0)
            match_count = sum(a1 == a2 for a1, a2 in zip(actions, episode.actions))
            this_match_rate = match_count / len(actions) if len(actions) > 0 else 0
            match_rates.append(this_match_rate)
            episode_return = sum(episode.rewards)
            test_returns.append(episode_return)
        metrics["test_returns_mean"] = float(np.mean(test_returns))
        metrics["test_returns_std"] = float(np.std(test_returns))
        metrics["policy_match_rate_mean"] = float(np.mean(match_rates))
        metrics["policy_match_rate_std"] = float(np.std(match_rates))
        print(
            f"Calculated test returns: mean={metrics['test_returns_mean']:.4f}, std={metrics['test_returns_std']:.4f}"
        )
        print(
            f"Policy match rate: mean={metrics['policy_match_rate_mean']:.4f}, std={metrics['policy_match_rate_std']:.4f}"
        )
        if test_returns:
            max_return_idx = np.argmax(test_returns)
            min_return_idx = np.argmin(test_returns)
            print(
                f"Highest return episode: {max_return_idx} with return {test_returns[max_return_idx].item():.4f}"
            )
            print(
                f"Lowest return episode: {min_return_idx} with return {test_returns[min_return_idx].item():.4f}"
            )
    except Exception as e:
        print(f"Warning: Could not evaluate: {e}")
        print("Skipping evaluation phase")
    return metrics


def add_training_params_to_metrics(metrics, args):
    metrics["alpha"] = args.alpha
    metrics["epochs"] = args.epochs
    metrics["batch_size"] = args.batch
    metrics["learning_rate"] = args.lr
    metrics["gamma"] = args.gamma
    return metrics


def analyze_predictions(model, test_episodes, top_n=5):
    print(f"Analyzing model predictions on {len(test_episodes)} test episodes...")
    if not hasattr(model, "predict"):
        print("Model doesn't have predict method, skipping prediction analysis")
        return
    try:
        episodes_to_analyze = min(top_n, len(test_episodes))
        for i in range(episodes_to_analyze):
            episode = test_episodes[i]
            print(f"\nAnalyzing Episode {i}:")
            observations = episode.observations
            episode_rewards = episode.rewards
            episode_actions = episode.actions
            predicted_actions = []
            confidence_scores = []
            for obs in observations:
                current_best_action_for_step = -1
                current_confidence_for_step = 0.0
                try:
                    current_best_action_for_step = model.predict(obs.reshape(1, -1))[0]
                    q_values_for_all_actions_list = []
                    current_obs_batch = obs.reshape(1, -1)
                    if hasattr(model, "action_size") and model.action_size is not None:
                        num_actions = model.action_size
                        for act_idx in range(num_actions):
                            action_batch_for_q = np.array([[act_idx]], dtype=np.int64)
                            q_val = model.predict_value(current_obs_batch, action_batch_for_q)
                            q_values_for_all_actions_list.append(q_val.item())
                        if len(q_values_for_all_actions_list) > 1:
                            sorted_q_values = np.sort(np.array(q_values_for_all_actions_list))[::-1]
                            current_confidence_for_step = sorted_q_values[0] - sorted_q_values[1]
                    else:
                        print(
                            f"DEBUG: model.action_size not available. Confidence will be 0.0 for this step."
                        )
                    predicted_actions.append(current_best_action_for_step)
                    confidence_scores.append(current_confidence_for_step)
                except Exception as e:
                    print(f"DEBUG: Exception during model prediction/Q-value retrieval: {e}")
                    print(f"DEBUG: Observation causing error: {obs}")
                    print(f"DEBUG: Observation shape: {obs.shape}, dtype: {obs.dtype}")
                    if np.isnan(obs).any():
                        print(f"DEBUG: NaN values found in observation: {obs[np.isnan(obs)]}")
                    if np.isinf(obs).any():
                        print(f"DEBUG: Infinite values found in observation: {obs[np.isinf(obs)]}")
                    predicted_actions.append(-1)
                    confidence_scores.append(0.0)
            match_count = sum(a1 == a2 for a1, a2 in zip(predicted_actions, episode_actions))
            match_rate = match_count / len(observations) if len(observations) > 0 else 0
            episode_return = sum(episode_rewards)
            print(f"  Length: {len(observations)} steps")
            print(f"  Return: {episode_return.item():.4f}")
            print(f"  Action match rate: {match_rate.item():.4f} ({match_count}/{len(observations)})")
            if confidence_scores:
                print(f"  Avg confidence: {np.mean(confidence_scores).item():.4f}")
                print(f"  Min confidence: {np.min(confidence_scores).item():.4f}")
                print(f"  Max confidence: {np.max(confidence_scores).item():.4f}")
            disagreements = [
                (i, episode_actions[i], predicted_actions[i])
                for i in range(len(observations))
                if episode_actions[i] != predicted_actions[i]
            ]
            if disagreements:
                print(f"  Sample disagreements (step, actual, predicted):")
                for step, actual, predicted in disagreements[:5]:
                    print(f"    Step {step}: actual={actual}, predicted={predicted}")
    except Exception as e:
        print(f"Error during prediction analysis: {e}")