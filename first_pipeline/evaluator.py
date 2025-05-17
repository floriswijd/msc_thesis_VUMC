#!/usr/bin/env python3
# -----------------------------------------------------------
# evaluator.py  --  Evaluation for HFNC CQL model
# -----------------------------------------------------------
import numpy as np

def evaluate_model(model, test_episodes):
    """Evaluate the trained model on test episodes"""
    print("Evaluating on test episodes...")
    metrics = {}
    
    try:
        # Check if the model has predict method
        if not hasattr(model, 'predict'):
            print("Model doesn't have predict method, skipping direct evaluation")
            return metrics
            
        print("Evaluating using direct prediction...")
        
        # Calculate average returns and action match rate
        test_returns = []
        match_rates = []
        
        for episode in test_episodes:
            observations = episode.observations
            actions = []
            
            # For each observation, get the predicted action
            for obs in observations:
                try:
                    # Handle potential issues with model.predict
                    action = model.predict(obs.reshape(1, -1))[0]
                    actions.append(action)
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    # Use a default action (e.g. 0) when prediction fails
                    actions.append(0)
            
            # Compare to actual actions in the episode
            match_count = sum(a1 == a2 for a1, a2 in zip(actions, episode.actions))
            this_match_rate = match_count / len(actions) if len(actions) > 0 else 0
            match_rates.append(this_match_rate)
            
            # Calculate episode return
            episode_return = sum(episode.rewards)
            test_returns.append(episode_return)
        
        # Calculate and store metrics
        metrics["test_returns_mean"] = float(np.mean(test_returns))
        metrics["test_returns_std"] = float(np.std(test_returns))
        metrics["policy_match_rate_mean"] = float(np.mean(match_rates))
        metrics["policy_match_rate_std"] = float(np.std(match_rates))
        
        print(f"Calculated test returns: mean={metrics['test_returns_mean']:.4f}, std={metrics['test_returns_std']:.4f}")
        print(f"Policy match rate: mean={metrics['policy_match_rate_mean']:.4f}, std={metrics['policy_match_rate_std']:.4f}")
        
        # Look for episodes with very high or very low returns
        if test_returns:
            max_return_idx = np.argmax(test_returns)
            min_return_idx = np.argmin(test_returns)
            print(f"Highest return episode: {max_return_idx} with return {test_returns[max_return_idx]:.4f}")
            print(f"Lowest return episode: {min_return_idx} with return {test_returns[min_return_idx]:.4f}")

    except Exception as e:
        print(f"Warning: Could not evaluate: {e}")
        print("Skipping evaluation phase")
    
    return metrics

def add_training_params_to_metrics(metrics, args):
    """Add training parameters to metrics"""
    metrics["alpha"] = args.alpha
    metrics["epochs"] = args.epochs
    metrics["batch_size"] = args.batch
    metrics["learning_rate"] = args.lr
    metrics["gamma"] = args.gamma
    
    return metrics

def analyze_predictions(model, test_episodes, top_n=5):
    """Analyze model predictions on test episodes to detect issues"""
    print(f"Analyzing model predictions on {len(test_episodes)} test episodes...")
    
    if not hasattr(model, 'predict'):
        print("Model doesn't have predict method, skipping prediction analysis")
        return
    
    try:
        # Select a few episodes for detailed analysis
        episodes_to_analyze = min(top_n, len(test_episodes))
        
        for i in range(episodes_to_analyze):
            episode = test_episodes[i]
            print(f"\nAnalyzing Episode {i}:")
            
            observations = episode.observations
            episode_rewards = episode.rewards
            episode_actions = episode.actions
            
            # Get model predictions
            predicted_actions = []
            confidence_scores = []
            
            for obs in observations:
                # If model supports q-values, we can see the confidence in each action
                try:
                    q_values = model.predict_value(obs.reshape(1, -1))[0]
                    action = model.predict(obs.reshape(1, -1))[0]
                    
                    predicted_actions.append(action)
                    
                    # Calculate confidence as difference between best and second best action
                    if len(q_values) > 1:
                        sorted_q = np.sort(q_values)[::-1]  # Sort in descending order
                        confidence = sorted_q[0] - sorted_q[1]
                        confidence_scores.append(confidence)
                except Exception as e:
                    predicted_actions.append(-1)  # Marker for failed prediction
                    confidence_scores.append(0)
            
            # Calculate statistics
            match_count = sum(a1 == a2 for a1, a2 in zip(predicted_actions, episode_actions))
            match_rate = match_count / len(observations) if len(observations) > 0 else 0
            episode_return = sum(episode_rewards)
            
            # Print summary
            print(f"  Length: {len(observations)} steps")
            print(f"  Return: {episode_return:.4f}")
            print(f"  Action match rate: {match_rate:.4f} ({match_count}/{len(observations)})")
            
            if confidence_scores:
                print(f"  Avg confidence: {np.mean(confidence_scores):.4f}")
                print(f"  Min confidence: {np.min(confidence_scores):.4f}")
                print(f"  Max confidence: {np.max(confidence_scores):.4f}")
            
            # Find disagreements
            disagreements = [(i, episode_actions[i], predicted_actions[i]) 
                           for i in range(len(observations)) 
                           if episode_actions[i] != predicted_actions[i]]
            
            if disagreements:
                print(f"  Sample disagreements (step, actual, predicted):")
                for step, actual, predicted in disagreements[:5]:  # Show at most 5
                    print(f"    Step {step}: actual={actual}, predicted={predicted}")
    
    except Exception as e:
        print(f"Error during prediction analysis: {e}")