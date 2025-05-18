#!/usr/bin/env python3
# -----------------------------------------------------------
# evaluator.py  --  Evaluation for HFNC CQL model
# 
# This module handles the evaluation of trained CQL models:
# - Calculating performance metrics on test episodes
# - Analyzing model predictions and policy match rates
# - Adding training parameters to metrics for complete reporting
# 
# Proper evaluation is crucial for understanding model performance
# and diagnosing potential issues in the training process.
# -----------------------------------------------------------
import numpy as np

def evaluate_model(model, test_episodes):
    """
    Evaluate the trained model on test episodes to assess performance.
    
    This function calculates key metrics to evaluate the model:
    1. Average returns (sum of rewards) on test episodes
    2. Action match rate between predicted and actual actions
    
    These metrics help understand both how well the model matches the
    clinician behavior and the expected outcomes from following the policy.
    
    Args:
        model (DiscreteCQL): The trained CQL model to evaluate
        test_episodes (list): List of Episode objects from the test set
        
    Returns:
        dict: Dictionary containing evaluation metrics:
            - test_returns_mean: Average return across test episodes
            - test_returns_std: Standard deviation of returns
            - policy_match_rate_mean: Average agreement with actual actions
            - policy_match_rate_std: Standard deviation of match rates
    
    Note:
        Missing the predict method or errors during prediction will be
        handled gracefully with appropriate error messages.
    """
    print("Evaluating on test episodes...")
    metrics = {}
    
    try:
        # Check if the model has predict method
        # This ensures compatibility with different versions of d3rlpy
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
            
            # For each observation, get the predicted action from the model
            # This shows what the model would do in each state
            for obs in observations:
                try:
                    # Handle potential issues with model.predict
                    # Reshape is needed as predict expects a batch dimension
                    action = model.predict(obs.reshape(1, -1))[0]
                    actions.append(action)
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    # Use a default action (e.g. 0) when prediction fails
                    actions.append(0)
            
            # Compare to actual actions in the episode to calculate match rate
            # Higher match rates indicate better imitation of clinician behavior
            match_count = sum(a1 == a2 for a1, a2 in zip(actions, episode.actions))
            this_match_rate = match_count / len(actions) if len(actions) > 0 else 0
            match_rates.append(this_match_rate)
            
            # Calculate episode return to evaluate expected policy performance
            # Higher returns suggest better patient outcomes under the policy
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
        # This can help identify outlier cases for further analysis
        if test_returns:
            max_return_idx = np.argmax(test_returns)
            min_return_idx = np.argmin(test_returns)
            # Ensure that the values from test_returns are Python scalars before formatting
            print(f"Highest return episode: {max_return_idx} with return {test_returns[max_return_idx].item():.4f}")
            print(f"Lowest return episode: {min_return_idx} with return {test_returns[min_return_idx].item():.4f}")

    except Exception as e:
        print(f"Warning: Could not evaluate: {e}")
        print("Skipping evaluation phase")
    
    return metrics

def add_training_params_to_metrics(metrics, args):
    """
    Add training parameters to metrics for complete reporting.
    
    This ensures that the hyperparameters used for training are
    saved alongside the resulting metrics, which is essential for
    reproducibility and hyperparameter optimization.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics
        args (argparse.Namespace): Parsed command line arguments
                                  containing training parameters
        
    Returns:
        dict: Updated metrics dictionary with training parameters added
        
    Note:
        This function is useful for maintaining a record of all experiment
        settings alongside their results.
    """
    metrics["alpha"] = args.alpha        # CQL conservatism parameter
    metrics["epochs"] = args.epochs      # Number of training epochs
    metrics["batch_size"] = args.batch   # Batch size used for updates
    metrics["learning_rate"] = args.lr   # Learning rate for optimizer
    metrics["gamma"] = args.gamma        # Discount factor
    
    return metrics

def analyze_predictions(model, test_episodes, top_n=5):
    """
    Analyze model predictions on test episodes to detect issues and provide insights.
    
    This function performs a detailed analysis of the model's predictions
    on a subset of test episodes, looking at:
    1. Action match rates (agreement with actual clinical decisions)
    2. Confidence in predictions (difference between best and second best Q-values)
    3. Specific disagreements between model and actual actions
    
    Args:
        model (DiscreteCQL): The trained CQL model to analyze
        test_episodes (list): List of Episode objects from the test set
        top_n (int): Number of episodes to analyze in detail
                    Default is 5
                    
    Side effects:
        Prints detailed analysis of model predictions to console
        
    Note:
        This function is particularly valuable for diagnosing issues with 
        the model's policy and understanding where and why it disagrees
        with clinician decisions.
    """
    print(f"Analyzing model predictions on {len(test_episodes)} test episodes...")
    
    if not hasattr(model, 'predict'):
        print("Model doesn't have predict method, skipping prediction analysis")
        return
    
    try:
        # Select a few episodes for detailed analysis
        # Limiting to top_n keeps the output manageable
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
                # This helps diagnose uncertain predictions
                
                # Initialize defaults for current step
                current_best_action_for_step = -1  # Default if prediction fails
                current_confidence_for_step = 0.0    # Default confidence

                try:
                    # Get the model's chosen (best) action
                    current_best_action_for_step = model.predict(obs.reshape(1, -1))[0]

                    # Get Q-values for all actions to calculate confidence
                    q_values_for_all_actions_list = []
                    current_obs_batch = obs.reshape(1, -1)  # Prepare observation batch

                    if hasattr(model, 'action_size') and model.action_size is not None:
                        num_actions = model.action_size
                        for act_idx in range(num_actions):
                            # Shape action as (batch_size=1, action_dimensionality=1) for predict_value
                            action_batch_for_q = np.array([[act_idx]], dtype=np.int64)
                            q_val = model.predict_value(current_obs_batch, action_batch_for_q)
                            # q_val is likely a numpy array like array([value]), so use .item()
                            q_values_for_all_actions_list.append(q_val.item())
                        
                        if len(q_values_for_all_actions_list) > 1:
                            sorted_q_values = np.sort(np.array(q_values_for_all_actions_list))[::-1]
                            current_confidence_for_step = sorted_q_values[0] - sorted_q_values[1]
                        # If only 1 action, confidence is typically 0.0 or undefined.
                        # current_confidence_for_step is already 0.0 by default, covering this.
                    else:
                        # model.action_size not available, confidence remains 0.0
                        print(f"DEBUG: model.action_size not available. Confidence will be 0.0 for this step.")
                    
                    predicted_actions.append(current_best_action_for_step)
                    confidence_scores.append(current_confidence_for_step)

                except Exception as e:
                    # This 'e' is the exception from predict, predict_value, or the Q-value loop
                    print(f"DEBUG: Exception during model prediction/Q-value retrieval: {e}")
                    print(f"DEBUG: Observation causing error: {obs}")
                    print(f"DEBUG: Observation shape: {obs.shape}, dtype: {obs.dtype}")
                    if np.isnan(obs).any():
                        print(f"DEBUG: NaN values found in observation: {obs[np.isnan(obs)]}")
                    if np.isinf(obs).any():
                        print(f"DEBUG: Infinite values found in observation: {obs[np.isinf(obs)]}")
                    
                    predicted_actions.append(-1)  # Marker for failed prediction
                    confidence_scores.append(0.0) # Default confidence on error
            
            # Calculate statistics
            match_count = sum(a1 == a2 for a1, a2 in zip(predicted_actions, episode_actions))
            match_rate = match_count / len(observations) if len(observations) > 0 else 0
            episode_return = sum(episode_rewards)
            
            # Print summary
            print(f"  Length: {len(observations)} steps")
            print(f"  Return: {episode_return.item():.4f}")
            print(f"  Action match rate: {match_rate.item():.4f} ({match_count}/{len(observations)})")

            if confidence_scores:
                # np.mean, np.min, np.max on a list of numpy scalars return a numpy scalar. Convert to Python scalar.
                print(f"  Avg confidence: {np.mean(confidence_scores).item():.4f}")
                print(f"  Min confidence: {np.min(confidence_scores).item():.4f}")
                print(f"  Max confidence: {np.max(confidence_scores).item():.4f}")
            
            # Find disagreements between model and actual actions
            # These are the most interesting cases to analyze
            disagreements = [(i, episode_actions[i], predicted_actions[i]) 
                           for i in range(len(observations)) 
                           if episode_actions[i] != predicted_actions[i]]
            
            if disagreements:
                print(f"  Sample disagreements (step, actual, predicted):")
                for step, actual, predicted in disagreements[:5]:  # Show at most 5
                    print(f"    Step {step}: actual={actual}, predicted={predicted}")
    
    except Exception as e:
        print(f"Error during prediction analysis: {e}")