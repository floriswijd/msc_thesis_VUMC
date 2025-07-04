#!/usr/bin/env python3
"""
Feature importance analysis for HFNC CQL model using multiple approaches
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not available")

class CQLFeatureImportanceAnalyzer:
    """Comprehensive feature importance analysis for CQL models"""
    
    def __init__(self, model, state_columns, config_path=None):
        self.model = model
        self.state_columns = state_columns
        self.config_path = config_path
        self.results = {}
    
    def analyze_feature_importance(self, test_episodes, save_dir="feature_importance_analysis"):
        """Run comprehensive feature importance analysis"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print("🔍 Running comprehensive feature importance analysis...")
        print(f"📋 Total features in dataset: {len(self.state_columns)}")
        print(f"📋 Features: {self.state_columns}")
        
        # Extract data for analysis
        states, actions = self._extract_data(test_episodes)
        
        # 1. SHAP Analysis (if available)
        if SHAP_AVAILABLE:
            shap_results = self._shap_analysis(states, actions, save_dir)
            self.results['shap'] = shap_results
        
        # 2. Permutation Importance
        perm_results = self._permutation_importance(states, actions, save_dir)
        self.results['permutation'] = perm_results
        
        # 3. Q-value Sensitivity Analysis
        qvalue_results = self._qvalue_sensitivity_analysis(states, save_dir)
        self.results['qvalue_sensitivity'] = qvalue_results
        
        # 4. Gradient-based Importance
        grad_results = self._gradient_based_importance(states, save_dir)
        self.results['gradient_based'] = grad_results
        
        # 5. Generate visualizations
        self._generate_plots(save_dir)
        
        # 6. Generate summary report
        self._generate_importance_report(save_dir)
        
        return self.results
    
    def _extract_data(self, test_episodes, max_samples=5000):
        """Extract state-action pairs for analysis"""
        states_list = []
        actions_list = []
        
        for episode in test_episodes:
            for state, action in zip(episode.observations, episode.actions):
                states_list.append(state)
                actions_list.append(action)
                
                if len(states_list) >= max_samples:
                    break
            if len(states_list) >= max_samples:
                break
        
        return np.array(states_list), np.array(actions_list)
    
    def _shap_analysis(self, states, actions, save_dir):
        """SHAP-based feature importance analysis with GradientExplainer for PyTorch"""
        print("🎯 Running SHAP analysis with GradientExplainer...")
        
        try:
            import torch
            import shap
            
            # Access the Q-function from d3rlpy's DiscreteCQLImpl
            if hasattr(self.model, '_impl') and self.model._impl is not None:
                # d3rlpy 2.x uses _impl
                q_func_list = self.model._impl.q_function  # This is a ModuleList
                print("✅ Accessed Q-function via _impl.q_function")
            elif hasattr(self.model, 'impl') and self.model.impl is not None:
                # d3rlpy 1.x uses impl
                q_func_list = self.model.impl.q_function   # This is a ModuleList
                print("✅ Accessed Q-function via impl.q_function")
            else:
                raise AttributeError("Cannot access underlying PyTorch Q-function")
            
            # Handle ModuleList - take the first Q-network
            if isinstance(q_func_list, torch.nn.ModuleList):
                q_func = q_func_list[0]  # Use the first Q-network
                print(f"📊 Using first Q-network from ModuleList: {type(q_func)}")
            else:
                q_func = q_func_list
                
            # Verify we have a PyTorch module
            if not isinstance(q_func, torch.nn.Module):
                raise TypeError(f"Q-function is not a PyTorch module: {type(q_func)}")
            
            print(f"📊 Q-function type: {type(q_func)}")
            print(f"📊 Q-function device: {next(q_func.parameters()).device}")
            
            # Enhanced sampling for better accuracy
            sample_size = min(1000, len(states))  # Increased from 100 to 500
            background_size = min(400, len(states))  # Increased from 50 to 200
            
            # Ensure we have enough data for non-overlapping samples
            total_needed = sample_size + background_size
            if total_needed > len(states):
                # Scale down proportionally if not enough data
                ratio = len(states) / total_needed
                sample_size = int(sample_size * ratio)
                background_size = int(background_size * ratio)
        
            # Enhanced background sampling with fallbacks
            try:
                background_states, background_indices = self._create_representative_background(
                    states, actions, background_size
                )
                print("✅ Using action-stratified background sampling")
            except:
                try:
                    background_states, background_indices = self._clinical_stratified_sampling(
                        states, background_size
                    )
                    print("✅ Using clinical-stratified background sampling")
                except:
                    # Final fallback to random sampling
                    background_indices = np.random.choice(len(states), background_size, replace=False)
                    background_states = states[background_indices]
                    print("✅ Using random background sampling (fallback)")
        
            # Then, select explanation samples from remaining data
            remaining_indices = np.setdiff1d(np.arange(len(states)), background_indices)
            if len(remaining_indices) >= sample_size:
                sample_indices = np.random.choice(remaining_indices, sample_size, replace=False)
            else:
                # If not enough remaining, use what we have
                sample_indices = remaining_indices
                sample_size = len(remaining_indices)
        
            states_sample = states[sample_indices]
            
            print(f"📊 Background samples: {len(background_states)}")
            print(f"📊 Explanation samples: {len(states_sample)}")
            print(f"📊 Sample overlap: {len(np.intersect1d(background_indices, sample_indices))} (should be 0)")
            
            # Get device from Q-function
            device = next(q_func.parameters()).device
            
            # Convert to PyTorch tensors on correct device
            background_tensor = torch.tensor(background_states, dtype=torch.float32).to(device)
            states_tensor = torch.tensor(states_sample, dtype=torch.float32).to(device)
            
            # Create a proper wrapper for the Q-function that works with GradientExplainer
            class QValueWrapper(torch.nn.Module):
                def __init__(self, q_network):
                    super().__init__()
                    self.q_network = q_network
                    
                def forward(self, x):
                    """Return Q-values - ensure proper shape for SHAP"""
                    # Get Q-values - d3rlpy returns QFunctionOutput object
                    q_output = self.q_network(x)
                    
                    # Extract the actual tensor from QFunctionOutput
                    if hasattr(q_output, 'q_value'):
                        q_values = q_output.q_value  # Shape: (batch_size, n_actions)
                    elif hasattr(q_output, 'values'):
                        q_values = q_output.values
                    elif hasattr(q_output, 'tensor'):
                        q_values = q_output.tensor
                    elif isinstance(q_output, torch.Tensor):
                        q_values = q_output
                    else:
                        raise TypeError(f"Could not extract tensor from Q-function output: {type(q_output)}")
                    
                    # Return the Q-value for the best action but keep 2D shape for SHAP
                    best_q_values = torch.max(q_values, dim=1)[0]  # Shape: (batch_size,)
                    
                    # SHAP expects 2D output, so reshape to (batch_size, 1)
                    return best_q_values.unsqueeze(1)
        except Exception as e:
            print(f"⚠️  SHAP analysis failed: {e}")
            return {
                'feature_importance': pd.DataFrame(),
                'shap_values': np.array([]),
                'method': 'SHAP GradientExplainer (Enhanced Accuracy)',
                'q_function_type': str(type(q_func)),
                'device': str(device) if 'device' in locals() else 'unknown'
            }
        # Create wrapper model
        wrapper_model = QValueWrapper(q_func)
        wrapper_model.eval()
        
        # Test the complete wrapper
        print("🔧 Testing complete wrapper...")
        with torch.no_grad():
            test_input = background_tensor[:2]
            test_wrapper_output = wrapper_model(test_input)
            print(f"📊 Wrapper output shape: {test_wrapper_output.shape}")
            print(f"📊 Wrapper output sample: {test_wrapper_output[:3]}")

        # SHAP Analysis - NOW INSIDE THE TRY BLOCK!
        print("🔧 Creating SHAP GradientExplainer...")
        explainer = shap.GradientExplainer(wrapper_model, background_tensor)
        shap_values = explainer.shap_values(states_tensor)
        
        # Convert to numpy if it's a tensor
        if isinstance(shap_values, torch.Tensor):
            shap_values = shap_values.detach().cpu().numpy()
        
        # If shap_values is a list, take first element
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            if isinstance(shap_values, torch.Tensor):
                shap_values = shap_values.detach().cpu().numpy()
        
        # Handle 3D SHAP values (samples, features, outputs) -> (samples, features)
        if shap_values.ndim == 3:
            shap_values = shap_values.squeeze(-1)  # Remove last dimension if it's 1
        
        print(f"📊 SHAP values shape: {shap_values.shape}")
        
        # Calculate feature importance (absolute mean SHAP values)
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.state_columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Generate visualizations
        self._generate_shap_plots(shap_values, states_sample, importance_df, save_dir)
        
        print(f"✅ SHAP analysis completed successfully!")
        print(f"📊 Top 5 most important features:")
        for i, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
            print(f"   {i}. {row['feature']}: {row['importance']:.4f}")
        
        return {
            'feature_importance': importance_df,
            'shap_values': shap_values,
            'method': 'SHAP GradientExplainer (Enhanced Accuracy)',
            'q_function_type': str(type(q_func)),
            'device': str(device)
        }
        
    

    def _generate_shap_plots(self, shap_values, states_sample, importance_df, save_dir):
        """Generate SHAP visualizations"""
        try:
            # SHAP summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values, 
                states_sample, 
                feature_names=self.state_columns, 
                show=False,
                max_display=20
            )
            plt.title('SHAP Summary Plot - Feature Importance for Q-Value Prediction')
            plt.tight_layout()
            plt.savefig(save_dir / 'shap_summary_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # SHAP waterfall plot for first prediction
            try:
                plt.figure(figsize=(10, 8))
                explanation = shap.Explanation(
                    values=shap_values[0],
                    base_values=0.0,  # Assume zero baseline for Q-values
                    data=states_sample[0],
                    feature_names=self.state_columns
                )
                shap.waterfall_plot(explanation, show=False, max_display=15)
                plt.title('SHAP Waterfall Plot - Single Prediction Example')
                plt.tight_layout()
                plt.savefig(save_dir / 'shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"⚠️ Could not generate waterfall plot: {e}")
            
            # Feature importance bar plot
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)
            y_pos = np.arange(len(top_features))
            
            plt.barh(y_pos, top_features['importance'])
            plt.yticks(y_pos, top_features['feature'])
            plt.xlabel('Mean |SHAP Value|')
            plt.title('SHAP Feature Importance - Top 20 Features')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / 'shap_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save to CSV
            importance_df.to_csv(save_dir / 'shap_feature_importance.csv', index=False)
            
        except Exception as e:
            print(f"⚠️ Error generating SHAP plots: {e}")
    
    # def _permutation_importance(self, states, actions, save_dir):
    #     """Permutation-based feature importance"""
    #     print("🔄 Running permutation importance analysis...")
        
    #     try:
    #         # Create a surrogate model to approximate the CQL policy
    #         rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            
    #         # Get predicted actions from CQL model
    #         predicted_actions = []
    #         for state in states:
    #             pred_action = self.model.predict(state.reshape(1, -1))[0]
    #             predicted_actions.append(pred_action)
            
    #         # Train surrogate model
    #         rf_model.fit(states, predicted_actions)
        
        #         # Calculate permutation importance
    #         perm_importance = permutation_importance(
    #             rf_model, states, predicted_actions, 
    #             n_repeats=10, random_state=42, scoring='accuracy'
    #         )
            
    #         # Create importance dataframe
    #         importance_df = pd.DataFrame({
    #             'feature': self.state_columns,
    #             'importance_mean': perm_importance.importances_mean,
    #             'importance_std': perm_importance.importances_std
    #         }).sort_values('importance_mean', ascending=False)
            
    #         return {
    #             'feature_importance': importance_df,
    #             'method': 'Permutation Importance'
            # }
            
        # except Exception as e:
        #     print(f"Permutation importance analysis failed: {e}")
        #     return None

    
    def _permutation_importance(self, states, actions, save_dir):
        """Direct permutation on CQL model (no surrogate)"""
        print("🔄 Running direct permutation importance analysis...")
        
        # Get baseline predictions
        baseline_actions = []
        for state in states:
            action = self.model.predict(state.reshape(1, -1))[0]
            baseline_actions.append(action)
        baseline_actions = np.array(baseline_actions)
        
        flip_rates = []
        for feat_idx, feature in enumerate(self.state_columns):
            # Shuffle this feature across states
            states_shuffled = states.copy()
            np.random.shuffle(states_shuffled[:, feat_idx])
            
            # Get predictions on shuffled data
            shuffled_actions = []
            for state in states_shuffled:
                action = self.model.predict(state.reshape(1, -1))[0]
                shuffled_actions.append(action)
            shuffled_actions = np.array(shuffled_actions)
            
            # Calculate action flip rate
            flip_rate = (baseline_actions != shuffled_actions).mean()
            flip_rates.append(flip_rate)
        
        # Create results
        importance_df = pd.DataFrame({
            'feature': self.state_columns,
            'action_flip_rate': flip_rates
        }).sort_values('action_flip_rate', ascending=False)
        
        return {
            'feature_importance': importance_df,
            'method': 'Direct Permutation (Action Flip Rate)'
        }
    
    def _qvalue_sensitivity_analysis(self, states, save_dir):
        """Analyze Q-value sensitivity to feature changes"""
        print("📈 Running Q-value sensitivity analysis...")
        
        if not hasattr(self.model, 'predict_value'):
            print("Model doesn't support Q-value prediction, skipping Q-value sensitivity")
            return None
        
        try:
            sensitivity_scores = {feature: [] for feature in self.state_columns}
            
            # Sample states for analysis
            sample_size = min(50, len(states))
            sample_indices = np.random.choice(len(states), sample_size, replace=False)
            
            for idx in sample_indices:
                state = states[idx].copy()
                
                # Get baseline Q-values
                baseline_q_values = []
                for action in range(12):
                    q_val = self.model.predict_value(
                        state.reshape(1, -1), 
                        np.array([[action]], dtype=np.int64)
                    ).item()
                    baseline_q_values.append(q_val)
                
                # Test sensitivity for each feature
                for feat_idx, feature in enumerate(self.state_columns):
                    original_value = state[feat_idx]
                    
                    # Perturb feature by ±10%
                    perturbations = [0.9, 1.1] if original_value != 0 else [-0.1, 0.1]
                    
                    max_q_change = 0
                    for perturbation in perturbations:
                        perturbed_state = state.copy()
                        if original_value != 0:
                            perturbed_state[feat_idx] = original_value * perturbation
                        else:
                            perturbed_state[feat_idx] = perturbation
                        
                        # Get Q-values for perturbed state
                        perturbed_q_values = []
                        for action in range(12):
                            q_val = self.model.predict_value(
                                perturbed_state.reshape(1, -1), 
                                np.array([[action]], dtype=np.int64)
                            ).item()
                            perturbed_q_values.append(q_val)
                        
                        # Calculate Q-value change
                        q_change = np.abs(np.array(perturbed_q_values) - np.array(baseline_q_values)).max()
                        max_q_change = max(max_q_change, q_change)
                    
                    sensitivity_scores[feature].append(max_q_change)
            
            # Calculate average sensitivity
            avg_sensitivity = {
                feature: np.mean(scores) for feature, scores in sensitivity_scores.items()
            }
            
            # Create sensitivity dataframe
            sensitivity_df = pd.DataFrame({
                'feature': list(avg_sensitivity.keys()),
                'q_value_sensitivity': list(avg_sensitivity.values())
            }).sort_values('q_value_sensitivity', ascending=False)
            
            return {
                'feature_sensitivity': sensitivity_df,
                'method': 'Q-value Sensitivity'
            }
            
        except Exception as e:
            print(f"Q-value sensitivity analysis failed: {e}")
            return None
    
    def _gradient_based_importance(self, states, save_dir):
        """Gradient-based feature importance (numerical approximation)"""
        print("🎯 Running gradient-based importance analysis...")
        
        try:
            feature_gradients = {feature: [] for feature in self.state_columns}
            
            # Sample states for gradient analysis
            sample_size = min(20, len(states))
            sample_indices = np.random.choice(len(states), sample_size, replace=False)
            
            for idx in sample_indices:
                state = states[idx]
                
                # Get predicted action
                predicted_action = self.model.predict(state.reshape(1, -1))[0]
                
                # Compute numerical gradients (approximation)
                gradients = []
                for feat_idx in range(len(state)):
                    # Small perturbation
                    epsilon = 1e-5
                    
                    # Forward pass
                    state_plus = state.copy()
                    state_plus[feat_idx] += epsilon
                    
                    state_minus = state.copy()
                    state_minus[feat_idx] -= epsilon
                    
                    # Get Q-values (if available) or use action probabilities
                    if hasattr(self.model, 'predict_value'):
                        q_plus = self.model.predict_value(
                            state_plus.reshape(1, -1), 
                            np.array([[predicted_action]], dtype=np.int64)
                        ).item()
                        q_minus = self.model.predict_value(
                            state_minus.reshape(1, -1), 
                            np.array([[predicted_action]], dtype=np.int64)
                        ).item()
                        
                        gradient = (q_plus - q_minus) / (2 * epsilon)
                    else:
                        # Use action prediction difference as proxy
                        action_plus = self.model.predict(state_plus.reshape(1, -1))[0]
                        action_minus = self.model.predict(state_minus.reshape(1, -1))[0]
                        gradient = 1.0 if action_plus != action_minus else 0.0
                    
                    gradients.append(abs(gradient))
                
                # Store gradients for each feature
                for feat_idx, gradient in enumerate(gradients):
                    feature_gradients[self.state_columns[feat_idx]].append(gradient)
            
            # Calculate average gradient magnitude
            avg_gradients = {
                feature: np.mean(grads) for feature, grads in feature_gradients.items()
            }
            
            # Create gradient importance dataframe
            gradient_df = pd.DataFrame({
                'feature': list(avg_gradients.keys()),
                'gradient_importance': list(avg_gradients.values())
            }).sort_values('gradient_importance', ascending=False)
            
            return {
                'feature_importance': gradient_df,
                'method': 'Gradient-based'
            }
            
        except Exception as e:
            print(f"Gradient-based analysis failed: {e}")
            return None
    
    def _generate_plots(self, save_dir):
        """Generate feature importance visualizations"""
        print("📊 Generating feature importance visualizations...")
        
        # Combined importance plot
        available_methods = [method for method in ['shap', 'permutation', 'qvalue_sensitivity', 'gradient_based'] 
                           if method in self.results and self.results[method] is not None]
        
        if not available_methods:
            print("No successful analysis methods to plot")
            return
        
        n_methods = len(available_methods)
        n_cols = min(2, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8*n_rows))
        if n_methods == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        method_names = {
            'shap': 'SHAP',
            'permutation': 'Permutation',
            'qvalue_sensitivity': 'Q-value Sensitivity',
            'gradient_based': 'Gradient-based'
        }
        
        for idx, method in enumerate(available_methods):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            results = self.results[method]
            method_name = method_names[method]
            
            if 'feature_importance' in results:
                df = results['feature_importance']
                importance_col = df.columns[1]  # Get importance column name
                
                # Plot top 15 features
                top_features = df.head(15)
                y_pos = np.arange(len(top_features))
                
                ax.barh(y_pos, top_features[importance_col])
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_features['feature'], fontsize=8)
                ax.set_xlabel('Importance Score')
                ax.set_title(f'{method_name} Feature Importance')
                ax.invert_yaxis()
                
            elif 'feature_sensitivity' in results:
                df = results['feature_sensitivity']
                top_features = df.head(15)
                y_pos = np.arange(len(top_features))
                
                ax.barh(y_pos, top_features['q_value_sensitivity'])
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_features['feature'], fontsize=8)
                ax.set_xlabel('Sensitivity Score')
                ax.set_title(f'{method_name} Feature Importance')
                ax.invert_yaxis()
        
        # Hide empty subplots
        if n_methods < n_rows * n_cols:
            for idx in range(n_methods, n_rows * n_cols):
                row = idx // n_cols
                col = idx % n_cols
                ax = axes[row, col] if n_rows > 1 else axes[col]
                ax.set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'comprehensive_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Individual method plots
        for method in available_methods:
            self._plot_individual_method(method, save_dir)
    
    def _plot_individual_method(self, method, save_dir):
        """Plot detailed results for individual method"""
        if method not in self.results or self.results[method] is None:
            return
            
        results = self.results[method]
        
        plt.figure(figsize=(12, max(8, len(self.state_columns) * 0.3)))
        
        if 'feature_importance' in results:
            df = results['feature_importance']
            importance_col = df.columns[1]
            
            # Plot all features
            y_pos = np.arange(len(df))
            plt.barh(y_pos, df[importance_col])
            plt.yticks(y_pos, df['feature'])
            plt.xlabel('Importance Score')
            plt.title(f'{results["method"]} - All Features')
            plt.gca().invert_yaxis()
            
        elif 'feature_sensitivity' in results:
            df = results['feature_sensitivity']
            
            y_pos = np.arange(len(df))
            plt.barh(y_pos, df['q_value_sensitivity'])
            plt.yticks(y_pos, df['feature'])
            plt.xlabel('Sensitivity Score')
            plt.title(f'{results["method"]} - All Features')
            plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{method}_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_importance_report(self, save_dir):
        """Generate comprehensive feature importance report"""
        report_path = save_dir / 'feature_importance_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Feature Importance Analysis Report: HFNC CQL Model\n\n")
            f.write("## Executive Summary\n")
            f.write("This report provides a comprehensive analysis of feature importance in the ")
            f.write("Conservative Q-Learning (CQL) model for High-Flow Nasal Cannula (HFNC) parameter optimization.\n\n")
            
            f.write(f"**Total Features Analyzed**: {len(self.state_columns)}\n\n")
            
            # Method summaries
            f.write("## Analysis Methods\n\n")
            for method, results in self.results.items():
                if results is not None:
                    f.write(f"### {results['method']}\n")
                    
                    if 'feature_importance' in results:
                        df = results['feature_importance']
                        f.write(f"**Top 10 most important features:**\n")
                        for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
                            f.write(f"{i}. **{row['feature']}**: {row.iloc[1]:.4f}\n")
                    
                    elif 'feature_sensitivity' in results:
                        df = results['feature_sensitivity']
                        f.write(f"**Top 10 most sensitive features:**\n")
                        for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
                            f.write(f"{i}. **{row['feature']}**: {row['q_value_sensitivity']:.4f}\n")
                    
                    f.write("\n")
            
            # Combined ranking (if SHAP available)
            if 'shap' in self.results and self.results['shap'] is not None:
                f.write("## Key Findings\n")
                df = self.results['shap']['feature_importance']
                f.write("Based on SHAP analysis (most reliable method):\n\n")
                f.write("**Most Important Features for HFNC Parameter Decisions:**\n")
                for i, (_, row) in enumerate(df.head(5).iterrows(), 1):
                    f.write(f"- **{row['feature']}** (importance: {row['importance']:.4f})\n")
                
                f.write("\n**Clinical Interpretation:**\n")
                f.write("The model's decisions are primarily driven by the features listed above. ")
                f.write("These should be carefully monitored and validated in clinical practice.\n\n")
            
            f.write("## Recommendations\n")
            f.write("1. **Data Quality**: Ensure high-quality data collection for top-importance features\n")
            f.write("2. **Clinical Validation**: Validate that high-importance features align with clinical expectations\n")
            f.write("3. **Feature Engineering**: Consider creating composite features from important individual features\n")
            f.write("4. **Model Monitoring**: Track feature importance over time to detect model drift\n")

def analyze_feature_importance(model, test_episodes, state_columns, save_dir="feature_importance_analysis"):
    """Convenience function for feature importance analysis"""
    analyzer = CQLFeatureImportanceAnalyzer(model, state_columns)
    return analyzer.analyze_feature_importance(test_episodes, save_dir)