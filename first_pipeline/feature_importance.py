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
    print("⚠️  SHAP not available. Install with: pip install shap")

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
        """SHAP-based feature importance analysis"""
        print("🎯 Running SHAP analysis...")
        
        try:
            # Create a wrapper function for the model
            def model_predict_proba(X):
                """Convert CQL model to probability distribution over actions"""
                batch_size = X.shape[0]
                action_probs = np.zeros((batch_size, 12))  # Assuming 12 actions
                
                for i, state in enumerate(X):
                    try:
                        # Get Q-values for all actions
                        q_values = []
                        for action in range(12):
                            if hasattr(self.model, 'predict_value'):
                                q_val = self.model.predict_value(
                                    state.reshape(1, -1), 
                                    np.array([[action]], dtype=np.int64)
                                ).item()
                                q_values.append(q_val)
                            else:
                                # Fallback: predict action and create one-hot
                                pred_action = self.model.predict(state.reshape(1, -1))[0]
                                q_values = [1.0 if a == pred_action else 0.0 for a in range(12)]
                                break
                        
                        # Convert Q-values to probabilities using softmax
                        if len(q_values) == 12:
                            q_values = np.array(q_values)
                            # Apply temperature scaling for better probability distribution
                            temperature = 1.0
                            exp_q = np.exp(q_values / temperature)
                            action_probs[i] = exp_q / np.sum(exp_q)
                        else:
                            # Uniform distribution as fallback
                            action_probs[i] = np.ones(12) / 12
                    except Exception as e:
                        print(f"Error in SHAP prediction for sample {i}: {e}")
                        action_probs[i] = np.ones(12) / 12
                
                return action_probs
            
            # Sample data for SHAP (computationally expensive)
            sample_size = min(100, len(states))
            sample_indices = np.random.choice(len(states), sample_size, replace=False)
            states_sample = states[sample_indices]
            
            # Create SHAP explainer
            explainer = shap.Explainer(model_predict_proba, states_sample[:50])
            shap_values = explainer(states_sample)
            
            # Calculate feature importance
            feature_importance = np.abs(shap_values.values).mean(axis=(0, 2))
            
            # Create SHAP summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values[:, :, 0], states_sample, 
                            feature_names=self.state_columns, show=False)
            plt.title('SHAP Summary Plot - Feature Importance for Action 0')
            plt.tight_layout()
            plt.savefig(save_dir / 'shap_summary_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Feature importance ranking
            importance_df = pd.DataFrame({
                'feature': self.state_columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            return {
                'feature_importance': importance_df,
                'shap_values': shap_values,
                'method': 'SHAP'
            }
            
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            return None
    
    def _permutation_importance(self, states, actions, save_dir):
        """Permutation-based feature importance"""
        print("🔄 Running permutation importance analysis...")
        
        try:
            # Create a surrogate model to approximate the CQL policy
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Get predicted actions from CQL model
            predicted_actions = []
            for state in states:
                pred_action = self.model.predict(state.reshape(1, -1))[0]
                predicted_actions.append(pred_action)
            
            # Train surrogate model
            rf_model.fit(states, predicted_actions)
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                rf_model, states, predicted_actions, 
                n_repeats=10, random_state=42, scoring='accuracy'
            )
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': self.state_columns,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            return {
                'feature_importance': importance_df,
                'method': 'Permutation Importance'
            }
            
        except Exception as e:
            print(f"Permutation importance analysis failed: {e}")
            return None
    
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