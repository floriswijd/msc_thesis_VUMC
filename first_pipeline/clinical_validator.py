#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

class HFNCClinicalValidator:
    """Clinical validation specifically for HFNC parameter optimization models"""
    
    def __init__(self, model):
        self.model = model
        
        # HFNC clinical parameters and safe ranges (based on literature)
        self.hfnc_parameters = {
            'flow_rate': {
                'safe_min': 10,     # L/min - minimum therapeutic flow
                'safe_max': 60,     # L/min - maximum safe flow
                'optimal_range': (20, 50),  # L/min - most commonly used range
                'pediatric_max': 25,  # L/min - maximum for pediatric patients
                'unit': 'L/min'
            },
            'fio2': {
                'safe_min': 0.21,   # Room air minimum
                'safe_max': 1.0,    # 100% oxygen maximum
                'optimal_range': (0.21, 0.6),  # Avoid oxygen toxicity
                'unit': 'fraction'
            },
            'temperature': {
                'safe_min': 31,     # °C - minimum to avoid discomfort
                'safe_max': 37,     # °C - body temperature maximum
                'optimal_range': (34, 37),  # °C - optimal comfort range
                'unit': '°C'
            }
        }
        
        # Clinical outcomes that indicate treatment success
        self.outcome_indicators = {
            'respiratory_rate_improvement': {'threshold': -5, 'better': 'lower'},
            'spo2_improvement': {'threshold': 2, 'better': 'higher'},
            'comfort_score_improvement': {'threshold': 1, 'better': 'higher'},
            'treatment_tolerance': {'threshold': 0.8, 'better': 'higher'}
        }
    
    def validate_clinical_safety(self, test_episodes, save_dir="clinical_validation"):
        """Comprehensive clinical safety validation for HFNC model"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print("🏥 Running HFNC-specific clinical validation...")
        
        safety_results = {}
        
        # 1. Parameter range adherence
        safety_results['parameter_safety'] = self._validate_parameter_ranges(test_episodes)
        
        # 2. Clinical appropriateness scoring
        safety_results['clinical_appropriateness'] = self._assess_clinical_appropriateness(test_episodes)
        
        # 3. Patient safety indicators
        safety_results['patient_safety'] = self._evaluate_patient_safety(test_episodes)
        
        # 4. Treatment effectiveness prediction
        safety_results['treatment_effectiveness'] = self._evaluate_treatment_effectiveness(test_episodes)
        
        # 5. Adverse event prediction
        safety_results['adverse_events'] = self._predict_adverse_events(test_episodes)
        
        # 6. Generate clinical visualizations
        self._generate_clinical_plots(test_episodes, save_dir)
        
        # 7. Generate clinical report
        self._generate_clinical_report(safety_results, save_dir)
        
        return safety_results
    
    def _validate_parameter_ranges(self, test_episodes):
        """Validate that predicted parameters fall within clinically safe ranges"""
        print("📋 Validating parameter safety ranges...")
        
        violations = {param: 0 for param in self.hfnc_parameters.keys()}
        total_predictions = 0
        parameter_distributions = {param: [] for param in self.hfnc_parameters.keys()}
        
        for episode in test_episodes:
            for obs in episode.observations:
                try:
                    # Get model prediction
                    action = self.model.predict(obs.reshape(1, -1))[0]
                    
                    # Convert action to HFNC parameters (this would depend on your action encoding)
                    hfnc_params = self._decode_action_to_hfnc_params(action, obs)
                    
                    total_predictions += 1
                    
                    # Check each parameter against safety ranges
                    for param, value in hfnc_params.items():
                        if param in self.hfnc_parameters:
                            parameter_distributions[param].append(value)
                            
                            safe_range = self.hfnc_parameters[param]
                            if value < safe_range['safe_min'] or value > safe_range['safe_max']:
                                violations[param] += 1
                                
                except Exception as e:
                    continue
        
        # Calculate violation rates
        violation_rates = {
            param: violations[param] / total_predictions if total_predictions > 0 else 0
            for param in violations.keys()
        }
        
        return {
            'violation_rates': violation_rates,
            'total_violations': sum(violations.values()),
            'total_predictions': total_predictions,
            'parameter_distributions': parameter_distributions,
            'safety_score': 1.0 - (sum(violations.values()) / (total_predictions * len(violations)) if total_predictions > 0 else 0)
        }
    
    def _assess_clinical_appropriateness(self, test_episodes):
        """Assess clinical appropriateness of parameter choices given patient state"""
        print("🩺 Assessing clinical appropriateness...")
        
        appropriateness_scores = []
        contextual_decisions = []
        
        for episode in test_episodes:
            for i, obs in enumerate(episode.observations):
                try:
                    # Get model prediction
                    action = self.model.predict(obs.reshape(1, -1))[0]
                    clinician_action = episode.actions[i]
                    
                    # Decode to HFNC parameters
                    model_params = self._decode_action_to_hfnc_params(action, obs)
                    clinician_params = self._decode_action_to_hfnc_params(clinician_action, obs)
                    
                    # Assess appropriateness based on patient state
                    appropriateness = self._score_parameter_appropriateness(model_params, obs)
                    appropriateness_scores.append(appropriateness)
                    
                    # Compare with clinician decision
                    contextual_decisions.append({
                        'model_appropriate': appropriateness > 0.7,
                        'clinician_action': clinician_action,
                        'model_action': action,
                        'patient_severity': self._assess_patient_severity(obs)
                    })
                    
                except Exception as e:
                    continue
        
        return {
            'mean_appropriateness': float(np.mean(appropriateness_scores)) if appropriateness_scores else 0,
            'std_appropriateness': float(np.std(appropriateness_scores)) if appropriateness_scores else 0,
            'high_appropriateness_rate': sum(1 for s in appropriateness_scores if s > 0.8) / len(appropriateness_scores) if appropriateness_scores else 0,
            'contextual_decisions': contextual_decisions[:50]  # Sample for analysis
        }
    
    def _evaluate_patient_safety(self, test_episodes):
        """Evaluate patient safety indicators"""
        print("🛡️ Evaluating patient safety indicators...")
        
        safety_violations = []
        risk_assessments = []
        
        for episode in test_episodes:
            episode_safety_score = 0
            episode_risk_factors = []
            
            for obs in episode.observations:
                try:
                    action = self.model.predict(obs.reshape(1, -1))[0]
                    hfnc_params = self._decode_action_to_hfnc_params(action, obs)
                    
                    # Check for specific safety concerns
                    safety_checks = self._perform_safety_checks(hfnc_params, obs)
                    
                    if safety_checks['high_risk']:
                        safety_violations.append(safety_checks['risk_factors'])
                    
                    episode_safety_score += safety_checks['safety_score']
                    episode_risk_factors.extend(safety_checks['risk_factors'])
                    
                except Exception as e:
                    continue
            
            if len(episode.observations) > 0:
                avg_episode_safety = episode_safety_score / len(episode.observations)
                risk_assessments.append({
                    'safety_score': avg_episode_safety,
                    'risk_factors': list(set(episode_risk_factors))  # Unique risk factors
                })
        
        return {
            'overall_safety_score': float(np.mean([r['safety_score'] for r in risk_assessments])) if risk_assessments else 0,
            'high_risk_episodes': len([r for r in risk_assessments if r['safety_score'] < 0.6]),
            'common_risk_factors': self._identify_common_risk_factors(safety_violations),
            'safety_violation_rate': len(safety_violations) / sum(len(ep.observations) for ep in test_episodes) if test_episodes else 0
        }
    
    def _evaluate_treatment_effectiveness(self, test_episodes):
        """Evaluate predicted treatment effectiveness"""
        print("📈 Evaluating treatment effectiveness...")
        
        effectiveness_scores = []
        outcome_predictions = []
        
        for episode in test_episodes:
            predicted_outcomes = []
            actual_outcomes = episode.rewards  # Assuming rewards represent outcomes
            
            for i, obs in enumerate(episode.observations):
                try:
                    action = self.model.predict(obs.reshape(1, -1))[0]
                    
                    # Predict treatment effectiveness based on action and state
                    effectiveness = self._predict_treatment_outcome(action, obs)
                    effectiveness_scores.append(effectiveness)
                    
                    predicted_outcomes.append(effectiveness)
                    
                except Exception as e:
                    predicted_outcomes.append(0.5)  # Neutral prediction
            
            # Compare predicted vs actual outcomes
            if len(predicted_outcomes) == len(actual_outcomes):
                correlation = np.corrcoef(predicted_outcomes, actual_outcomes)[0, 1] if len(predicted_outcomes) > 1 else 0
                outcome_predictions.append(correlation)
        
        return {
            'mean_effectiveness_score': float(np.mean(effectiveness_scores)) if effectiveness_scores else 0,
            'prediction_accuracy': float(np.mean(outcome_predictions)) if outcome_predictions else 0,
            'high_effectiveness_rate': sum(1 for s in effectiveness_scores if s > 0.7) / len(effectiveness_scores) if effectiveness_scores else 0
        }
    
    def _predict_adverse_events(self, test_episodes):
        """Predict potential adverse events from model decisions"""
        print("⚠️ Predicting adverse event risks...")
        
        adverse_event_risks = []
        risk_categories = {
            'oxygen_toxicity': 0,
            'respiratory_distress': 0,
            'patient_discomfort': 0,
            'treatment_failure': 0
        }
        
        for episode in test_episodes:
            for obs in episode.observations:
                try:
                    action = self.model.predict(obs.reshape(1, -1))[0]
                    hfnc_params = self._decode_action_to_hfnc_params(action, obs)
                    
                    # Assess risk for each type of adverse event
                    risks = self._assess_adverse_event_risks(hfnc_params, obs)
                    
                    for risk_type, risk_level in risks.items():
                        if risk_level > 0.3:  # Threshold for concerning risk
                            risk_categories[risk_type] += 1
                    
                    adverse_event_risks.append(max(risks.values()))
                    
                except Exception as e:
                    continue
        
        total_decisions = sum(len(ep.observations) for ep in test_episodes)
        
        return {
            'overall_adverse_event_risk': float(np.mean(adverse_event_risks)) if adverse_event_risks else 0,
            'high_risk_decisions': sum(1 for risk in adverse_event_risks if risk > 0.5),
            'risk_category_rates': {
                cat: count / total_decisions if total_decisions > 0 else 0 
                for cat, count in risk_categories.items()
            },
            'risk_distribution': adverse_event_risks[:100]  # Sample for analysis
        }
    
    def _decode_action_to_hfnc_params(self, action, obs):
        """Convert discrete action to HFNC parameters (customize based on your encoding)"""
        # This is a placeholder - you'll need to implement based on your specific action encoding
        # For example, if actions represent discrete combinations of flow rate and FiO2:
        
        # Simple example mapping (you should replace with your actual mapping)
        if isinstance(action, (int, np.integer)):
            # Assuming actions 0-9 map to different flow rates and FiO2 combinations
            flow_rates = np.linspace(15, 50, 10)  # 10 different flow rates
            fio2_levels = [0.21, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            
            flow_idx = action % 10
            fio2_idx = min(action // 10, 9)
            
            return {
                'flow_rate': flow_rates[flow_idx],
                'fio2': fio2_levels[fio2_idx],
                'temperature': 37.0  # Assuming fixed temperature
            }
        else:
            # Default safe parameters
            return {
                'flow_rate': 30.0,
                'fio2': 0.4,
                'temperature': 37.0
            }
    
    def _score_parameter_appropriateness(self, params, patient_state):
        """Score how appropriate the parameters are for the patient state"""
        # Placeholder scoring logic - implement based on clinical guidelines
        score = 1.0
        
        # Example: Reduce score if FiO2 is too high for patient's condition
        if params['fio2'] > 0.6:
            score -= 0.2
        
        # Example: Reduce score if flow rate is inappropriate for patient size/condition
        if params['flow_rate'] > 50 or params['flow_rate'] < 15:
            score -= 0.3
        
        return max(0, score)
    
    def _assess_patient_severity(self, obs):
        """Assess patient severity from observation (implement based on your features)"""
        # Placeholder - implement based on your state features
        return "moderate"  # Could be "mild", "moderate", "severe"
    
    def _perform_safety_checks(self, params, obs):
        """Perform comprehensive safety checks"""
        safety_score = 1.0
        risk_factors = []
        
        # Check parameter ranges
        for param, value in params.items():
            if param in self.hfnc_parameters:
                ranges = self.hfnc_parameters[param]
                if value < ranges['safe_min'] or value > ranges['safe_max']:
                    safety_score -= 0.3
                    risk_factors.append(f"{param}_out_of_range")
        
        # Check for parameter combinations that might be risky
        if params['fio2'] > 0.8 and params['flow_rate'] > 45:
            safety_score -= 0.2
            risk_factors.append("high_fio2_high_flow_combination")
        
        return {
            'safety_score': max(0, safety_score),
            'high_risk': safety_score < 0.6,
            'risk_factors': risk_factors
        }
    
    def _predict_treatment_outcome(self, action, obs):
        """Predict treatment effectiveness based on action and patient state"""
        # Placeholder - implement based on your domain knowledge
        # This could use clinical scores, vital signs improvement predictions, etc.
        return 0.7  # Placeholder effectiveness score
    
    def _assess_adverse_event_risks(self, params, obs):
        """Assess risks for different types of adverse events"""
        risks = {
            'oxygen_toxicity': 0.0,
            'respiratory_distress': 0.0,
            'patient_discomfort': 0.0,
            'treatment_failure': 0.0
        }
        
        # Oxygen toxicity risk
        if params['fio2'] > 0.6:
            risks['oxygen_toxicity'] = (params['fio2'] - 0.6) / 0.4
        
        # Respiratory distress risk
        if params['flow_rate'] > 50:
            risks['respiratory_distress'] = (params['flow_rate'] - 50) / 10
        
        # Patient discomfort risk
        if params['temperature'] > 37 or params['temperature'] < 34:
            risks['patient_discomfort'] = 0.3
        
        # Treatment failure risk (inverse of effectiveness)
        effectiveness = self._predict_treatment_outcome(None, obs)
        risks['treatment_failure'] = 1.0 - effectiveness
        
        return risks
    
    def _identify_common_risk_factors(self, safety_violations):
        """Identify most common risk factors across violations"""
        all_factors = []
        for violation in safety_violations:
            all_factors.extend(violation)
        
        from collections import Counter
        return dict(Counter(all_factors).most_common(5))
    
    def _generate_clinical_plots(self, test_episodes, save_dir):
        """Generate clinical validation visualizations"""
        print("📊 Generating clinical validation plots...")
        
        # Set clinical style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Parameter safety distribution
        self._plot_parameter_safety_distribution(test_episodes, save_dir)
        
        # 2. Clinical appropriateness heatmap
        self._plot_clinical_appropriateness_heatmap(test_episodes, save_dir)
        
        # 3. Adverse event risk assessment
        self._plot_adverse_event_risks(test_episodes, save_dir)
        
        # 4. Treatment effectiveness comparison
        self._plot_treatment_effectiveness(test_episodes, save_dir)
    
    def _plot_parameter_safety_distribution(self, test_episodes, save_dir):
        """Plot distribution of parameters with safety ranges"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Collect parameter values
        flow_rates = []
        fio2_values = []
        
        for episode in test_episodes[:20]:  # Sample for efficiency
            for obs in episode.observations:
                try:
                    action = self.model.predict(obs.reshape(1, -1))[0]
                    params = self._decode_action_to_hfnc_params(action, obs)
                    flow_rates.append(params['flow_rate'])
                    fio2_values.append(params['fio2'])
                except:
                    continue
        
        # Plot flow rate distribution
        axes[0, 0].hist(flow_rates, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(self.hfnc_parameters['flow_rate']['safe_min'], color='red', linestyle='--', label='Safety limits')
        axes[0, 0].axvline(self.hfnc_parameters['flow_rate']['safe_max'], color='red', linestyle='--')
        axes[0, 0].axvspan(self.hfnc_parameters['flow_rate']['optimal_range'][0], 
                          self.hfnc_parameters['flow_rate']['optimal_range'][1], 
                          alpha=0.2, color='green', label='Optimal range')
        axes[0, 0].set_title('Flow Rate Distribution')
        axes[0, 0].set_xlabel('Flow Rate (L/min)')
        axes[0, 0].legend()
        
        # Plot FiO2 distribution
        axes[0, 1].hist(fio2_values, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(self.hfnc_parameters['fio2']['safe_min'], color='red', linestyle='--', label='Safety limits')
        axes[0, 1].axvline(self.hfnc_parameters['fio2']['safe_max'], color='red', linestyle='--')
        axes[0, 1].axvspan(self.hfnc_parameters['fio2']['optimal_range'][0], 
                          self.hfnc_parameters['fio2']['optimal_range'][1], 
                          alpha=0.2, color='green', label='Optimal range')
        axes[0, 1].set_title('FiO2 Distribution')
        axes[0, 1].set_xlabel('FiO2 (fraction)')
        axes[0, 1].legend()
        
        # Plot parameter correlation
        axes[1, 0].scatter(flow_rates, fio2_values, alpha=0.6)
        axes[1, 0].set_xlabel('Flow Rate (L/min)')
        axes[1, 0].set_ylabel('FiO2 (fraction)')
        axes[1, 0].set_title('Flow Rate vs FiO2 Correlation')
        
        # Plot safety score distribution
        safety_scores = []
        for i in range(min(100, len(flow_rates))):
            params = {'flow_rate': flow_rates[i], 'fio2': fio2_values[i], 'temperature': 37.0}
            safety_check = self._perform_safety_checks(params, np.zeros(10))  # Dummy obs
            safety_scores.append(safety_check['safety_score'])
        
        axes[1, 1].hist(safety_scores, bins=15, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(0.6, color='red', linestyle='--', label='Risk threshold')
        axes[1, 1].set_title('Safety Score Distribution')
        axes[1, 1].set_xlabel('Safety Score')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'parameter_safety_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_clinical_appropriateness_heatmap(self, test_episodes, save_dir):
        """Plot clinical appropriateness heatmap"""
        # Simplified heatmap - adapt based on your specific use case
        pass
    
    def _plot_adverse_event_risks(self, test_episodes, save_dir):
        """Plot adverse event risk distribution"""
        # Implementation would depend on your specific adverse event definitions
        pass
    
    def _plot_treatment_effectiveness(self, test_episodes, save_dir):
        """Plot treatment effectiveness comparison"""
        # Implementation would depend on your effectiveness metrics
        pass

def validate_clinical_safety(model, test_episodes, save_dir="clinical_validation"):
    """Convenience function for clinical validation"""
    validator = HFNCClinicalValidator(model)
    return validator.validate_clinical_safety(test_episodes, save_dir)