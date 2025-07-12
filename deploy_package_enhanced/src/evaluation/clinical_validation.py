"""
Clinical validation tools for EEG harmful brain activity classification.
Validates model performance against clinical standards and expert annotations.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path


@dataclass
class ClinicalValidationResult:
    """Container for clinical validation results."""
    validation_type: str
    agreement_score: float
    clinical_metrics: Dict[str, float]
    expert_comparison: Optional[Dict[str, float]]
    recommendations: List[str]
    metadata: Dict[str, any]


class ClinicalAgreementValidator:
    """
    Validates model predictions against expert annotations.
    Measures inter-rater agreement and clinical concordance.
    """
    
    def __init__(self, class_names: List[str], 
                 critical_classes: List[str] = None):
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.critical_classes = critical_classes or ['Seizure', 'LPD', 'GPD']
        
    def validate_against_experts(self,
                                model_predictions: np.ndarray,
                                expert_annotations: Dict[str, np.ndarray],
                                weights: Optional[Dict[str, float]] = None) -> ClinicalValidationResult:
        """
        Validate model against multiple expert annotations.
        
        Args:
            model_predictions: Model's predicted labels
            expert_annotations: Dict of expert_name -> expert_labels
            weights: Optional weights for each expert
            
        Returns:
            ClinicalValidationResult
        """
        # Default equal weights
        if weights is None:
            weights = {expert: 1.0 / len(expert_annotations) 
                      for expert in expert_annotations}
            
        # Compute agreement with each expert
        expert_agreements = {}
        for expert_name, expert_labels in expert_annotations.items():
            agreement = self._compute_agreement(model_predictions, expert_labels)
            expert_agreements[expert_name] = agreement
            
        # Weighted average agreement
        weighted_agreement = sum(
            agreement['cohen_kappa'] * weights[expert]
            for expert, agreement in expert_agreements.items()
        )
        
        # Compute consensus labels if multiple experts
        if len(expert_annotations) > 1:
            consensus_labels = self._compute_consensus(expert_annotations)
            consensus_agreement = self._compute_agreement(
                model_predictions, consensus_labels
            )
        else:
            consensus_agreement = list(expert_agreements.values())[0]
            
        # Clinical metrics
        clinical_metrics = self._compute_clinical_metrics(
            model_predictions, consensus_labels if len(expert_annotations) > 1 
            else list(expert_annotations.values())[0]
        )
        
        # Generate recommendations
        recommendations = self._generate_clinical_recommendations(
            consensus_agreement, clinical_metrics
        )
        
        return ClinicalValidationResult(
            validation_type='expert_agreement',
            agreement_score=weighted_agreement,
            clinical_metrics=clinical_metrics,
            expert_comparison=expert_agreements,
            recommendations=recommendations,
            metadata={
                'n_experts': len(expert_annotations),
                'consensus_agreement': consensus_agreement
            }
        )
    
    def _compute_agreement(self, pred1: np.ndarray, pred2: np.ndarray) -> Dict[str, float]:
        """Compute various agreement metrics between two sets of predictions."""
        # Cohen's Kappa
        kappa = cohen_kappa_score(pred1, pred2)
        
        # Simple agreement
        simple_agreement = np.mean(pred1 == pred2)
        
        # Class-specific agreement
        class_agreements = {}
        for i, class_name in enumerate(self.class_names):
            mask = pred2 == i  # Where expert labeled as this class
            if mask.any():
                class_agreements[class_name] = np.mean(pred1[mask] == i)
            else:
                class_agreements[class_name] = None
                
        # Critical class agreement
        critical_mask = np.isin(pred2, [self.class_names.index(c) 
                                       for c in self.critical_classes 
                                       if c in self.class_names])
        critical_agreement = np.mean(pred1[critical_mask] == pred2[critical_mask]) if critical_mask.any() else None
        
        return {
            'cohen_kappa': kappa,
            'simple_agreement': simple_agreement,
            'class_agreements': class_agreements,
            'critical_agreement': critical_agreement
        }
    
    def _compute_consensus(self, expert_annotations: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute consensus labels from multiple experts."""
        # Stack all annotations
        all_annotations = np.stack(list(expert_annotations.values()))
        
        # Majority voting
        consensus = stats.mode(all_annotations, axis=0)[0].squeeze()
        
        return consensus
    
    def _compute_clinical_metrics(self, predictions: np.ndarray, 
                                 ground_truth: np.ndarray) -> Dict[str, float]:
        """Compute clinically relevant metrics."""
        metrics = {}
        
        # Critical finding detection
        critical_indices = [self.class_names.index(c) 
                           for c in self.critical_classes 
                           if c in self.class_names]
        
        if critical_indices:
            critical_true = np.isin(ground_truth, critical_indices)
            critical_pred = np.isin(predictions, critical_indices)
            
            tp = np.sum(critical_true & critical_pred)
            fn = np.sum(critical_true & ~critical_pred)
            fp = np.sum(~critical_true & critical_pred)
            tn = np.sum(~critical_true & ~critical_pred)
            
            metrics['critical_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['critical_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['critical_ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['critical_npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
            
        # Seizure-specific metrics if applicable
        if 'Seizure' in self.class_names:
            seizure_idx = self.class_names.index('Seizure')
            seizure_metrics = self._compute_seizure_metrics(
                predictions, ground_truth, seizure_idx
            )
            metrics.update(seizure_metrics)
            
        return metrics
    
    def _compute_seizure_metrics(self, predictions: np.ndarray,
                                ground_truth: np.ndarray,
                                seizure_idx: int) -> Dict[str, float]:
        """Compute seizure-specific clinical metrics."""
        seizure_true = (ground_truth == seizure_idx)
        seizure_pred = (predictions == seizure_idx)
        
        # Seizure burden (proportion of time in seizure)
        true_burden = np.mean(seizure_true)
        pred_burden = np.mean(seizure_pred)
        burden_error = abs(pred_burden - true_burden) / (true_burden + 1e-10)
        
        # Event-based metrics (simplified)
        true_events = self._count_events(seizure_true)
        pred_events = self._count_events(seizure_pred)
        
        return {
            'seizure_burden_error': burden_error,
            'true_seizure_events': true_events,
            'predicted_seizure_events': pred_events,
            'seizure_event_error': abs(pred_events - true_events) / (true_events + 1)
        }
    
    def _count_events(self, binary_sequence: np.ndarray) -> int:
        """Count number of events in binary sequence."""
        # Count transitions from 0 to 1
        padded = np.pad(binary_sequence, (1, 0), constant_values=0)
        transitions = np.diff(padded.astype(int))
        return np.sum(transitions == 1)
    
    def _generate_clinical_recommendations(self, agreement: Dict[str, float],
                                         clinical_metrics: Dict[str, float]) -> List[str]:
        """Generate clinical recommendations based on validation results."""
        recommendations = []
        
        # Agreement-based recommendations
        kappa = agreement.get('cohen_kappa', 0)
        if kappa < 0.4:
            recommendations.append(
                "Poor agreement with clinical experts (κ < 0.4). "
                "Model requires significant improvement or retraining."
            )
        elif kappa < 0.6:
            recommendations.append(
                "Moderate agreement with clinical experts (κ = {:.2f}). "
                "Consider additional training on disagreement cases.".format(kappa)
            )
        elif kappa < 0.8:
            recommendations.append(
                "Good agreement with clinical experts (κ = {:.2f}). "
                "Model shows clinically acceptable performance.".format(kappa)
            )
        else:
            recommendations.append(
                "Excellent agreement with clinical experts (κ = {:.2f}). "
                "Model ready for clinical validation studies.".format(kappa)
            )
            
        # Critical finding recommendations
        if 'critical_sensitivity' in clinical_metrics:
            sens = clinical_metrics['critical_sensitivity']
            if sens < 0.9:
                recommendations.append(
                    f"Critical finding sensitivity ({sens:.1%}) below clinical threshold. "
                    f"Prioritize improving detection of critical patterns."
                )
                
        # Seizure-specific recommendations
        if 'seizure_burden_error' in clinical_metrics:
            burden_error = clinical_metrics['seizure_burden_error']
            if burden_error > 0.2:
                recommendations.append(
                    f"High seizure burden estimation error ({burden_error:.1%}). "
                    f"Consider temporal modeling improvements."
                )
                
        return recommendations


class ClinicalTrialSimulator:
    """
    Simulates clinical trial scenarios to evaluate model performance.
    Tests model under realistic clinical conditions.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def simulate_screening_trial(self,
                                patient_data: List[Dict],
                                screening_criteria: Dict[str, any],
                                expert_labels: Optional[Dict[str, np.ndarray]] = None) -> ClinicalValidationResult:
        """
        Simulate a screening trial scenario.
        
        Args:
            patient_data: List of patient data dictionaries
            screening_criteria: Criteria for positive screening
            expert_labels: Optional expert labels for comparison
            
        Returns:
            ClinicalValidationResult
        """
        screening_results = []
        expert_screening = [] if expert_labels else None
        
        for patient in patient_data:
            # Model screening
            model_result = self._screen_patient(patient, screening_criteria)
            screening_results.append(model_result)
            
            # Expert screening if available
            if expert_labels and patient['id'] in expert_labels:
                expert_result = self._screen_patient_expert(
                    expert_labels[patient['id']], screening_criteria
                )
                expert_screening.append(expert_result)
                
        # Compute screening metrics
        screening_metrics = self._compute_screening_metrics(
            screening_results, expert_screening
        )
        
        # Clinical trial metrics
        trial_metrics = {
            'patients_screened': len(patient_data),
            'positive_screens': sum(screening_results),
            'screening_rate': np.mean(screening_results),
            'screening_consistency': self._compute_screening_consistency(screening_results)
        }
        
        recommendations = self._generate_screening_recommendations(
            screening_metrics, trial_metrics
        )
        
        return ClinicalValidationResult(
            validation_type='screening_trial',
            agreement_score=screening_metrics.get('screening_agreement', 0),
            clinical_metrics=trial_metrics,
            expert_comparison=screening_metrics if expert_screening else None,
            recommendations=recommendations,
            metadata={'screening_criteria': screening_criteria}
        )
    
    def simulate_monitoring_trial(self,
                                 continuous_data_loader: torch.utils.data.DataLoader,
                                 monitoring_duration_hours: float = 24.0,
                                 alarm_criteria: Dict[str, any] = None) -> ClinicalValidationResult:
        """
        Simulate continuous monitoring trial.
        
        Args:
            continuous_data_loader: Loader with continuous EEG data
            monitoring_duration_hours: Duration of monitoring
            alarm_criteria: Criteria for raising clinical alarms
            
        Returns:
            ClinicalValidationResult
        """
        if alarm_criteria is None:
            alarm_criteria = {
                'seizure_probability': 0.8,
                'critical_pattern_probability': 0.7,
                'min_duration_seconds': 10
            }
            
        monitoring_results = {
            'total_alarms': 0,
            'false_alarms': 0,
            'missed_events': 0,
            'alarm_latencies': [],
            'monitoring_segments': 0
        }
        
        current_alarm_state = None
        alarm_start_time = None
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(continuous_data_loader):
                eeg_data = batch['eeg'].to(self.device)
                true_labels = batch['label'].cpu().numpy()
                
                # Get model predictions
                outputs = self.model(eeg_data)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                
                # Check alarm criteria
                for i in range(len(probs)):
                    alarm_triggered = self._check_alarm_criteria(
                        probs[i], alarm_criteria
                    )
                    
                    # Update monitoring results
                    monitoring_results['monitoring_segments'] += 1
                    
                    # Handle alarm state transitions
                    if alarm_triggered and current_alarm_state is None:
                        current_alarm_state = 'active'
                        alarm_start_time = batch_idx * len(batch) + i
                        monitoring_results['total_alarms'] += 1
                        
                    elif not alarm_triggered and current_alarm_state == 'active':
                        current_alarm_state = None
                        
        # Compute alarm metrics
        segments_per_hour = 360  # Assuming 10-second segments
        alarm_rate = monitoring_results['total_alarms'] / (
            monitoring_results['monitoring_segments'] / segments_per_hour
        )
        
        clinical_metrics = {
            'alarm_rate_per_hour': alarm_rate,
            'total_monitoring_hours': monitoring_results['monitoring_segments'] / segments_per_hour,
            'mean_alarm_duration': np.mean(monitoring_results['alarm_latencies']) if monitoring_results['alarm_latencies'] else 0
        }
        
        recommendations = self._generate_monitoring_recommendations(clinical_metrics)
        
        return ClinicalValidationResult(
            validation_type='monitoring_trial',
            agreement_score=0.0,  # Not applicable for monitoring
            clinical_metrics=clinical_metrics,
            expert_comparison=None,
            recommendations=recommendations,
            metadata={
                'alarm_criteria': alarm_criteria,
                'monitoring_duration_hours': monitoring_duration_hours
            }
        )
    
    def _screen_patient(self, patient_data: Dict, criteria: Dict) -> bool:
        """Screen a patient using the model."""
        # Process patient EEG data
        eeg_tensor = torch.tensor(patient_data['eeg_data']).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(eeg_tensor.unsqueeze(0))
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
        # Apply screening criteria
        if 'min_seizure_probability' in criteria:
            seizure_idx = patient_data.get('seizure_class_idx', 1)
            if probs[seizure_idx] >= criteria['min_seizure_probability']:
                return True
                
        if 'any_abnormal_probability' in criteria:
            abnormal_prob = 1 - probs[0]  # Assuming class 0 is normal
            if abnormal_prob >= criteria['any_abnormal_probability']:
                return True
                
        return False
    
    def _screen_patient_expert(self, expert_labels: np.ndarray, criteria: Dict) -> bool:
        """Screen based on expert labels."""
        # Apply same criteria to expert labels
        if 'min_seizure_probability' in criteria:
            # For expert labels, check if seizure present
            seizure_present = np.any(expert_labels == criteria.get('seizure_class_idx', 1))
            if seizure_present:
                return True
                
        if 'any_abnormal_probability' in criteria:
            abnormal_present = np.any(expert_labels != 0)
            if abnormal_present:
                return True
                
        return False
    
    def _compute_screening_metrics(self, model_results: List[bool],
                                  expert_results: Optional[List[bool]]) -> Dict[str, float]:
        """Compute screening performance metrics."""
        metrics = {}
        
        if expert_results:
            # Compare with expert screening
            agreement = np.mean(np.array(model_results) == np.array(expert_results))
            
            # Screening performance
            tp = sum(m and e for m, e in zip(model_results, expert_results))
            fp = sum(m and not e for m, e in zip(model_results, expert_results))
            fn = sum(not m and e for m, e in zip(model_results, expert_results))
            tn = sum(not m and not e for m, e in zip(model_results, expert_results))
            
            metrics['screening_agreement'] = agreement
            metrics['screening_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['screening_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['screening_ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            
        return metrics
    
    def _compute_screening_consistency(self, results: List[bool]) -> float:
        """Compute consistency of screening decisions."""
        # Simple measure of how consistent screening is across patients
        positive_rate = np.mean(results)
        # Consistency is high if rate is very high or very low
        consistency = 1 - 4 * positive_rate * (1 - positive_rate)
        return consistency
    
    def _check_alarm_criteria(self, probabilities: np.ndarray, criteria: Dict) -> bool:
        """Check if current predictions meet alarm criteria."""
        # Check seizure probability
        if 'seizure_probability' in criteria:
            seizure_idx = 1  # Assuming standard class order
            if probabilities[seizure_idx] >= criteria['seizure_probability']:
                return True
                
        # Check critical pattern probability
        if 'critical_pattern_probability' in criteria:
            critical_indices = [1, 2, 3, 4, 5]  # Seizure, LPD, GPD, LRDA, GRDA
            critical_prob = probabilities[critical_indices].max()
            if critical_prob >= criteria['critical_pattern_probability']:
                return True
                
        return False
    
    def _generate_screening_recommendations(self, screening_metrics: Dict,
                                          trial_metrics: Dict) -> List[str]:
        """Generate recommendations for screening trial."""
        recommendations = []
        
        # Screening rate recommendations
        screening_rate = trial_metrics.get('screening_rate', 0)
        if screening_rate > 0.5:
            recommendations.append(
                f"High positive screening rate ({screening_rate:.1%}). "
                f"Consider adjusting screening thresholds for specificity."
            )
        elif screening_rate < 0.1:
            recommendations.append(
                f"Low positive screening rate ({screening_rate:.1%}). "
                f"Model may be too conservative for screening application."
            )
            
        # Agreement recommendations
        if screening_metrics:
            agreement = screening_metrics.get('screening_agreement', 0)
            if agreement < 0.8:
                recommendations.append(
                    f"Screening decisions show low agreement with experts ({agreement:.1%}). "
                    f"Review screening criteria and model calibration."
                )
                
        return recommendations
    
    def _generate_monitoring_recommendations(self, clinical_metrics: Dict) -> List[str]:
        """Generate recommendations for monitoring trial."""
        recommendations = []
        
        # Alarm rate recommendations
        alarm_rate = clinical_metrics.get('alarm_rate_per_hour', 0)
        if alarm_rate > 10:
            recommendations.append(
                f"High alarm rate ({alarm_rate:.1f}/hour) may cause alarm fatigue. "
                f"Consider adjusting alarm thresholds or implementing alarm management."
            )
        elif alarm_rate < 0.1:
            recommendations.append(
                f"Very low alarm rate ({alarm_rate:.2f}/hour). "
                f"Verify model sensitivity to critical events."
            )
            
        return recommendations


class ClinicalPerformanceAnalyzer:
    """
    Analyzes model performance from clinical perspective.
    Provides clinically meaningful performance metrics and insights.
    """
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.clinical_groupings = self._define_clinical_groupings()
        
    def _define_clinical_groupings(self) -> Dict[str, List[str]]:
        """Define clinical groupings of EEG patterns."""
        return {
            'normal': ['Other'],
            'seizure': ['Seizure', 'SZ'],
            'periodic': ['LPD', 'GPD', 'LRDA', 'GRDA'],
            'epileptiform': ['Seizure', 'LPD', 'GPD'],
            'rhythmic': ['LRDA', 'GRDA'],
            'lateralized': ['LPD', 'LRDA'],
            'generalized': ['GPD', 'GRDA']
        }
        
    def analyze_clinical_performance(self,
                                   predictions: np.ndarray,
                                   ground_truth: np.ndarray,
                                   probabilities: Optional[np.ndarray] = None,
                                   patient_metadata: Optional[pd.DataFrame] = None) -> Dict[str, any]:
        """
        Comprehensive clinical performance analysis.
        
        Args:
            predictions: Model predictions
            ground_truth: True labels
            probabilities: Prediction probabilities
            patient_metadata: Optional patient information
            
        Returns:
            Dict with clinical performance analysis
        """
        analysis = {
            'pattern_detection': self._analyze_pattern_detection(predictions, ground_truth),
            'clinical_grouping_performance': self._analyze_clinical_groupings(
                predictions, ground_truth
            ),
            'error_analysis': self._analyze_clinical_errors(predictions, ground_truth),
            'confidence_analysis': self._analyze_prediction_confidence(
                predictions, ground_truth, probabilities
            ) if probabilities is not None else None
        }
        
        # Patient subgroup analysis if metadata available
        if patient_metadata is not None:
            analysis['subgroup_analysis'] = self._analyze_patient_subgroups(
                predictions, ground_truth, patient_metadata
            )
            
        # Generate clinical insights
        analysis['clinical_insights'] = self._generate_clinical_insights(analysis)
        
        return analysis
    
    def _analyze_pattern_detection(self, predictions: np.ndarray,
                                  ground_truth: np.ndarray) -> Dict[str, float]:
        """Analyze detection performance for each EEG pattern."""
        pattern_metrics = {}
        
        for i, pattern in enumerate(self.class_names):
            pattern_true = (ground_truth == i)
            pattern_pred = (predictions == i)
            
            if pattern_true.any():
                tp = np.sum(pattern_true & pattern_pred)
                fn = np.sum(pattern_true & ~pattern_pred)
                fp = np.sum(~pattern_true & pattern_pred)
                
                sensitivity = tp / (tp + fn)
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                
                pattern_metrics[pattern] = {
                    'sensitivity': sensitivity,
                    'ppv': ppv,
                    'f1': 2 * sensitivity * ppv / (sensitivity + ppv) if (sensitivity + ppv) > 0 else 0,
                    'prevalence': pattern_true.mean(),
                    'detection_rate': pattern_pred.mean()
                }
            else:
                pattern_metrics[pattern] = {
                    'sensitivity': None,
                    'ppv': None,
                    'f1': None,
                    'prevalence': 0,
                    'detection_rate': pattern_pred.mean()
                }
                
        return pattern_metrics
    
    def _analyze_clinical_groupings(self, predictions: np.ndarray,
                                   ground_truth: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Analyze performance on clinical groupings."""
        grouping_metrics = {}
        
        for group_name, patterns in self.clinical_groupings.items():
            # Get indices for this clinical group
            group_indices = [i for i, name in enumerate(self.class_names) 
                           if name in patterns]
            
            if not group_indices:
                continue
                
            # Binary classification for this group
            group_true = np.isin(ground_truth, group_indices)
            group_pred = np.isin(predictions, group_indices)
            
            if group_true.any():
                tp = np.sum(group_true & group_pred)
                fn = np.sum(group_true & ~group_pred)
                fp = np.sum(~group_true & group_pred)
                tn = np.sum(~group_true & ~group_pred)
                
                grouping_metrics[group_name] = {
                    'sensitivity': tp / (tp + fn),
                    'specificity': tn / (tn + fp),
                    'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
                    'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
                    'accuracy': (tp + tn) / len(predictions)
                }
            else:
                grouping_metrics[group_name] = {
                    'sensitivity': None,
                    'specificity': None,
                    'ppv': None,
                    'npv': None,
                    'accuracy': None
                }
                
        return grouping_metrics
    
    def _analyze_clinical_errors(self, predictions: np.ndarray,
                                ground_truth: np.ndarray) -> Dict[str, any]:
        """Analyze clinically significant errors."""
        # Confusion matrix
        cm = confusion_matrix(ground_truth, predictions)
        
        # Identify critical misclassifications
        critical_errors = []
        
        # Missing seizures
        seizure_indices = [i for i, name in enumerate(self.class_names) 
                          if 'Seizure' in name or 'SZ' in name]
        
        for sz_idx in seizure_indices:
            missed_seizures = cm[sz_idx, :].sum() - cm[sz_idx, sz_idx]
            if missed_seizures > 0:
                critical_errors.append({
                    'error_type': 'missed_seizure',
                    'count': missed_seizures,
                    'pattern': self.class_names[sz_idx]
                })
                
        # False seizure detections
        for sz_idx in seizure_indices:
            false_seizures = cm[:, sz_idx].sum() - cm[sz_idx, sz_idx]
            if false_seizures > 0:
                critical_errors.append({
                    'error_type': 'false_seizure',
                    'count': false_seizures,
                    'pattern': self.class_names[sz_idx]
                })
                
        # Confusion between periodic and rhythmic patterns
        periodic_indices = [i for i, name in enumerate(self.class_names) 
                           if name in ['LPD', 'GPD']]
        rhythmic_indices = [i for i, name in enumerate(self.class_names) 
                           if name in ['LRDA', 'GRDA']]
        
        periodic_rhythmic_confusion = 0
        for p_idx in periodic_indices:
            for r_idx in rhythmic_indices:
                periodic_rhythmic_confusion += cm[p_idx, r_idx] + cm[r_idx, p_idx]
                
        return {
            'confusion_matrix': cm,
            'critical_errors': critical_errors,
            'periodic_rhythmic_confusion': periodic_rhythmic_confusion,
            'total_errors': len(predictions) - np.trace(cm)
        }
    
    def _analyze_prediction_confidence(self, predictions: np.ndarray,
                                     ground_truth: np.ndarray,
                                     probabilities: np.ndarray) -> Dict[str, float]:
        """Analyze prediction confidence in clinical context."""
        correct_mask = predictions == ground_truth
        confidences = probabilities.max(axis=1)
        
        # Confidence by correctness
        correct_confidence = confidences[correct_mask].mean()
        incorrect_confidence = confidences[~correct_mask].mean()
        
        # Confidence for critical patterns
        critical_indices = [i for i, name in enumerate(self.class_names)
                           if name in ['Seizure', 'LPD', 'GPD']]
        
        critical_confidences = []
        for idx in critical_indices:
            mask = ground_truth == idx
            if mask.any():
                critical_confidences.extend(confidences[mask])
                
        return {
            'mean_confidence': confidences.mean(),
            'correct_prediction_confidence': correct_confidence,
            'incorrect_prediction_confidence': incorrect_confidence,
            'confidence_gap': correct_confidence - incorrect_confidence,
            'critical_pattern_confidence': np.mean(critical_confidences) if critical_confidences else None
        }
    
    def _analyze_patient_subgroups(self, predictions: np.ndarray,
                                  ground_truth: np.ndarray,
                                  patient_metadata: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze performance across patient subgroups."""
        subgroup_analysis = {}
        
        # Age groups
        if 'age' in patient_metadata.columns:
            age_groups = pd.cut(patient_metadata['age'], 
                               bins=[0, 18, 65, 100],
                               labels=['pediatric', 'adult', 'elderly'])
            
            for group in age_groups.unique():
                mask = age_groups == group
                if mask.any():
                    accuracy = np.mean(predictions[mask] == ground_truth[mask])
                    subgroup_analysis[f'age_{group}'] = {
                        'n_samples': mask.sum(),
                        'accuracy': accuracy
                    }
                    
        # Gender
        if 'gender' in patient_metadata.columns:
            for gender in patient_metadata['gender'].unique():
                mask = patient_metadata['gender'] == gender
                if mask.any():
                    accuracy = np.mean(predictions[mask] == ground_truth[mask])
                    subgroup_analysis[f'gender_{gender}'] = {
                        'n_samples': mask.sum(),
                        'accuracy': accuracy
                    }
                    
        return subgroup_analysis
    
    def _generate_clinical_insights(self, analysis: Dict) -> List[str]:
        """Generate clinical insights from analysis."""
        insights = []
        
        # Pattern detection insights
        pattern_detection = analysis.get('pattern_detection', {})
        for pattern, metrics in pattern_detection.items():
            if metrics['sensitivity'] is not None and metrics['sensitivity'] < 0.8:
                insights.append(
                    f"Low sensitivity for {pattern} detection ({metrics['sensitivity']:.1%}). "
                    f"Clinical review recommended."
                )
                
        # Clinical grouping insights
        grouping_performance = analysis.get('clinical_grouping_performance', {})
        if 'seizure' in grouping_performance:
            seizure_perf = grouping_performance['seizure']
            if seizure_perf['sensitivity'] and seizure_perf['sensitivity'] < 0.9:
                insights.append(
                    f"Seizure detection sensitivity below clinical standard "
                    f"({seizure_perf['sensitivity']:.1%}). Enhancement needed."
                )
                
        # Error analysis insights
        error_analysis = analysis.get('error_analysis', {})
        critical_errors = error_analysis.get('critical_errors', [])
        for error in critical_errors:
            if error['error_type'] == 'missed_seizure':
                insights.append(
                    f"Model missed {error['count']} seizure cases. "
                    f"Critical safety concern requiring immediate attention."
                )
                
        return insights


class ClinicalReportGenerator:
    """
    Generates comprehensive clinical validation reports.
    Creates formatted reports suitable for clinical review.
    """
    
    def __init__(self):
        self.report_sections = []
        
    def add_validation_results(self, results: ClinicalValidationResult, 
                              section_name: str):
        """Add validation results to report."""
        self.report_sections.append({
            'name': section_name,
            'results': results
        })
        
    def generate_report(self, save_path: str, 
                       study_info: Optional[Dict] = None) -> str:
        """
        Generate comprehensive clinical validation report.
        
        Args:
            save_path: Path to save report
            study_info: Optional study information
            
        Returns:
            Path to generated report
        """
        # Create report structure
        report = {
            'title': 'Clinical Validation Report - EEG Harmful Brain Activity Classification',
            'date': datetime.now().strftime('%Y-%m-%d'),
            'study_info': study_info or {},
            'executive_summary': self._generate_executive_summary(),
            'detailed_results': self._format_detailed_results(),
            'clinical_recommendations': self._compile_recommendations(),
            'appendices': self._generate_appendices()
        }
        
        # Save as JSON
        json_path = Path(save_path).with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Generate HTML report
        html_path = self._generate_html_report(report, save_path)
        
        # Generate visualizations
        self._generate_report_visualizations(save_path)
        
        return str(html_path)
    
    def _generate_executive_summary(self) -> Dict[str, any]:
        """Generate executive summary of validation results."""
        summary = {
            'total_validations': len(self.report_sections),
            'overall_performance': 'Pending',
            'key_findings': [],
            'critical_issues': []
        }
        
        # Aggregate key metrics
        agreement_scores = []
        for section in self.report_sections:
            results = section['results']
            if results.agreement_score > 0:
                agreement_scores.append(results.agreement_score)
                
            # Check for critical issues
            for rec in results.recommendations:
                if 'critical' in rec.lower() or 'poor' in rec.lower():
                    summary['critical_issues'].append({
                        'validation': section['name'],
                        'issue': rec
                    })
                    
        # Overall performance assessment
        if agreement_scores:
            avg_agreement = np.mean(agreement_scores)
            if avg_agreement >= 0.8:
                summary['overall_performance'] = 'Excellent'
            elif avg_agreement >= 0.6:
                summary['overall_performance'] = 'Good'
            elif avg_agreement >= 0.4:
                summary['overall_performance'] = 'Moderate'
            else:
                summary['overall_performance'] = 'Poor'
                
        # Key findings
        for section in self.report_sections:
            results = section['results']
            if results.clinical_metrics:
                key_metric = max(results.clinical_metrics.items(), 
                               key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0)
                summary['key_findings'].append({
                    'validation': section['name'],
                    'finding': f"{key_metric[0]}: {key_metric[1]:.3f}"
                })
                
        return summary
    
    def _format_detailed_results(self) -> List[Dict]:
        """Format detailed results for each validation."""
        detailed = []
        
        for section in self.report_sections:
            detailed.append({
                'section_name': section['name'],
                'validation_type': section['results'].validation_type,
                'agreement_score': section['results'].agreement_score,
                'clinical_metrics': section['results'].clinical_metrics,
                'expert_comparison': section['results'].expert_comparison,
                'metadata': section['results'].metadata
            })
            
        return detailed
    
    def _compile_recommendations(self) -> List[Dict]:
        """Compile all recommendations with priority."""
        all_recommendations = []
        
        for section in self.report_sections:
            for rec in section['results'].recommendations:
                priority = 'high' if any(word in rec.lower() 
                                       for word in ['critical', 'poor', 'immediate']) else 'medium'
                                       
                all_recommendations.append({
                    'source': section['name'],
                    'recommendation': rec,
                    'priority': priority
                })
                
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        all_recommendations.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return all_recommendations
    
    def _generate_appendices(self) -> Dict:
        """Generate appendices with technical details."""
        return {
            'validation_methods': {
                'expert_agreement': 'Cohen\'s Kappa coefficient for inter-rater agreement',
                'screening_trial': 'Simulated patient screening scenario',
                'monitoring_trial': 'Continuous EEG monitoring simulation'
            },
            'metric_definitions': {
                'sensitivity': 'True positive rate - ability to detect condition when present',
                'specificity': 'True negative rate - ability to exclude condition when absent',
                'ppv': 'Positive predictive value - probability of condition given positive test',
                'npv': 'Negative predictive value - probability of no condition given negative test'
            }
        }
    
    def _generate_html_report(self, report_data: Dict, save_path: str) -> Path:
        """Generate HTML version of report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_data['title']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                .summary {{ background-color: #f0f0f0; padding: 20px; margin: 20px 0; }}
                .critical {{ color: red; font-weight: bold; }}
                .good {{ color: green; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; }}
                .high-priority {{ border-left: 5px solid red; }}
                .medium-priority {{ border-left: 5px solid orange; }}
            </style>
        </head>
        <body>
            <h1>{report_data['title']}</h1>
            <p><strong>Date:</strong> {report_data['date']}</p>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p><strong>Overall Performance:</strong> 
                   <span class="{'good' if report_data['executive_summary']['overall_performance'] in ['Excellent', 'Good'] else 'critical'}">
                   {report_data['executive_summary']['overall_performance']}
                   </span>
                </p>
                <p><strong>Total Validations:</strong> {report_data['executive_summary']['total_validations']}</p>
                
                <h3>Key Findings</h3>
                <ul>
                """
        
        for finding in report_data['executive_summary']['key_findings']:
            html_content += f"<li><strong>{finding['validation']}:</strong> {finding['finding']}</li>"
            
        html_content += """
                </ul>
            </div>
            
            <h2>Detailed Results</h2>
        """
        
        for result in report_data['detailed_results']:
            html_content += f"""
            <h3>{result['section_name']}</h3>
            <p><strong>Validation Type:</strong> {result['validation_type']}</p>
            <p><strong>Agreement Score:</strong> {result['agreement_score']:.3f}</p>
            
            <h4>Clinical Metrics</h4>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
            """
            
            for metric, value in result['clinical_metrics'].items():
                html_content += f"<tr><td>{metric}</td><td>{value:.3f if isinstance(value, float) else value}</td></tr>"
                
            html_content += """
            </table>
            """
            
        html_content += """
            <h2>Clinical Recommendations</h2>
        """
        
        for rec in report_data['clinical_recommendations']:
            priority_class = f"{rec['priority']}-priority"
            html_content += f"""
            <div class="recommendation {priority_class}">
                <strong>{rec['source']}:</strong> {rec['recommendation']}
            </div>
            """
            
        html_content += """
        </body>
        </html>
        """
        
        html_path = Path(save_path).with_suffix('.html')
        with open(html_path, 'w') as f:
            f.write(html_content)
            
        return html_path
    
    def _generate_report_visualizations(self, save_path: str):
        """Generate visualizations for the report."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Agreement scores plot
        ax = axes[0, 0]
        sections = [s['name'] for s in self.report_sections]
        scores = [s['results'].agreement_score for s in self.report_sections]
        
        bars = ax.bar(range(len(sections)), scores)
        for i, (bar, score) in enumerate(zip(bars, scores)):
            color = 'green' if score >= 0.8 else 'orange' if score >= 0.6 else 'red'
            bar.set_color(color)
            
        ax.set_xticks(range(len(sections)))
        ax.set_xticklabels(sections, rotation=45, ha='right')
        ax.set_ylabel('Agreement Score')
        ax.set_title('Clinical Agreement by Validation Type')
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5)
        
        # Other visualizations would go in remaining subplots
        
        plt.tight_layout()
        viz_path = Path(save_path).with_suffix('.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close() 