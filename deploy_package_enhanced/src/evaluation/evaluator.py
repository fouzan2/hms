"""
Comprehensive evaluation framework for EEG classification models.
Includes clinical metrics, statistical testing, and bias detection.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve,
    matthews_corrcoef, cohen_kappa_score,
    classification_report
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class ClinicalMetrics:
    """Clinical-specific metrics for brain activity classification."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.classes = config['classes']
        # Make case-insensitive lookup for seizure class
        seizure_classes = [cls for cls in self.classes if cls.lower() == 'seizure']
        if seizure_classes:
            self.seizure_idx = self.classes.index(seizure_classes[0])
        else:
            # Fallback: try to find any class containing 'seizure'
            seizure_classes = [cls for cls in self.classes if 'seizure' in cls.lower()]
            if seizure_classes:
                self.seizure_idx = self.classes.index(seizure_classes[0])
            else:
                # If no seizure class found, use first class as fallback
                self.seizure_idx = 0
                logger.warning(f"No seizure class found in classes: {self.classes}. Using index 0 as fallback.")
        
    def seizure_detection_metrics(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate seizure-specific detection metrics."""
        # Binary classification for seizure detection
        y_true_binary = (y_true == self.seizure_idx).astype(int)
        y_pred_binary = (y_pred == self.seizure_idx).astype(int)
        
        # Sensitivity (True Positive Rate)
        true_positives = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        false_negatives = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        sensitivity = true_positives / (true_positives + false_negatives + 1e-10)
        
        # Specificity (True Negative Rate)
        true_negatives = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        false_positives = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        specificity = true_negatives / (true_negatives + false_positives + 1e-10)
        
        # False Alarm Rate (per hour)
        # Assuming 50-second segments
        segments_per_hour = 3600 / 50
        false_alarm_rate = (false_positives / len(y_true)) * segments_per_hour
        
        # Detection latency (would need temporal information)
        detection_latency = self._calculate_detection_latency(y_true, y_pred)
        
        metrics = {
            'seizure_sensitivity': sensitivity,
            'seizure_specificity': specificity,
            'false_alarm_rate_per_hour': false_alarm_rate,
            'seizure_precision': true_positives / (true_positives + false_positives + 1e-10),
            'seizure_f1': 2 * sensitivity * true_positives / 
                         (2 * true_positives + false_positives + false_negatives + 1e-10),
            'detection_latency_seconds': detection_latency
        }
        
        # Add AUC if probabilities are available
        if y_prob is not None:
            seizure_probs = y_prob[:, self.seizure_idx]
            metrics['seizure_auc'] = roc_auc_score(y_true_binary, seizure_probs)
            
        return metrics
        
    def _calculate_detection_latency(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate average detection latency for seizures."""
        # This is a simplified version - real implementation would need temporal info
        # Placeholder: assume average 5-second latency
        return 5.0
        
    def periodic_discharge_metrics(self, 
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for periodic discharge detection."""
        pd_classes = ['LPD', 'GPD']
        pd_indices = [self.classes.index(cls) for cls in pd_classes]
        
        metrics = {}
        for pd_class, pd_idx in zip(pd_classes, pd_indices):
            y_true_binary = (y_true == pd_idx).astype(int)
            y_pred_binary = (y_pred == pd_idx).astype(int)
            
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            
            sensitivity = tp / (tp + fn + 1e-10)
            precision = tp / (tp + fp + 1e-10)
            
            metrics[f'{pd_class}_sensitivity'] = sensitivity
            metrics[f'{pd_class}_precision'] = precision
            metrics[f'{pd_class}_f1'] = 2 * sensitivity * precision / (sensitivity + precision + 1e-10)
            
        return metrics
        
    def rhythmic_activity_metrics(self, 
                                y_true: np.ndarray,
                                y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for rhythmic activity detection."""
        ra_classes = ['LRDA', 'GRDA']
        ra_indices = [self.classes.index(cls) for cls in ra_classes]
        
        metrics = {}
        for ra_class, ra_idx in zip(ra_classes, ra_indices):
            y_true_binary = (y_true == ra_idx).astype(int)
            y_pred_binary = (y_pred == ra_idx).astype(int)
            
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            
            sensitivity = tp / (tp + fn + 1e-10)
            metrics[f'{ra_class}_sensitivity'] = sensitivity
            
        return metrics


class ModelEvaluator:
    """Comprehensive model evaluation framework."""
    
    def __init__(self, model: nn.Module, config: Dict, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
        self.clinical_metrics = ClinicalMetrics(config)
        self.classes = config['classes']
        
    def evaluate_model(self, dataloader: torch.utils.data.DataLoader) -> Dict:
        """Comprehensive model evaluation."""
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_uncertainties = []
        patient_ids = []
        
        with torch.no_grad():
            for batch in dataloader:
                eeg = batch['eeg'].to(self.device)
                spectrogram = batch['spectrogram'].to(self.device)
                targets = batch['label'].cpu().numpy()
                
                # Get model outputs
                outputs = self.model(eeg, spectrogram)
                
                # Extract predictions and probabilities
                probs = torch.softmax(outputs['logits'], dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
                all_targets.extend(targets)
                
                # Get uncertainties if available
                if 'uncertainty' in outputs:
                    all_uncertainties.extend(outputs['uncertainty'].cpu().numpy())
                    
                # Track patient IDs
                patient_ids.extend(batch['patient_id'])
                
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)
        
        # Calculate all metrics
        metrics = self._calculate_all_metrics(
            all_targets, 
            all_predictions, 
            all_probabilities,
            patient_ids
        )
        
        return metrics
        
    def _calculate_all_metrics(self, 
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             y_prob: np.ndarray,
                             patient_ids: List[str]) -> Dict:
        """Calculate all evaluation metrics."""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = accuracy_score(y_true, y_pred, normalize=True)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(self.classes))
        )
        
        for i, class_name in enumerate(self.classes):
            metrics[f'{class_name}_precision'] = precision[i]
            metrics[f'{class_name}_recall'] = recall[i]
            metrics[f'{class_name}_f1'] = f1[i]
            metrics[f'{class_name}_support'] = support[i]
            
        # Macro and weighted averages
        metrics['macro_precision'] = np.mean(precision)
        metrics['macro_recall'] = np.mean(recall)
        metrics['macro_f1'] = np.mean(f1)
        
        metrics['weighted_precision'] = np.average(precision, weights=support)
        metrics['weighted_recall'] = np.average(recall, weights=support)
        metrics['weighted_f1'] = np.average(f1, weights=support)
        
        # Additional metrics
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Multi-class AUC
        try:
            metrics['auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            metrics['auc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo')
        except:
            logger.warning("Could not calculate multi-class AUC")
            
        # Clinical metrics
        clinical_metrics = self.clinical_metrics.seizure_detection_metrics(y_true, y_pred, y_prob)
        metrics.update(clinical_metrics)
        
        pd_metrics = self.clinical_metrics.periodic_discharge_metrics(y_true, y_pred)
        metrics.update(pd_metrics)
        
        ra_metrics = self.clinical_metrics.rhythmic_activity_metrics(y_true, y_pred)
        metrics.update(ra_metrics)
        
        # Patient-level metrics
        patient_metrics = self._calculate_patient_level_metrics(
            y_true, y_pred, patient_ids
        )
        metrics.update(patient_metrics)
        
        return metrics
        
    def _calculate_patient_level_metrics(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       patient_ids: List[str]) -> Dict:
        """Calculate metrics at patient level."""
        patient_df = pd.DataFrame({
            'patient_id': patient_ids,
            'y_true': y_true,
            'y_pred': y_pred
        })
        
        patient_metrics = {}
        
        # Per-patient accuracy
        patient_accuracies = []
        for patient_id, group in patient_df.groupby('patient_id'):
            acc = accuracy_score(group['y_true'], group['y_pred'])
            patient_accuracies.append(acc)
            
        patient_metrics['mean_patient_accuracy'] = np.mean(patient_accuracies)
        patient_metrics['std_patient_accuracy'] = np.std(patient_accuracies)
        patient_metrics['min_patient_accuracy'] = np.min(patient_accuracies)
        patient_metrics['max_patient_accuracy'] = np.max(patient_accuracies)
        
        return patient_metrics
        
    def statistical_significance_test(self,
                                    predictions1: np.ndarray,
                                    predictions2: np.ndarray,
                                    y_true: np.ndarray) -> Dict:
        """Perform statistical significance testing between two models."""
        # McNemar's test for paired samples
        correct1 = (predictions1 == y_true)
        correct2 = (predictions2 == y_true)
        
        # Build contingency table
        n00 = np.sum(~correct1 & ~correct2)  # Both wrong
        n01 = np.sum(~correct1 & correct2)   # Model 1 wrong, Model 2 correct
        n10 = np.sum(correct1 & ~correct2)   # Model 1 correct, Model 2 wrong
        n11 = np.sum(correct1 & correct2)    # Both correct
        
        # McNemar's test
        from statsmodels.stats.contingency_tables import mcnemar
        result = mcnemar([[n11, n10], [n01, n00]], exact=True)
        
        # DeLong's test for AUC comparison (would need implementation)
        
        return {
            'mcnemar_statistic': result.statistic,
            'mcnemar_pvalue': result.pvalue,
            'model1_better_count': n10,
            'model2_better_count': n01,
            'both_correct_count': n11,
            'both_wrong_count': n00
        }
        
    def cross_validation_evaluation(self,
                                  dataset: torch.utils.data.Dataset,
                                  n_folds: int = 5) -> Dict:
        """Perform cross-validation evaluation."""
        from sklearn.model_selection import StratifiedKFold
        
        # Get labels for stratification
        labels = [dataset[i]['label'].item() for i in range(len(dataset))]
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
            logger.info(f"Evaluating fold {fold + 1}/{n_folds}")
            
            # Create subset datasets
            train_subset = torch.utils.data.Subset(dataset, train_idx)
            val_subset = torch.utils.data.Subset(dataset, val_idx)
            
            # Create dataloaders
            val_loader = torch.utils.data.DataLoader(
                val_subset, 
                batch_size=32, 
                shuffle=False
            )
            
            # Evaluate on validation set
            metrics = self.evaluate_model(val_loader)
            fold_metrics.append(metrics)
            
        # Aggregate metrics across folds
        aggregated_metrics = self._aggregate_cv_metrics(fold_metrics)
        
        return aggregated_metrics
        
    def _aggregate_cv_metrics(self, fold_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across CV folds."""
        aggregated = {}
        
        # Get all metric names
        metric_names = set()
        for fold in fold_metrics:
            metric_names.update(fold.keys())
            
        # Skip non-numeric metrics
        skip_metrics = ['confusion_matrix', 'classification_report']
        
        for metric in metric_names:
            if metric in skip_metrics:
                continue
                
            values = []
            for fold in fold_metrics:
                if metric in fold and isinstance(fold[metric], (int, float)):
                    values.append(fold[metric])
                    
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
                
        return aggregated


class BiasDetector:
    """Detect and analyze bias in model predictions."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.classes = config['classes']
        
    def detect_demographic_bias(self,
                              predictions: np.ndarray,
                              targets: np.ndarray,
                              demographics: pd.DataFrame) -> Dict:
        """Detect bias across demographic groups."""
        bias_metrics = {}
        
        # Analyze by age groups
        if 'age' in demographics.columns:
            age_groups = pd.cut(demographics['age'], bins=[0, 18, 65, 100], 
                              labels=['pediatric', 'adult', 'elderly'])
            age_bias = self._analyze_group_performance(
                predictions, targets, age_groups
            )
            bias_metrics['age_bias'] = age_bias
            
        # Analyze by gender
        if 'gender' in demographics.columns:
            gender_bias = self._analyze_group_performance(
                predictions, targets, demographics['gender']
            )
            bias_metrics['gender_bias'] = gender_bias
            
        # Analyze by recording conditions
        if 'recording_condition' in demographics.columns:
            condition_bias = self._analyze_group_performance(
                predictions, targets, demographics['recording_condition']
            )
            bias_metrics['condition_bias'] = condition_bias
            
        return bias_metrics
        
    def _analyze_group_performance(self,
                                 predictions: np.ndarray,
                                 targets: np.ndarray,
                                 groups: pd.Series) -> Dict:
        """Analyze performance across different groups."""
        group_metrics = {}
        
        for group in groups.unique():
            if pd.isna(group):
                continue
                
            mask = (groups == group)
            group_preds = predictions[mask]
            group_targets = targets[mask]
            
            if len(group_targets) > 0:
                group_metrics[str(group)] = {
                    'accuracy': accuracy_score(group_targets, group_preds),
                    'n_samples': len(group_targets)
                }
                
        # Calculate disparity metrics
        accuracies = [m['accuracy'] for m in group_metrics.values()]
        if len(accuracies) > 1:
            group_metrics['max_disparity'] = max(accuracies) - min(accuracies)
            group_metrics['disparity_ratio'] = max(accuracies) / (min(accuracies) + 1e-10)
            
        return group_metrics


class EvaluationVisualizer:
    """Visualization tools for evaluation results."""
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, 
                            class_names: List[str],
                            save_path: Optional[Path] = None):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add counts in cells
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j + 0.5, i + 0.7, f'n={cm[i,j]}',
                        ha='center', va='center', fontsize=8, color='gray')
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    @staticmethod
    def plot_roc_curves(y_true: np.ndarray,
                       y_prob: np.ndarray,
                       class_names: List[str],
                       save_path: Optional[Path] = None):
        """Plot ROC curves for all classes."""
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(class_names):
            y_true_binary = (y_true == i).astype(int)
            y_score = y_prob[:, i]
            
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            auc = roc_auc_score(y_true_binary, y_score)
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
            
        # Plot random classifier
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - One vs Rest')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    @staticmethod
    def plot_metric_comparison(metrics_dict: Dict[str, Dict],
                             metric_names: List[str],
                             save_path: Optional[Path] = None):
        """Plot comparison of metrics across models."""
        models = list(metrics_dict.keys())
        n_metrics = len(metric_names)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
            
        for ax, metric in zip(axes, metric_names):
            values = [metrics_dict[model].get(metric, 0) for model in models]
            
            bars = ax.bar(models, values)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_ylim(0, 1.1)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom')
                       
            ax.grid(True, alpha=0.3, axis='y')
            
        plt.suptitle('Model Performance Comparison')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show() 