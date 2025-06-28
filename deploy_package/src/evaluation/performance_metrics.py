"""
Comprehensive performance metrics for EEG harmful brain activity classification.
Includes both technical ML metrics and clinical evaluation measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve,
    matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score, average_precision_score,
    precision_recall_curve, auc, f1_score,
    multilabel_confusion_matrix, classification_report
)
from scipy import stats
import warnings
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Container for metric computation results."""
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    per_class_values: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, any]] = None


class PerformanceMetrics:
    """
    Comprehensive performance metrics for EEG classification.
    Includes standard ML metrics and clinical-specific measures.
    """
    
    def __init__(self, class_names: List[str], 
                 seizure_classes: Optional[List[str]] = None,
                 critical_classes: Optional[List[str]] = None):
        self.class_names = class_names
        self.n_classes = len(class_names)
        
        # Define clinically important classes
        self.seizure_classes = seizure_classes or ['Seizure', 'SZ']
        self.critical_classes = critical_classes or ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA']
        
        # Map class names to indices
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.seizure_indices = [self.class_to_idx[c] for c in self.seizure_classes 
                               if c in self.class_to_idx]
        self.critical_indices = [self.class_to_idx[c] for c in self.critical_classes 
                                if c in self.class_to_idx]
        
    def compute_all_metrics(self,
                           y_true: np.ndarray,
                           y_pred: np.ndarray,
                           y_prob: Optional[np.ndarray] = None,
                           sample_weights: Optional[np.ndarray] = None) -> Dict[str, MetricResult]:
        """
        Compute comprehensive set of metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            sample_weights: Sample weights (optional)
            
        Returns:
            Dictionary of metric names to MetricResult objects
        """
        metrics = {}
        
        # Basic classification metrics
        metrics.update(self._compute_basic_metrics(y_true, y_pred, sample_weights))
        
        # Per-class metrics
        metrics.update(self._compute_per_class_metrics(y_true, y_pred, sample_weights))
        
        # Clinical metrics
        metrics.update(self._compute_clinical_metrics(y_true, y_pred, y_prob))
        
        # Probability-based metrics (if available)
        if y_prob is not None:
            metrics.update(self._compute_probability_metrics(y_true, y_prob))
            
        # Statistical metrics
        metrics.update(self._compute_statistical_metrics(y_true, y_pred))
        
        return metrics
    
    def _compute_basic_metrics(self, y_true, y_pred, sample_weights=None):
        """Compute basic classification metrics."""
        metrics = {}
        
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred, sample_weight=sample_weights)
        metrics['accuracy'] = MetricResult(
            value=accuracy,
            metadata={'description': 'Overall classification accuracy'}
        )
        
        # Balanced accuracy
        balanced_acc = balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weights)
        metrics['balanced_accuracy'] = MetricResult(
            value=balanced_acc,
            metadata={'description': 'Balanced accuracy accounting for class imbalance'}
        )
        
        # Macro-averaged metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', sample_weight=sample_weights
        )
        
        metrics['macro_precision'] = MetricResult(value=precision)
        metrics['macro_recall'] = MetricResult(value=recall)
        metrics['macro_f1'] = MetricResult(value=f1)
        
        # Weighted-averaged metrics
        w_precision, w_recall, w_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', sample_weight=sample_weights
        )
        
        metrics['weighted_precision'] = MetricResult(value=w_precision)
        metrics['weighted_recall'] = MetricResult(value=w_recall)
        metrics['weighted_f1'] = MetricResult(value=w_f1)
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(y_true, y_pred, sample_weight=sample_weights)
        metrics['cohen_kappa'] = MetricResult(
            value=kappa,
            metadata={'description': 'Inter-rater agreement metric'}
        )
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_true, y_pred, sample_weight=sample_weights)
        metrics['matthews_corrcoef'] = MetricResult(
            value=mcc,
            metadata={'description': 'Balanced metric for binary and multi-class'}
        )
        
        return metrics
    
    def _compute_per_class_metrics(self, y_true, y_pred, sample_weights=None):
        """Compute per-class performance metrics."""
        metrics = {}
        
        # Per-class precision, recall, f1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(self.n_classes),
            sample_weight=sample_weights
        )
        
        per_class_precision = {self.class_names[i]: precision[i] for i in range(self.n_classes)}
        per_class_recall = {self.class_names[i]: recall[i] for i in range(self.n_classes)}
        per_class_f1 = {self.class_names[i]: f1[i] for i in range(self.n_classes)}
        per_class_support = {self.class_names[i]: int(support[i]) for i in range(self.n_classes)}
        
        metrics['per_class_precision'] = MetricResult(
            value=np.mean(precision),
            per_class_values=per_class_precision
        )
        
        metrics['per_class_recall'] = MetricResult(
            value=np.mean(recall),
            per_class_values=per_class_recall
        )
        
        metrics['per_class_f1'] = MetricResult(
            value=np.mean(f1),
            per_class_values=per_class_f1
        )
        
        metrics['per_class_support'] = MetricResult(
            value=np.sum(support),
            per_class_values=per_class_support
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(self.n_classes))
        metrics['confusion_matrix'] = MetricResult(
            value=0,  # Placeholder
            metadata={'matrix': cm, 'class_names': self.class_names}
        )
        
        return metrics
    
    def _compute_clinical_metrics(self, y_true, y_pred, y_prob=None):
        """Compute clinical-specific metrics."""
        metrics = {}
        
        # Seizure detection metrics
        if self.seizure_indices:
            seizure_metrics = self._compute_seizure_metrics(y_true, y_pred, y_prob)
            metrics.update(seizure_metrics)
            
        # Critical class detection
        if self.critical_indices:
            critical_metrics = self._compute_critical_class_metrics(y_true, y_pred, y_prob)
            metrics.update(critical_metrics)
            
        # False negative rate for critical conditions
        critical_fn_rate = self._compute_critical_false_negative_rate(y_true, y_pred)
        metrics['critical_false_negative_rate'] = MetricResult(
            value=critical_fn_rate,
            metadata={'critical_classes': self.critical_classes}
        )
        
        return metrics
    
    def _compute_seizure_metrics(self, y_true, y_pred, y_prob=None):
        """Compute seizure-specific detection metrics."""
        metrics = {}
        
        # Binary seizure detection
        y_true_seizure = np.isin(y_true, self.seizure_indices).astype(int)
        y_pred_seizure = np.isin(y_pred, self.seizure_indices).astype(int)
        
        # Sensitivity (recall for seizures)
        tp = np.sum((y_true_seizure == 1) & (y_pred_seizure == 1))
        fn = np.sum((y_true_seizure == 1) & (y_pred_seizure == 0))
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        metrics['seizure_sensitivity'] = MetricResult(
            value=sensitivity,
            metadata={'description': 'Ability to detect seizures when present'}
        )
        
        # Specificity
        tn = np.sum((y_true_seizure == 0) & (y_pred_seizure == 0))
        fp = np.sum((y_true_seizure == 0) & (y_pred_seizure == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        metrics['seizure_specificity'] = MetricResult(
            value=specificity,
            metadata={'description': 'Ability to correctly identify non-seizures'}
        )
        
        # Precision (PPV)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['seizure_precision'] = MetricResult(value=precision)
        
        # F1 score
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        metrics['seizure_f1'] = MetricResult(value=f1)
        
        # False alarm rate (per hour, assuming 10-second segments)
        segments_per_hour = 360  # 3600 seconds / 10 seconds
        far = (fp / len(y_true)) * segments_per_hour
        metrics['seizure_false_alarm_rate_per_hour'] = MetricResult(
            value=far,
            metadata={'description': 'False seizure detections per hour'}
        )
        
        # AUC if probabilities available
        if y_prob is not None and self.seizure_indices:
            seizure_probs = y_prob[:, self.seizure_indices].max(axis=1)
            try:
                auc_score = roc_auc_score(y_true_seizure, seizure_probs)
                metrics['seizure_auc'] = MetricResult(value=auc_score)
            except:
                pass
                
        return metrics
    
    def _compute_critical_class_metrics(self, y_true, y_pred, y_prob=None):
        """Compute metrics for critical EEG patterns."""
        metrics = {}
        
        # Binary critical detection
        y_true_critical = np.isin(y_true, self.critical_indices).astype(int)
        y_pred_critical = np.isin(y_pred, self.critical_indices).astype(int)
        
        # Sensitivity for critical patterns
        tp = np.sum((y_true_critical == 1) & (y_pred_critical == 1))
        fn = np.sum((y_true_critical == 1) & (y_pred_critical == 0))
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        metrics['critical_pattern_sensitivity'] = MetricResult(
            value=sensitivity,
            metadata={'patterns': self.critical_classes}
        )
        
        # Precision for critical patterns
        fp = np.sum((y_true_critical == 0) & (y_pred_critical == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        metrics['critical_pattern_precision'] = MetricResult(value=precision)
        
        return metrics
    
    def _compute_critical_false_negative_rate(self, y_true, y_pred):
        """Compute false negative rate for critical conditions."""
        critical_mask = np.isin(y_true, self.critical_indices)
        if not np.any(critical_mask):
            return 0.0
            
        critical_errors = y_true[critical_mask] != y_pred[critical_mask]
        return np.mean(critical_errors)
    
    def _compute_probability_metrics(self, y_true, y_prob):
        """Compute metrics based on predicted probabilities."""
        metrics = {}
        
        # Multi-class AUC (One-vs-Rest)
        try:
            auc_ovr = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            metrics['auc_macro'] = MetricResult(value=auc_ovr)
        except:
            pass
            
        # Weighted AUC
        try:
            auc_weighted = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            metrics['auc_weighted'] = MetricResult(value=auc_weighted)
        except:
            pass
            
        # Per-class AUC
        per_class_auc = {}
        for i, class_name in enumerate(self.class_names):
            y_true_binary = (y_true == i).astype(int)
            if len(np.unique(y_true_binary)) > 1:  # Need both classes
                try:
                    auc_score = roc_auc_score(y_true_binary, y_prob[:, i])
                    per_class_auc[class_name] = auc_score
                except:
                    pass
                    
        if per_class_auc:
            metrics['per_class_auc'] = MetricResult(
                value=np.mean(list(per_class_auc.values())),
                per_class_values=per_class_auc
            )
            
        # Average precision (area under PR curve)
        per_class_ap = {}
        for i, class_name in enumerate(self.class_names):
            y_true_binary = (y_true == i).astype(int)
            if len(np.unique(y_true_binary)) > 1:
                try:
                    ap_score = average_precision_score(y_true_binary, y_prob[:, i])
                    per_class_ap[class_name] = ap_score
                except:
                    pass
                    
        if per_class_ap:
            metrics['average_precision'] = MetricResult(
                value=np.mean(list(per_class_ap.values())),
                per_class_values=per_class_ap
            )
            
        # Brier score (calibration)
        brier_score = self._compute_brier_score(y_true, y_prob)
        metrics['brier_score'] = MetricResult(
            value=brier_score,
            metadata={'description': 'Lower is better, measures calibration'}
        )
        
        # Expected Calibration Error
        ece = self._compute_expected_calibration_error(y_true, y_prob)
        metrics['expected_calibration_error'] = MetricResult(
            value=ece,
            metadata={'description': 'Measures probability calibration'}
        )
        
        return metrics
    
    def _compute_brier_score(self, y_true, y_prob):
        """Compute multi-class Brier score."""
        # One-hot encode true labels
        y_true_onehot = np.zeros((len(y_true), self.n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1
        
        # Brier score
        return np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))
    
    def _compute_expected_calibration_error(self, y_true, y_prob, n_bins=10):
        """Compute Expected Calibration Error."""
        # Get predicted classes and confidences
        y_pred = np.argmax(y_prob, axis=1)
        confidences = np.max(y_prob, axis=1)
        
        # Bin predictions by confidence
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.astype(float).mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (y_pred[in_bin] == y_true[in_bin]).astype(float).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece
    
    def _compute_statistical_metrics(self, y_true, y_pred):
        """Compute statistical significance metrics."""
        metrics = {}
        
        # McNemar test preparation (for comparing models)
        # This is metadata for later statistical testing
        correct_predictions = (y_true == y_pred)
        metrics['correct_predictions'] = MetricResult(
            value=np.mean(correct_predictions),
            metadata={'predictions': correct_predictions}
        )
        
        # Class distribution metrics
        true_dist = np.bincount(y_true, minlength=self.n_classes) / len(y_true)
        pred_dist = np.bincount(y_pred, minlength=self.n_classes) / len(y_pred)
        
        # KL divergence between true and predicted distributions
        kl_div = stats.entropy(true_dist + 1e-10, pred_dist + 1e-10)
        metrics['class_distribution_kl_divergence'] = MetricResult(
            value=kl_div,
            metadata={'true_dist': true_dist, 'pred_dist': pred_dist}
        )
        
        return metrics
    
    def compute_confidence_intervals(self,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   metric_name: str,
                                   n_bootstrap: int = 1000,
                                   confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Compute confidence intervals using bootstrap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            metric_name: Name of metric to compute CI for
            n_bootstrap: Number of bootstrap iterations
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n_samples = len(y_true)
        metric_values = []
        
        # Define metric computation functions
        metric_functions = {
            'accuracy': lambda yt, yp: accuracy_score(yt, yp),
            'balanced_accuracy': lambda yt, yp: balanced_accuracy_score(yt, yp),
            'f1_macro': lambda yt, yp: f1_score(yt, yp, average='macro'),
            'seizure_sensitivity': lambda yt, yp: self._compute_seizure_sensitivity(yt, yp)
        }
        
        if metric_name not in metric_functions:
            raise ValueError(f"Confidence interval not implemented for {metric_name}")
            
        metric_func = metric_functions[metric_name]
        
        # Bootstrap sampling
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            try:
                metric_value = metric_func(y_true_boot, y_pred_boot)
                metric_values.append(metric_value)
            except:
                continue
                
        # Compute percentiles
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(metric_values, lower_percentile)
        upper_bound = np.percentile(metric_values, upper_percentile)
        
        return lower_bound, upper_bound
    
    def _compute_seizure_sensitivity(self, y_true, y_pred):
        """Helper to compute seizure sensitivity for bootstrap."""
        y_true_seizure = np.isin(y_true, self.seizure_indices).astype(int)
        y_pred_seizure = np.isin(y_pred, self.seizure_indices).astype(int)
        
        tp = np.sum((y_true_seizure == 1) & (y_pred_seizure == 1))
        fn = np.sum((y_true_seizure == 1) & (y_pred_seizure == 0))
        
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0


class TemporalMetrics:
    """
    Metrics for evaluating temporal aspects of predictions.
    Important for continuous EEG monitoring scenarios.
    """
    
    def __init__(self, segment_duration: float = 10.0):
        self.segment_duration = segment_duration  # seconds
        
    def compute_temporal_metrics(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               timestamps: np.ndarray,
                               patient_ids: Optional[np.ndarray] = None) -> Dict[str, MetricResult]:
        """
        Compute temporal performance metrics.
        
        Args:
            y_true: True labels over time
            y_pred: Predicted labels over time
            timestamps: Timestamps for each prediction
            patient_ids: Patient IDs for grouped analysis
            
        Returns:
            Dictionary of temporal metrics
        """
        metrics = {}
        
        # Detection latency
        latency_metrics = self._compute_detection_latency(y_true, y_pred, timestamps)
        metrics.update(latency_metrics)
        
        # Temporal consistency
        consistency_metrics = self._compute_temporal_consistency(y_pred, timestamps)
        metrics.update(consistency_metrics)
        
        # Event-based metrics
        event_metrics = self._compute_event_metrics(y_true, y_pred, timestamps)
        metrics.update(event_metrics)
        
        return metrics
    
    def _compute_detection_latency(self, y_true, y_pred, timestamps):
        """Compute detection latency for events."""
        metrics = {}
        
        # Find seizure events (assuming class 1 is seizure)
        seizure_class = 1
        
        # Detect onset of seizures in ground truth
        true_onsets = self._find_event_onsets(y_true, seizure_class)
        
        if len(true_onsets) == 0:
            return metrics
            
        # For each true onset, find detection time
        detection_latencies = []
        
        for onset_idx in true_onsets:
            # Look for detection within reasonable window (e.g., 60 seconds)
            window_size = int(60 / self.segment_duration)
            window_end = min(onset_idx + window_size, len(y_pred))
            
            # Find first detection in window
            detection_indices = np.where(y_pred[onset_idx:window_end] == seizure_class)[0]
            
            if len(detection_indices) > 0:
                detection_idx = onset_idx + detection_indices[0]
                latency = (timestamps[detection_idx] - timestamps[onset_idx])
                detection_latencies.append(latency)
            else:
                # Missed detection
                detection_latencies.append(np.inf)
                
        # Compute latency statistics
        finite_latencies = [l for l in detection_latencies if l != np.inf]
        
        if finite_latencies:
            metrics['mean_detection_latency'] = MetricResult(
                value=np.mean(finite_latencies),
                metadata={'unit': 'seconds'}
            )
            
            metrics['median_detection_latency'] = MetricResult(
                value=np.median(finite_latencies),
                metadata={'unit': 'seconds'}
            )
            
        # Detection rate
        detection_rate = len(finite_latencies) / len(true_onsets)
        metrics['event_detection_rate'] = MetricResult(value=detection_rate)
        
        return metrics
    
    def _find_event_onsets(self, labels, event_class):
        """Find onset indices of events."""
        # Detect transitions to event class
        is_event = (labels == event_class).astype(int)
        diff = np.diff(np.concatenate([[0], is_event]))
        onsets = np.where(diff == 1)[0]
        return onsets
    
    def _compute_temporal_consistency(self, y_pred, timestamps):
        """Compute temporal consistency of predictions."""
        metrics = {}
        
        # Prediction stability (frequency of class changes)
        changes = np.sum(np.diff(y_pred) != 0)
        change_rate = changes / (len(y_pred) - 1)
        
        metrics['prediction_change_rate'] = MetricResult(
            value=change_rate,
            metadata={'description': 'Frequency of prediction changes'}
        )
        
        # Average duration of consistent predictions
        consistent_segments = self._find_consistent_segments(y_pred)
        if consistent_segments:
            durations = [seg[1] - seg[0] for seg in consistent_segments]
            avg_duration = np.mean(durations) * self.segment_duration
            
            metrics['avg_consistent_prediction_duration'] = MetricResult(
                value=avg_duration,
                metadata={'unit': 'seconds'}
            )
            
        return metrics
    
    def _find_consistent_segments(self, predictions):
        """Find segments of consistent predictions."""
        segments = []
        start_idx = 0
        current_class = predictions[0]
        
        for i in range(1, len(predictions)):
            if predictions[i] != current_class:
                segments.append((start_idx, i))
                start_idx = i
                current_class = predictions[i]
                
        # Add final segment
        segments.append((start_idx, len(predictions)))
        
        return segments
    
    def _compute_event_metrics(self, y_true, y_pred, timestamps):
        """Compute event-based metrics."""
        metrics = {}
        
        # For each class, compute event-level metrics
        for class_idx in range(int(y_true.max()) + 1):
            if class_idx == 0:  # Skip background/normal class
                continue
                
            # Find true and predicted events
            true_events = self._extract_events(y_true, class_idx)
            pred_events = self._extract_events(y_pred, class_idx)
            
            # Event-level precision and recall
            tp, fp, fn = self._match_events(true_events, pred_events)
            
            event_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            event_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            metrics[f'class_{class_idx}_event_precision'] = MetricResult(value=event_precision)
            metrics[f'class_{class_idx}_event_recall'] = MetricResult(value=event_recall)
            
        return metrics
    
    def _extract_events(self, labels, event_class):
        """Extract event segments from labels."""
        events = []
        in_event = False
        start_idx = 0
        
        for i, label in enumerate(labels):
            if label == event_class and not in_event:
                # Event start
                start_idx = i
                in_event = True
            elif label != event_class and in_event:
                # Event end
                events.append((start_idx, i))
                in_event = False
                
        # Handle event extending to end
        if in_event:
            events.append((start_idx, len(labels)))
            
        return events
    
    def _match_events(self, true_events, pred_events, overlap_threshold=0.5):
        """Match predicted events to true events based on overlap."""
        tp = 0
        matched_true = set()
        matched_pred = set()
        
        # For each predicted event, find best matching true event
        for pred_idx, (pred_start, pred_end) in enumerate(pred_events):
            best_overlap = 0
            best_true_idx = None
            
            for true_idx, (true_start, true_end) in enumerate(true_events):
                if true_idx in matched_true:
                    continue
                    
                # Compute overlap
                overlap_start = max(pred_start, true_start)
                overlap_end = min(pred_end, true_end)
                
                if overlap_end > overlap_start:
                    overlap_duration = overlap_end - overlap_start
                    true_duration = true_end - true_start
                    overlap_ratio = overlap_duration / true_duration
                    
                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_true_idx = true_idx
                        
            # If sufficient overlap, count as true positive
            if best_overlap >= overlap_threshold:
                tp += 1
                matched_true.add(best_true_idx)
                matched_pred.add(pred_idx)
                
        fp = len(pred_events) - len(matched_pred)
        fn = len(true_events) - len(matched_true)
        
        return tp, fp, fn 