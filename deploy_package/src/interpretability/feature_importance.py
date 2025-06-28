"""
Feature importance analysis for EEG harmful brain activity classification.
Provides various methods to understand which features drive model predictions.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
import shap
from scipy.stats import spearmanr
import json
import os


@dataclass
class FeatureImportanceResult:
    """Container for feature importance results."""
    importance_scores: np.ndarray
    feature_names: List[str]
    importance_std: Optional[np.ndarray]
    method: str
    metadata: Dict[str, any]
    
    def __len__(self):
        """Return the number of features."""
        return len(self.importance_scores)
        
    def __getitem__(self, idx):
        """Get feature importance by index."""
        return self.importance_scores[idx]
        
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        indices = np.argsort(self.importance_scores)[-n:][::-1]
        return [(self.feature_names[i], self.importance_scores[i]) for i in indices]


class PermutationImportance:
    """
    Permutation importance for model-agnostic feature importance.
    Measures importance by shuffling features and observing performance drop.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def compute_importance(self,
                           X: torch.Tensor,
                           y: torch.Tensor,
                           n_repeats: int = 10,
                           scoring: str = 'accuracy',
                           batch_size: int = 32) -> FeatureImportanceResult:
        """
        Compute permutation importance.
        
        Args:
            X: Input features
            y: True labels
            n_repeats: Number of permutation repeats
            scoring: Scoring metric ('accuracy', 'auc', 'custom')
            batch_size: Batch size for evaluation
            
        Returns:
            FeatureImportanceResult with importance scores
        """
        n_features = X.shape[-1]
        baseline_score = self._compute_baseline_score(X, y, scoring, batch_size)
        
        importances = np.zeros((n_repeats, n_features))
        
        for repeat in tqdm(range(n_repeats), desc="Permutation repeats"):
            for feature_idx in range(n_features):
                # Create permuted copy
                X_permuted = X.clone()
                
                # Permute feature across samples
                perm_indices = torch.randperm(X.shape[0])
                if len(X.shape) == 3:  # (batch, channels, features)
                    X_permuted[:, :, feature_idx] = X[perm_indices, :, feature_idx]
                else:  # (batch, features)
                    X_permuted[:, feature_idx] = X[perm_indices, feature_idx]
                
                # Compute score with permuted feature
                permuted_score = self._compute_score(
                    X_permuted, y, scoring, batch_size
                )
                
                # Importance is the decrease in score
                importances[repeat, feature_idx] = baseline_score - permuted_score
                
        # Compute mean and std
        mean_importances = importances.mean(axis=0)
        std_importances = importances.std(axis=0)
        
        # Generate feature names if not provided
        feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        return FeatureImportanceResult(
            importance_scores=mean_importances,
            feature_names=feature_names,
            importance_std=std_importances,
            method='permutation_importance',
            metadata={
                'n_repeats': n_repeats,
                'scoring': scoring,
                'baseline_score': baseline_score
            }
        )
    
    def _compute_baseline_score(self, X, y, scoring, batch_size):
        """Compute baseline model performance."""
        return self._compute_score(X, y, scoring, batch_size)
    
    def _compute_score(self, X, y, scoring, batch_size):
        """Compute model performance score."""
        predictions = []
        
        # Process in batches
        n_batches = (len(X) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X))
                batch = X[start_idx:end_idx].to(self.device)
                
                pred = self.model(batch)
                predictions.append(pred.cpu())
                
        predictions = torch.cat(predictions)
        
        # Compute score based on metric
        if scoring == 'accuracy':
            pred_classes = predictions.argmax(dim=-1)
            score = accuracy_score(y.cpu().numpy(), pred_classes.numpy())
        elif scoring == 'auc':
            if predictions.shape[-1] == 2:
                # Binary classification
                probs = torch.softmax(predictions, dim=-1)[:, 1]
                score = roc_auc_score(y.cpu().numpy(), probs.numpy())
            else:
                # Multi-class - use average AUC
                probs = torch.softmax(predictions, dim=-1).numpy()
                y_true = y.cpu().numpy()
                scores = []
                for class_idx in range(predictions.shape[-1]):
                    y_binary = (y_true == class_idx).astype(int)
                    if y_binary.sum() > 0:  # Skip if class not present
                        scores.append(roc_auc_score(y_binary, probs[:, class_idx]))
                score = np.mean(scores)
        else:
            raise ValueError(f"Unknown scoring metric: {scoring}")
            
        return score


class PartialDependencePlotter:
    """
    Partial dependence plots showing feature-prediction relationships.
    Reveals how features affect predictions while marginalizing over other features.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def plot_partial_dependence(self,
                                X: torch.Tensor,
                                feature_indices: List[int],
                                feature_names: Optional[List[str]] = None,
                                grid_resolution: int = 100,
                                percentiles: Tuple[float, float] = (0.05, 0.95),
                                save_path: Optional[str] = None):
        """
        Create partial dependence plots.
        
        Args:
            X: Input data
            feature_indices: Indices of features to plot
            feature_names: Names of features
            grid_resolution: Number of grid points
            percentiles: Percentiles for grid range
            save_path: Path to save plots
        """
        n_features = len(feature_indices)
        fig, axes = plt.subplots(
            (n_features + 1) // 2, 2, 
            figsize=(12, 4 * ((n_features + 1) // 2))
        )
        
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        for idx, feature_idx in enumerate(feature_indices):
            # Compute partial dependence
            pd_result = self._compute_partial_dependence(
                X, feature_idx, grid_resolution, percentiles
            )
            
            # Plot
            ax = axes[idx]
            ax.plot(pd_result['grid_values'], pd_result['pdp_mean'], 
                   'b-', linewidth=2, label='Mean')
            ax.fill_between(
                pd_result['grid_values'],
                pd_result['pdp_mean'] - pd_result['pdp_std'],
                pd_result['pdp_mean'] + pd_result['pdp_std'],
                alpha=0.3, label='±1 std'
            )
            
            # Add individual conditional expectation curves
            if pd_result['ice_curves'] is not None:
                for ice_curve in pd_result['ice_curves'][:50]:  # Show max 50
                    ax.plot(pd_result['grid_values'], ice_curve, 
                           'gray', alpha=0.1, linewidth=0.5)
                    
            feature_name = (feature_names[feature_idx] if feature_names 
                          else f"Feature {feature_idx}")
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Partial Dependence')
            ax.set_title(f'Partial Dependence: {feature_name}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        # Remove extra subplots
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _compute_partial_dependence(self, X, feature_idx, grid_resolution, percentiles):
        """Compute partial dependence for a single feature."""
        # Create grid
        feature_values = X[:, feature_idx] if len(X.shape) == 2 else X[:, :, feature_idx]
        min_val = torch.quantile(feature_values.flatten(), percentiles[0])
        max_val = torch.quantile(feature_values.flatten(), percentiles[1])
        grid_values = torch.linspace(min_val, max_val, grid_resolution)
        
        pdp_values = []
        ice_curves = []
        
        # Compute PDP for each grid point
        for grid_val in tqdm(grid_values, desc=f"Computing PDP for feature {feature_idx}"):
            X_modified = X.clone()
            
            # Set feature to grid value
            if len(X.shape) == 3:
                X_modified[:, :, feature_idx] = grid_val
            else:
                X_modified[:, feature_idx] = grid_val
                
            # Get predictions
            with torch.no_grad():
                predictions = []
                batch_size = 32
                
                for i in range(0, len(X_modified), batch_size):
                    batch = X_modified[i:i+batch_size].to(self.device)
                    pred = self.model(batch)
                    predictions.append(pred.cpu())
                    
                predictions = torch.cat(predictions)
                
            # For classification, use probability of positive class
            if predictions.shape[-1] > 1:
                predictions = torch.softmax(predictions, dim=-1)[:, 1]
                
            pdp_values.append(predictions.numpy())
            
        pdp_values = np.array(pdp_values).T  # Shape: (n_samples, n_grid)
        
        # Compute mean and std
        pdp_mean = pdp_values.mean(axis=0)
        pdp_std = pdp_values.std(axis=0)
        
        # Store some ICE curves
        n_ice = min(100, len(X))
        ice_indices = np.random.choice(len(X), n_ice, replace=False)
        ice_curves = pdp_values[ice_indices]
        
        return {
            'grid_values': grid_values.numpy(),
            'pdp_mean': pdp_mean,
            'pdp_std': pdp_std,
            'ice_curves': ice_curves
        }


class AccumulatedLocalEffects:
    """
    Accumulated Local Effects (ALE) for unbiased feature importance.
    Handles correlated features better than PDP.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def compute_ale(self,
                    X: torch.Tensor,
                    feature_idx: int,
                    n_bins: int = 20) -> Dict[str, np.ndarray]:
        """
        Compute Accumulated Local Effects.
        
        Args:
            X: Input data
            feature_idx: Feature index to analyze
            n_bins: Number of bins for discretization
            
        Returns:
            Dictionary with ALE values and bin centers
        """
        # Sort data by feature value
        feature_values = X[:, feature_idx] if len(X.shape) == 2 else X[:, :, feature_idx].mean(dim=1)
        sorted_indices = torch.argsort(feature_values)
        X_sorted = X[sorted_indices]
        
        # Create bins
        n_samples = len(X)
        bin_size = n_samples // n_bins
        
        ale_values = []
        bin_centers = []
        
        for bin_idx in range(n_bins):
            start_idx = bin_idx * bin_size
            end_idx = start_idx + bin_size if bin_idx < n_bins - 1 else n_samples
            
            if end_idx <= start_idx:
                continue
                
            # Get samples in bin
            X_bin = X_sorted[start_idx:end_idx]
            
            # Compute local effects
            local_effects = self._compute_local_effects(
                X_bin, feature_idx, start_idx, end_idx, feature_values[sorted_indices]
            )
            
            ale_values.append(local_effects.mean())
            
            # Compute bin center
            bin_feature_values = feature_values[sorted_indices[start_idx:end_idx]]
            bin_centers.append(bin_feature_values.mean().item())
            
        # Accumulate effects
        ale_values = np.array(ale_values)
        ale_accumulated = np.cumsum(ale_values)
        
        # Center ALE
        ale_accumulated -= ale_accumulated.mean()
        
        return {
            'ale_values': ale_accumulated,
            'bin_centers': np.array(bin_centers),
            'feature_idx': feature_idx
        }
    
    def _compute_local_effects(self, X_bin, feature_idx, start_idx, end_idx, sorted_feature_values):
        """Compute local effects within a bin."""
        # Get feature values at bin boundaries
        if start_idx > 0:
            lower_bound = (sorted_feature_values[start_idx - 1] + 
                          sorted_feature_values[start_idx]) / 2
        else:
            lower_bound = sorted_feature_values[start_idx]
            
        if end_idx < len(sorted_feature_values):
            upper_bound = (sorted_feature_values[end_idx - 1] + 
                          sorted_feature_values[end_idx]) / 2
        else:
            upper_bound = sorted_feature_values[end_idx - 1]
            
        # Create copies with feature at boundaries
        X_lower = X_bin.clone()
        X_upper = X_bin.clone()
        
        if len(X_bin.shape) == 3:
            X_lower[:, :, feature_idx] = lower_bound
            X_upper[:, :, feature_idx] = upper_bound
        else:
            X_lower[:, feature_idx] = lower_bound
            X_upper[:, feature_idx] = upper_bound
            
        # Compute predictions
        with torch.no_grad():
            pred_lower = self.model(X_lower.to(self.device)).cpu()
            pred_upper = self.model(X_upper.to(self.device)).cpu()
            
        # Compute differences
        if pred_lower.shape[-1] > 1:  # Classification
            pred_lower = torch.softmax(pred_lower, dim=-1)[:, 1]
            pred_upper = torch.softmax(pred_upper, dim=-1)[:, 1]
            
        local_effects = (pred_upper - pred_lower).numpy()
        
        return local_effects


class ClinicalFeatureValidator:
    """
    Validate feature importance against clinical domain knowledge.
    Ensures model focuses on clinically relevant patterns.
    """
    
    def __init__(self, clinical_knowledge_path: Optional[str] = None):
        self.clinical_features = self._load_clinical_knowledge(clinical_knowledge_path)
        
    def _load_clinical_knowledge(self, path):
        """Load clinical feature importance priors."""
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        else:
            # Default clinical features for EEG
            return {
                'frequency_bands': {
                    'delta': {'range': [0.5, 4], 'importance': 'high'},
                    'theta': {'range': [4, 8], 'importance': 'high'},
                    'alpha': {'range': [8, 13], 'importance': 'medium'},
                    'beta': {'range': [13, 30], 'importance': 'medium'},
                    'gamma': {'range': [30, 50], 'importance': 'low'}
                },
                'temporal_patterns': {
                    'spikes': {'importance': 'critical'},
                    'sharp_waves': {'importance': 'high'},
                    'slow_waves': {'importance': 'high'}
                },
                'spatial_patterns': {
                    'focal': {'importance': 'high'},
                    'generalized': {'importance': 'high'},
                    'lateralized': {'importance': 'medium'}
                }
            }
            
    def validate_importance(self,
                            feature_importance_result: FeatureImportanceResult,
                            feature_metadata: Optional[Dict] = None) -> Dict[str, any]:
        """
        Validate feature importance against clinical knowledge.
        
        Args:
            feature_importance_result: Computed feature importance
            feature_metadata: Metadata about features (e.g., frequency, channel)
            
        Returns:
            Validation results and recommendations
        """
        importance_scores = feature_importance_result.importance_scores
        feature_names = feature_importance_result.feature_names
        
        validation_results = {
            'clinical_alignment_score': 0.0,
            'missing_critical_features': [],
            'unexpected_important_features': [],
            'recommendations': []
        }
        
        if feature_metadata:
            # Check frequency band importance
            freq_importance = self._validate_frequency_importance(
                importance_scores, feature_names, feature_metadata
            )
            validation_results['frequency_validation'] = freq_importance
            
            # Check spatial pattern importance
            spatial_importance = self._validate_spatial_importance(
                importance_scores, feature_names, feature_metadata
            )
            validation_results['spatial_validation'] = spatial_importance
            
        # Compute overall alignment score
        alignment_scores = []
        
        # Check if high-importance clinical features are captured
        top_k = int(len(importance_scores) * 0.2)  # Top 20% features
        top_indices = np.argsort(importance_scores)[-top_k:]
        
        for idx in top_indices:
            if self._is_clinically_important(feature_names[idx], feature_metadata):
                alignment_scores.append(1.0)
            else:
                alignment_scores.append(0.5)
                validation_results['unexpected_important_features'].append(
                    feature_names[idx]
                )
                
        validation_results['clinical_alignment_score'] = np.mean(alignment_scores)
        
        # Generate recommendations
        if validation_results['clinical_alignment_score'] < 0.7:
            validation_results['recommendations'].append(
                "Model may be focusing on non-clinical patterns. Consider domain-guided training."
            )
            
        return validation_results
    
    def _validate_frequency_importance(self, importance_scores, feature_names, metadata):
        """Validate importance of frequency-domain features."""
        freq_validation = {}
        
        for band_name, band_info in self.clinical_features['frequency_bands'].items():
            band_features = [
                i for i, name in enumerate(feature_names)
                if metadata.get(name, {}).get('frequency_range') == band_info['range']
            ]
            
            if band_features:
                mean_importance = np.mean([importance_scores[i] for i in band_features])
                expected_importance = band_info['importance']
                
                freq_validation[band_name] = {
                    'mean_importance': mean_importance,
                    'expected': expected_importance,
                    'aligned': self._check_alignment(mean_importance, expected_importance)
                }
                
        return freq_validation
    
    def _validate_spatial_importance(self, importance_scores, feature_names, metadata):
        """Validate importance of spatial patterns."""
        spatial_validation = {}
        
        # Group features by channel/location
        channel_importance = {}
        for i, name in enumerate(feature_names):
            channel = metadata.get(name, {}).get('channel')
            if channel:
                if channel not in channel_importance:
                    channel_importance[channel] = []
                channel_importance[channel].append(importance_scores[i])
                
        # Analyze spatial patterns
        if channel_importance:
            # Check for focal vs distributed importance
            importance_variance = np.var([np.mean(scores) for scores in channel_importance.values()])
            spatial_validation['pattern'] = 'focal' if importance_variance > 0.1 else 'distributed'
            spatial_validation['channel_importance'] = {
                ch: np.mean(scores) for ch, scores in channel_importance.items()
            }
            
        return spatial_validation
    
    def _is_clinically_important(self, feature_name, metadata):
        """Check if a feature is clinically important."""
        if not metadata or feature_name not in metadata:
            return False
            
        feature_meta = metadata[feature_name]
        
        # Check frequency bands
        freq_range = feature_meta.get('frequency_range')
        if freq_range:
            for band_info in self.clinical_features['frequency_bands'].values():
                if (freq_range[0] >= band_info['range'][0] and 
                    freq_range[1] <= band_info['range'][1] and
                    band_info['importance'] in ['high', 'critical']):
                    return True
                    
        # Check other clinical patterns
        pattern_type = feature_meta.get('pattern_type')
        if pattern_type:
            for pattern_category in self.clinical_features.values():
                if isinstance(pattern_category, dict) and pattern_type in pattern_category:
                    if pattern_category[pattern_type].get('importance') in ['high', 'critical']:
                        return True
                        
        return False
    
    def _check_alignment(self, actual_importance, expected_importance):
        """Check if actual importance aligns with expected."""
        importance_levels = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0}
        expected_score = importance_levels.get(expected_importance, 0.5)
        
        # Normalize actual importance to [0, 1]
        actual_normalized = actual_importance / (actual_importance + 1)
        
        return abs(actual_normalized - expected_score) < 0.3


class FeatureInteractionAnalyzer:
    """
    Analyze interactions between features.
    Identifies synergistic effects in model predictions.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def analyze_interactions(self,
                             X: torch.Tensor,
                             feature_pairs: Optional[List[Tuple[int, int]]] = None,
                             max_pairs: int = 10) -> Dict[str, any]:
        """
        Analyze feature interactions.
        
        Args:
            X: Input data
            feature_pairs: Specific pairs to analyze (None for automatic)
            max_pairs: Maximum pairs to analyze if automatic
            
        Returns:
            Dictionary with interaction analysis results
        """
        if feature_pairs is None:
            # Select top interacting pairs based on correlation
            feature_pairs = self._select_feature_pairs(X, max_pairs)
            
        interaction_results = {}
        
        for feat1, feat2 in tqdm(feature_pairs, desc="Analyzing interactions"):
            interaction_strength = self._compute_interaction_strength(
                X, feat1, feat2
            )
            
            interaction_results[f"feature_{feat1}_x_feature_{feat2}"] = {
                'interaction_strength': interaction_strength,
                'feature_indices': (feat1, feat2),
                'interaction_plot': self._create_interaction_plot(X, feat1, feat2)
            }
            
        # Rank interactions
        ranked_interactions = sorted(
            interaction_results.items(),
            key=lambda x: x[1]['interaction_strength'],
            reverse=True
        )
        
        return {
            'interactions': interaction_results,
            'ranked_interactions': ranked_interactions,
            'strongest_interaction': ranked_interactions[0] if ranked_interactions else None
        }
    
    def _select_feature_pairs(self, X, max_pairs):
        """Automatically select feature pairs for interaction analysis."""
        n_features = X.shape[-1]
        
        # Compute feature correlations
        if len(X.shape) == 3:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
            
        correlations = np.corrcoef(X_flat.T)
        
        # Select pairs with moderate correlation (potential interactions)
        pairs = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = abs(correlations[i, j])
                if 0.3 < corr < 0.8:  # Moderate correlation
                    pairs.append((i, j, corr))
                    
        # Sort by correlation and select top pairs
        pairs.sort(key=lambda x: x[2], reverse=True)
        return [(p[0], p[1]) for p in pairs[:max_pairs]]
    
    def _compute_interaction_strength(self, X, feat1, feat2):
        """Compute strength of interaction between two features."""
        # Use H-statistic approach
        # Compare joint effect vs sum of individual effects
        
        # Individual effects
        effect1 = self._compute_individual_effect(X, feat1)
        effect2 = self._compute_individual_effect(X, feat2)
        
        # Joint effect
        joint_effect = self._compute_joint_effect(X, feat1, feat2)
        
        # Interaction is deviation from additivity
        expected_joint = effect1 + effect2
        interaction = joint_effect - expected_joint
        
        # Normalize by total variance
        total_var = joint_effect.var()
        if total_var > 0:
            interaction_strength = (interaction ** 2).mean() / total_var
        else:
            interaction_strength = 0.0
            
        return interaction_strength
    
    def _compute_individual_effect(self, X, feature_idx):
        """Compute effect of individual feature."""
        # Create copies with feature at different values
        X_low = X.clone()
        X_high = X.clone()
        
        feature_values = X[:, feature_idx] if len(X.shape) == 2 else X[:, :, feature_idx]
        low_val = torch.quantile(feature_values.flatten(), 0.25)
        high_val = torch.quantile(feature_values.flatten(), 0.75)
        
        if len(X.shape) == 3:
            X_low[:, :, feature_idx] = low_val
            X_high[:, :, feature_idx] = high_val
        else:
            X_low[:, feature_idx] = low_val
            X_high[:, feature_idx] = high_val
            
        # Compute predictions
        with torch.no_grad():
            pred_low = self.model(X_low.to(self.device)).cpu()
            pred_high = self.model(X_high.to(self.device)).cpu()
            
        if pred_low.shape[-1] > 1:
            pred_low = torch.softmax(pred_low, dim=-1)[:, 1]
            pred_high = torch.softmax(pred_high, dim=-1)[:, 1]
            
        return (pred_high - pred_low).numpy()
    
    def _compute_joint_effect(self, X, feat1, feat2):
        """Compute joint effect of two features."""
        # Create 2x2 design
        X_ll = X.clone()  # low-low
        X_lh = X.clone()  # low-high
        X_hl = X.clone()  # high-low
        X_hh = X.clone()  # high-high
        
        # Get quartile values
        if len(X.shape) == 3:
            values1 = X[:, :, feat1].flatten()
            values2 = X[:, :, feat2].flatten()
        else:
            values1 = X[:, feat1]
            values2 = X[:, feat2]
            
        low1 = torch.quantile(values1, 0.25)
        high1 = torch.quantile(values1, 0.75)
        low2 = torch.quantile(values2, 0.25)
        high2 = torch.quantile(values2, 0.75)
        
        # Set values
        if len(X.shape) == 3:
            X_ll[:, :, feat1] = low1
            X_ll[:, :, feat2] = low2
            X_lh[:, :, feat1] = low1
            X_lh[:, :, feat2] = high2
            X_hl[:, :, feat1] = high1
            X_hl[:, :, feat2] = low2
            X_hh[:, :, feat1] = high1
            X_hh[:, :, feat2] = high2
        else:
            X_ll[:, feat1] = low1
            X_ll[:, feat2] = low2
            X_lh[:, feat1] = low1
            X_lh[:, feat2] = high2
            X_hl[:, feat1] = high1
            X_hl[:, feat2] = low2
            X_hh[:, feat1] = high1
            X_hh[:, feat2] = high2
            
        # Get predictions
        with torch.no_grad():
            pred_ll = self.model(X_ll.to(self.device)).cpu()
            pred_lh = self.model(X_lh.to(self.device)).cpu()
            pred_hl = self.model(X_hl.to(self.device)).cpu()
            pred_hh = self.model(X_hh.to(self.device)).cpu()
            
        if pred_ll.shape[-1] > 1:
            pred_ll = torch.softmax(pred_ll, dim=-1)[:, 1]
            pred_lh = torch.softmax(pred_lh, dim=-1)[:, 1]
            pred_hl = torch.softmax(pred_hl, dim=-1)[:, 1]
            pred_hh = torch.softmax(pred_hh, dim=-1)[:, 1]
            
        # Interaction effect: (hh - hl) - (lh - ll)
        interaction = (pred_hh - pred_hl) - (pred_lh - pred_ll)
        
        return interaction.numpy()
    
    def _create_interaction_plot(self, X, feat1, feat2):
        """Create interaction plot data."""
        # This returns data for plotting, actual plotting done elsewhere
        n_grid = 20
        
        if len(X.shape) == 3:
            values1 = X[:, :, feat1].flatten()
            values2 = X[:, :, feat2].flatten()
        else:
            values1 = X[:, feat1]
            values2 = X[:, feat2]
            
        grid1 = torch.linspace(values1.min(), values1.max(), n_grid)
        grid2 = torch.linspace(values2.min(), values2.max(), n_grid)
        
        interaction_matrix = np.zeros((n_grid, n_grid))
        
        # Sample a subset of data for efficiency
        n_samples = min(100, len(X))
        sample_indices = torch.randperm(len(X))[:n_samples]
        X_sample = X[sample_indices]
        
        for i, val1 in enumerate(grid1):
            for j, val2 in enumerate(grid2):
                X_temp = X_sample.clone()
                
                if len(X.shape) == 3:
                    X_temp[:, :, feat1] = val1
                    X_temp[:, :, feat2] = val2
                else:
                    X_temp[:, feat1] = val1
                    X_temp[:, feat2] = val2
                    
                with torch.no_grad():
                    pred = self.model(X_temp.to(self.device)).cpu()
                    
                if pred.shape[-1] > 1:
                    pred = torch.softmax(pred, dim=-1)[:, 1]
                    
                interaction_matrix[i, j] = pred.mean().item()
                
        return {
            'grid1': grid1.numpy(),
            'grid2': grid2.numpy(),
            'interaction_matrix': interaction_matrix,
            'feature_indices': (feat1, feat2)
        }


class FeatureImportanceReporter:
    """
    Generate comprehensive feature importance reports.
    Creates visualizations and summaries for clinical interpretation.
    """
    
    def __init__(self):
        self.report_data = {}
        
    def generate_report(self,
                        importance_results: List[FeatureImportanceResult],
                        clinical_validation: Optional[Dict] = None,
                        interaction_analysis: Optional[Dict] = None,
                        save_path: Optional[str] = None):
        """
        Generate comprehensive feature importance report.
        
        Args:
            importance_results: List of importance results from different methods
            clinical_validation: Clinical validation results
            interaction_analysis: Feature interaction analysis
            save_path: Path to save report
        """
        # Create figure with subplots
        n_methods = len(importance_results)
        fig = plt.figure(figsize=(16, 6 * ((n_methods + 2) // 2)))
        
        # Plot importance scores from each method
        for idx, result in enumerate(importance_results):
            ax = plt.subplot((n_methods + 2) // 2, 2, idx + 1)
            self._plot_importance_scores(ax, result)
            
        # Add clinical validation plot if available
        if clinical_validation:
            ax = plt.subplot((n_methods + 2) // 2, 2, n_methods + 1)
            self._plot_clinical_validation(ax, clinical_validation)
            
        # Add interaction heatmap if available
        if interaction_analysis:
            ax = plt.subplot((n_methods + 2) // 2, 2, n_methods + 2)
            self._plot_interactions(ax, interaction_analysis)
            
        plt.tight_layout()
        
        if save_path:
            # Save figure
            fig_path = save_path.replace('.html', '_figure.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            
            # Generate HTML report
            self._generate_html_report(
                importance_results, clinical_validation, 
                interaction_analysis, save_path
            )
            
        plt.close()
        
    def _plot_importance_scores(self, ax, result):
        """Plot feature importance scores."""
        # Select top features
        n_features = len(result.importance_scores)
        n_show = min(20, n_features)
        
        top_indices = np.argsort(result.importance_scores)[-n_show:]
        top_scores = result.importance_scores[top_indices]
        top_names = [result.feature_names[i] for i in top_indices]
        
        # Create horizontal bar plot
        y_pos = np.arange(n_show)
        
        bars = ax.barh(y_pos, top_scores)
        
        # Add error bars if available
        if result.importance_std is not None:
            top_std = result.importance_std[top_indices]
            ax.errorbar(top_scores, y_pos, xerr=top_std, fmt='none', 
                       color='black', capsize=3)
            
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Feature Importance - {result.method}')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Color bars by importance
        colors = plt.cm.RdYlBu_r(top_scores / top_scores.max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            
    def _plot_clinical_validation(self, ax, validation):
        """Plot clinical validation results."""
        # Create alignment visualization
        if 'frequency_validation' in validation:
            freq_data = validation['frequency_validation']
            
            bands = list(freq_data.keys())
            importances = [freq_data[b]['mean_importance'] for b in bands]
            expected = [freq_data[b]['expected'] for b in bands]
            
            x = np.arange(len(bands))
            width = 0.35
            
            ax.bar(x - width/2, importances, width, label='Actual', alpha=0.7)
            
            # Convert expected to numeric
            expected_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'critical': 1.0}
            expected_numeric = [expected_map.get(e, 0.5) for e in expected]
            ax.bar(x + width/2, expected_numeric, width, label='Expected', alpha=0.7)
            
            ax.set_xlabel('Frequency Band')
            ax.set_ylabel('Importance')
            ax.set_title('Clinical Alignment - Frequency Bands')
            ax.set_xticks(x)
            ax.set_xticklabels(bands)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
    def _plot_interactions(self, ax, interaction_analysis):
        """Plot feature interactions."""
        # Create interaction strength heatmap
        interactions = interaction_analysis['interactions']
        
        # Extract interaction matrix
        n_features = int(np.sqrt(len(interactions))) + 1
        interaction_matrix = np.zeros((n_features, n_features))
        
        for interaction_name, data in interactions.items():
            feat1, feat2 = data['feature_indices']
            strength = data['interaction_strength']
            interaction_matrix[feat1, feat2] = strength
            interaction_matrix[feat2, feat1] = strength
            
        # Plot heatmap
        sns.heatmap(interaction_matrix, ax=ax, cmap='YlOrRd', 
                   cbar_kws={'label': 'Interaction Strength'})
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('Feature Index')
        ax.set_title('Feature Interaction Strengths')
        
    def _generate_html_report(self, importance_results, clinical_validation,
                              interaction_analysis, save_path):
        """Generate HTML report with all results."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Feature Importance Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                .section { margin-bottom: 30px; }
                .metric { background-color: #f0f0f0; padding: 10px; margin: 5px 0; }
                .warning { background-color: #fff3cd; padding: 10px; margin: 10px 0; }
                .success { background-color: #d4edda; padding: 10px; margin: 10px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <h1>Feature Importance Analysis Report</h1>
        """
        
        # Add summary section
        html_content += self._create_summary_section(
            importance_results, clinical_validation
        )
        
        # Add detailed results for each method
        for result in importance_results:
            html_content += self._create_method_section(result)
            
        # Add clinical validation section
        if clinical_validation:
            html_content += self._create_clinical_section(clinical_validation)
            
        # Add interaction analysis section
        if interaction_analysis:
            html_content += self._create_interaction_section(interaction_analysis)
            
        # Add recommendations
        html_content += self._create_recommendations_section(
            importance_results, clinical_validation
        )
        
        html_content += """
        </body>
        </html>
        """
        
        # Save HTML file
        with open(save_path, 'w') as f:
            f.write(html_content)
            
    def _create_summary_section(self, importance_results, clinical_validation):
        """Create summary section for HTML report."""
        section = "<div class='section'><h2>Executive Summary</h2>"
        
        # Count consistently important features across methods
        all_features = set()
        top_features_by_method = {}
        
        for result in importance_results:
            top_k = min(10, len(result.importance_scores))
            top_indices = np.argsort(result.importance_scores)[-top_k:]
            top_features = [result.feature_names[i] for i in top_indices]
            top_features_by_method[result.method] = set(top_features)
            all_features.update(top_features)
            
        # Find features that appear in multiple methods
        consistent_features = []
        for feature in all_features:
            count = sum(1 for features in top_features_by_method.values() 
                       if feature in features)
            if count >= len(importance_results) // 2:
                consistent_features.append(feature)
                
        section += f"<div class='metric'>Total features analyzed: {len(all_features)}</div>"
        section += f"<div class='metric'>Consistently important features: {len(consistent_features)}</div>"
        
        if clinical_validation:
            score = clinical_validation.get('clinical_alignment_score', 0)
            if score > 0.7:
                section += f"<div class='success'>Clinical alignment score: {score:.2f} - Good alignment with clinical knowledge</div>"
            else:
                section += f"<div class='warning'>Clinical alignment score: {score:.2f} - Low alignment with clinical knowledge</div>"
                
        section += "</div>"
        return section
        
    def _create_method_section(self, result):
        """Create section for individual method results."""
        section = f"<div class='section'><h2>{result.method} Results</h2>"
        
        # Add metadata
        section += "<h3>Method Configuration</h3>"
        for key, value in result.metadata.items():
            section += f"<div class='metric'>{key}: {value}</div>"
            
        # Add top features table
        section += "<h3>Top 10 Most Important Features</h3>"
        section += "<table><tr><th>Rank</th><th>Feature</th><th>Importance Score</th>"
        
        if result.importance_std is not None:
            section += "<th>Std Dev</th>"
            
        section += "</tr>"
        
        top_indices = np.argsort(result.importance_scores)[-10:][::-1]
        
        for rank, idx in enumerate(top_indices, 1):
            section += f"<tr><td>{rank}</td>"
            section += f"<td>{result.feature_names[idx]}</td>"
            section += f"<td>{result.importance_scores[idx]:.4f}</td>"
            
            if result.importance_std is not None:
                section += f"<td>{result.importance_std[idx]:.4f}</td>"
                
            section += "</tr>"
            
        section += "</table></div>"
        return section
        
    def _create_clinical_section(self, validation):
        """Create clinical validation section."""
        section = "<div class='section'><h2>Clinical Validation</h2>"
        
        score = validation.get('clinical_alignment_score', 0)
        section += f"<h3>Overall Clinical Alignment Score: {score:.2f}</h3>"
        
        # Add frequency validation
        if 'frequency_validation' in validation:
            section += "<h3>Frequency Band Analysis</h3>"
            section += "<table><tr><th>Band</th><th>Expected Importance</th><th>Actual Importance</th><th>Aligned</th></tr>"
            
            for band, data in validation['frequency_validation'].items():
                aligned = "✓" if data['aligned'] else "✗"
                section += f"<tr><td>{band}</td>"
                section += f"<td>{data['expected']}</td>"
                section += f"<td>{data['mean_importance']:.3f}</td>"
                section += f"<td>{aligned}</td></tr>"
                
            section += "</table>"
            
        # Add recommendations
        if validation.get('recommendations'):
            section += "<h3>Recommendations</h3>"
            for rec in validation['recommendations']:
                section += f"<div class='warning'>{rec}</div>"
                
        section += "</div>"
        return section
        
    def _create_interaction_section(self, interaction_analysis):
        """Create feature interaction section."""
        section = "<div class='section'><h2>Feature Interactions</h2>"
        
        # Add strongest interactions
        section += "<h3>Top 5 Strongest Feature Interactions</h3>"
        section += "<table><tr><th>Rank</th><th>Feature Pair</th><th>Interaction Strength</th></tr>"
        
        for rank, (name, data) in enumerate(
            interaction_analysis['ranked_interactions'][:5], 1
        ):
            feat1, feat2 = data['feature_indices']
            section += f"<tr><td>{rank}</td>"
            section += f"<td>Feature {feat1} × Feature {feat2}</td>"
            section += f"<td>{data['interaction_strength']:.4f}</td></tr>"
            
        section += "</table></div>"
        return section
        
    def _create_recommendations_section(self, importance_results, clinical_validation):
        """Create recommendations section."""
        section = "<div class='section'><h2>Recommendations</h2>"
        
        recommendations = []
        
        # Check for method agreement
        if len(importance_results) > 1:
            # Compare top features across methods
            top_features_sets = []
            for result in importance_results:
                top_k = min(10, len(result.importance_scores))
                top_indices = np.argsort(result.importance_scores)[-top_k:]
                top_features = set(result.feature_names[i] for i in top_indices)
                top_features_sets.append(top_features)
                
            # Calculate overlap
            intersection = set.intersection(*top_features_sets)
            union = set.union(*top_features_sets)
            jaccard = len(intersection) / len(union) if union else 0
            
            if jaccard < 0.5:
                recommendations.append(
                    "Low agreement between different importance methods. "
                    "Consider ensemble approach or investigate method-specific biases."
                )
                
        # Clinical alignment recommendations
        if clinical_validation:
            score = clinical_validation.get('clinical_alignment_score', 0)
            if score < 0.7:
                recommendations.append(
                    "Model shows low alignment with clinical knowledge. "
                    "Consider incorporating domain knowledge into training or feature engineering."
                )
                
            if clinical_validation.get('unexpected_important_features'):
                recommendations.append(
                    f"Model identifies {len(clinical_validation['unexpected_important_features'])} "
                    "unexpected important features. Review these for potential insights or artifacts."
                )
                
        # General recommendations
        recommendations.append(
            "Regularly validate feature importance against new clinical knowledge and expert feedback."
        )
        recommendations.append(
            "Consider temporal stability analysis to ensure importance patterns are consistent over time."
        )
        
        for rec in recommendations:
            section += f"<div class='metric'>• {rec}</div>"
            
        section += "</div>"
        return section


class FeatureStabilityAnalyzer:
    """
    Analyze stability of feature importance across different conditions.
    Ensures robust and reliable feature importance estimates.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def analyze_stability(self,
                          X: torch.Tensor,
                          y: torch.Tensor,
                          n_bootstrap: int = 100,
                          subsample_ratio: float = 0.8) -> Dict[str, any]:
        """
        Analyze feature importance stability.
        
        Args:
            X: Input features
            y: Labels
            n_bootstrap: Number of bootstrap samples
            subsample_ratio: Ratio of data to use in each bootstrap
            
        Returns:
            Stability analysis results
        """
        n_samples = len(X)
        n_features = X.shape[-1]
        subsample_size = int(n_samples * subsample_ratio)
        
        # Store importance scores from each bootstrap
        bootstrap_importances = []
        
        # Create importance calculator
        perm_importance = PermutationImportance(self.model, self.device)
        
        for bootstrap_idx in tqdm(range(n_bootstrap), desc="Bootstrap sampling"):
            # Sample with replacement
            indices = torch.randint(0, n_samples, (subsample_size,))
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Compute importance
            result = perm_importance.compute_importance(
                X_bootstrap, y_bootstrap, n_repeats=5
            )
            
            bootstrap_importances.append(result.importance_scores)
            
        bootstrap_importances = np.array(bootstrap_importances)
        
        # Compute stability metrics
        stability_results = {
            'mean_importance': bootstrap_importances.mean(axis=0),
            'std_importance': bootstrap_importances.std(axis=0),
            'cv_importance': self._compute_cv(bootstrap_importances),
            'rank_stability': self._compute_rank_stability(bootstrap_importances),
            'selection_stability': self._compute_selection_stability(bootstrap_importances),
            'confidence_intervals': self._compute_confidence_intervals(bootstrap_importances)
        }
        
        # Identify stable vs unstable features
        stable_features, unstable_features = self._identify_stable_features(
            stability_results
        )
        
        stability_results['stable_features'] = stable_features
        stability_results['unstable_features'] = unstable_features
        
        return stability_results
    
    def _compute_cv(self, importances):
        """Compute coefficient of variation for each feature."""
        mean_imp = importances.mean(axis=0)
        std_imp = importances.std(axis=0)
        
        # Avoid division by zero
        cv = np.zeros_like(mean_imp)
        non_zero = mean_imp > 0
        cv[non_zero] = std_imp[non_zero] / mean_imp[non_zero]
        
        return cv
    
    def _compute_rank_stability(self, importances):
        """Compute stability of feature rankings."""
        n_bootstrap = importances.shape[0]
        n_features = importances.shape[1]
        
        # Get ranks for each bootstrap
        ranks = np.zeros((n_bootstrap, n_features))
        for i in range(n_bootstrap):
            ranks[i] = n_features - np.argsort(np.argsort(importances[i]))
            
        # Compute rank variance for each feature
        rank_variance = ranks.var(axis=0)
        
        # Normalize by maximum possible variance
        max_variance = (n_features ** 2 - 1) / 12  # Variance of uniform distribution
        rank_stability = 1 - (rank_variance / max_variance)
        
        return rank_stability
    
    def _compute_selection_stability(self, importances, top_k=10):
        """Compute stability of top-k feature selection."""
        n_bootstrap = importances.shape[0]
        
        # Get top-k features for each bootstrap
        top_features_list = []
        for i in range(n_bootstrap):
            top_indices = np.argsort(importances[i])[-top_k:]
            top_features_list.append(set(top_indices))
            
        # Compute pairwise Jaccard similarity
        similarities = []
        for i in range(n_bootstrap):
            for j in range(i + 1, n_bootstrap):
                intersection = len(top_features_list[i] & top_features_list[j])
                union = len(top_features_list[i] | top_features_list[j])
                if union > 0:
                    similarities.append(intersection / union)
                    
        return np.mean(similarities) if similarities else 0.0
    
    def _compute_confidence_intervals(self, importances, confidence=0.95):
        """Compute confidence intervals using bootstrap percentiles."""
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bounds = np.percentile(importances, lower_percentile, axis=0)
        upper_bounds = np.percentile(importances, upper_percentile, axis=0)
        
        return {
            'lower': lower_bounds,
            'upper': upper_bounds,
            'confidence_level': confidence
        }
    
    def _identify_stable_features(self, stability_results, cv_threshold=0.5):
        """Identify stable and unstable features."""
        cv_scores = stability_results['cv_importance']
        rank_stability = stability_results['rank_stability']
        
        # Features are stable if they have low CV and high rank stability
        stable_mask = (cv_scores < cv_threshold) & (rank_stability > 0.7)
        
        stable_features = np.where(stable_mask)[0].tolist()
        unstable_features = np.where(~stable_mask)[0].tolist()
        
        return stable_features, unstable_features 