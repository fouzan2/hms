"""
Ensemble model combining ResNet1D-GRU and EfficientNet with meta-learning.
Implements stacking ensemble with advanced combination strategies.
Enhanced with dynamic weighting, Bayesian averaging, and diversity measures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import entropy
from scipy.special import softmax
import joblib
import logging
from pathlib import Path
import yaml

from .resnet1d_gru import ResNet1D_GRU
from .efficientnet_spectrogram import EfficientNetSpectrogram

logger = logging.getLogger(__name__)


class AttentionFusion(nn.Module):
    """Attention-based fusion mechanism for ensemble predictions."""
    
    def __init__(self, num_models: int, num_classes: int, hidden_dim: int = 256):
        super(AttentionFusion, self).__init__()
        
        self.num_models = num_models
        self.num_classes = num_classes
        
        # Attention weights generator
        self.attention = nn.Sequential(
            nn.Linear(num_models * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_models),
            nn.Softmax(dim=1)
        )
        
        # Confidence estimator
        self.confidence = nn.Sequential(
            nn.Linear(num_models * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse predictions from multiple models.
        
        Args:
            predictions: List of tensors, each (batch, num_classes)
            
        Returns:
            fused_prediction: (batch, num_classes)
            confidence: (batch, 1)
        """
        # Stack predictions
        stacked = torch.stack(predictions, dim=1)  # (batch, num_models, num_classes)
        batch_size = stacked.shape[0]
        
        # Flatten for attention
        flattened = stacked.view(batch_size, -1)  # (batch, num_models * num_classes)
        
        # Generate attention weights
        attention_weights = self.attention(flattened)  # (batch, num_models)
        attention_weights = attention_weights.unsqueeze(-1)  # (batch, num_models, 1)
        
        # Apply weighted fusion
        weighted_preds = stacked * attention_weights  # (batch, num_models, num_classes)
        fused = weighted_preds.sum(dim=1)  # (batch, num_classes)
        
        # Estimate confidence
        confidence = self.confidence(flattened)  # (batch, 1)
        
        return fused, confidence


class BayesianModelAveraging(nn.Module):
    """Bayesian Model Averaging for uncertainty-aware ensemble."""
    
    def __init__(self, num_models: int, num_classes: int, temperature: float = 1.0):
        super(BayesianModelAveraging, self).__init__()
        self.num_models = num_models
        self.num_classes = num_classes
        self.temperature = temperature
        
        # Learnable model weights with temperature scaling
        self.model_weights = nn.Parameter(torch.ones(num_models) / num_models)
        self.weight_temperature = nn.Parameter(torch.tensor(temperature))
        
    def forward(self, predictions: List[torch.Tensor], 
                uncertainties: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform Bayesian model averaging.
        
        Args:
            predictions: List of probability tensors (batch, num_classes)
            uncertainties: List of uncertainty tensors (batch, num_classes)
            
        Returns:
            Dictionary with averaged predictions and uncertainties
        """
        # Stack predictions and uncertainties
        pred_stack = torch.stack(predictions, dim=1)  # (batch, num_models, num_classes)
        unc_stack = torch.stack(uncertainties, dim=1)  # (batch, num_models, num_classes)
        
        # Compute model weights based on uncertainties
        avg_uncertainty = unc_stack.mean(dim=2)  # (batch, num_models)
        confidence = 1 / (1 + avg_uncertainty)  # Higher confidence for lower uncertainty
        
        # Apply temperature scaling to model weights
        scaled_weights = F.softmax(self.model_weights / self.weight_temperature, dim=0)
        
        # Combine with confidence-based weights
        dynamic_weights = confidence * scaled_weights.unsqueeze(0)
        dynamic_weights = dynamic_weights / dynamic_weights.sum(dim=1, keepdim=True)
        
        # Weighted average of predictions
        weighted_pred = (pred_stack * dynamic_weights.unsqueeze(-1)).sum(dim=1)
        
        # Propagate uncertainty
        # Uncertainty = weighted uncertainty + variance of predictions
        weighted_unc = (unc_stack * dynamic_weights.unsqueeze(-1)).sum(dim=1)
        pred_variance = ((pred_stack - weighted_pred.unsqueeze(1)) ** 2 * 
                        dynamic_weights.unsqueeze(-1)).sum(dim=1)
        total_uncertainty = weighted_unc + pred_variance
        
        return {
            'prediction': weighted_pred,
            'uncertainty': total_uncertainty,
            'model_weights': dynamic_weights
        }


class EnsembleDiversity:
    """Measures and optimizes ensemble diversity."""
    
    @staticmethod
    def pairwise_disagreement(predictions: List[torch.Tensor]) -> float:
        """Calculate pairwise disagreement between models."""
        n_models = len(predictions)
        if n_models < 2:
            return 0.0
            
        disagreements = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                pred_i = torch.argmax(predictions[i], dim=1)
                pred_j = torch.argmax(predictions[j], dim=1)
                disagreement = (pred_i != pred_j).float().mean().item()
                disagreements.append(disagreement)
                
        return np.mean(disagreements)
    
    @staticmethod
    def entropy_diversity(predictions: List[torch.Tensor]) -> float:
        """Calculate entropy-based diversity measure."""
        # Stack predictions
        stacked = torch.stack(predictions, dim=0)  # (n_models, batch, n_classes)
        
        # Calculate entropy for each sample
        entropies = []
        for i in range(stacked.shape[1]):
            sample_preds = stacked[:, i, :].cpu().numpy()
            avg_pred = np.mean(sample_preds, axis=0)
            ent = entropy(avg_pred)
            entropies.append(ent)
            
        return np.mean(entropies)
    
    @staticmethod
    def correlation_diversity(features: List[torch.Tensor]) -> float:
        """Calculate feature correlation-based diversity."""
        if len(features) < 2:
            return 1.0
            
        correlations = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                feat_i = features[i].flatten()
                feat_j = features[j].flatten()
                
                # Compute correlation
                corr = torch.corrcoef(torch.stack([feat_i, feat_j]))[0, 1]
                correlations.append(abs(corr.item()))
                
        # Lower correlation means higher diversity
        return 1.0 - np.mean(correlations)


class AdaptiveEnsembleSelection(nn.Module):
    """Dynamically select best subset of models based on input characteristics."""
    
    def __init__(self, num_models: int, feature_dim: int, num_classes: int):
        super(AdaptiveEnsembleSelection, self).__init__()
        
        self.num_models = num_models
        self.num_classes = num_classes
        
        # Input characteristics analyzer
        self.input_analyzer = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Model selector (outputs selection probabilities)
        self.model_selector = nn.Sequential(
            nn.Linear(64, num_models),
            nn.Sigmoid()
        )
        
        # Gumbel-Softmax temperature for differentiable selection
        self.temperature = 1.0
        
    def forward(self, input_features: torch.Tensor, 
                predictions: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Select best models for given input.
        
        Args:
            input_features: Features characterizing the input
            predictions: List of model predictions
            
        Returns:
            Dictionary with selected predictions and selection weights
        """
        # Analyze input characteristics
        input_repr = self.input_analyzer(input_features)
        
        # Get selection probabilities
        selection_probs = self.model_selector(input_repr)
        
        # Apply Gumbel-Softmax for differentiable selection
        if self.training:
            # Add Gumbel noise for exploration
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(selection_probs) + 1e-8) + 1e-8)
            selection_weights = F.softmax((torch.log(selection_probs + 1e-8) + gumbel_noise) / self.temperature, dim=-1)
        else:
            # Hard selection during inference
            selection_weights = (selection_probs > 0.5).float()
            selection_weights = selection_weights / (selection_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Apply selection weights
        stacked_preds = torch.stack(predictions, dim=1)  # (batch, num_models, num_classes)
        selected_pred = (stacked_preds * selection_weights.unsqueeze(-1)).sum(dim=1)
        
        return {
            'prediction': selected_pred,
            'selection_weights': selection_weights
        }


class StackingEnsemble(nn.Module):
    """Enhanced stacking ensemble with neural network meta-learner."""
    
    def __init__(self, num_base_models: int, num_classes: int, 
                 feature_dim: int, hidden_dim: int = 512):
        super(StackingEnsemble, self).__init__()
        
        # Input: concatenated predictions and features from all models
        input_dim = num_base_models * (num_classes + feature_dim)
        
        # Multi-head meta-learner for different aspects
        self.prediction_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Uncertainty estimation with aleatoric and epistemic components
        self.aleatoric_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.epistemic_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Diversity regularization
        self.diversity_weight = 0.1
        
    def forward(self, predictions: List[torch.Tensor], 
                features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Enhanced stacking with uncertainty decomposition.
        """
        # Concatenate all inputs
        all_inputs = []
        for pred, feat in zip(predictions, features):
            all_inputs.extend([pred, feat])
            
        stacked_input = torch.cat(all_inputs, dim=1)
        
        # Get predictions
        final_prediction = self.prediction_head(stacked_input)
        
        # Uncertainty estimation
        log_aleatoric = self.aleatoric_head(stacked_input)
        log_epistemic = self.epistemic_head(stacked_input)
        
        aleatoric_uncertainty = torch.exp(log_aleatoric)
        epistemic_uncertainty = torch.exp(log_epistemic)
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        # Calculate diversity loss for regularization
        diversity_score = EnsembleDiversity.pairwise_disagreement(predictions)
        diversity_loss = -self.diversity_weight * diversity_score
        
        return {
            'prediction': final_prediction,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'total_uncertainty': total_uncertainty,
            'diversity_loss': diversity_loss
        }


class HMSEnsembleModel(nn.Module):
    """Enhanced ensemble model for HMS brain activity classification."""
    
    def __init__(self, config: Dict):
        super(HMSEnsembleModel, self).__init__()
        
        self.config = config
        self.ensemble_config = config['models']['ensemble']
        self.num_classes = len(config['classes'])
        
        # Initialize base models
        self.resnet_gru = ResNet1D_GRU(config)
        self.efficientnet = EfficientNetSpectrogram(config)
        
        # Feature dimensions from base models
        resnet_feature_dim = config['models']['resnet1d_gru']['gru']['hidden_size'] * 2
        efficientnet_feature_dim = 1536  # EfficientNet-B3
        total_feature_dim = resnet_feature_dim + efficientnet_feature_dim
        
        # Initialize ensemble methods
        self.ensemble_method = self.ensemble_config['method']
        
        # Bayesian Model Averaging
        self.bayesian_averaging = BayesianModelAveraging(
            num_models=2,
            num_classes=self.num_classes,
            temperature=1.0
        )
        
        # Adaptive ensemble selection
        self.adaptive_selection = AdaptiveEnsembleSelection(
            num_models=2,
            feature_dim=32,  # Meta-feature dimension
            num_classes=self.num_classes
        )
        
        # Enhanced stacking ensemble
        self.stacking_ensemble = StackingEnsemble(
            num_base_models=2,
            num_classes=self.num_classes,
            feature_dim=total_feature_dim // 2,
            hidden_dim=512
        )
        
        # Calibration with temperature scaling
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Meta-features extractor
        self.meta_feature_extractor = self._build_meta_feature_extractor()
        
    def _build_meta_feature_extractor(self) -> nn.Module:
        """Build enhanced module to extract meta-features."""
        return nn.Sequential(
            nn.Linear(self.num_classes * 2 + 4, 128),  # +4 for diversity metrics
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
    def _calculate_diversity_metrics(self, predictions: List[torch.Tensor], 
                                   features: List[torch.Tensor]) -> torch.Tensor:
        """Calculate diversity metrics for the ensemble."""
        with torch.no_grad():
            disagreement = EnsembleDiversity.pairwise_disagreement(predictions)
            entropy_div = EnsembleDiversity.entropy_diversity(predictions)
            correlation_div = EnsembleDiversity.correlation_diversity(features)
            
            # Average prediction variance
            stacked = torch.stack(predictions, dim=1)
            avg_variance = stacked.var(dim=1).mean().item()
            
        return torch.tensor([disagreement, entropy_div, correlation_div, avg_variance], 
                          device=predictions[0].device)
        
    def forward(self, eeg_data: torch.Tensor, 
                spectrogram_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass through ensemble.
        """
        # Get predictions from base models
        resnet_output = self.resnet_gru(eeg_data)
        efficientnet_output = self.efficientnet(spectrogram_data)
        
        # Extract predictions and features
        resnet_logits = resnet_output['logits']
        efficientnet_logits = efficientnet_output['logits']
        
        # Temperature-scaled probabilities
        resnet_probs = F.softmax(resnet_logits / self.temperature, dim=1)
        efficientnet_probs = F.softmax(efficientnet_logits / self.temperature, dim=1)
        
        # Get uncertainties from models
        resnet_uncertainty = torch.ones_like(resnet_probs) * 0.1  # Placeholder
        efficientnet_uncertainty = efficientnet_output.get('log_variance', 
                                                         torch.zeros_like(efficientnet_probs))
        efficientnet_uncertainty = torch.exp(efficientnet_uncertainty)
        
        # Get features for stacking
        resnet_features = resnet_output['features']['resnet_output'].mean(dim=2)
        efficientnet_features = efficientnet_output['features']
        
        # Calculate diversity metrics
        diversity_metrics = self._calculate_diversity_metrics(
            [resnet_probs, efficientnet_probs],
            [resnet_features.flatten(1), efficientnet_features]
        )
        
        # Extract meta-features
        meta_input = torch.cat([
            resnet_probs, 
            efficientnet_probs,
            diversity_metrics.unsqueeze(0).expand(resnet_probs.size(0), -1)
        ], dim=1)
        meta_features = self.meta_feature_extractor(meta_input)
        
        # Apply different ensemble methods
        results = {}
        
        # Bayesian Model Averaging
        bayesian_output = self.bayesian_averaging(
            [resnet_probs, efficientnet_probs],
            [resnet_uncertainty, efficientnet_uncertainty]
        )
        
        # Adaptive Selection
        adaptive_output = self.adaptive_selection(
            meta_features,
            [resnet_probs, efficientnet_probs]
        )
        
        # Stacking Ensemble
        stacking_output = self.stacking_ensemble(
            [resnet_probs, efficientnet_probs],
            [resnet_features.flatten(1), efficientnet_features]
        )
        
        # Combine methods based on configuration
        if self.ensemble_method == 'bayesian':
            final_probs = bayesian_output['prediction']
            uncertainty = bayesian_output['uncertainty']
        elif self.ensemble_method == 'adaptive':
            final_probs = adaptive_output['prediction']
            uncertainty = torch.ones_like(final_probs[:, 0:1]) * 0.5
        elif self.ensemble_method == 'stacking':
            final_logits = stacking_output['prediction']
            final_probs = F.softmax(final_logits / self.temperature, dim=1)
            uncertainty = stacking_output['total_uncertainty']
        else:  # Weighted average of all methods
            final_probs = (bayesian_output['prediction'] + 
                          adaptive_output['prediction'] + 
                          F.softmax(stacking_output['prediction'], dim=1)) / 3
            uncertainty = (bayesian_output['uncertainty'] + 
                          stacking_output['total_uncertainty']) / 2
        
        # Convert probabilities to logits
        final_logits = torch.log(final_probs + 1e-8)
        
        return {
            'logits': final_logits,
            'probabilities': final_probs,
            'uncertainty': uncertainty,
            'aleatoric_uncertainty': stacking_output.get('aleatoric_uncertainty'),
            'epistemic_uncertainty': stacking_output.get('epistemic_uncertainty'),
            'resnet_logits': resnet_logits,
            'efficientnet_logits': efficientnet_logits,
            'attention_weights': resnet_output.get('attention_weights'),
            'model_weights': bayesian_output.get('model_weights'),
            'selection_weights': adaptive_output.get('selection_weights'),
            'meta_features': meta_features,
            'diversity_metrics': diversity_metrics,
            'diversity_loss': stacking_output.get('diversity_loss', 0),
            'individual_predictions': {
                'resnet': torch.argmax(resnet_logits, dim=1),
                'efficientnet': torch.argmax(efficientnet_logits, dim=1)
            }
        }
        
    def get_model_contributions(self) -> Dict[str, torch.Tensor]:
        """Get detailed contribution analysis of each model."""
        if hasattr(self, 'bayesian_averaging'):
            weights = self.bayesian_averaging.model_weights
            return {
                'resnet_gru': weights[0].item(),
                'efficientnet': weights[1].item(),
                'temperature': self.temperature.item()
            }
        else:
            return {
                'resnet_gru': 0.5,
                'efficientnet': 0.5,
                'temperature': 1.0
            }


class MetaLearner:
    """Traditional ML meta-learner for ensemble."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.meta_config = config['models']['ensemble']['meta_learner_params']
        self.meta_learner_type = config['models']['ensemble']['meta_learner']
        
        # Initialize meta-learner
        if self.meta_learner_type == 'xgboost':
            self.model = xgb.XGBClassifier(**self.meta_config)
        elif self.meta_learner_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(**self.meta_config)
        elif self.meta_learner_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000)
        elif self.meta_learner_type == 'random_forest':
            self.model = RandomForestClassifier(**self.meta_config)
        else:
            raise ValueError(f"Unknown meta-learner: {self.meta_learner_type}")
            
    def prepare_features(self, predictions: np.ndarray, 
                        features: Optional[np.ndarray] = None) -> np.ndarray:
        """Prepare features for meta-learner."""
        # predictions shape: (n_samples, n_models, n_classes)
        n_samples = predictions.shape[0]
        
        # Flatten predictions
        flat_predictions = predictions.reshape(n_samples, -1)
        
        # Add statistical features
        stats_features = []
        for i in range(n_samples):
            model_preds = predictions[i]
            
            # Agreement between models
            agreement = np.mean(np.argmax(model_preds, axis=1) == 
                              np.argmax(model_preds[0]))
            
            # Entropy of average prediction
            avg_pred = np.mean(model_preds, axis=0)
            entropy = -np.sum(avg_pred * np.log(avg_pred + 1e-8))
            
            # Max confidence
            max_conf = np.max(model_preds)
            
            # Variance across models
            var = np.mean(np.var(model_preds, axis=0))
            
            stats_features.append([agreement, entropy, max_conf, var])
            
        stats_features = np.array(stats_features)
        
        # Combine all features
        if features is not None:
            meta_features = np.concatenate([flat_predictions, stats_features, features], axis=1)
        else:
            meta_features = np.concatenate([flat_predictions, stats_features], axis=1)
            
        return meta_features
        
    def fit(self, predictions: np.ndarray, labels: np.ndarray, 
            features: Optional[np.ndarray] = None):
        """Train the meta-learner."""
        X = self.prepare_features(predictions, features)
        self.model.fit(X, labels)
        
    def predict(self, predictions: np.ndarray, 
                features: Optional[np.ndarray] = None) -> np.ndarray:
        """Make predictions using meta-learner."""
        X = self.prepare_features(predictions, features)
        return self.model.predict(X)
        
    def predict_proba(self, predictions: np.ndarray,
                     features: Optional[np.ndarray] = None) -> np.ndarray:
        """Get prediction probabilities."""
        X = self.prepare_features(predictions, features)
        return self.model.predict_proba(X)
        
    def save(self, path: Path):
        """Save meta-learner model."""
        joblib.dump(self.model, path)
        
    def load(self, path: Path):
        """Load meta-learner model."""
        self.model = joblib.load(path) 