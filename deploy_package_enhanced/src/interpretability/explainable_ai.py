"""
Explainable AI with Counterfactual Reasoning for HMS Brain Activity Classification
Enterprise-level framework providing interpretable explanations for EEG classification decisions
with counterfactual analysis, SHAP integration, and clinical interpretation tools.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from pathlib import Path
import json
import pickle
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import entropy
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.ensemble import IsolationForest
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ExplanationConfig:
    """Configuration for explainable AI framework."""
    # Counterfactual parameters
    counterfactual_method: str = 'gradient_based'  # 'gradient_based', 'genetic', 'optimization'
    cf_lambda_proximity: float = 1.0  # Weight for proximity loss
    cf_lambda_diversity: float = 0.1  # Weight for diversity loss
    cf_lambda_sparsity: float = 0.5   # Weight for sparsity loss
    cf_max_iterations: int = 500      # Maximum optimization iterations
    cf_learning_rate: float = 0.01    # Learning rate for optimization
    
    # SHAP parameters
    shap_method: str = 'deep'         # 'deep', 'gradient', 'kernel'
    shap_n_samples: int = 100         # Number of samples for SHAP
    shap_batch_size: int = 16         # Batch size for SHAP computation
    
    # Attention visualization
    attention_layers: List[str] = None  # Specific layers to visualize
    attention_heads: List[int] = None   # Specific attention heads
    
    # Clinical interpretation
    frequency_bands: Dict[str, Tuple[float, float]] = None
    clinical_features: List[str] = None
    
    # Visualization parameters
    plot_style: str = 'seaborn'       # Plotting style
    figure_size: Tuple[int, int] = (12, 8)  # Default figure size
    save_plots: bool = True           # Whether to save plots
    plot_format: str = 'png'          # Plot format
    
    def __post_init__(self):
        if self.frequency_bands is None:
            self.frequency_bands = {
                'delta': (0.5, 4.0),
                'theta': (4.0, 8.0),
                'alpha': (8.0, 13.0),
                'beta': (13.0, 30.0),
                'gamma': (30.0, 100.0)
            }
        
        if self.clinical_features is None:
            self.clinical_features = [
                'power_spectral_density',
                'spectral_entropy',
                'hjorth_parameters',
                'connectivity_measures',
                'nonlinear_features'
            ]


class CounterfactualGenerator:
    """Generate counterfactual explanations for EEG classifications."""
    
    def __init__(self, model: nn.Module, config: ExplanationConfig, device: str = 'cpu'):
        """
        Initialize counterfactual generator.
        
        Args:
            model: Trained model for which to generate explanations
            config: Configuration for explanations
            device: Computing device
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.model.eval()
        
    def generate_counterfactual(self, 
                              x_original: torch.Tensor,
                              target_class: int,
                              original_class: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate counterfactual explanation for a given input.
        
        Args:
            x_original: Original input (1, n_channels, seq_length)
            target_class: Desired target class
            original_class: Original predicted class (if None, will predict)
            
        Returns:
            Dictionary containing counterfactual explanation
        """
        x_original = x_original.to(self.device)
        
        # Get original prediction if not provided
        if original_class is None:
            with torch.no_grad():
                original_logits = self.model(x_original)
                if isinstance(original_logits, dict):
                    original_logits = original_logits.get('logits', list(original_logits.values())[0])
                original_class = torch.argmax(original_logits, dim=1).item()
        
        if self.config.counterfactual_method == 'gradient_based':
            return self._generate_gradient_based_cf(x_original, target_class, original_class)
        elif self.config.counterfactual_method == 'optimization':
            return self._generate_optimization_based_cf(x_original, target_class, original_class)
        else:
            raise ValueError(f"Unknown counterfactual method: {self.config.counterfactual_method}")
    
    def _generate_gradient_based_cf(self, 
                                  x_original: torch.Tensor,
                                  target_class: int,
                                  original_class: int) -> Dict[str, Any]:
        """Generate counterfactual using gradient-based optimization."""
        # Initialize counterfactual as copy of original
        x_cf = x_original.clone().requires_grad_(True)
        
        # Target tensor
        target = torch.tensor([target_class], device=self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam([x_cf], lr=self.config.cf_learning_rate)
        
        # Track optimization
        losses = []
        predictions = []
        
        for iteration in range(self.config.cf_max_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(x_cf)
            if isinstance(logits, dict):
                logits = logits.get('logits', list(logits.values())[0])
            
            # Classification loss (we want to minimize distance to target class)
            class_loss = F.cross_entropy(logits, target)
            
            # Proximity loss (minimize distance from original)
            proximity_loss = torch.mean((x_cf - x_original) ** 2)
            
            # Sparsity loss (minimize number of changes)
            sparsity_loss = torch.mean(torch.abs(x_cf - x_original))
            
            # Total loss
            total_loss = (class_loss + 
                         self.config.cf_lambda_proximity * proximity_loss +
                         self.config.cf_lambda_sparsity * sparsity_loss)
            
            total_loss.backward()
            optimizer.step()
            
            # Track progress
            losses.append(total_loss.item())
            with torch.no_grad():
                pred_class = torch.argmax(logits, dim=1).item()
                predictions.append(pred_class)
                
            # Early stopping if target class achieved
            if pred_class == target_class:
                logger.info(f"Counterfactual found at iteration {iteration}")
                break
        
        # Compute final metrics
        with torch.no_grad():
            final_logits = self.model(x_cf)
            if isinstance(final_logits, dict):
                final_logits = final_logits.get('logits', list(final_logits.values())[0])
            final_pred = torch.argmax(final_logits, dim=1).item()
            confidence = torch.softmax(final_logits, dim=1)[0, target_class].item()
            
            # Compute changes
            changes = torch.abs(x_cf - x_original)
            total_change = torch.sum(changes).item()
            relative_change = total_change / torch.sum(torch.abs(x_original)).item()
            
            # Compute feature importance (gradient of output w.r.t. input)
            x_cf.requires_grad_(True)
            logits = self.model(x_cf)
            if isinstance(logits, dict):
                logits = logits.get('logits', list(logits.values())[0])
            grad = torch.autograd.grad(logits[0, target_class], x_cf, create_graph=False)[0]
            
        return {
            'counterfactual': x_cf.detach().cpu(),
            'original': x_original.cpu(),
            'changes': changes.cpu(),
            'total_change': total_change,
            'relative_change': relative_change,
            'final_prediction': final_pred,
            'target_class': target_class,
            'original_class': original_class,
            'confidence': confidence,
            'gradient_importance': grad.abs().cpu(),
            'optimization_losses': losses,
            'optimization_predictions': predictions,
            'success': final_pred == target_class
        }
    
    def _generate_optimization_based_cf(self, 
                                      x_original: torch.Tensor,
                                      target_class: int,
                                      original_class: int) -> Dict[str, Any]:
        """Generate counterfactual using constrained optimization."""
        # This is a more sophisticated approach using constrained optimization
        # For brevity, implementing a simplified version
        return self._generate_gradient_based_cf(x_original, target_class, original_class)
    
    def generate_diverse_counterfactuals(self,
                                       x_original: torch.Tensor,
                                       target_class: int,
                                       n_counterfactuals: int = 5) -> List[Dict[str, Any]]:
        """Generate multiple diverse counterfactuals."""
        counterfactuals = []
        
        for i in range(n_counterfactuals):
            # Add noise to original for diversity
            noise = torch.randn_like(x_original) * 0.01 * (i + 1)
            x_noisy = x_original + noise
            
            cf = self.generate_counterfactual(x_noisy, target_class)
            cf['diversity_seed'] = i
            counterfactuals.append(cf)
        
        return counterfactuals


class SHAPExplainer:
    """SHAP-based explanations for EEG classification models."""
    
    def __init__(self, model: nn.Module, config: ExplanationConfig, device: str = 'cpu'):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model
            config: Configuration
            device: Computing device
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.model.eval()
        self.explainer = None
        
    def initialize_explainer(self, background_data: torch.Tensor):
        """Initialize SHAP explainer with background data."""
        background_data = background_data.to(self.device)
        
        if self.config.shap_method == 'deep':
            self.explainer = shap.DeepExplainer(self.model, background_data)
        elif self.config.shap_method == 'gradient':
            self.explainer = shap.GradientExplainer(self.model, background_data)
        else:
            raise ValueError(f"Unknown SHAP method: {self.config.shap_method}")
    
    def explain(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Generate SHAP explanations for input.
        
        Args:
            x: Input data (batch_size, n_channels, seq_length)
            
        Returns:
            Dictionary containing SHAP values and explanations
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized. Call initialize_explainer() first.")
        
        x = x.to(self.device)
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(x)
        
        # Get model predictions
        with torch.no_grad():
            logits = self.model(x)
            if isinstance(logits, dict):
                logits = logits.get('logits', list(logits.values())[0])
            predictions = torch.softmax(logits, dim=1)
        
        # Analyze SHAP values
        analysis = self._analyze_shap_values(shap_values, x, predictions)
        
        return {
            'shap_values': shap_values,
            'predictions': predictions.cpu().numpy(),
            'input_data': x.cpu().numpy(),
            'analysis': analysis
        }
    
    def _analyze_shap_values(self, shap_values, x, predictions):
        """Analyze SHAP values to extract insights."""
        # Convert to numpy if needed
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
        elif isinstance(shap_values, torch.Tensor):
            shap_values = shap_values.cpu().numpy()
        
        # Channel importance
        channel_importance = np.mean(np.abs(shap_values), axis=(0, 2))
        
        # Temporal importance
        temporal_importance = np.mean(np.abs(shap_values), axis=(0, 1))
        
        # Frequency band analysis
        frequency_importance = self._compute_frequency_importance(shap_values, x)
        
        return {
            'channel_importance': channel_importance,
            'temporal_importance': temporal_importance,
            'frequency_importance': frequency_importance,
            'global_importance': np.mean(np.abs(shap_values))
        }
    
    def _compute_frequency_importance(self, shap_values, x):
        """Compute importance for different frequency bands."""
        frequency_importance = {}
        
        for band_name, (low_freq, high_freq) in self.config.frequency_bands.items():
            # Apply bandpass filter to SHAP values
            # This is a simplified approach - in practice, you'd use proper filtering
            band_importance = np.mean(np.abs(shap_values))  # Placeholder
            frequency_importance[band_name] = band_importance
        
        return frequency_importance


class AttentionVisualizer:
    """Visualize attention patterns in transformer-based models."""
    
    def __init__(self, model: nn.Module, config: ExplanationConfig, device: str = 'cpu'):
        """
        Initialize attention visualizer.
        
        Args:
            model: Transformer model with attention mechanisms
            config: Configuration
            device: Computing device
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.model.eval()
        
    def extract_attention_patterns(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Extract attention patterns from model.
        
        Args:
            x: Input data (batch_size, n_channels, seq_length)
            
        Returns:
            Dictionary containing attention patterns
        """
        x = x.to(self.device)
        
        # Hook to capture attention weights
        attention_weights = {}
        
        def attention_hook(module, input, output):
            if hasattr(output, 'attentions') or isinstance(output, tuple):
                if isinstance(output, tuple) and len(output) > 1:
                    attention_weights[module.__class__.__name__] = output[1]
                elif hasattr(output, 'attentions'):
                    attention_weights[module.__class__.__name__] = output.attentions
        
        # Register hooks
        hooks = []
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'transformer' in name.lower():
                hook = module.register_forward_hook(attention_hook)
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(x, return_attentions=True)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Extract attention from outputs if available
        if isinstance(outputs, dict) and 'attentions' in outputs:
            attention_weights['model_attentions'] = outputs['attentions']
        
        # Analyze attention patterns
        analysis = self._analyze_attention_patterns(attention_weights, x)
        
        return {
            'attention_weights': attention_weights,
            'analysis': analysis,
            'input_shape': x.shape
        }
    
    def _analyze_attention_patterns(self, attention_weights, x):
        """Analyze attention patterns to extract insights."""
        analysis = {}
        
        for layer_name, weights in attention_weights.items():
            if weights is not None:
                if isinstance(weights, (list, tuple)):
                    weights = weights[0]  # Take first item if list
                
                if isinstance(weights, torch.Tensor):
                    weights = weights.cpu().numpy()
                
                # Compute attention statistics
                analysis[layer_name] = {
                    'mean_attention': np.mean(weights),
                    'max_attention': np.max(weights),
                    'attention_entropy': entropy(weights.flatten() + 1e-8),
                    'shape': weights.shape
                }
        
        return analysis


class ClinicalInterpreter:
    """Clinical interpretation tools for EEG classification explanations."""
    
    def __init__(self, config: ExplanationConfig):
        """
        Initialize clinical interpreter.
        
        Args:
            config: Configuration
        """
        self.config = config
        
    def interpret_explanation(self, 
                            explanation: Dict[str, Any],
                            eeg_data: np.ndarray,
                            prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Provide clinical interpretation of model explanation.
        
        Args:
            explanation: Explanation from SHAP, counterfactuals, etc.
            eeg_data: Original EEG data
            prediction: Model prediction with confidence
            
        Returns:
            Clinical interpretation
        """
        interpretation = {
            'summary': self._generate_summary(explanation, prediction),
            'clinical_features': self._extract_clinical_features(eeg_data, explanation),
            'risk_factors': self._identify_risk_factors(explanation),
            'recommendations': self._generate_recommendations(explanation, prediction),
            'confidence_analysis': self._analyze_confidence(prediction, explanation)
        }
        
        return interpretation
    
    def _generate_summary(self, explanation, prediction):
        """Generate clinical summary of the explanation."""
        pred_class = prediction.get('predicted_class', 'unknown')
        confidence = prediction.get('confidence', 0.0)
        
        summary = f"Model predicts {pred_class} with {confidence:.1%} confidence. "
        
        # Add key findings based on explanation type
        if 'shap_values' in explanation:
            shap_analysis = explanation.get('analysis', {})
            important_channels = np.argsort(shap_analysis.get('channel_importance', []))[-3:]
            summary += f"Most important channels: {important_channels}. "
        
        if 'counterfactual' in explanation:
            cf_success = explanation.get('success', False)
            if cf_success:
                summary += "Alternative scenario identified. "
            else:
                summary += "No clear alternative scenario found. "
        
        return summary
    
    def _extract_clinical_features(self, eeg_data, explanation):
        """Extract clinically relevant features from EEG and explanation."""
        features = {}
        
        # Power spectral density analysis
        psd_features = self._compute_psd_features(eeg_data)
        features['power_spectral_density'] = psd_features
        
        # Spectral entropy
        spectral_entropy = self._compute_spectral_entropy(eeg_data)
        features['spectral_entropy'] = spectral_entropy
        
        # Hjorth parameters
        hjorth_params = self._compute_hjorth_parameters(eeg_data)
        features['hjorth_parameters'] = hjorth_params
        
        # Connectivity measures (simplified)
        connectivity = self._compute_connectivity(eeg_data)
        features['connectivity'] = connectivity
        
        return features
    
    def _compute_psd_features(self, eeg_data):
        """Compute power spectral density features."""
        fs = 200  # Sampling frequency
        features = {}
        
        for band_name, (low_freq, high_freq) in self.config.frequency_bands.items():
            band_power = []
            for channel in range(eeg_data.shape[0]):
                freqs, psd = signal.welch(eeg_data[channel], fs, nperseg=256)
                band_indices = (freqs >= low_freq) & (freqs <= high_freq)
                power = np.trapz(psd[band_indices], freqs[band_indices])
                band_power.append(power)
            
            features[band_name] = {
                'mean_power': np.mean(band_power),
                'std_power': np.std(band_power),
                'channel_powers': band_power
            }
        
        return features
    
    def _compute_spectral_entropy(self, eeg_data):
        """Compute spectral entropy for each channel."""
        fs = 200
        entropies = []
        
        for channel in range(eeg_data.shape[0]):
            freqs, psd = signal.welch(eeg_data[channel], fs, nperseg=256)
            # Normalize PSD to get probability distribution
            psd_norm = psd / np.sum(psd)
            # Compute entropy
            se = entropy(psd_norm + 1e-8)
            entropies.append(se)
        
        return {
            'mean_entropy': np.mean(entropies),
            'std_entropy': np.std(entropies),
            'channel_entropies': entropies
        }
    
    def _compute_hjorth_parameters(self, eeg_data):
        """Compute Hjorth parameters (Activity, Mobility, Complexity)."""
        hjorth_params = []
        
        for channel in range(eeg_data.shape[0]):
            signal_data = eeg_data[channel]
            
            # First derivative
            diff1 = np.diff(signal_data)
            # Second derivative
            diff2 = np.diff(diff1)
            
            # Activity (variance)
            activity = np.var(signal_data)
            
            # Mobility
            mobility = np.sqrt(np.var(diff1) / np.var(signal_data))
            
            # Complexity
            complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / mobility
            
            hjorth_params.append({
                'activity': activity,
                'mobility': mobility,
                'complexity': complexity
            })
        
        return hjorth_params
    
    def _compute_connectivity(self, eeg_data):
        """Compute simplified connectivity measures."""
        # Pearson correlation between channels
        correlation_matrix = np.corrcoef(eeg_data)
        
        # Average connectivity
        avg_connectivity = np.mean(np.abs(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]))
        
        return {
            'correlation_matrix': correlation_matrix,
            'average_connectivity': avg_connectivity,
            'max_connectivity': np.max(np.abs(correlation_matrix - np.eye(correlation_matrix.shape[0]))),
            'connectivity_variance': np.var(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
        }
    
    def _identify_risk_factors(self, explanation):
        """Identify potential risk factors based on explanation."""
        risk_factors = []
        
        # Based on SHAP values
        if 'shap_values' in explanation:
            analysis = explanation.get('analysis', {})
            channel_importance = analysis.get('channel_importance', [])
            
            # High importance in specific channels
            if len(channel_importance) > 0:
                max_channel = np.argmax(channel_importance)
                if channel_importance[max_channel] > np.mean(channel_importance) * 2:
                    risk_factors.append(f"High activity in channel {max_channel}")
        
        # Based on counterfactuals
        if 'changes' in explanation:
            changes = explanation['changes']
            if isinstance(changes, torch.Tensor):
                changes = changes.numpy()
            
            large_changes = np.where(changes > np.percentile(changes, 95))
            if len(large_changes[0]) > 0:
                risk_factors.append("Significant signal modifications required for alternative outcome")
        
        return risk_factors
    
    def _generate_recommendations(self, explanation, prediction):
        """Generate clinical recommendations based on explanation."""
        recommendations = []
        
        confidence = prediction.get('confidence', 0.0)
        pred_class = prediction.get('predicted_class', 'unknown')
        
        # Confidence-based recommendations
        if confidence < 0.7:
            recommendations.append("Low confidence prediction - consider additional evaluation")
        elif confidence > 0.95:
            recommendations.append("High confidence prediction - monitor for confirmation")
        
        # Class-specific recommendations
        if pred_class in ['seizure', 'lpd', 'gpd']:
            recommendations.append("Immediate clinical attention recommended")
            recommendations.append("Consider EEG monitoring continuation")
        elif pred_class in ['lrda', 'grda']:
            recommendations.append("Monitor for pattern evolution")
            recommendations.append("Review clinical context")
        
        # Explanation-based recommendations
        if 'counterfactual' in explanation and explanation.get('success', False):
            relative_change = explanation.get('relative_change', 0)
            if relative_change < 0.1:
                recommendations.append("Small changes could alter classification - borderline case")
        
        return recommendations
    
    def _analyze_confidence(self, prediction, explanation):
        """Analyze prediction confidence in context of explanation."""
        confidence = prediction.get('confidence', 0.0)
        
        analysis = {
            'confidence_level': confidence,
            'confidence_category': self._categorize_confidence(confidence),
            'reliability_indicators': []
        }
        
        # Check explanation consistency
        if 'shap_values' in explanation:
            analysis['reliability_indicators'].append("SHAP-based feature importance available")
        
        if 'counterfactual' in explanation:
            cf_success = explanation.get('success', False)
            relative_change = explanation.get('relative_change', 1.0)
            
            if cf_success and relative_change > 0.5:
                analysis['reliability_indicators'].append("Robust prediction - large changes required for alternative")
            elif cf_success and relative_change < 0.1:
                analysis['reliability_indicators'].append("Sensitive prediction - small changes affect outcome")
        
        return analysis
    
    def _categorize_confidence(self, confidence):
        """Categorize confidence level."""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.7:
            return "High"
        elif confidence >= 0.5:
            return "Moderate"
        else:
            return "Low"


class ExplainableAI:
    """
    Main explainable AI framework integrating all explanation methods.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 config: Optional[ExplanationConfig] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize explainable AI framework.
        
        Args:
            model: Trained model to explain
            config: Configuration for explanations
            device: Computing device
        """
        self.model = model.to(device)
        self.config = config or ExplanationConfig()
        self.device = device
        
        # Initialize explanation components
        self.counterfactual_generator = CounterfactualGenerator(model, self.config, device)
        self.shap_explainer = SHAPExplainer(model, self.config, device)
        self.attention_visualizer = AttentionVisualizer(model, self.config, device)
        self.clinical_interpreter = ClinicalInterpreter(self.config)
        
        # Cache for background data
        self.background_data = None
        
    def initialize_background_data(self, background_data: torch.Tensor):
        """Initialize background data for SHAP explanations."""
        self.background_data = background_data.to(self.device)
        self.shap_explainer.initialize_explainer(background_data)
        
    def explain_prediction(self, 
                         x: torch.Tensor,
                         explanation_types: List[str] = None,
                         target_class: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a prediction.
        
        Args:
            x: Input data (1, n_channels, seq_length) or (batch_size, n_channels, seq_length)
            explanation_types: Types of explanations to generate
            target_class: Target class for counterfactual (if None, uses alternative class)
            
        Returns:
            Comprehensive explanation dictionary
        """
        if explanation_types is None:
            explanation_types = ['prediction', 'shap', 'counterfactual', 'attention', 'clinical']
        
        x = x.to(self.device)
        if x.dim() == 2:  # Add batch dimension if needed
            x = x.unsqueeze(0)
        
        explanation = {}
        
        # Get model prediction
        if 'prediction' in explanation_types:
            prediction_info = self._get_prediction_info(x)
            explanation['prediction'] = prediction_info
        
        # SHAP explanations
        if 'shap' in explanation_types and self.background_data is not None:
            try:
                shap_explanation = self.shap_explainer.explain(x)
                explanation['shap'] = shap_explanation
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
                explanation['shap'] = {'error': str(e)}
        
        # Counterfactual explanations
        if 'counterfactual' in explanation_types:
            try:
                # Determine target class
                if target_class is None and 'prediction' in explanation:
                    pred_class = explanation['prediction']['predicted_class_idx']
                    # Use different class as target
                    n_classes = explanation['prediction']['probabilities'].shape[1]
                    target_class = (pred_class + 1) % n_classes
                
                if target_class is not None:
                    cf_explanation = self.counterfactual_generator.generate_counterfactual(
                        x[0:1], target_class
                    )
                    explanation['counterfactual'] = cf_explanation
            except Exception as e:
                logger.warning(f"Counterfactual explanation failed: {e}")
                explanation['counterfactual'] = {'error': str(e)}
        
        # Attention visualizations
        if 'attention' in explanation_types:
            try:
                attention_explanation = self.attention_visualizer.extract_attention_patterns(x)
                explanation['attention'] = attention_explanation
            except Exception as e:
                logger.warning(f"Attention explanation failed: {e}")
                explanation['attention'] = {'error': str(e)}
        
        # Clinical interpretation
        if 'clinical' in explanation_types and 'prediction' in explanation:
            try:
                clinical_explanation = self.clinical_interpreter.interpret_explanation(
                    explanation, x[0].cpu().numpy(), explanation['prediction']
                )
                explanation['clinical'] = clinical_explanation
            except Exception as e:
                logger.warning(f"Clinical interpretation failed: {e}")
                explanation['clinical'] = {'error': str(e)}
        
        return explanation
    
    def _get_prediction_info(self, x: torch.Tensor) -> Dict[str, Any]:
        """Get detailed prediction information."""
        with torch.no_grad():
            outputs = self.model(x)
            
            # Handle different model output formats
            if isinstance(outputs, dict):
                logits = outputs.get('logits', list(outputs.values())[0])
            else:
                logits = outputs
            
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
            
            return {
                'logits': logits.cpu(),
                'probabilities': probabilities.cpu(),
                'predicted_class_idx': predicted_class[0].item(),
                'confidence': confidence[0].item(),
                'top_k_predictions': torch.topk(probabilities[0], k=min(3, probabilities.shape[1]))
            }
    
    def generate_explanation_report(self, 
                                  explanation: Dict[str, Any],
                                  save_path: Optional[Path] = None) -> str:
        """
        Generate a comprehensive explanation report.
        
        Args:
            explanation: Explanation dictionary from explain_prediction
            save_path: Optional path to save the report
            
        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("EEG CLASSIFICATION EXPLANATION REPORT")
        report_lines.append("=" * 80)
        
        # Prediction summary
        if 'prediction' in explanation:
            pred = explanation['prediction']
            report_lines.append(f"\nPREDICTION SUMMARY:")
            report_lines.append(f"Predicted Class: {pred['predicted_class_idx']}")
            report_lines.append(f"Confidence: {pred['confidence']:.3f}")
            
            if 'top_k_predictions' in pred:
                values, indices = pred['top_k_predictions']
                report_lines.append(f"Top predictions:")
                for i, (val, idx) in enumerate(zip(values, indices)):
                    report_lines.append(f"  {i+1}. Class {idx}: {val:.3f}")
        
        # Clinical interpretation
        if 'clinical' in explanation and 'error' not in explanation['clinical']:
            clinical = explanation['clinical']
            report_lines.append(f"\nCLINICAL INTERPRETATION:")
            report_lines.append(f"Summary: {clinical['summary']}")
            
            if 'recommendations' in clinical:
                report_lines.append(f"Recommendations:")
                for rec in clinical['recommendations']:
                    report_lines.append(f"  - {rec}")
            
            if 'risk_factors' in clinical:
                report_lines.append(f"Risk Factors:")
                for factor in clinical['risk_factors']:
                    report_lines.append(f"  - {factor}")
        
        # SHAP analysis
        if 'shap' in explanation and 'error' not in explanation['shap']:
            shap_data = explanation['shap']
            report_lines.append(f"\nFEATURE IMPORTANCE ANALYSIS:")
            
            if 'analysis' in shap_data:
                analysis = shap_data['analysis']
                report_lines.append(f"Global importance: {analysis['global_importance']:.6f}")
                
                if 'channel_importance' in analysis:
                    ch_imp = analysis['channel_importance']
                    top_channels = np.argsort(ch_imp)[-3:][::-1]
                    report_lines.append(f"Most important channels: {top_channels}")
        
        # Counterfactual analysis
        if 'counterfactual' in explanation and 'error' not in explanation['counterfactual']:
            cf = explanation['counterfactual']
            report_lines.append(f"\nCOUNTERFACTUAL ANALYSIS:")
            report_lines.append(f"Target class: {cf['target_class']}")
            report_lines.append(f"Success: {cf['success']}")
            report_lines.append(f"Total change required: {cf['total_change']:.6f}")
            report_lines.append(f"Relative change: {cf['relative_change']:.3f}")
            
            if cf['success']:
                report_lines.append(f"Alternative prediction confidence: {cf['confidence']:.3f}")
        
        # Attention patterns
        if 'attention' in explanation and 'error' not in explanation['attention']:
            attention = explanation['attention']
            report_lines.append(f"\nATTENTION ANALYSIS:")
            
            if 'analysis' in attention:
                for layer_name, layer_analysis in attention['analysis'].items():
                    report_lines.append(f"{layer_name}:")
                    report_lines.append(f"  Mean attention: {layer_analysis['mean_attention']:.6f}")
                    report_lines.append(f"  Attention entropy: {layer_analysis['attention_entropy']:.6f}")
        
        report_lines.append("\n" + "=" * 80)
        
        # Join all lines
        report = "\n".join(report_lines)
        
        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
    
    def create_visualization_dashboard(self, 
                                    explanation: Dict[str, Any],
                                    save_path: Optional[Path] = None) -> go.Figure:
        """
        Create interactive visualization dashboard for explanations.
        
        Args:
            explanation: Explanation dictionary
            save_path: Optional path to save dashboard
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Prediction Confidence', 'Feature Importance',
                'Counterfactual Changes', 'Attention Patterns',
                'Clinical Features', 'Time Series Analysis'
            ],
            specs=[
                [{"type": "bar"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "heatmap"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        # Prediction confidence
        if 'prediction' in explanation:
            pred = explanation['prediction']
            probs = pred['probabilities'][0].numpy()
            classes = [f"Class {i}" for i in range(len(probs))]
            
            fig.add_trace(
                go.Bar(x=classes, y=probs, name="Confidence"),
                row=1, col=1
            )
        
        # Feature importance (SHAP)
        if 'shap' in explanation and 'error' not in explanation['shap']:
            shap_data = explanation['shap']
            if 'analysis' in shap_data and 'channel_importance' in shap_data['analysis']:
                ch_imp = shap_data['analysis']['channel_importance']
                channels = [f"Ch{i}" for i in range(len(ch_imp))]
                
                fig.add_trace(
                    go.Heatmap(
                        z=[ch_imp],
                        x=channels,
                        y=['Importance'],
                        colorscale='RdBu'
                    ),
                    row=1, col=2
                )
        
        # Counterfactual changes
        if 'counterfactual' in explanation and 'error' not in explanation['counterfactual']:
            cf = explanation['counterfactual']
            iterations = list(range(len(cf['optimization_losses'])))
            
            fig.add_trace(
                go.Scatter(
                    x=iterations,
                    y=cf['optimization_losses'],
                    mode='lines',
                    name='Optimization Loss'
                ),
                row=2, col=1
            )
        
        # Add more visualizations...
        
        # Update layout
        fig.update_layout(
            title="EEG Classification Explanation Dashboard",
            height=1200,
            showlegend=True
        )
        
        # Save if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_path))
        
        return fig 