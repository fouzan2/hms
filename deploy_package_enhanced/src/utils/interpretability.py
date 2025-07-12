"""
Interpretability and explainability utilities for EEG classification models.
Includes gradient-based attribution, SHAP, attention visualization, and uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import shap
from captum.attr import (
    IntegratedGradients,
    GradientShap,
    LayerGradCam,
    LayerAttribution,
    Saliency,
    DeepLift,
    NoiseTunnel
)
# from captum.attr.visualization import visualize_image_attr
import logging

logger = logging.getLogger(__name__)


class ModelInterpreter:
    """Comprehensive model interpretation toolkit."""
    
    def __init__(self, model: nn.Module, config: Dict, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Initialize interpretation methods
        self._init_interpretation_methods()
        
    def _init_interpretation_methods(self):
        """Initialize various interpretation methods."""
        # For ResNet1D-GRU
        if hasattr(self.model, 'resnet_gru'):
            self.ig_resnet = IntegratedGradients(self._forward_resnet)
            self.saliency_resnet = Saliency(self._forward_resnet)
            
        # For EfficientNet
        if hasattr(self.model, 'efficientnet'):
            self.ig_efficientnet = IntegratedGradients(self._forward_efficientnet)
            
            # GradCAM for last conv layer
            if hasattr(self.model.efficientnet.backbone, 'features'):
                last_conv = self.model.efficientnet.backbone.features[-1]
                self.gradcam = LayerGradCam(
                    self._forward_efficientnet,
                    last_conv
                )
                
    def _forward_resnet(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Forward function for ResNet attribution."""
        # Create dummy spectrogram
        batch_size = eeg_data.shape[0]
        dummy_spec = torch.zeros(batch_size, 3, 224, 224).to(self.device)
        
        output = self.model(eeg_data, dummy_spec)
        return output['logits']
        
    def _forward_efficientnet(self, spectrogram_data: torch.Tensor) -> torch.Tensor:
        """Forward function for EfficientNet attribution."""
        # Create dummy EEG
        batch_size = spectrogram_data.shape[0]
        dummy_eeg = torch.zeros(batch_size, 19, 10000).to(self.device)
        
        output = self.model(dummy_eeg, spectrogram_data)
        return output['logits']
        
    def compute_integrated_gradients(self, 
                                   inputs: torch.Tensor,
                                   target_class: int,
                                   model_type: str = 'resnet') -> np.ndarray:
        """Compute integrated gradients attribution."""
        inputs = inputs.to(self.device)
        inputs.requires_grad = True
        
        if model_type == 'resnet':
            attributions = self.ig_resnet.attribute(
                inputs,
                target=target_class,
                n_steps=50
            )
        else:
            attributions = self.ig_efficientnet.attribute(
                inputs,
                target=target_class,
                n_steps=50
            )
            
        return attributions.cpu().numpy()
        
    def compute_gradcam(self,
                       spectrogram: torch.Tensor,
                       target_class: int) -> np.ndarray:
        """Compute GradCAM heatmap for spectrograms."""
        spectrogram = spectrogram.to(self.device)
        spectrogram.requires_grad = True
        
        attributions = self.gradcam.attribute(
            spectrogram,
            target=target_class
        )
        
        # Upsample to input size
        attributions = LayerAttribution.interpolate(
            attributions,
            spectrogram.shape[2:]
        )
        
        return attributions.cpu().numpy()
        
    def compute_attention_weights(self,
                                eeg_data: torch.Tensor,
                                spectrogram_data: torch.Tensor) -> Dict[str, np.ndarray]:
        """Extract attention weights from models."""
        eeg_data = eeg_data.to(self.device)
        spectrogram_data = spectrogram_data.to(self.device)
        
        with torch.no_grad():
            output = self.model(eeg_data, spectrogram_data)
            
        attention_weights = {}
        
        # Get ResNet-GRU attention
        if 'attention_weights' in output:
            attention_weights['temporal_attention'] = output['attention_weights'].cpu().numpy()
            
        # Get ensemble attention if available
        if hasattr(self.model, 'attention_fusion'):
            # Run through attention fusion
            resnet_probs = F.softmax(output['resnet_logits'], dim=1)
            efficientnet_probs = F.softmax(output['efficientnet_logits'], dim=1)
            
            with torch.no_grad():
                _, model_attention = self.model.attention_fusion([resnet_probs, efficientnet_probs])
                
            attention_weights['model_attention'] = model_attention.cpu().numpy()
            
        return attention_weights
        
    def compute_shap_values(self,
                          background_data: torch.Tensor,
                          test_data: torch.Tensor,
                          model_type: str = 'resnet') -> np.ndarray:
        """Compute SHAP values for model interpretation."""
        background_data = background_data.to(self.device)
        test_data = test_data.to(self.device)
        
        # Use GradientShap
        if model_type == 'resnet':
            explainer = GradientShap(self._forward_resnet)
            shap_values = explainer.attribute(
                test_data,
                baselines=background_data,
                n_samples=50
            )
        else:
            explainer = GradientShap(self._forward_efficientnet)
            shap_values = explainer.attribute(
                test_data,
                baselines=background_data,
                n_samples=50
            )
            
        return shap_values.cpu().numpy()


class UncertaintyQuantification:
    """Methods for uncertainty quantification in predictions."""
    
    def __init__(self, model: nn.Module, config: Dict, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
        # Get uncertainty configuration
        self.uncertainty_config = config['interpretability']['uncertainty']
        self.n_samples = self.uncertainty_config['n_samples']
        
    def enable_dropout(self):
        """Enable dropout layers for uncertainty estimation."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                
    def monte_carlo_dropout(self,
                          eeg_data: torch.Tensor,
                          spectrogram_data: torch.Tensor) -> Dict[str, np.ndarray]:
        """Perform Monte Carlo dropout for uncertainty estimation."""
        eeg_data = eeg_data.to(self.device)
        spectrogram_data = spectrogram_data.to(self.device)
        
        # Enable dropout
        self.enable_dropout()
        
        # Collect predictions
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                output = self.model(eeg_data, spectrogram_data)
                probs = F.softmax(output['logits'], dim=1)
                predictions.append(probs.cpu().numpy())
                
        predictions = np.array(predictions)  # (n_samples, batch, n_classes)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = np.mean(std_pred, axis=1)
        
        # Aleatoric uncertainty (data uncertainty)
        entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-8), axis=1)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + entropy
        
        return {
            'mean_prediction': mean_pred,
            'std_prediction': std_pred,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': entropy,
            'total_uncertainty': total_uncertainty,
            'all_predictions': predictions
        }
        
    def ensemble_uncertainty(self,
                           eeg_data: torch.Tensor,
                           spectrogram_data: torch.Tensor) -> Dict[str, np.ndarray]:
        """Calculate uncertainty from ensemble disagreement."""
        eeg_data = eeg_data.to(self.device)
        spectrogram_data = spectrogram_data.to(self.device)
        
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(eeg_data, spectrogram_data)
            
        # Get individual model predictions
        resnet_probs = F.softmax(output['resnet_logits'], dim=1).cpu().numpy()
        efficientnet_probs = F.softmax(output['efficientnet_logits'], dim=1).cpu().numpy()
        ensemble_probs = F.softmax(output['logits'], dim=1).cpu().numpy()
        
        # Calculate disagreement
        predictions = np.stack([resnet_probs, efficientnet_probs], axis=0)
        
        # Jensen-Shannon divergence between models
        mean_probs = np.mean(predictions, axis=0)
        js_divergence = np.zeros(len(mean_probs))
        
        for i in range(len(predictions)):
            kl_div = np.sum(predictions[i] * np.log(predictions[i] / mean_probs + 1e-8), axis=1)
            js_divergence += kl_div
            
        js_divergence /= len(predictions)
        
        # Prediction variance
        pred_variance = np.var(predictions, axis=0).mean(axis=1)
        
        return {
            'ensemble_prediction': ensemble_probs,
            'model_disagreement': js_divergence,
            'prediction_variance': pred_variance,
            'individual_predictions': {
                'resnet': resnet_probs,
                'efficientnet': efficientnet_probs
            }
        }


class VisualizationTools:
    """Tools for visualizing model interpretations."""
    
    @staticmethod
    def plot_eeg_attribution(eeg_signal: np.ndarray,
                           attribution: np.ndarray,
                           channel_names: List[str],
                           save_path: Optional[Path] = None):
        """Plot EEG signal with attribution overlay."""
        n_channels = len(channel_names)
        fig, axes = plt.subplots(n_channels, 1, figsize=(15, 2*n_channels))
        
        if n_channels == 1:
            axes = [axes]
            
        time = np.arange(eeg_signal.shape[1]) / 200  # 200 Hz sampling
        
        for i, (ax, ch_name) in enumerate(zip(axes, channel_names)):
            # Plot signal
            ax.plot(time, eeg_signal[i], 'b-', alpha=0.7, label='Signal')
            
            # Plot attribution
            ax2 = ax.twinx()
            ax2.fill_between(time, 0, attribution[i], 
                           color='red', alpha=0.3, label='Attribution')
            
            ax.set_ylabel(f'{ch_name}\nAmplitude')
            ax2.set_ylabel('Attribution')
            ax.set_xlabel('Time (s)')
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(loc='upper left')
                ax2.legend(loc='upper right')
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    @staticmethod
    def plot_spectrogram_attribution(spectrogram: np.ndarray,
                                   attribution: np.ndarray,
                                   save_path: Optional[Path] = None):
        """Plot spectrogram with attribution overlay."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original spectrogram
        im1 = axes[0].imshow(spectrogram[0], aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('Original Spectrogram')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Frequency')
        plt.colorbar(im1, ax=axes[0])
        
        # Attribution heatmap
        im2 = axes[1].imshow(attribution[0], aspect='auto', origin='lower', cmap='hot')
        axes[1].set_title('Attribution Heatmap')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Frequency')
        plt.colorbar(im2, ax=axes[1])
        
        # Overlay
        axes[2].imshow(spectrogram[0], aspect='auto', origin='lower', cmap='viridis', alpha=0.7)
        im3 = axes[2].imshow(attribution[0], aspect='auto', origin='lower', cmap='hot', alpha=0.5)
        axes[2].set_title('Attribution Overlay')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    @staticmethod
    def plot_attention_weights(attention_weights: Dict[str, np.ndarray],
                             save_path: Optional[Path] = None):
        """Plot various attention weights."""
        n_plots = len(attention_weights)
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 4))
        
        if n_plots == 1:
            axes = [axes]
            
        for ax, (name, weights) in zip(axes, attention_weights.items()):
            if weights.ndim == 1:
                ax.plot(weights)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Attention Weight')
            else:
                im = ax.imshow(weights, aspect='auto', cmap='Blues')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Sample')
                plt.colorbar(im, ax=ax)
                
            ax.set_title(f'{name.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    @staticmethod
    def plot_uncertainty_distribution(uncertainty_results: Dict[str, np.ndarray],
                                    save_path: Optional[Path] = None):
        """Plot uncertainty distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        # Prediction distribution
        mean_pred = uncertainty_results['mean_prediction'][0]
        std_pred = uncertainty_results['std_prediction'][0]
        classes = np.arange(len(mean_pred))
        
        axes[0].bar(classes, mean_pred, yerr=std_pred, capsize=5)
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Probability')
        axes[0].set_title('Prediction with Uncertainty')
        axes[0].set_xticks(classes)
        
        # Uncertainty types
        uncertainties = {
            'Epistemic': uncertainty_results['epistemic_uncertainty'],
            'Aleatoric': uncertainty_results['aleatoric_uncertainty'],
            'Total': uncertainty_results['total_uncertainty']
        }
        
        for i, (name, values) in enumerate(uncertainties.items(), 1):
            axes[i].hist(values, bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_xlabel(f'{name} Uncertainty')
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'{name} Uncertainty Distribution')
            axes[i].axvline(np.mean(values), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(values):.3f}')
            axes[i].legend()
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show() 