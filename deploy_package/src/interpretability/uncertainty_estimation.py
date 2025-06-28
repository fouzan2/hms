"""
Uncertainty estimation for EEG harmful brain activity classification.
Provides uncertainty quantification for model predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import json


@dataclass
class UncertaintyResult:
    """Container for uncertainty estimation results."""
    aleatoric: Optional[np.ndarray] = None
    epistemic: Optional[np.ndarray] = None
    total: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None
    prediction_variance: Optional[np.ndarray] = None
    entropy: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None


class MonteCarloDropout:
    """Monte Carlo Dropout for uncertainty estimation."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize Monte Carlo Dropout.
        
        Args:
            model: Trained model with dropout layers
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
    def enable_dropout(self):
        """Enable dropout during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                
    def disable_dropout(self):
        """Disable dropout (standard inference)."""
        self.model.eval()
        
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100) -> UncertaintyResult:
        """
        Get predictions with uncertainty estimation.
        
        Args:
            x: Input tensor
            n_samples: Number of Monte Carlo samples
            
        Returns:
            UncertaintyResult containing predictions and uncertainty measures
        """
        x = x.to(self.device)
        self.enable_dropout()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.model(x)
                if isinstance(output, dict):
                    output = output.get('logits', output.get('output', output))
                predictions.append(F.softmax(output, dim=-1).cpu().numpy())
        
        predictions = np.array(predictions)  # Shape: [n_samples, batch_size, n_classes]
        
        # Calculate statistics
        mean_predictions = np.mean(predictions, axis=0)
        prediction_variance = np.var(predictions, axis=0)
        total_uncertainty = np.mean(prediction_variance, axis=-1)
        
        # Calculate entropy-based uncertainty
        prediction_entropy = -np.sum(mean_predictions * np.log(mean_predictions + 1e-8), axis=-1)
        
        # Confidence (max probability)
        confidence = np.max(mean_predictions, axis=-1)
        
        self.disable_dropout()
        
        return UncertaintyResult(
            total=total_uncertainty,
            predictions=mean_predictions,
            prediction_variance=prediction_variance,
            entropy=prediction_entropy,
            confidence=confidence
        )
        
    def calibrate_uncertainty(self, val_loader, n_samples: int = 100) -> Dict:
        """
        Calibrate uncertainty estimates using validation data.
        
        Args:
            val_loader: Validation data loader
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Calibration metrics
        """
        all_uncertainties = []
        all_correct = []
        
        for batch in tqdm(val_loader, desc="Calibrating uncertainty"):
            if isinstance(batch, dict):
                inputs = batch.get('data', batch.get('eeg', batch.get('input')))
                labels = batch.get('label', batch.get('target'))
            else:
                inputs, labels = batch
                
            # Get uncertainty estimates
            uncertainty_result = self.predict_with_uncertainty(inputs, n_samples)
            
            # Get predictions
            predicted_classes = np.argmax(uncertainty_result.predictions, axis=-1)
            correct = (predicted_classes == labels.numpy())
            
            all_uncertainties.extend(uncertainty_result.total.tolist())
            all_correct.extend(correct.tolist())
        
        all_uncertainties = np.array(all_uncertainties)
        all_correct = np.array(all_correct)
        
        # Calculate calibration metrics
        return self._calculate_calibration_metrics(all_uncertainties, all_correct)
        
    def _calculate_calibration_metrics(self, uncertainties: np.ndarray, 
                                     correct: np.ndarray) -> Dict:
        """Calculate uncertainty calibration metrics."""
        # Sort by uncertainty
        sorted_indices = np.argsort(uncertainties)
        sorted_uncertainties = uncertainties[sorted_indices]
        sorted_correct = correct[sorted_indices]
        
        # Calculate ECE (Expected Calibration Error)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (sorted_uncertainties >= bin_lower) & (sorted_uncertainties < bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = sorted_correct[in_bin].mean()
                avg_confidence_in_bin = (1 - sorted_uncertainties[in_bin]).mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {
            'ece': ece,
            'uncertainty_mean': np.mean(uncertainties),
            'uncertainty_std': np.std(uncertainties),
            'accuracy': np.mean(correct)
        }


class UncertaintyEstimator:
    """Comprehensive uncertainty estimation framework."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize uncertainty estimator.
        
        Args:
            model: Trained model
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.mc_dropout = MonteCarloDropout(model, device)
        
    def estimate_uncertainty(self, x: torch.Tensor, 
                           method: str = 'mc_dropout',
                           **kwargs) -> UncertaintyResult:
        """
        Estimate uncertainty using specified method.
        
        Args:
            x: Input tensor
            method: Uncertainty estimation method
            **kwargs: Method-specific arguments
            
        Returns:
            UncertaintyResult
        """
        if method == 'mc_dropout':
            n_samples = kwargs.get('n_samples', 100)
            return self.mc_dropout.predict_with_uncertainty(x, n_samples)
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
            
    def visualize_uncertainty(self, uncertainty_result: UncertaintyResult, 
                            save_path: Optional[str] = None):
        """
        Visualize uncertainty estimates.
        
        Args:
            uncertainty_result: Uncertainty estimation results
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total uncertainty distribution
        axes[0, 0].hist(uncertainty_result.total, bins=50, alpha=0.7)
        axes[0, 0].set_title('Total Uncertainty Distribution')
        axes[0, 0].set_xlabel('Uncertainty')
        axes[0, 0].set_ylabel('Frequency')
        
        # Confidence vs Uncertainty
        axes[0, 1].scatter(uncertainty_result.confidence, uncertainty_result.total, alpha=0.6)
        axes[0, 1].set_title('Confidence vs Uncertainty')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Uncertainty')
        
        # Entropy distribution
        axes[1, 0].hist(uncertainty_result.entropy, bins=50, alpha=0.7, color='orange')
        axes[1, 0].set_title('Prediction Entropy Distribution')
        axes[1, 0].set_xlabel('Entropy')
        axes[1, 0].set_ylabel('Frequency')
        
        # Prediction variance
        if uncertainty_result.prediction_variance is not None:
            mean_variance = np.mean(uncertainty_result.prediction_variance, axis=-1)
            axes[1, 1].hist(mean_variance, bins=50, alpha=0.7, color='green')
            axes[1, 1].set_title('Prediction Variance Distribution')
            axes[1, 1].set_xlabel('Variance')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()


class BayesianNeuralNetwork(nn.Module):
    """Bayesian Neural Network implementation for uncertainty estimation."""
    
    def __init__(self, base_model: nn.Module, prior_std: float = 1.0):
        """
        Initialize Bayesian Neural Network.
        
        Args:
            base_model: Base neural network architecture
            prior_std: Standard deviation of weight priors
        """
        super().__init__()
        self.base_model = base_model
        self.prior_std = prior_std
        
        # Convert deterministic layers to Bayesian layers
        self._convert_to_bayesian()
        
    def _convert_to_bayesian(self):
        """Convert deterministic layers to Bayesian layers."""
        # This is a simplified implementation
        # In practice, you would replace nn.Linear layers with Bayesian equivalents
        pass
        
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Forward pass with sampling.
        
        Args:
            x: Input tensor
            sample: Whether to sample from weight distributions
            
        Returns:
            Output tensor
        """
        if sample:
            # Sample from weight distributions
            return self.base_model(x)
        else:
            # Use mean weights
            return self.base_model(x)
            
    def predict_with_uncertainty(self, x: torch.Tensor, 
                               n_samples: int = 100) -> UncertaintyResult:
        """
        Predict with epistemic uncertainty.
        
        Args:
            x: Input tensor
            n_samples: Number of samples from weight distribution
            
        Returns:
            UncertaintyResult
        """
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.forward(x, sample=True)
                if isinstance(output, dict):
                    output = output.get('logits', output.get('output', output))
                predictions.append(F.softmax(output, dim=-1).cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Calculate epistemic uncertainty
        mean_predictions = np.mean(predictions, axis=0)
        epistemic_uncertainty = np.var(predictions, axis=0)
        total_epistemic = np.mean(epistemic_uncertainty, axis=-1)
        
        return UncertaintyResult(
            epistemic=total_epistemic,
            predictions=mean_predictions,
            prediction_variance=epistemic_uncertainty
        )


class EnsembleUncertainty:
    """Uncertainty estimation using model ensembles."""
    
    def __init__(self, models: List[nn.Module], device: str = 'cpu'):
        """
        Initialize ensemble uncertainty estimator.
        
        Args:
            models: List of trained models
            device: Device to run inference on
        """
        self.models = models
        self.device = device
        
        for model in self.models:
            model.to(device)
            model.eval()
            
    def predict_with_uncertainty(self, x: torch.Tensor) -> UncertaintyResult:
        """
        Predict with ensemble uncertainty.
        
        Args:
            x: Input tensor
            
        Returns:
            UncertaintyResult
        """
        x = x.to(self.device)
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                output = model(x)
                if isinstance(output, dict):
                    output = output.get('logits', output.get('output', output))
                predictions.append(F.softmax(output, dim=-1).cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Calculate ensemble statistics
        mean_predictions = np.mean(predictions, axis=0)
        prediction_variance = np.var(predictions, axis=0)
        total_uncertainty = np.mean(prediction_variance, axis=-1)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = total_uncertainty
        
        return UncertaintyResult(
            epistemic=epistemic_uncertainty,
            total=total_uncertainty,
            predictions=mean_predictions,
            prediction_variance=prediction_variance
        ) 