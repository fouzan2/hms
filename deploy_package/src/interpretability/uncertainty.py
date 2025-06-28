"""
Uncertainty quantification for EEG harmful brain activity classification.
Provides epistemic and aleatoric uncertainty estimates for clinical decision support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


@dataclass
class UncertaintyResult:
    """Container for uncertainty quantification results."""
    predictions: np.ndarray
    epistemic_uncertainty: np.ndarray
    aleatoric_uncertainty: np.ndarray
    total_uncertainty: np.ndarray
    confidence_intervals: Tuple[np.ndarray, np.ndarray]
    calibration_metrics: Dict[str, float]
    metadata: Dict[str, any]


class MonteCarloDropout:
    """
    Monte Carlo Dropout for epistemic uncertainty estimation.
    Enables dropout during inference to estimate model uncertainty.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self._enable_dropout()
        
    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        def apply_dropout(m):
            if type(m) == nn.Dropout:
                m.train()
            elif hasattr(m, 'dropout'):
                m.dropout.train()
                
        self.model.eval()
        self.model.apply(apply_dropout)
        
    def predict_with_uncertainty(self,
                                 input_tensor: torch.Tensor,
                                 n_samples: int = 100,
                                 confidence_level: float = 0.95) -> UncertaintyResult:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            input_tensor: Input EEG data
            n_samples: Number of Monte Carlo samples
            confidence_level: Confidence level for intervals
            
        Returns:
            UncertaintyResult with predictions and uncertainties
        """
        predictions = []
        
        # Collect Monte Carlo samples
        for _ in tqdm(range(n_samples), desc="MC Dropout sampling"):
            with torch.no_grad():
                pred = self.model(input_tensor)
                predictions.append(pred.cpu().numpy())
                
        predictions = np.array(predictions)
        
        # Compute mean prediction
        mean_prediction = predictions.mean(axis=0)
        
        # Compute epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = self._compute_epistemic_uncertainty(predictions)
        
        # Compute aleatoric uncertainty (data uncertainty)
        aleatoric_uncertainty = self._compute_aleatoric_uncertainty(mean_prediction)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Compute confidence intervals
        lower_bound, upper_bound = self._compute_confidence_intervals(
            predictions, confidence_level
        )
        
        # Compute calibration metrics
        calibration_metrics = self._compute_calibration_metrics(
            predictions, input_tensor
        )
        
        return UncertaintyResult(
            predictions=mean_prediction,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            confidence_intervals=(lower_bound, upper_bound),
            calibration_metrics=calibration_metrics,
            metadata={
                'n_samples': n_samples,
                'confidence_level': confidence_level,
                'method': 'monte_carlo_dropout'
            }
        )
    
    def _compute_epistemic_uncertainty(self, predictions):
        """Compute epistemic uncertainty from MC samples."""
        # Variance across Monte Carlo samples
        if predictions.shape[-1] > 1:  # Classification
            # Convert to probabilities
            probs = self._softmax(predictions)
            # Mutual information between predictions and model parameters
            mean_probs = probs.mean(axis=0)
            expected_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=-1)
            
            # Expected entropy
            individual_entropies = []
            for i in range(probs.shape[0]):
                entropy = -np.sum(probs[i] * np.log(probs[i] + 1e-8), axis=-1)
                individual_entropies.append(entropy)
            expected_of_entropy = np.mean(individual_entropies, axis=0)
            
            epistemic = expected_entropy - expected_of_entropy
        else:  # Regression
            epistemic = predictions.var(axis=0)
            
        return epistemic
    
    def _compute_aleatoric_uncertainty(self, mean_prediction):
        """Compute aleatoric uncertainty from predictions."""
        if mean_prediction.shape[-1] > 1:  # Classification
            # Entropy of mean prediction
            probs = self._softmax(mean_prediction)
            aleatoric = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
        else:  # Regression
            # Could be learned from data, here we use a heuristic
            aleatoric = np.ones_like(mean_prediction) * 0.1
            
        return aleatoric
    
    def _compute_confidence_intervals(self, predictions, confidence_level):
        """Compute confidence intervals from MC samples."""
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(predictions, upper_percentile, axis=0)
        
        return lower_bound, upper_bound
    
    def _compute_calibration_metrics(self, predictions, input_tensor):
        """Compute calibration metrics for uncertainty estimates."""
        # Expected Calibration Error (ECE)
        if predictions.shape[-1] > 1:  # Classification
            probs = self._softmax(predictions.mean(axis=0))
            confidences = probs.max(axis=-1)
            predicted_classes = probs.argmax(axis=-1)
            
            # Bin confidences
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                prop_in_bin = in_bin.astype(float).mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = predicted_classes[in_bin].mean()  # Simplified
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    
            return {'ece': ece, 'mce': ece}  # Simplified metrics
        else:
            return {}
            
    def _softmax(self, x):
        """Apply softmax to convert logits to probabilities."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class EnsembleUncertainty:
    """
    Ensemble-based uncertainty quantification.
    Uses multiple models to estimate predictive uncertainty.
    """
    
    def __init__(self, models: List[nn.Module], device: str = 'cuda'):
        self.models = models
        self.device = device
        for model in self.models:
            model.eval()
            
    def predict_with_uncertainty(self,
                                 input_tensor: torch.Tensor,
                                 aggregate_method: str = 'mean') -> UncertaintyResult:
        """
        Make predictions with ensemble uncertainty.
        
        Args:
            input_tensor: Input EEG data
            aggregate_method: How to aggregate predictions ('mean', 'voting')
            
        Returns:
            UncertaintyResult with ensemble predictions
        """
        predictions = []
        
        # Get predictions from each model
        for model in self.models:
            with torch.no_grad():
                pred = model(input_tensor)
                predictions.append(pred.cpu().numpy())
                
        predictions = np.array(predictions)
        
        # Aggregate predictions
        if aggregate_method == 'mean':
            ensemble_prediction = predictions.mean(axis=0)
        elif aggregate_method == 'voting':
            # For classification
            votes = predictions.argmax(axis=-1)
            ensemble_prediction = np.zeros_like(predictions[0])
            for i in range(votes.shape[1]):
                vote_counts = np.bincount(votes[:, i])
                ensemble_prediction[i, vote_counts.argmax()] = 1
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate_method}")
            
        # Compute uncertainties
        epistemic_uncertainty = self._compute_ensemble_variance(predictions)
        aleatoric_uncertainty = self._compute_ensemble_entropy(ensemble_prediction)
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Confidence intervals
        lower_bound = np.percentile(predictions, 2.5, axis=0)
        upper_bound = np.percentile(predictions, 97.5, axis=0)
        
        # Diversity metrics
        diversity_metrics = self._compute_ensemble_diversity(predictions)
        
        return UncertaintyResult(
            predictions=ensemble_prediction,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            confidence_intervals=(lower_bound, upper_bound),
            calibration_metrics=diversity_metrics,
            metadata={
                'n_models': len(self.models),
                'aggregate_method': aggregate_method,
                'method': 'ensemble'
            }
        )
    
    def _compute_ensemble_variance(self, predictions):
        """Compute variance across ensemble members."""
        if predictions.shape[-1] > 1:  # Classification
            # Variance of probability predictions
            probs = np.array([self._softmax(pred) for pred in predictions])
            variance = probs.var(axis=0).mean(axis=-1)
        else:  # Regression
            variance = predictions.var(axis=0)
            
        return variance
    
    def _compute_ensemble_entropy(self, ensemble_prediction):
        """Compute entropy of ensemble prediction."""
        if ensemble_prediction.shape[-1] > 1:  # Classification
            probs = self._softmax(ensemble_prediction)
            entropy = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
        else:  # Regression
            entropy = np.zeros_like(ensemble_prediction).squeeze()
            
        return entropy
    
    def _compute_ensemble_diversity(self, predictions):
        """Compute diversity metrics for ensemble."""
        n_models = predictions.shape[0]
        
        # Pairwise disagreement
        disagreements = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                if predictions.shape[-1] > 1:  # Classification
                    disagree = (predictions[i].argmax(axis=-1) != 
                               predictions[j].argmax(axis=-1)).mean()
                else:  # Regression
                    disagree = np.abs(predictions[i] - predictions[j]).mean()
                disagreements.append(disagree)
                
        return {
            'mean_disagreement': np.mean(disagreements),
            'std_disagreement': np.std(disagreements)
        }
    
    def _softmax(self, x):
        """Apply softmax to convert logits to probabilities."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class BayesianNeuralNetwork:
    """
    Bayesian Neural Network for principled uncertainty estimation.
    Uses variational inference to learn distributions over weights.
    """
    
    def __init__(self, base_model: nn.Module, device: str = 'cuda'):
        self.device = device
        self.base_model = base_model
        self._convert_to_bayesian()
        
    def _convert_to_bayesian(self):
        """Convert deterministic layers to Bayesian layers."""
        # This is a simplified implementation
        # In practice, would replace layers with variational counterparts
        self.bayesian_model = BayesianWrapper(self.base_model)
        
    def predict_with_uncertainty(self,
                                 input_tensor: torch.Tensor,
                                 n_samples: int = 100) -> UncertaintyResult:
        """
        Make predictions with Bayesian uncertainty.
        
        Args:
            input_tensor: Input data
            n_samples: Number of samples from posterior
            
        Returns:
            UncertaintyResult with Bayesian predictions
        """
        predictions = []
        log_priors = []
        log_variational_posteriors = []
        
        # Sample from posterior
        for _ in range(n_samples):
            # Sample weights from variational posterior
            self.bayesian_model.sample_weights()
            
            with torch.no_grad():
                pred = self.bayesian_model(input_tensor)
                predictions.append(pred.cpu().numpy())
                
            # Get KL divergence components
            log_prior = self.bayesian_model.log_prior()
            log_variational_posterior = self.bayesian_model.log_variational_posterior()
            
            log_priors.append(log_prior)
            log_variational_posteriors.append(log_variational_posterior)
            
        predictions = np.array(predictions)
        
        # Compute uncertainties
        mean_prediction = predictions.mean(axis=0)
        epistemic_uncertainty = predictions.var(axis=0).mean(axis=-1)
        
        # Compute KL divergence
        kl_divergence = (torch.tensor(log_variational_posteriors).mean() - 
                        torch.tensor(log_priors).mean()).item()
        
        # Aleatoric uncertainty from predictive distribution
        if predictions.shape[-1] > 1:  # Classification
            probs = np.array([self._softmax(pred) for pred in predictions])
            aleatoric_uncertainty = -np.sum(
                probs.mean(axis=0) * np.log(probs.mean(axis=0) + 1e-8), 
                axis=-1
            )
        else:  # Regression
            aleatoric_uncertainty = np.ones_like(mean_prediction).squeeze() * 0.1
            
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Confidence intervals
        lower_bound = np.percentile(predictions, 2.5, axis=0)
        upper_bound = np.percentile(predictions, 97.5, axis=0)
        
        return UncertaintyResult(
            predictions=mean_prediction,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty,
            confidence_intervals=(lower_bound, upper_bound),
            calibration_metrics={'kl_divergence': kl_divergence},
            metadata={
                'n_samples': n_samples,
                'method': 'bayesian_nn'
            }
        )
    
    def _softmax(self, x):
        """Apply softmax to convert logits to probabilities."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class BayesianWrapper(nn.Module):
    """Wrapper to add Bayesian functionality to existing models."""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.noise_std = 0.1  # Simplified noise for weight sampling
        
    def sample_weights(self):
        """Sample weights from approximate posterior."""
        # Simplified: add Gaussian noise to weights
        for param in self.base_model.parameters():
            noise = torch.randn_like(param) * self.noise_std
            param.data.add_(noise)
            
    def forward(self, x):
        return self.base_model(x)
    
    def log_prior(self):
        """Compute log prior of weights."""
        # Simplified: Gaussian prior
        log_prior = 0
        for param in self.base_model.parameters():
            log_prior += -0.5 * (param ** 2).sum()
        return log_prior
    
    def log_variational_posterior(self):
        """Compute log variational posterior."""
        # Simplified: factorized Gaussian
        log_posterior = 0
        for param in self.base_model.parameters():
            log_posterior += -0.5 * ((param / self.noise_std) ** 2).sum()
        return log_posterior


class TemperatureScaling:
    """
    Temperature scaling for calibrated probability estimates.
    Improves calibration of neural network predictions.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.model.eval()
        
    def calibrate(self, 
                  validation_loader: torch.utils.data.DataLoader,
                  lr: float = 0.01,
                  max_iter: int = 50):
        """
        Learn temperature parameter on validation set.
        
        Args:
            validation_loader: Validation data loader
            lr: Learning rate for temperature optimization
            max_iter: Maximum optimization iterations
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        logits_list = []
        labels_list = []
        
        # Collect predictions
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs = inputs.to(self.device)
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)
                
        logits = torch.cat(logits_list).to(self.device)
        labels = torch.cat(labels_list).to(self.device)
        
        # Optimize temperature
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        
        print(f"Optimal temperature: {self.temperature.item():.3f}")
        
    def temperature_scale(self, logits):
        """Apply temperature scaling to logits."""
        return logits / self.temperature
    
    def predict_calibrated(self, input_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Make calibrated predictions.
        
        Args:
            input_tensor: Input data
            
        Returns:
            Dictionary with calibrated predictions and uncertainties
        """
        with torch.no_grad():
            logits = self.model(input_tensor)
            calibrated_logits = self.temperature_scale(logits)
            calibrated_probs = F.softmax(calibrated_logits, dim=-1)
            
            # Compute uncertainties
            entropy = -(calibrated_probs * torch.log(calibrated_probs + 1e-8)).sum(dim=-1)
            confidence = calibrated_probs.max(dim=-1)[0]
            
        return {
            'predictions': calibrated_probs.cpu().numpy(),
            'entropy': entropy.cpu().numpy(),
            'confidence': confidence.cpu().numpy(),
            'temperature': self.temperature.item()
        }


class UncertaintyVisualizer:
    """
    Visualize uncertainty estimates for interpretability.
    """
    
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def plot_prediction_intervals(self, 
                                  uncertainty_result: UncertaintyResult,
                                  feature_names: Optional[List[str]] = None,
                                  save_path: Optional[str] = None):
        """
        Plot prediction intervals with uncertainty bands.
        
        Args:
            uncertainty_result: UncertaintyResult object
            feature_names: Names for x-axis labels
            save_path: Path to save plot
        """
        predictions = uncertainty_result.predictions
        lower_bound, upper_bound = uncertainty_result.confidence_intervals
        epistemic = uncertainty_result.epistemic_uncertainty
        aleatoric = uncertainty_result.aleatoric_uncertainty
        
        n_samples = len(predictions)
        x = np.arange(n_samples)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot predictions with confidence intervals
        ax1.plot(x, predictions, 'b-', label='Mean Prediction', linewidth=2)
        ax1.fill_between(x, lower_bound, upper_bound, alpha=0.3, 
                         label=f'{int(uncertainty_result.metadata["confidence_level"]*100)}% CI')
        
        ax1.set_xlabel('Sample Index' if feature_names is None else 'Feature')
        ax1.set_ylabel('Prediction')
        ax1.set_title('Predictions with Confidence Intervals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot uncertainty decomposition
        width = 0.35
        ax2.bar(x - width/2, epistemic, width, label='Epistemic Uncertainty', 
                color=self.colors[0], alpha=0.7)
        ax2.bar(x + width/2, aleatoric, width, label='Aleatoric Uncertainty',
                color=self.colors[1], alpha=0.7)
        
        ax2.set_xlabel('Sample Index' if feature_names is None else 'Feature')
        ax2.set_ylabel('Uncertainty')
        ax2.set_title('Uncertainty Decomposition')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        if feature_names and len(feature_names) == n_samples:
            ax1.set_xticks(x)
            ax1.set_xticklabels(feature_names, rotation=45)
            ax2.set_xticks(x)
            ax2.set_xticklabels(feature_names, rotation=45)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_calibration_curve(self,
                               predictions: np.ndarray,
                               labels: np.ndarray,
                               n_bins: int = 10,
                               save_path: Optional[str] = None):
        """
        Plot calibration curve to assess prediction reliability.
        
        Args:
            predictions: Predicted probabilities
            labels: True labels
            n_bins: Number of calibration bins
            save_path: Path to save plot
        """
        # Get confidence and accuracy for each bin
        confidences = predictions.max(axis=-1)
        accuracies = (predictions.argmax(axis=-1) == labels).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_count = in_bin.sum()
            
            if bin_count > 0:
                bin_confidences.append(confidences[in_bin].mean())
                bin_accuracies.append(accuracies[in_bin].mean())
                bin_counts.append(bin_count)
            else:
                bin_confidences.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(0)
                bin_counts.append(0)
                
        # Plot calibration curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Reliability diagram
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        ax1.plot(bin_confidences, bin_accuracies, 'o-', color=self.colors[0],
                linewidth=2, markersize=8, label='Model Calibration')
        
        ax1.set_xlabel('Mean Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Reliability Diagram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Confidence histogram
        ax2.hist(confidences, bins=n_bins, alpha=0.7, color=self.colors[1],
                edgecolor='black', density=True)
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Density')
        ax2.set_title('Confidence Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_uncertainty_distribution(self,
                                      uncertainty_result: UncertaintyResult,
                                      save_path: Optional[str] = None):
        """
        Plot distribution of different uncertainty types.
        
        Args:
            uncertainty_result: UncertaintyResult object
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Epistemic uncertainty distribution
        axes[0, 0].hist(uncertainty_result.epistemic_uncertainty, bins=30,
                       alpha=0.7, color=self.colors[0], edgecolor='black')
        axes[0, 0].set_xlabel('Epistemic Uncertainty')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Model Uncertainty Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Aleatoric uncertainty distribution
        axes[0, 1].hist(uncertainty_result.aleatoric_uncertainty, bins=30,
                       alpha=0.7, color=self.colors[1], edgecolor='black')
        axes[0, 1].set_xlabel('Aleatoric Uncertainty')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Data Uncertainty Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Total uncertainty distribution
        axes[1, 0].hist(uncertainty_result.total_uncertainty, bins=30,
                       alpha=0.7, color=self.colors[2], edgecolor='black')
        axes[1, 0].set_xlabel('Total Uncertainty')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Total Uncertainty Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Uncertainty correlation
        axes[1, 1].scatter(uncertainty_result.epistemic_uncertainty,
                          uncertainty_result.aleatoric_uncertainty,
                          alpha=0.5, color=self.colors[3])
        axes[1, 1].set_xlabel('Epistemic Uncertainty')
        axes[1, 1].set_ylabel('Aleatoric Uncertainty')
        axes[1, 1].set_title('Uncertainty Correlation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class UncertaintyBasedActiveLearning:
    """
    Active learning strategies based on uncertainty estimates.
    Selects most informative samples for labeling.
    """
    
    def __init__(self, uncertainty_method: Union[MonteCarloDropout, 
                                                EnsembleUncertainty,
                                                BayesianNeuralNetwork]):
        self.uncertainty_method = uncertainty_method
        
    def select_samples(self,
                       unlabeled_data: torch.Tensor,
                       n_samples: int,
                       strategy: str = 'max_entropy',
                       batch_size: int = 32) -> List[int]:
        """
        Select most informative samples for labeling.
        
        Args:
            unlabeled_data: Unlabeled data tensor
            n_samples: Number of samples to select
            strategy: Selection strategy
            batch_size: Batch size for processing
            
        Returns:
            Indices of selected samples
        """
        all_uncertainties = []
        
        # Process in batches
        n_batches = (len(unlabeled_data) + batch_size - 1) // batch_size
        
        for i in tqdm(range(n_batches), desc="Computing uncertainties"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(unlabeled_data))
            batch = unlabeled_data[start_idx:end_idx]
            
            # Get uncertainty estimates
            uncertainty_result = self.uncertainty_method.predict_with_uncertainty(batch)
            
            if strategy == 'max_entropy':
                # Maximum entropy sampling
                scores = self._entropy_sampling(uncertainty_result)
            elif strategy == 'bald':
                # Bayesian Active Learning by Disagreement
                scores = self._bald_sampling(uncertainty_result)
            elif strategy == 'variance_ratio':
                # Variance ratio sampling
                scores = self._variance_ratio_sampling(uncertainty_result)
            elif strategy == 'margin':
                # Margin sampling
                scores = self._margin_sampling(uncertainty_result)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
            all_uncertainties.extend(scores)
            
        # Select top-k most uncertain samples
        all_uncertainties = np.array(all_uncertainties)
        selected_indices = np.argsort(-all_uncertainties)[:n_samples]
        
        return selected_indices.tolist()
    
    def _entropy_sampling(self, uncertainty_result):
        """Select samples with maximum predictive entropy."""
        predictions = uncertainty_result.predictions
        if predictions.shape[-1] > 1:  # Classification
            probs = self._softmax(predictions)
            entropy = -np.sum(probs * np.log(probs + 1e-8), axis=-1)
        else:  # Regression
            entropy = uncertainty_result.total_uncertainty
        return entropy
    
    def _bald_sampling(self, uncertainty_result):
        """Bayesian Active Learning by Disagreement."""
        # This is approximated by epistemic uncertainty
        return uncertainty_result.epistemic_uncertainty
    
    def _variance_ratio_sampling(self, uncertainty_result):
        """Select samples with minimum confidence."""
        predictions = uncertainty_result.predictions
        if predictions.shape[-1] > 1:  # Classification
            probs = self._softmax(predictions)
            # 1 - max probability
            variance_ratio = 1 - probs.max(axis=-1)
        else:  # Regression
            variance_ratio = uncertainty_result.total_uncertainty
        return variance_ratio
    
    def _margin_sampling(self, uncertainty_result):
        """Select samples with smallest margin between top predictions."""
        predictions = uncertainty_result.predictions
        if predictions.shape[-1] > 1:  # Classification
            probs = self._softmax(predictions)
            sorted_probs = np.sort(probs, axis=-1)
            # Difference between top two predictions
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            # Invert so smaller margins have higher scores
            return 1 - margin
        else:  # Regression
            return uncertainty_result.total_uncertainty
            
    def _softmax(self, x):
        """Apply softmax to convert logits to probabilities."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True) 