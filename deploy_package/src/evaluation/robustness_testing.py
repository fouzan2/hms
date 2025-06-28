"""
Model robustness testing for EEG classification.
Tests model performance under various challenging conditions and perturbations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import warnings
from tqdm import tqdm
from scipy import signal
import pandas as pd


@dataclass
class RobustnessResult:
    """Container for robustness test results."""
    test_name: str
    baseline_performance: float
    perturbed_performance: float
    performance_drop: float
    severity_levels: Dict[str, float]
    metadata: Dict[str, any]


class NoiseRobustnessTester:
    """
    Tests model robustness to various types of noise.
    Simulates real-world EEG recording conditions.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def test_gaussian_noise(self,
                           dataloader: torch.utils.data.DataLoader,
                           eval_fn: Callable,
                           noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2]) -> RobustnessResult:
        """
        Test robustness to Gaussian noise.
        
        Args:
            dataloader: Test data loader
            eval_fn: Evaluation function (model, loader) -> metrics
            noise_levels: Noise standard deviations to test
            
        Returns:
            RobustnessResult with performance at each noise level
        """
        # Baseline performance
        baseline_metrics = eval_fn(self.model, dataloader)
        baseline_perf = self._extract_primary_metric(baseline_metrics)
        
        severity_results = {}
        
        for noise_level in tqdm(noise_levels, desc="Testing Gaussian noise"):
            # Create noisy dataloader
            noisy_loader = self._create_noisy_loader(
                dataloader, 
                lambda x: self._add_gaussian_noise(x, noise_level)
            )
            
            # Evaluate on noisy data
            noisy_metrics = eval_fn(self.model, noisy_loader)
            noisy_perf = self._extract_primary_metric(noisy_metrics)
            
            severity_results[f'sigma_{noise_level}'] = noisy_perf
            
        # Calculate average performance drop
        avg_perturbed = np.mean(list(severity_results.values()))
        performance_drop = (baseline_perf - avg_perturbed) / baseline_perf
        
        return RobustnessResult(
            test_name='gaussian_noise',
            baseline_performance=baseline_perf,
            perturbed_performance=avg_perturbed,
            performance_drop=performance_drop,
            severity_levels=severity_results,
            metadata={'noise_type': 'gaussian', 'noise_levels': noise_levels}
        )
    
    def test_electrode_noise(self,
                            dataloader: torch.utils.data.DataLoader,
                            eval_fn: Callable,
                            dropout_rates: List[float] = [0.1, 0.2, 0.3, 0.5]) -> RobustnessResult:
        """
        Test robustness to electrode dropout/failure.
        
        Args:
            dataloader: Test data loader
            eval_fn: Evaluation function
            dropout_rates: Fraction of channels to drop
            
        Returns:
            RobustnessResult
        """
        baseline_metrics = eval_fn(self.model, dataloader)
        baseline_perf = self._extract_primary_metric(baseline_metrics)
        
        severity_results = {}
        
        for dropout_rate in tqdm(dropout_rates, desc="Testing electrode dropout"):
            noisy_loader = self._create_noisy_loader(
                dataloader,
                lambda x: self._simulate_electrode_dropout(x, dropout_rate)
            )
            
            noisy_metrics = eval_fn(self.model, noisy_loader)
            noisy_perf = self._extract_primary_metric(noisy_metrics)
            
            severity_results[f'dropout_{dropout_rate}'] = noisy_perf
            
        avg_perturbed = np.mean(list(severity_results.values()))
        performance_drop = (baseline_perf - avg_perturbed) / baseline_perf
        
        return RobustnessResult(
            test_name='electrode_dropout',
            baseline_performance=baseline_perf,
            perturbed_performance=avg_perturbed,
            performance_drop=performance_drop,
            severity_levels=severity_results,
            metadata={'dropout_rates': dropout_rates}
        )
    
    def test_powerline_interference(self,
                                   dataloader: torch.utils.data.DataLoader,
                                   eval_fn: Callable,
                                   frequencies: List[float] = [50.0, 60.0],
                                   amplitudes: List[float] = [0.1, 0.5, 1.0]) -> RobustnessResult:
        """
        Test robustness to powerline interference.
        
        Args:
            dataloader: Test data loader
            eval_fn: Evaluation function
            frequencies: Powerline frequencies (Hz)
            amplitudes: Interference amplitudes
            
        Returns:
            RobustnessResult
        """
        baseline_metrics = eval_fn(self.model, dataloader)
        baseline_perf = self._extract_primary_metric(baseline_metrics)
        
        severity_results = {}
        
        for freq in frequencies:
            for amp in amplitudes:
                noisy_loader = self._create_noisy_loader(
                    dataloader,
                    lambda x: self._add_powerline_noise(x, freq, amp)
                )
                
                noisy_metrics = eval_fn(self.model, noisy_loader)
                noisy_perf = self._extract_primary_metric(noisy_metrics)
                
                severity_results[f'{freq}Hz_amp{amp}'] = noisy_perf
                
        avg_perturbed = np.mean(list(severity_results.values()))
        performance_drop = (baseline_perf - avg_perturbed) / baseline_perf
        
        return RobustnessResult(
            test_name='powerline_interference',
            baseline_performance=baseline_perf,
            perturbed_performance=avg_perturbed,
            performance_drop=performance_drop,
            severity_levels=severity_results,
            metadata={'frequencies': frequencies, 'amplitudes': amplitudes}
        )
    
    def _add_gaussian_noise(self, x: torch.Tensor, std: float) -> torch.Tensor:
        """Add Gaussian noise to signal."""
        noise = torch.randn_like(x) * std
        return x + noise
    
    def _simulate_electrode_dropout(self, x: torch.Tensor, dropout_rate: float) -> torch.Tensor:
        """Simulate electrode dropout by zeroing channels."""
        x_noisy = x.clone()
        n_channels = x.shape[1] if len(x.shape) > 2 else x.shape[0]
        n_dropout = int(n_channels * dropout_rate)
        
        # Randomly select channels to drop
        dropout_channels = torch.randperm(n_channels)[:n_dropout]
        
        if len(x.shape) == 3:  # (batch, channels, time)
            x_noisy[:, dropout_channels, :] = 0
        elif len(x.shape) == 2:  # (channels, time)
            x_noisy[dropout_channels, :] = 0
            
        return x_noisy
    
    def _add_powerline_noise(self, x: torch.Tensor, freq: float, amplitude: float) -> torch.Tensor:
        """Add powerline interference."""
        # Assuming sampling rate is stored or default
        fs = 250.0  # Hz, typical EEG sampling rate
        
        if len(x.shape) == 3:
            batch_size, n_channels, n_samples = x.shape
            t = torch.arange(n_samples, device=x.device) / fs
            noise = amplitude * torch.sin(2 * np.pi * freq * t)
            noise = noise.unsqueeze(0).unsqueeze(0).expand(batch_size, n_channels, -1)
        else:
            n_samples = x.shape[-1]
            t = torch.arange(n_samples, device=x.device) / fs
            noise = amplitude * torch.sin(2 * np.pi * freq * t)
            
        return x + noise
    
    def _create_noisy_loader(self, original_loader, noise_fn):
        """Create a dataloader with noise applied."""
        class NoisyDataset(torch.utils.data.Dataset):
            def __init__(self, base_loader, transform_fn):
                self.data = []
                for batch in base_loader:
                    self.data.append(batch)
                self.transform_fn = transform_fn
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                batch = self.data[idx]
                # Apply noise to EEG data
                noisy_batch = batch.copy()
                if 'eeg' in noisy_batch:
                    noisy_batch['eeg'] = self.transform_fn(noisy_batch['eeg'])
                return noisy_batch
                
        noisy_dataset = NoisyDataset(original_loader, noise_fn)
        return torch.utils.data.DataLoader(
            noisy_dataset,
            batch_size=1,
            shuffle=False
        )
    
    def _extract_primary_metric(self, metrics: Dict) -> float:
        """Extract primary performance metric from results."""
        # Priority order for metrics
        if 'accuracy' in metrics:
            return metrics['accuracy']
        elif 'balanced_accuracy' in metrics:
            return metrics['balanced_accuracy']
        elif 'macro_f1' in metrics:
            return metrics['macro_f1']
        else:
            # Return first numeric metric found
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    return value
            return 0.0


class AdversarialRobustnessTester:
    """
    Tests model robustness to adversarial attacks.
    Ensures model is not easily fooled by crafted inputs.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        
    def test_fgsm_attack(self,
                        dataloader: torch.utils.data.DataLoader,
                        eval_fn: Callable,
                        epsilon_values: List[float] = [0.01, 0.05, 0.1, 0.2]) -> RobustnessResult:
        """
        Test robustness to Fast Gradient Sign Method attack.
        
        Args:
            dataloader: Test data loader
            eval_fn: Evaluation function
            epsilon_values: Attack strengths
            
        Returns:
            RobustnessResult
        """
        self.model.eval()
        
        # Baseline performance
        baseline_metrics = eval_fn(self.model, dataloader)
        baseline_perf = baseline_metrics.get('accuracy', 0.0)
        
        severity_results = {}
        
        for epsilon in tqdm(epsilon_values, desc="Testing FGSM attack"):
            correct = 0
            total = 0
            
            for batch in dataloader:
                eeg_data = batch['eeg'].to(self.device).requires_grad_(True)
                labels = batch['label'].to(self.device)
                
                # Generate adversarial examples
                adv_eeg = self._fgsm_attack(eeg_data, labels, epsilon)
                
                # Evaluate on adversarial examples
                with torch.no_grad():
                    if 'spectrogram' in batch:
                        # Recompute spectrogram for adversarial EEG
                        outputs = self.model(adv_eeg, batch['spectrogram'].to(self.device))
                    else:
                        outputs = self.model(adv_eeg)
                        
                    predictions = outputs.argmax(dim=1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                    
            accuracy = correct / total
            severity_results[f'epsilon_{epsilon}'] = accuracy
            
        avg_perturbed = np.mean(list(severity_results.values()))
        performance_drop = (baseline_perf - avg_perturbed) / baseline_perf
        
        return RobustnessResult(
            test_name='fgsm_attack',
            baseline_performance=baseline_perf,
            perturbed_performance=avg_perturbed,
            performance_drop=performance_drop,
            severity_levels=severity_results,
            metadata={'attack_type': 'FGSM', 'epsilon_values': epsilon_values}
        )
    
    def _fgsm_attack(self, data: torch.Tensor, target: torch.Tensor, 
                     epsilon: float) -> torch.Tensor:
        """Generate adversarial examples using FGSM."""
        # Forward pass
        output = self.model(data)
        loss = F.cross_entropy(output, target)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Collect gradients
        data_grad = data.grad.data
        
        # Create adversarial example
        sign_data_grad = data_grad.sign()
        perturbed_data = data + epsilon * sign_data_grad
        
        # Clamp to valid range (assuming normalized data)
        perturbed_data = torch.clamp(perturbed_data, -3, 3)
        
        return perturbed_data


class DistributionShiftTester:
    """
    Tests model robustness to distribution shifts.
    Evaluates performance on out-of-distribution data.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def test_demographic_shift(self,
                              train_loader: torch.utils.data.DataLoader,
                              test_loaders: Dict[str, torch.utils.data.DataLoader],
                              eval_fn: Callable) -> Dict[str, RobustnessResult]:
        """
        Test performance across different demographic groups.
        
        Args:
            train_loader: Training data loader
            test_loaders: Dict of test loaders by demographic group
            eval_fn: Evaluation function
            
        Returns:
            Dict of RobustnessResults by demographic
        """
        # Baseline on training distribution
        baseline_metrics = eval_fn(self.model, train_loader)
        baseline_perf = baseline_metrics.get('accuracy', 0.0)
        
        results = {}
        
        for demo_name, demo_loader in test_loaders.items():
            demo_metrics = eval_fn(self.model, demo_loader)
            demo_perf = demo_metrics.get('accuracy', 0.0)
            
            perf_drop = (baseline_perf - demo_perf) / baseline_perf
            
            results[demo_name] = RobustnessResult(
                test_name=f'demographic_shift_{demo_name}',
                baseline_performance=baseline_perf,
                perturbed_performance=demo_perf,
                performance_drop=perf_drop,
                severity_levels={demo_name: demo_perf},
                metadata={'demographic': demo_name}
            )
            
        return results
    
    def test_temporal_shift(self,
                           early_data_loader: torch.utils.data.DataLoader,
                           late_data_loader: torch.utils.data.DataLoader,
                           eval_fn: Callable) -> RobustnessResult:
        """
        Test robustness to temporal distribution shift.
        
        Args:
            early_data_loader: Data from early time period
            late_data_loader: Data from later time period
            eval_fn: Evaluation function
            
        Returns:
            RobustnessResult
        """
        # Performance on early data
        early_metrics = eval_fn(self.model, early_data_loader)
        early_perf = early_metrics.get('accuracy', 0.0)
        
        # Performance on late data
        late_metrics = eval_fn(self.model, late_data_loader)
        late_perf = late_metrics.get('accuracy', 0.0)
        
        perf_drop = (early_perf - late_perf) / early_perf
        
        return RobustnessResult(
            test_name='temporal_shift',
            baseline_performance=early_perf,
            perturbed_performance=late_perf,
            performance_drop=perf_drop,
            severity_levels={'early': early_perf, 'late': late_perf},
            metadata={'shift_type': 'temporal'}
        )
    
    def test_site_shift(self,
                       source_site_loader: torch.utils.data.DataLoader,
                       target_site_loaders: Dict[str, torch.utils.data.DataLoader],
                       eval_fn: Callable) -> Dict[str, RobustnessResult]:
        """
        Test robustness to data from different recording sites.
        
        Args:
            source_site_loader: Source site data
            target_site_loaders: Dict of target site loaders
            eval_fn: Evaluation function
            
        Returns:
            Dict of RobustnessResults by site
        """
        # Baseline on source site
        source_metrics = eval_fn(self.model, source_site_loader)
        source_perf = source_metrics.get('accuracy', 0.0)
        
        results = {}
        
        for site_name, site_loader in target_site_loaders.items():
            site_metrics = eval_fn(self.model, site_loader)
            site_perf = site_metrics.get('accuracy', 0.0)
            
            perf_drop = (source_perf - site_perf) / source_perf
            
            results[site_name] = RobustnessResult(
                test_name=f'site_shift_{site_name}',
                baseline_performance=source_perf,
                perturbed_performance=site_perf,
                performance_drop=perf_drop,
                severity_levels={site_name: site_perf},
                metadata={'source_site': 'source', 'target_site': site_name}
            )
            
        return results


class CalibrationTester:
    """
    Tests model calibration under various conditions.
    Ensures prediction probabilities are well-calibrated.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def test_calibration(self,
                        dataloader: torch.utils.data.DataLoader,
                        n_bins: int = 10) -> Dict[str, float]:
        """
        Test model calibration using reliability diagrams.
        
        Args:
            dataloader: Test data loader
            n_bins: Number of bins for calibration
            
        Returns:
            Dict with calibration metrics
        """
        all_probs = []
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                eeg = batch['eeg'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get predictions
                if 'spectrogram' in batch:
                    outputs = self.model(eeg, batch['spectrogram'].to(self.device))
                else:
                    outputs = self.model(eeg)
                    
                probs = F.softmax(outputs, dim=1)
                predictions = probs.argmax(dim=1)
                
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
                all_predictions.append(predictions.cpu())
                
        # Concatenate results
        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        
        # Compute calibration metrics
        ece = self._expected_calibration_error(
            all_probs, all_labels, all_predictions, n_bins
        )
        
        mce = self._maximum_calibration_error(
            all_probs, all_labels, all_predictions, n_bins
        )
        
        brier_score = self._brier_score(all_probs, all_labels)
        
        # Compute per-class calibration
        per_class_ece = {}
        n_classes = all_probs.shape[1]
        
        for class_idx in range(n_classes):
            class_probs = all_probs[:, class_idx]
            class_labels = (all_labels == class_idx).float()
            class_ece = self._binary_expected_calibration_error(
                class_probs, class_labels, n_bins
            )
            per_class_ece[f'class_{class_idx}_ece'] = class_ece
            
        return {
            'expected_calibration_error': ece,
            'maximum_calibration_error': mce,
            'brier_score': brier_score,
            **per_class_ece
        }
    
    def _expected_calibration_error(self, probs, labels, predictions, n_bins):
        """Compute Expected Calibration Error."""
        confidences = probs.max(dim=1)[0]
        accuracies = (predictions == labels).float()
        
        ece = 0.0
        for bin_idx in range(n_bins):
            bin_lower = bin_idx / n_bins
            bin_upper = (bin_idx + 1) / n_bins
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                acc_in_bin = accuracies[in_bin].mean()
                conf_in_bin = confidences[in_bin].mean()
                ece += prop_in_bin * torch.abs(acc_in_bin - conf_in_bin)
                
        return ece.item()
    
    def _maximum_calibration_error(self, probs, labels, predictions, n_bins):
        """Compute Maximum Calibration Error."""
        confidences = probs.max(dim=1)[0]
        accuracies = (predictions == labels).float()
        
        mce = 0.0
        for bin_idx in range(n_bins):
            bin_lower = bin_idx / n_bins
            bin_upper = (bin_idx + 1) / n_bins
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.any():
                acc_in_bin = accuracies[in_bin].mean()
                conf_in_bin = confidences[in_bin].mean()
                mce = max(mce, torch.abs(acc_in_bin - conf_in_bin).item())
                
        return mce
    
    def _brier_score(self, probs, labels):
        """Compute Brier score."""
        n_classes = probs.shape[1]
        one_hot_labels = F.one_hot(labels, n_classes).float()
        
        return ((probs - one_hot_labels) ** 2).sum(dim=1).mean().item()
    
    def _binary_expected_calibration_error(self, probs, labels, n_bins):
        """Compute ECE for binary classification."""
        ece = 0.0
        
        for bin_idx in range(n_bins):
            bin_lower = bin_idx / n_bins
            bin_upper = (bin_idx + 1) / n_bins
            
            in_bin = (probs > bin_lower) & (probs <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                acc_in_bin = labels[in_bin].mean()
                conf_in_bin = probs[in_bin].mean()
                ece += prop_in_bin * torch.abs(acc_in_bin - conf_in_bin)
                
        return ece.item()


class ConsistencyTester:
    """
    Tests prediction consistency across similar inputs.
    Ensures stable predictions for clinically similar cases.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def test_augmentation_consistency(self,
                                     dataloader: torch.utils.data.DataLoader,
                                     augmentations: List[Callable],
                                     n_trials: int = 5) -> Dict[str, float]:
        """
        Test consistency under data augmentations.
        
        Args:
            dataloader: Test data loader
            augmentations: List of augmentation functions
            n_trials: Number of trials per augmentation
            
        Returns:
            Dict with consistency metrics
        """
        consistency_scores = []
        
        for batch in tqdm(dataloader, desc="Testing augmentation consistency"):
            eeg = batch['eeg'].to(self.device)
            
            # Get original predictions
            with torch.no_grad():
                if 'spectrogram' in batch:
                    orig_outputs = self.model(eeg, batch['spectrogram'].to(self.device))
                else:
                    orig_outputs = self.model(eeg)
                    
                orig_probs = F.softmax(orig_outputs, dim=1)
                orig_preds = orig_probs.argmax(dim=1)
                
            # Test each augmentation
            for aug_fn in augmentations:
                aug_consistencies = []
                
                for _ in range(n_trials):
                    # Apply augmentation
                    aug_eeg = aug_fn(eeg)
                    
                    # Get augmented predictions
                    with torch.no_grad():
                        if 'spectrogram' in batch:
                            # Recompute spectrogram for augmented data
                            aug_outputs = self.model(aug_eeg, batch['spectrogram'].to(self.device))
                        else:
                            aug_outputs = self.model(aug_eeg)
                            
                        aug_probs = F.softmax(aug_outputs, dim=1)
                        aug_preds = aug_probs.argmax(dim=1)
                        
                    # Compute consistency
                    pred_consistency = (orig_preds == aug_preds).float().mean()
                    prob_consistency = 1 - (orig_probs - aug_probs).abs().mean()
                    
                    aug_consistencies.append({
                        'pred_consistency': pred_consistency.item(),
                        'prob_consistency': prob_consistency.item()
                    })
                    
                consistency_scores.append(aug_consistencies)
                
        # Aggregate results
        all_pred_consistencies = []
        all_prob_consistencies = []
        
        for batch_scores in consistency_scores:
            for score in batch_scores:
                all_pred_consistencies.append(score['pred_consistency'])
                all_prob_consistencies.append(score['prob_consistency'])
                
        return {
            'mean_prediction_consistency': np.mean(all_pred_consistencies),
            'std_prediction_consistency': np.std(all_pred_consistencies),
            'mean_probability_consistency': np.mean(all_prob_consistencies),
            'std_probability_consistency': np.std(all_prob_consistencies),
            'min_prediction_consistency': np.min(all_pred_consistencies),
            'min_probability_consistency': np.min(all_prob_consistencies)
        }
    
    def test_temporal_consistency(self,
                                 sequential_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Test temporal consistency of predictions.
        
        Args:
            sequential_loader: Loader with temporally sequential data
            
        Returns:
            Dict with temporal consistency metrics
        """
        prediction_changes = []
        confidence_changes = []
        
        prev_pred = None
        prev_conf = None
        
        with torch.no_grad():
            for batch in sequential_loader:
                eeg = batch['eeg'].to(self.device)
                
                if 'spectrogram' in batch:
                    outputs = self.model(eeg, batch['spectrogram'].to(self.device))
                else:
                    outputs = self.model(eeg)
                    
                probs = F.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)
                confs = probs.max(dim=1)[0]
                
                if prev_pred is not None:
                    # Compute changes
                    pred_change = (preds != prev_pred).float().mean()
                    conf_change = (confs - prev_conf).abs().mean()
                    
                    prediction_changes.append(pred_change.item())
                    confidence_changes.append(conf_change.item())
                    
                prev_pred = preds
                prev_conf = confs
                
        return {
            'mean_prediction_change_rate': np.mean(prediction_changes),
            'mean_confidence_change': np.mean(confidence_changes),
            'max_prediction_change_rate': np.max(prediction_changes),
            'max_confidence_change': np.max(confidence_changes)
        }


class RobustnessReporter:
    """
    Generates comprehensive robustness evaluation reports.
    """
    
    def __init__(self):
        self.results = {}
        
    def add_results(self, test_category: str, results: Union[RobustnessResult, Dict]):
        """Add test results to report."""
        if test_category not in self.results:
            self.results[test_category] = []
            
        if isinstance(results, dict):
            self.results[test_category].extend(results.values())
        else:
            self.results[test_category].append(results)
            
    def generate_report(self, save_path: str):
        """Generate comprehensive robustness report."""
        report = {
            'summary': self._generate_summary(),
            'detailed_results': self._format_detailed_results(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save as JSON
        import json
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Generate visualizations
        self._generate_visualizations(save_path.replace('.json', '_plots.png'))
        
        return report
    
    def _generate_summary(self) -> Dict:
        """Generate summary statistics."""
        summary = {
            'total_tests': sum(len(results) for results in self.results.values()),
            'test_categories': list(self.results.keys()),
            'average_performance_drops': {}
        }
        
        for category, results in self.results.items():
            drops = [r.performance_drop for r in results if hasattr(r, 'performance_drop')]
            if drops:
                summary['average_performance_drops'][category] = np.mean(drops)
                
        # Identify critical vulnerabilities
        critical_tests = []
        for category, results in self.results.items():
            for result in results:
                if hasattr(result, 'performance_drop') and result.performance_drop > 0.2:
                    critical_tests.append({
                        'category': category,
                        'test': result.test_name,
                        'drop': result.performance_drop
                    })
                    
        summary['critical_vulnerabilities'] = critical_tests
        
        return summary
    
    def _format_detailed_results(self) -> Dict:
        """Format detailed results for each test."""
        detailed = {}
        
        for category, results in self.results.items():
            detailed[category] = []
            
            for result in results:
                if isinstance(result, RobustnessResult):
                    detailed[category].append({
                        'test_name': result.test_name,
                        'baseline_performance': result.baseline_performance,
                        'perturbed_performance': result.perturbed_performance,
                        'performance_drop': result.performance_drop,
                        'severity_levels': result.severity_levels,
                        'metadata': result.metadata
                    })
                else:
                    detailed[category].append(result)
                    
        return detailed
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check noise robustness
        noise_results = self.results.get('noise_robustness', [])
        for result in noise_results:
            if hasattr(result, 'performance_drop') and result.performance_drop > 0.15:
                recommendations.append(
                    f"Model shows significant vulnerability to {result.test_name}. "
                    f"Consider data augmentation or adversarial training."
                )
                
        # Check calibration
        calibration_results = self.results.get('calibration', {})
        if isinstance(calibration_results, dict):
            ece = calibration_results.get('expected_calibration_error', 0)
            if ece > 0.1:
                recommendations.append(
                    f"Model calibration is poor (ECE={ece:.3f}). "
                    f"Consider temperature scaling or other calibration methods."
                )
                
        # Check consistency
        consistency_results = self.results.get('consistency', {})
        if isinstance(consistency_results, dict):
            pred_consistency = consistency_results.get('mean_prediction_consistency', 1.0)
            if pred_consistency < 0.8:
                recommendations.append(
                    f"Model shows poor prediction consistency ({pred_consistency:.2f}). "
                    f"Consider consistency regularization during training."
                )
                
        if not recommendations:
            recommendations.append("Model shows good robustness across all tested conditions.")
            
        return recommendations
    
    def _generate_visualizations(self, save_path: str):
        """Generate visualization plots for robustness results."""
        import matplotlib.pyplot as plt
        
        n_categories = len(self.results)
        fig, axes = plt.subplots(2, (n_categories + 1) // 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (category, results) in enumerate(self.results.items()):
            ax = axes[idx]
            
            # Extract performance drops
            test_names = []
            perf_drops = []
            
            for result in results:
                if hasattr(result, 'performance_drop'):
                    test_names.append(result.test_name)
                    perf_drops.append(result.performance_drop * 100)  # Convert to percentage
                    
            if test_names:
                # Create bar plot
                bars = ax.bar(range(len(test_names)), perf_drops)
                
                # Color bars based on severity
                colors = ['green' if p < 10 else 'orange' if p < 20 else 'red' for p in perf_drops]
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                    
                ax.set_xticks(range(len(test_names)))
                ax.set_xticklabels(test_names, rotation=45, ha='right')
                ax.set_ylabel('Performance Drop (%)')
                ax.set_title(f'{category.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3, axis='y')
                
        # Remove empty subplots
        for idx in range(len(self.results), len(axes)):
            fig.delaxes(axes[idx])
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() 