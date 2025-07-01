"""
Adaptive Preprocessing for EEG Data
Enterprise-level implementation that automatically optimizes preprocessing parameters
based on signal characteristics, noise profiles, and data quality.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import json
import pickle
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.stats import median_abs_deviation
import hashlib
from collections import deque
import time
import mne
from mne.preprocessing import ICA

from .signal_quality import SignalQualityAssessor, QualityMetrics
from .signal_filters import EEGFilter, AdvancedDenoiser
from .feature_extraction import EEGFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveParameters:
    """Optimized preprocessing parameters."""
    # Filtering parameters
    highpass_freq: float
    lowpass_freq: float
    notch_freq: Optional[float]
    notch_quality: float
    
    # Artifact removal parameters
    ica_components: int
    ica_method: str
    artifact_threshold: float
    
    # Denoising parameters
    wavelet_type: str
    wavelet_level: int
    denoising_threshold: float
    
    # Bad channel handling
    interpolation_method: str
    bad_channel_threshold: float
    correlation_threshold: float
    
    # Normalization
    normalization_method: str
    robust_scale: bool
    
    # Quality score
    expected_quality_improvement: float
    optimization_confidence: float


class DataProfiler:
    """Analyzes EEG data characteristics for adaptive preprocessing."""
    
    def __init__(self, sampling_rate: float = 200.0):
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
        
    def profile_data(self, eeg_data: np.ndarray, 
                    channel_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Comprehensive data profiling for preprocessing optimization.
        
        Args:
            eeg_data: Raw EEG data (channels x samples)
            channel_names: Optional channel names
            
        Returns:
            Dictionary containing data profile
        """
        profile = {}
        
        # Basic statistics
        profile['basic_stats'] = self._compute_basic_stats(eeg_data)
        
        # Frequency characteristics
        profile['frequency_profile'] = self._analyze_frequency_content(eeg_data)
        
        # Noise analysis
        profile['noise_profile'] = self._analyze_noise_characteristics(eeg_data)
        
        # Artifact detection
        profile['artifact_profile'] = self._detect_artifact_types(eeg_data)
        
        # Channel quality
        profile['channel_quality'] = self._assess_channel_quality(eeg_data, channel_names)
        
        # Stationarity analysis
        profile['stationarity'] = self._analyze_stationarity(eeg_data)
        
        # Connectivity patterns
        profile['connectivity'] = self._analyze_connectivity(eeg_data)
        
        return profile
    
    def _compute_basic_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Compute basic statistical measures."""
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'median': float(np.median(data)),
            'mad': float(stats.median_abs_deviation(data.flatten())),
            'skewness': float(stats.skew(data.flatten())),
            'kurtosis': float(stats.kurtosis(data.flatten())),
            'range': float(np.ptp(data)),
            'iqr': float(np.percentile(data, 75) - np.percentile(data, 25))
        }
    
    def _analyze_frequency_content(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze frequency domain characteristics."""
        # Compute PSD for each channel
        freqs, psd = signal.welch(data, fs=self.sampling_rate, nperseg=min(256, data.shape[1]//4))
        avg_psd = np.mean(psd, axis=0)
        
        # Frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        band_powers = {}
        # Use power only within our frequency bands of interest (0.5-50 Hz)
        freq_mask = (freqs >= 0.5) & (freqs <= 50)
        if not np.any(freq_mask):
            # Fallback if no frequencies in range
            total_power = np.sum(avg_psd)
        else:
            total_power = np.sum(avg_psd[freq_mask])
        
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs <= high)
            band_power = np.sum(avg_psd[band_mask])
            band_powers[band_name] = float(band_power / (total_power + 1e-10))
        
        # Dominant frequency
        dominant_freq = freqs[np.argmax(avg_psd)]
        
        # Spectral edge frequency
        cumsum_psd = np.cumsum(avg_psd)
        total_power_full = np.sum(avg_psd)
        sef_95_indices = np.where(cumsum_psd >= 0.95 * total_power_full)[0]
        sef_95 = freqs[sef_95_indices[0]] if len(sef_95_indices) > 0 else freqs[-1]
        
        return {
            'band_powers': band_powers,
            'dominant_frequency': float(dominant_freq),
            'spectral_edge_95': float(sef_95),
            'spectral_entropy': float(stats.entropy(avg_psd / total_power_full + 1e-10)),
            'high_freq_power': float(np.sum(avg_psd[freqs > 30]) / total_power_full)
        }
    
    def _analyze_noise_characteristics(self, data: np.ndarray) -> Dict[str, float]:
        """Analyze different types of noise in the signal."""
        noise_profile = {}
        
        # Line noise detection (50/60 Hz)
        for line_freq in [50, 60]:
            freqs, psd = signal.welch(data, fs=self.sampling_rate)
            freq_idx = np.argmin(np.abs(freqs - line_freq))
            
            # Check peak at line frequency
            if freq_idx > 2 and freq_idx < len(freqs) - 2:
                surrounding_power = np.mean(psd[:, [freq_idx-2, freq_idx+2]], axis=1)
                line_power = psd[:, freq_idx]
                line_noise_ratio = np.mean(line_power / (surrounding_power + 1e-10))
                noise_profile[f'line_noise_{line_freq}hz'] = float(line_noise_ratio)
        
        # High-frequency noise (muscle artifacts)
        hf_power = self._calculate_high_freq_noise(data)
        noise_profile['high_freq_noise'] = float(hf_power)
        
        # Low-frequency drift
        lf_drift = self._calculate_low_freq_drift(data)
        noise_profile['low_freq_drift'] = float(lf_drift)
        
        # White noise estimation
        white_noise = self._estimate_white_noise(data)
        noise_profile['white_noise_level'] = float(white_noise)
        
        return noise_profile
    
    def _detect_artifact_types(self, data: np.ndarray) -> Dict[str, float]:
        """Detect different types of artifacts."""
        artifacts = {}
        
        # Eye blink detection (frontal channels)
        eye_blink_score = self._detect_eye_blinks(data)
        artifacts['eye_blink_score'] = float(eye_blink_score)
        
        # Movement artifacts
        movement_score = self._detect_movement_artifacts(data)
        artifacts['movement_score'] = float(movement_score)
        
        # Electrode pop detection
        pop_score = self._detect_electrode_pops(data)
        artifacts['electrode_pop_score'] = float(pop_score)
        
        # Muscle artifacts
        muscle_score = self._detect_muscle_artifacts(data)
        artifacts['muscle_artifact_score'] = float(muscle_score)
        
        return artifacts
    
    def _assess_channel_quality(self, data: np.ndarray, 
                               channel_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Assess quality of individual channels."""
        n_channels = data.shape[0]
        
        # Channel variance
        channel_vars = np.var(data, axis=1)
        
        # Channel correlations
        corr_matrix = np.corrcoef(data)
        
        # Dead channel detection
        dead_channels = []
        for i in range(n_channels):
            if channel_vars[i] < 1e-10 or np.all(np.abs(corr_matrix[i, :]) < 0.1):
                ch_name = channel_names[i] if channel_names else f"ch_{i}"
                dead_channels.append(ch_name)
        
        # Noisy channel detection
        var_threshold = np.median(channel_vars) + 3 * stats.median_abs_deviation(channel_vars)
        noisy_channels = []
        for i in range(n_channels):
            if channel_vars[i] > var_threshold:
                ch_name = channel_names[i] if channel_names else f"ch_{i}"
                noisy_channels.append(ch_name)
        
        return {
            'dead_channels': dead_channels,
            'noisy_channels': noisy_channels,
            'channel_variance_ratio': float(np.max(channel_vars) / (np.min(channel_vars) + 1e-10)),
            'mean_correlation': float(np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])))
        }
    
    def _analyze_stationarity(self, data: np.ndarray) -> Dict[str, float]:
        """Analyze signal stationarity."""
        # Divide signal into segments
        n_segments = 10
        segment_length = data.shape[1] // n_segments
        
        segment_stats = []
        for i in range(n_segments):
            start = i * segment_length
            end = (i + 1) * segment_length
            segment = data[:, start:end]
            
            segment_stats.append({
                'mean': np.mean(segment),
                'std': np.std(segment),
                'power': np.mean(segment ** 2)
            })
        
        # Check variation in statistics
        mean_variation = np.std([s['mean'] for s in segment_stats])
        std_variation = np.std([s['std'] for s in segment_stats])
        power_variation = np.std([s['power'] for s in segment_stats])
        
        return {
            'mean_stationarity': float(1 / (1 + mean_variation)),
            'variance_stationarity': float(1 / (1 + std_variation)),
            'power_stationarity': float(1 / (1 + power_variation))
        }
    
    def _analyze_connectivity(self, data: np.ndarray) -> Dict[str, float]:
        """Analyze channel connectivity patterns."""
        # Coherence analysis
        n_channels = data.shape[0]
        coherence_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                f, Cxy = signal.coherence(data[i], data[j], fs=self.sampling_rate)
                # Average coherence in physiological bands
                physiological_mask = (f >= 0.5) & (f <= 40)
                coherence_matrix[i, j] = np.mean(Cxy[physiological_mask])
                coherence_matrix[j, i] = coherence_matrix[i, j]
        
        return {
            'mean_coherence': float(np.mean(coherence_matrix[np.triu_indices_from(coherence_matrix, k=1)])),
            'coherence_std': float(np.std(coherence_matrix[np.triu_indices_from(coherence_matrix, k=1)])),
            'network_density': float(np.sum(coherence_matrix > 0.5) / (n_channels * (n_channels - 1)))
        }
    
    # Helper methods for noise and artifact detection
    def _calculate_high_freq_noise(self, data: np.ndarray) -> float:
        """Calculate high-frequency noise level."""
        freqs, psd = signal.welch(data, fs=self.sampling_rate)
        hf_mask = freqs > 40
        if np.any(hf_mask):
            hf_power = np.mean(psd[:, hf_mask])
            total_power = np.mean(psd)
            return hf_power / total_power
        return 0.0
    
    def _calculate_low_freq_drift(self, data: np.ndarray) -> float:
        """Calculate low-frequency drift."""
        # Detrend and compare power
        detrended = signal.detrend(data, axis=1)
        drift_power = np.mean((data - detrended) ** 2)
        signal_power = np.mean(data ** 2)
        return drift_power / (signal_power + 1e-10)
    
    def _estimate_white_noise(self, data: np.ndarray) -> float:
        """Estimate white noise level."""
        # Use high-frequency spectrum flatness as indicator
        freqs, psd = signal.welch(data, fs=self.sampling_rate)
        hf_mask = freqs > 30
        if np.any(hf_mask):
            hf_psd = psd[:, hf_mask]
            flatness = np.mean(np.std(hf_psd, axis=1) / (np.mean(hf_psd, axis=1) + 1e-10))
            return 1 - flatness  # More flat = more white noise
        return 0.0
    
    def _detect_eye_blinks(self, data: np.ndarray) -> float:
        """Detect eye blink artifacts."""
        # Frontal channels typically show eye blinks
        frontal_data = data[:4] if data.shape[0] >= 4 else data
        
        # Look for characteristic slow waves
        filtered = signal.butter(4, [0.5, 5], btype='band', fs=self.sampling_rate, output='sos')
        slow_waves = signal.sosfiltfilt(filtered, frontal_data)
        
        # Detect high amplitude slow waves
        threshold = 3 * np.std(slow_waves)
        peaks = signal.find_peaks(np.abs(slow_waves).max(axis=0), height=threshold)[0]
        
        return len(peaks) / (data.shape[1] / self.sampling_rate)  # Blinks per second
    
    def _detect_movement_artifacts(self, data: np.ndarray) -> float:
        """Detect movement artifacts."""
        # Large amplitude, low-frequency artifacts
        filtered = signal.butter(4, 10, btype='low', fs=self.sampling_rate, output='sos')
        low_freq = signal.sosfiltfilt(filtered, data)
        
        # Detect sudden changes
        diff = np.diff(low_freq, axis=1)
        movement_score = np.mean(np.abs(diff) > 3 * np.std(diff))
        
        return movement_score
    
    def _detect_electrode_pops(self, data: np.ndarray) -> float:
        """Detect electrode pop artifacts."""
        # Sudden spikes in single channels
        pop_count = 0
        for ch in range(data.shape[0]):
            # Detect outliers
            z_scores = np.abs(stats.zscore(data[ch]))
            pops = np.sum(z_scores > 5)
            pop_count += pops
            
        return pop_count / (data.shape[0] * data.shape[1])
    
    def _detect_muscle_artifacts(self, data: np.ndarray) -> float:
        """Detect muscle artifacts."""
        # High-frequency activity
        freqs, psd = signal.welch(data, fs=self.sampling_rate)
        muscle_band = (20 < freqs) & (freqs < 40)
        
        if np.any(muscle_band):
            muscle_power = np.mean(psd[:, muscle_band])
            total_power = np.mean(psd)
            return muscle_power / total_power
        return 0.0


class PreprocessingOptimizer(nn.Module):
    """Neural network for optimizing preprocessing parameters."""
    
    def __init__(self, profile_dim: int = 50, hidden_dim: int = 256):
        super().__init__()
        
        # Encoder for data profile
        self.profile_encoder = nn.Sequential(
            nn.Linear(profile_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Parameter prediction heads
        self.filter_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # highpass, lowpass, notch_freq, notch_q
        )
        
        self.artifact_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # ica_components, method_idx, threshold
        )
        
        self.denoise_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # wavelet_idx, level, threshold
        )
        
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # expected_improvement, confidence
        )
        
        # Available methods
        self.ica_methods = ['fastica', 'infomax', 'picard']
        self.wavelet_types = ['db4', 'sym5', 'coif3', 'bior3.5']
        
    def forward(self, profile_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict optimal preprocessing parameters."""
        # Encode profile
        encoded = self.profile_encoder(profile_features)
        
        # Predict parameters
        filter_params = self.filter_head(encoded)
        artifact_params = self.artifact_head(encoded)
        denoise_params = self.denoise_head(encoded)
        quality_metrics = self.quality_head(encoded)
        
        return {
            'filter': filter_params,
            'artifact': artifact_params,
            'denoise': denoise_params,
            'quality': quality_metrics
        }
    
    def decode_parameters(self, predictions: Dict[str, torch.Tensor]) -> AdaptiveParameters:
        """Convert network outputs to preprocessing parameters."""
        # Extract first sample from batch and convert to numpy
        filter_params = predictions['filter'][0].detach().cpu().numpy()
        artifact_params = predictions['artifact'][0].detach().cpu().numpy()
        denoise_params = predictions['denoise'][0].detach().cpu().numpy()
        quality_metrics = predictions['quality'][0].detach().cpu().numpy()
        
        # Decode filter parameters
        highpass = float(np.clip(filter_params[0] * 2, 0.1, 2.0))  # 0.1-2 Hz
        lowpass = float(np.clip(filter_params[1] * 50 + 40, 40, 90))  # 40-90 Hz
        notch_freq = float(filter_params[2] * 10 + 55) if filter_params[2] > 0.5 else None  # 50-65 Hz
        notch_q = float(np.clip(filter_params[3] * 30 + 10, 10, 40))  # Q factor 10-40
        
        # Decode artifact parameters - be more conservative with ICA components
        ica_components = int(np.clip(artifact_params[0] * 15 + 5, 5, 20))  # 5-20 instead of 5-30
        ica_method = self.ica_methods[int(np.clip(artifact_params[1] * len(self.ica_methods), 0, len(self.ica_methods)-1))]
        artifact_threshold = float(np.clip(artifact_params[2] * 3 + 1, 0.5, 4.0))
        
        # Decode denoising parameters
        wavelet_idx = int(np.clip(denoise_params[0] * len(self.wavelet_types), 0, len(self.wavelet_types)-1))
        wavelet_type = self.wavelet_types[wavelet_idx]
        wavelet_level = int(np.clip(denoise_params[1] * 5 + 3, 3, 8))
        denoising_threshold = float(np.clip(denoise_params[2] * 0.5 + 0.1, 0.05, 0.6))
        
        # Quality metrics - use torch functions since these are still tensors
        expected_improvement = float(torch.sigmoid(torch.tensor(quality_metrics[0])))
        confidence = float(torch.sigmoid(torch.tensor(quality_metrics[1])))
        
        return AdaptiveParameters(
            highpass_freq=highpass,
            lowpass_freq=lowpass,
            notch_freq=notch_freq,
            notch_quality=notch_q,
            ica_components=ica_components,
            ica_method=ica_method,
            artifact_threshold=artifact_threshold,
            wavelet_type=wavelet_type,
            wavelet_level=wavelet_level,
            denoising_threshold=denoising_threshold,
            interpolation_method='spherical',
            bad_channel_threshold=3.0,
            correlation_threshold=0.1,
            normalization_method='robust',
            robust_scale=True,
            expected_quality_improvement=expected_improvement,
            optimization_confidence=confidence
        )


class AdaptivePreprocessor:
    """
    Enterprise-level adaptive preprocessing system.
    Automatically optimizes preprocessing parameters based on data characteristics.
    """
    
    def __init__(self, config: Dict, 
                 model_path: Optional[Path] = None,
                 cache_size: int = 1000,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize adaptive preprocessor.
        
        Args:
            config: Configuration dictionary
            model_path: Path to trained optimizer model
            cache_size: Size of parameter cache
            device: Device for neural network
        """
        self.config = config
        self.device = device
        self.sampling_rate = config.get('sampling_rate', 200.0)
        
        # Initialize components
        self.data_profiler = DataProfiler(self.sampling_rate)
        self.quality_assessor = SignalQualityAssessor(self.sampling_rate)
        
        # Initialize optimizer network
        self.optimizer = PreprocessingOptimizer().to(device)
        if model_path and model_path.exists():
            self.load_optimizer(model_path)
        else:
            logger.warning("No pre-trained optimizer found. Using default parameters.")
            
        # Initialize cache for similar data
        self.parameter_cache = deque(maxlen=cache_size)
        self.cache_hits = 0
        self.cache_requests = 0
        
        # Import EEGPreprocessor here to avoid circular import
        from .eeg_preprocessor import EEGPreprocessor
        
        # Initialize base preprocessors with default params
        self.base_preprocessor = EEGPreprocessor(config)
        
        # Create EEGFilter with proper config format
        filter_config = {
            'bandpass': {
                'lowcut': 0.5,
                'highcut': 40.0,
                'order': 4
            },
            'notch': {
                'freqs': [60.0],
                'quality': 30,
                'enabled': True
            }
        }
        self.base_filter = EEGFilter(
            sampling_rate=self.sampling_rate,
            config=filter_config
        )
        self.base_denoiser = AdvancedDenoiser(sampling_rate=self.sampling_rate)
        
        # Metrics tracking
        self.processing_metrics = {
            'total_processed': 0,
            'quality_improvements': [],
            'processing_times': [],
            'parameter_adaptations': []
        }
        
    def preprocess(self, eeg_data: np.ndarray,
                  channel_names: Optional[List[str]] = None,
                  force_default: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Adaptively preprocess EEG data.
        
        Args:
            eeg_data: Raw EEG data (channels x samples)
            channel_names: Optional channel names
            force_default: Force use of default parameters
            
        Returns:
            Tuple of (preprocessed_data, processing_info)
        """
        start_time = time.time()
        
        # Assess initial quality
        initial_quality = self.quality_assessor.assess_quality(eeg_data, channel_names)
        
        if force_default or initial_quality.overall_quality_score > 0.25:
            # Use default preprocessing for high-quality data
            processed_data = self.base_preprocessor.preprocess(eeg_data)
            processing_info = {
                'method': 'default',
                'initial_quality': initial_quality.overall_quality_score,
                'final_quality': initial_quality.overall_quality_score,
                'parameters': 'default'
            }
        else:
            # Adaptive preprocessing for lower quality data
            processed_data, processing_info = self._adaptive_preprocess(
                eeg_data, channel_names, initial_quality
            )
            
        # Update metrics
        processing_time = time.time() - start_time
        self.processing_metrics['total_processed'] += 1
        self.processing_metrics['processing_times'].append(processing_time)
        
        # Add timing to processing info
        processing_info['processing_time'] = processing_time
        processing_info['cache_hit_rate'] = self.cache_hits / max(1, self.cache_requests)
        
        return processed_data, processing_info
    
    def _adaptive_preprocess(self, eeg_data: np.ndarray,
                           channel_names: Optional[List[str]],
                           initial_quality: QualityMetrics) -> Tuple[np.ndarray, Dict]:
        """
        Perform adaptive preprocessing with parameter optimization.
        """
        # Profile the data
        data_profile = self.data_profiler.profile_data(eeg_data, channel_names)
        
        # Add actual channel count to profile
        data_profile['n_channels'] = eeg_data.shape[0]
        
        # Check cache for similar data
        cached_params = self._check_cache(data_profile)
        if cached_params is not None:
            self.cache_hits += 1
            parameters = cached_params
        else:
            # Optimize parameters using neural network
            parameters = self._optimize_parameters(data_profile)
            
            # Cache the parameters
            self._cache_parameters(data_profile, parameters)
            
        # Apply adaptive preprocessing
        processed_data = self._apply_preprocessing(eeg_data, parameters, channel_names)
        
        # Assess final quality
        final_quality = self.quality_assessor.assess_quality(processed_data, channel_names)
        
        # Calculate improvement
        quality_improvement = final_quality.overall_quality_score - initial_quality.overall_quality_score
        self.processing_metrics['quality_improvements'].append(quality_improvement)
        
        # Prepare processing info
        processing_info = {
            'method': 'adaptive',
            'initial_quality': initial_quality.overall_quality_score,
            'final_quality': final_quality.overall_quality_score,
            'quality_improvement': quality_improvement,
            'parameters': asdict(parameters),
            'data_profile': self._summarize_profile(data_profile),
            'optimization_confidence': parameters.optimization_confidence
        }
        
        # Log if quality didn't improve as expected
        if quality_improvement < parameters.expected_quality_improvement * 0.5:
            logger.warning(f"Quality improvement ({quality_improvement:.3f}) below expected "
                         f"({parameters.expected_quality_improvement:.3f})")
            
        return processed_data, processing_info
    
    def _optimize_parameters(self, data_profile: Dict[str, Any]) -> AdaptiveParameters:
        """Optimize preprocessing parameters using neural network."""
        # Convert profile to feature vector
        profile_features = self._profile_to_features(data_profile)
        profile_tensor = torch.tensor(profile_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Set model to eval mode to handle BatchNorm with single samples
        self.optimizer.eval()
        
        # Get predictions from optimizer
        with torch.no_grad():
            predictions = self.optimizer(profile_tensor)
            
        # Decode to parameters
        parameters = self.optimizer.decode_parameters(predictions)
        
        # Apply heuristic adjustments based on profile
        parameters = self._apply_heuristic_adjustments(parameters, data_profile)
        
        return parameters
    
    def _profile_to_features(self, profile: Dict[str, Any]) -> np.ndarray:
        """Convert data profile to feature vector for neural network."""
        features = []
        
        # Basic statistics (8 features)
        basic_stats = profile['basic_stats']
        features.extend([
            basic_stats['mean'], basic_stats['std'],
            basic_stats['median'], basic_stats['mad'],
            basic_stats['skewness'], basic_stats['kurtosis'],
            basic_stats['range'], basic_stats['iqr']
        ])
        
        # Frequency profile (10 features)
        freq_profile = profile['frequency_profile']
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            features.append(freq_profile['band_powers'][band])
        features.extend([
            freq_profile['dominant_frequency'],
            freq_profile['spectral_edge_95'],
            freq_profile['spectral_entropy'],
            freq_profile['high_freq_power'],
            0.0  # Padding
        ])
        
        # Noise profile (5 features)
        noise_profile = profile['noise_profile']
        features.extend([
            noise_profile.get('line_noise_50hz', 0),
            noise_profile.get('line_noise_60hz', 0),
            noise_profile['high_freq_noise'],
            noise_profile['low_freq_drift'],
            noise_profile['white_noise_level']
        ])
        
        # Artifact profile (4 features)
        artifact_profile = profile['artifact_profile']
        features.extend([
            artifact_profile['eye_blink_score'],
            artifact_profile['movement_score'],
            artifact_profile['electrode_pop_score'],
            artifact_profile['muscle_artifact_score']
        ])
        
        # Channel quality (4 features)
        channel_quality = profile['channel_quality']
        features.extend([
            len(channel_quality['dead_channels']),
            len(channel_quality['noisy_channels']),
            channel_quality['channel_variance_ratio'],
            channel_quality['mean_correlation']
        ])
        
        # Stationarity (3 features)
        stationarity = profile['stationarity']
        features.extend([
            stationarity['mean_stationarity'],
            stationarity['variance_stationarity'],
            stationarity['power_stationarity']
        ])
        
        # Connectivity (3 features)
        connectivity = profile['connectivity']
        features.extend([
            connectivity['mean_coherence'],
            connectivity['coherence_std'],
            connectivity['network_density']
        ])
        
        # Pad to expected dimension (50)
        while len(features) < 50:
            features.append(0.0)
            
        return np.array(features[:50], dtype=np.float32)
    
    def _apply_heuristic_adjustments(self, parameters: AdaptiveParameters,
                                    profile: Dict[str, Any]) -> AdaptiveParameters:
        """Apply domain-specific heuristic adjustments to optimized parameters."""
        # Adjust based on line noise
        line_noise_50 = profile['noise_profile'].get('line_noise_50hz', 0)
        line_noise_60 = profile['noise_profile'].get('line_noise_60hz', 0)
        
        if line_noise_50 > 2.0 and (parameters.notch_freq is None or abs(parameters.notch_freq - 50) > 5):
            parameters.notch_freq = 50.0
        elif line_noise_60 > 2.0 and (parameters.notch_freq is None or abs(parameters.notch_freq - 60) > 5):
            parameters.notch_freq = 60.0
            
        # Adjust ICA components based on channel count
        # Get actual channel count from profile
        n_total_channels = profile.get('n_channels', 19)  # Default to 19 if not specified
        
        # Calculate available channels
        channel_quality = profile['channel_quality']
        n_dead_channels = len(channel_quality.get('dead_channels', []))
        
        # Good channels = total - dead channels
        n_good_channels = n_total_channels - n_dead_channels
        
        # ICA components must be less than the number of good channels
        max_ica_components = max(0, n_good_channels - 1)
        parameters.ica_components = min(parameters.ica_components, max_ica_components)
        
        # If we have very few good channels, disable ICA
        if n_good_channels < 8:
            parameters.ica_components = 0  # Disable ICA for too few channels
            
        # Increase artifact threshold for high movement recordings
        if profile['artifact_profile']['movement_score'] > 0.3:
            parameters.artifact_threshold *= 1.5
            
        return parameters
    
    def _apply_preprocessing(self, eeg_data: np.ndarray,
                           parameters: AdaptiveParameters,
                           channel_names: Optional[List[str]]) -> np.ndarray:
        """Apply preprocessing with optimized parameters."""
        processed = eeg_data.copy()
        nyquist = self.sampling_rate / 2
        
        # Step 1: Initial bad channel detection and interpolation
        if parameters.interpolation_method != 'none':
            # This would use MNE or custom interpolation
            # For now, we'll skip actual interpolation
            pass
            
        # Step 2: Filtering
        if parameters.highpass_freq > 0:
            sos = signal.butter(4, parameters.highpass_freq, 'high', 
                              fs=self.sampling_rate, output='sos')
            processed = signal.sosfiltfilt(sos, processed, axis=1)
            
        if parameters.lowpass_freq < nyquist:
            sos = signal.butter(4, parameters.lowpass_freq, 'low',
                              fs=self.sampling_rate, output='sos')
            processed = signal.sosfiltfilt(sos, processed, axis=1)
            
        if parameters.notch_freq is not None:
            Q = parameters.notch_quality
            b, a = signal.iirnotch(parameters.notch_freq, Q, self.sampling_rate)
            processed = signal.filtfilt(b, a, processed, axis=1)
            
        # Step 3: Artifact removal (use ICA only if we have sufficient channels and components > 0)
        if parameters.ica_components > 0 and processed.shape[0] > parameters.ica_components:
            try:
                # Create MNE Raw object for ICA
                if channel_names is None:
                    channel_names = [f'CH_{i}' for i in range(processed.shape[0])]
                    
                # Truncate channel names if necessary
                channel_names = channel_names[:processed.shape[0]]
                
                # Create MNE info and raw object
                info = mne.create_info(
                    ch_names=channel_names,
                    sfreq=self.sampling_rate,
                    ch_types='eeg',
                    verbose=False
                )
                raw = mne.io.RawArray(processed, info, verbose=False)
                
                # Apply ICA
                ica = ICA(
                    n_components=parameters.ica_components,
                    method=parameters.ica_method,
                    random_state=42,
                    max_iter=200,  # Reduced for speed
                    verbose=False
                )
                
                # Fit ICA with error handling
                ica.fit(raw, picks='eeg', verbose=False)
                
                # Apply ICA (simplified - no automatic artifact detection for speed)
                raw_clean = ica.apply(raw.copy(), verbose=False)
                processed = raw_clean.get_data()
                
                logger.debug(f"Applied ICA with {parameters.ica_components} components")
                
            except Exception as e:
                logger.warning(f"ICA failed: {e}. Skipping ICA step.")
                # Continue without ICA
                
        else:
            logger.debug(f"Skipping ICA: components={parameters.ica_components}, channels={processed.shape[0]}")
            
        # Step 4: Simple artifact detection (non-ICA based)
        artifact_mask = self._detect_artifacts_simple(processed, parameters.artifact_threshold)
        
        # Step 5: Wavelet denoising (simplified)
        if parameters.wavelet_type and parameters.wavelet_level > 0:
            # Simplified wavelet denoising
            # In practice, would use pywt for proper denoising
            pass
            
        # Step 6: Normalization
        if parameters.normalization_method == 'robust':
            if parameters.robust_scale:
                median = np.median(processed, axis=1, keepdims=True)
                mad = median_abs_deviation(processed, axis=1)
                mad = mad.reshape(-1, 1)
                processed = (processed - median) / (mad + 1e-10)
        else:
            mean = np.mean(processed, axis=1, keepdims=True)
            std = np.std(processed, axis=1, keepdims=True)
            processed = (processed - mean) / (std + 1e-10)
            
        return processed
    
    def _detect_artifacts_simple(self, data: np.ndarray, threshold: float) -> np.ndarray:
        """Simple artifact detection for preprocessing."""
        # Z-score based detection
        z_scores = np.abs(stats.zscore(data, axis=1))
        return np.any(z_scores > threshold, axis=0)
    
    def _check_cache(self, profile: Dict[str, Any]) -> Optional[AdaptiveParameters]:
        """Check if similar profile exists in cache."""
        self.cache_requests += 1
        
        profile_hash = self._hash_profile(profile)
        
        for cached_hash, cached_params in self.parameter_cache:
            if cached_hash == profile_hash:
                return cached_params
                
        return None
    
    def _cache_parameters(self, profile: Dict[str, Any], parameters: AdaptiveParameters):
        """Cache parameters for similar data."""
        profile_hash = self._hash_profile(profile)
        self.parameter_cache.append((profile_hash, parameters))
        
    def _hash_profile(self, profile: Dict[str, Any]) -> str:
        """Create hash of data profile for caching."""
        # Convert profile to feature vector
        features = self._profile_to_features(profile)
        
        # Quantize features for more cache hits
        quantized = np.round(features, decimals=2)
        
        # Create hash
        return hashlib.md5(quantized.tobytes()).hexdigest()
    
    def _summarize_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of data profile for logging."""
        return {
            'dominant_frequency': profile['frequency_profile']['dominant_frequency'],
            'primary_band': max(profile['frequency_profile']['band_powers'].items(), 
                               key=lambda x: x[1])[0],
            'noise_level': np.mean([v for k, v in profile['noise_profile'].items() 
                                   if isinstance(v, (int, float))]),
            'artifact_level': np.mean([v for v in profile['artifact_profile'].values()]),
            'bad_channels': len(profile['channel_quality']['dead_channels']) + 
                           len(profile['channel_quality']['noisy_channels'])
        }
    
    def train_optimizer(self, training_data: List[Tuple[np.ndarray, QualityMetrics]],
                       epochs: int = 100, learning_rate: float = 1e-3):
        """
        Train the parameter optimizer on labeled data.
        
        Args:
            training_data: List of (eeg_data, quality_metrics) tuples
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        optimizer = torch.optim.Adam(self.optimizer.parameters(), lr=learning_rate)
        
        logger.info(f"Training adaptive preprocessor on {len(training_data)} samples")
        
        for epoch in range(epochs):
            total_loss = 0
            
            for eeg_data, target_quality in training_data:
                # Profile the data
                profile = self.data_profiler.profile_data(eeg_data)
                features = self._profile_to_features(profile)
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Forward pass
                predictions = self.optimizer(features_tensor)
                parameters = self.optimizer.decode_parameters(predictions)
                
                # Apply preprocessing
                processed = self._apply_preprocessing(eeg_data, parameters, None)
                
                # Assess quality
                achieved_quality = self.quality_assessor.assess_quality(processed)
                
                # Compute loss (quality improvement)
                quality_diff = achieved_quality.overall_quality_score - target_quality.overall_quality_score
                loss = -quality_diff  # Negative because we want to maximize improvement
                
                # Add regularization for reasonable parameters
                param_loss = self._parameter_regularization_loss(parameters)
                total_loss += loss + 0.1 * param_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss_tensor = torch.tensor(loss, requires_grad=True)
                loss_tensor.backward()
                optimizer.step()
                
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Average loss = {total_loss / len(training_data):.4f}")
                
    def _parameter_regularization_loss(self, params: AdaptiveParameters) -> float:
        """Compute regularization loss for reasonable parameters."""
        loss = 0.0
        
        # Penalize extreme filter cutoffs
        if params.highpass_freq < 0.1 or params.highpass_freq > 2.0:
            loss += abs(params.highpass_freq - 0.5)
            
        if params.lowpass_freq < 30 or params.lowpass_freq > 100:
            loss += abs(params.lowpass_freq - 50) / 50
            
        # Penalize too many ICA components
        if params.ica_components > 25:
            loss += (params.ica_components - 25) / 10
            
        return loss
    
    def save_optimizer(self, path: Path):
        """Save trained optimizer model."""
        torch.save({
            'model_state_dict': self.optimizer.state_dict(),
            'cache': list(self.parameter_cache),
            'metrics': self.processing_metrics
        }, path)
        logger.info(f"Saved adaptive preprocessor to {path}")
        
    def load_optimizer(self, path: Path):
        """Load trained optimizer model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.optimizer.load_state_dict(checkpoint['model_state_dict'])
        
        if 'cache' in checkpoint:
            self.parameter_cache = deque(checkpoint['cache'], maxlen=self.parameter_cache.maxlen)
            
        if 'metrics' in checkpoint:
            self.processing_metrics.update(checkpoint['metrics'])
            
        logger.info(f"Loaded adaptive preprocessor from {path}")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get preprocessing metrics."""
        metrics = self.processing_metrics.copy()
        
        if metrics['quality_improvements']:
            metrics['avg_quality_improvement'] = np.mean(metrics['quality_improvements'])
            metrics['quality_improvement_std'] = np.std(metrics['quality_improvements'])
            
        if metrics['processing_times']:
            metrics['avg_processing_time'] = np.mean(metrics['processing_times'])
            
        metrics['cache_hit_rate'] = self.cache_hits / max(1, self.cache_requests)
        
        return metrics 