#!/usr/bin/env python3
"""
Comprehensive feature extraction for EEG signals.

This module provides:
- Time-domain features (statistical moments, entropy)
- Frequency-domain features (PSD, spectral edges)
- Time-frequency features (wavelet coefficients)
- Connectivity features (coherence, PLV)
- Nonlinear features (Lyapunov, correlation dimension)
- Clinical features (spike detection, sharp waves)
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from scipy import signal, stats
from scipy.signal import hilbert, find_peaks
from scipy.spatial.distance import pdist, squareform
import pywt
import antropy as ant
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class FeatureSet:
    """Container for extracted features."""
    time_features: Dict[str, np.ndarray]
    frequency_features: Dict[str, np.ndarray]
    time_frequency_features: Dict[str, np.ndarray]
    connectivity_features: Dict[str, np.ndarray]
    nonlinear_features: Dict[str, np.ndarray]
    clinical_features: Dict[str, np.ndarray]
    feature_names: List[str]
    feature_vector: np.ndarray  # Concatenated features


class EEGFeatureExtractor:
    """Comprehensive EEG feature extraction."""
    
    def __init__(self, sampling_rate: float = 200.0, config: Optional[Dict] = None):
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
        
        # Default configuration
        self.config = {
            # Feature selection
            'time_features': [
                'mean', 'std', 'skewness', 'kurtosis',
                'peak_to_peak', 'rms', 'zero_crossings',
                'entropy', 'complexity'
            ],
            'frequency_features': [
                'band_power', 'peak_frequency', 'spectral_edge',
                'spectral_entropy', 'spectral_centroid'
            ],
            'time_frequency_features': [
                'wavelet_energy', 'wavelet_entropy'
            ],
            'connectivity_features': [
                'coherence', 'plv', 'correlation'
            ],
            'nonlinear_features': [
                'hurst', 'dfa', 'sample_entropy'
            ],
            'clinical_features': [
                'spike_rate', 'sharp_wave_rate'
            ],
            
            # Frequency bands
            'frequency_bands': {
                'delta': (0.5, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 50)
            },
            
            # Wavelet settings
            'wavelet': {
                'name': 'db4',
                'levels': 5
            },
            
            # Clinical detection settings
            'spike_detection': {
                'min_amplitude': 50e-6,  # 50 µV
                'max_duration': 0.07,    # 70 ms
                'sharpness_threshold': 2.0
            },
            
            # Feature normalization
            'normalize': True,
            'normalization_method': 'robust'  # 'standard', 'minmax', 'robust'
        }
        
        if config:
            self.config.update(config)
    
    def extract_features(self, data: np.ndarray,
                        channel_names: Optional[List[str]] = None) -> FeatureSet:
        """
        Extract comprehensive feature set from EEG data.
        
        Args:
            data: EEG data of shape (n_channels, n_samples)
            channel_names: Optional list of channel names
            
        Returns:
            FeatureSet object containing all extracted features
        """
        n_channels, n_samples = data.shape
        
        if channel_names is None:
            channel_names = [f'ch_{i}' for i in range(n_channels)]
        
        logger.info(f"Extracting features from {n_channels} channels, {n_samples} samples")
        
        # Extract different feature types
        time_features = self._extract_time_features(data)
        frequency_features = self._extract_frequency_features(data)
        time_frequency_features = self._extract_time_frequency_features(data)
        connectivity_features = self._extract_connectivity_features(data)
        nonlinear_features = self._extract_nonlinear_features(data)
        clinical_features = self._extract_clinical_features(data)
        
        # Create feature vector
        feature_vector, feature_names = self._concatenate_features(
            time_features, frequency_features, time_frequency_features,
            connectivity_features, nonlinear_features, clinical_features,
            channel_names=channel_names
        )
        
        # Normalize if requested
        if self.config['normalize']:
            feature_vector = self._normalize_features(feature_vector)
        
        return FeatureSet(
            time_features=time_features,
            frequency_features=frequency_features,
            time_frequency_features=time_frequency_features,
            connectivity_features=connectivity_features,
            nonlinear_features=nonlinear_features,
            clinical_features=clinical_features,
            feature_names=feature_names,
            feature_vector=feature_vector
        )
    
    def _extract_time_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract time-domain features."""
        features = {}
        n_channels = data.shape[0]
        
        # Basic statistical features
        if 'mean' in self.config['time_features']:
            features['mean'] = np.mean(data, axis=1)
        
        if 'std' in self.config['time_features']:
            features['std'] = np.std(data, axis=1)
        
        if 'skewness' in self.config['time_features']:
            features['skewness'] = stats.skew(data, axis=1)
        
        if 'kurtosis' in self.config['time_features']:
            features['kurtosis'] = stats.kurtosis(data, axis=1)
        
        if 'peak_to_peak' in self.config['time_features']:
            features['peak_to_peak'] = np.ptp(data, axis=1)
        
        if 'rms' in self.config['time_features']:
            features['rms'] = np.sqrt(np.mean(data**2, axis=1))
        
        # Zero crossings
        if 'zero_crossings' in self.config['time_features']:
            zero_crossings = np.zeros(n_channels)
            for ch in range(n_channels):
                zero_crossings[ch] = np.sum(np.diff(np.signbit(data[ch, :])))
            features['zero_crossings'] = zero_crossings
        
        # Entropy measures
        if 'entropy' in self.config['time_features']:
            entropy = np.zeros(n_channels)
            for ch in range(n_channels):
                entropy[ch] = ant.spectral_entropy(
                    data[ch, :], sf=self.sampling_rate, method='welch'
                )
            features['entropy'] = entropy
        
        # Complexity measures
        if 'complexity' in self.config['time_features']:
            complexity = np.zeros(n_channels)
            for ch in range(n_channels):
                complexity[ch] = ant.hjorth_params(data[ch, :])[1]  # Hjorth complexity
            features['complexity'] = complexity
        
        return features
    
    def _extract_frequency_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract frequency-domain features."""
        features = {}
        n_channels = data.shape[0]
        
        # Calculate PSD for all channels
        freqs, psd = signal.welch(data, fs=self.sampling_rate,
                                 nperseg=min(256, data.shape[1]//4),
                                 axis=1)
        
        # Band power features
        if 'band_power' in self.config['frequency_features']:
            for band_name, (low, high) in self.config['frequency_bands'].items():
                band_mask = (freqs >= low) & (freqs <= high)
                if np.any(band_mask):
                    band_power = np.mean(psd[:, band_mask], axis=1)
                    features[f'band_power_{band_name}'] = band_power
                    
                    # Relative band power
                    total_power = np.mean(psd, axis=1)
                    features[f'relative_band_power_{band_name}'] = band_power / (total_power + 1e-10)
        
        # Peak frequency
        if 'peak_frequency' in self.config['frequency_features']:
            peak_freq = np.zeros(n_channels)
            for ch in range(n_channels):
                peak_idx = np.argmax(psd[ch, :])
                peak_freq[ch] = freqs[peak_idx]
            features['peak_frequency'] = peak_freq
        
        # Spectral edge frequency (95% of power)
        if 'spectral_edge' in self.config['frequency_features']:
            spectral_edge = np.zeros(n_channels)
            for ch in range(n_channels):
                cumsum_psd = np.cumsum(psd[ch, :])
                cumsum_psd = cumsum_psd / cumsum_psd[-1]
                edge_idx = np.argmax(cumsum_psd >= 0.95)
                spectral_edge[ch] = freqs[edge_idx]
            features['spectral_edge_95'] = spectral_edge
        
        # Spectral entropy
        if 'spectral_entropy' in self.config['frequency_features']:
            spectral_entropy = np.zeros(n_channels)
            for ch in range(n_channels):
                # Normalize PSD to probability distribution
                psd_norm = psd[ch, :] / np.sum(psd[ch, :])
                spectral_entropy[ch] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            features['spectral_entropy'] = spectral_entropy
        
        # Spectral centroid
        if 'spectral_centroid' in self.config['frequency_features']:
            spectral_centroid = np.zeros(n_channels)
            for ch in range(n_channels):
                spectral_centroid[ch] = np.sum(freqs * psd[ch, :]) / np.sum(psd[ch, :])
            features['spectral_centroid'] = spectral_centroid
        
        return features
    
    def _extract_time_frequency_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract time-frequency domain features using wavelets."""
        features = {}
        n_channels = data.shape[0]
        
        wavelet_name = self.config['wavelet']['name']
        wavelet_levels = self.config['wavelet']['levels']
        
        # Wavelet decomposition features
        if 'wavelet_energy' in self.config['time_frequency_features']:
            # Energy in each wavelet band
            for level in range(1, wavelet_levels + 1):
                energy = np.zeros(n_channels)
                for ch in range(n_channels):
                    coeffs = pywt.wavedec(data[ch, :], wavelet_name, level=level)
                    # Energy of detail coefficients at this level
                    if level <= len(coeffs) - 1:
                        energy[ch] = np.sum(coeffs[level]**2)
                features[f'wavelet_energy_level_{level}'] = energy
        
        # Wavelet entropy
        if 'wavelet_entropy' in self.config['time_frequency_features']:
            wavelet_entropy = np.zeros(n_channels)
            for ch in range(n_channels):
                coeffs = pywt.wavedec(data[ch, :], wavelet_name, level=wavelet_levels)
                # Calculate entropy across wavelet coefficients
                energies = [np.sum(c**2) for c in coeffs]
                total_energy = np.sum(energies)
                if total_energy > 0:
                    probs = np.array(energies) / total_energy
                    wavelet_entropy[ch] = -np.sum(probs * np.log2(probs + 1e-10))
            features['wavelet_entropy'] = wavelet_entropy
        
        return features
    
    def _extract_connectivity_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract inter-channel connectivity features."""
        features = {}
        n_channels = data.shape[0]
        
        # Only compute if we have multiple channels
        if n_channels < 2:
            return features
        
        # Correlation matrix
        if 'correlation' in self.config['connectivity_features']:
            corr_matrix = np.corrcoef(data)
            # Extract upper triangle (excluding diagonal)
            upper_tri_indices = np.triu_indices(n_channels, k=1)
            features['correlation'] = corr_matrix[upper_tri_indices]
            features['mean_correlation'] = np.array([np.mean(np.abs(features['correlation']))])
        
        # Phase Locking Value (PLV)
        if 'plv' in self.config['connectivity_features']:
            plv_matrix = self._calculate_plv(data)
            upper_tri_indices = np.triu_indices(n_channels, k=1)
            features['plv'] = plv_matrix[upper_tri_indices]
            features['mean_plv'] = np.array([np.mean(features['plv'])])
        
        # Coherence (simplified - just for alpha band)
        if 'coherence' in self.config['connectivity_features']:
            coherence_values = []
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    f, Cxy = signal.coherence(data[i, :], data[j, :],
                                            fs=self.sampling_rate,
                                            nperseg=min(128, data.shape[1]//4))
                    # Average coherence in alpha band
                    alpha_mask = (f >= 8) & (f <= 13)
                    if np.any(alpha_mask):
                        coherence_values.append(np.mean(Cxy[alpha_mask]))
            
            if coherence_values:
                features['alpha_coherence'] = np.array(coherence_values)
                features['mean_alpha_coherence'] = np.array([np.mean(coherence_values)])
        
        return features
    
    def _calculate_plv(self, data: np.ndarray) -> np.ndarray:
        """Calculate Phase Locking Value between channels."""
        n_channels, n_samples = data.shape
        plv_matrix = np.zeros((n_channels, n_channels))
        
        # Get instantaneous phase using Hilbert transform
        phases = np.zeros_like(data)
        for ch in range(n_channels):
            analytic_signal = hilbert(data[ch, :])
            phases[ch, :] = np.angle(analytic_signal)
        
        # Calculate PLV for each channel pair
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                phase_diff = phases[i, :] - phases[j, :]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_matrix[i, j] = plv
                plv_matrix[j, i] = plv
        
        return plv_matrix
    
    def _extract_nonlinear_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract nonlinear dynamics features."""
        features = {}
        n_channels = data.shape[0]
        
        # Hurst exponent
        if 'hurst' in self.config['nonlinear_features']:
            hurst = np.zeros(n_channels)
            for ch in range(n_channels):
                try:
                    hurst[ch] = ant.detrended_fluctuation(data[ch, :])
                except:
                    hurst[ch] = 0.5  # Default to random walk
            features['hurst_exponent'] = hurst
        
        # Detrended Fluctuation Analysis
        if 'dfa' in self.config['nonlinear_features']:
            dfa = np.zeros(n_channels)
            for ch in range(n_channels):
                try:
                    dfa[ch] = ant.detrended_fluctuation(data[ch, :])
                except:
                    dfa[ch] = 0.5
            features['dfa'] = dfa
        
        # Sample entropy
        if 'sample_entropy' in self.config['nonlinear_features']:
            sample_entropy = np.zeros(n_channels)
            for ch in range(n_channels):
                try:
                    sample_entropy[ch] = ant.sample_entropy(data[ch, :])
                except:
                    sample_entropy[ch] = 0.0
            features['sample_entropy'] = sample_entropy
        
        return features
    
    def _extract_clinical_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract clinically relevant EEG features."""
        features = {}
        n_channels = data.shape[0]
        
        # Spike detection
        if 'spike_rate' in self.config['clinical_features']:
            spike_rates = np.zeros(n_channels)
            for ch in range(n_channels):
                spikes = self._detect_spikes(data[ch, :])
                spike_rates[ch] = len(spikes) / (data.shape[1] / self.sampling_rate)
            features['spike_rate'] = spike_rates
        
        # Sharp wave detection
        if 'sharp_wave_rate' in self.config['clinical_features']:
            sharp_wave_rates = np.zeros(n_channels)
            for ch in range(n_channels):
                sharp_waves = self._detect_sharp_waves(data[ch, :])
                sharp_wave_rates[ch] = len(sharp_waves) / (data.shape[1] / self.sampling_rate)
            features['sharp_wave_rate'] = sharp_wave_rates
        
        return features
    
    def _detect_spikes(self, signal: np.ndarray) -> List[int]:
        """Detect epileptiform spikes in EEG signal."""
        spike_config = self.config['spike_detection']
        
        # Calculate signal derivative (sharpness)
        derivative = np.diff(signal)
        
        # Find peaks in absolute derivative (sharp changes)
        peaks, properties = find_peaks(
            np.abs(derivative),
            height=spike_config['min_amplitude'],
            width=(1, int(spike_config['max_duration'] * self.sampling_rate))
        )
        
        # Filter by sharpness
        spikes = []
        for peak in peaks:
            if peak > 0 and peak < len(signal) - 1:
                # Check sharpness (second derivative)
                sharpness = abs(derivative[peak] - derivative[peak-1])
                if sharpness > spike_config['sharpness_threshold'] * np.std(derivative):
                    spikes.append(peak)
        
        return spikes
    
    def _detect_sharp_waves(self, signal: np.ndarray) -> List[int]:
        """Detect sharp waves (slower than spikes)."""
        # Similar to spike detection but with different parameters
        # Sharp waves are typically 70-200ms in duration
        
        # Smooth signal slightly
        smoothed = signal  # Could apply light smoothing here
        
        # Find peaks
        peaks, _ = find_peaks(
            np.abs(smoothed),
            height=30e-6,  # 30 µV
            width=(int(0.07 * self.sampling_rate), int(0.2 * self.sampling_rate))
        )
        
        return peaks.tolist()
    
    def _concatenate_features(self, *feature_dicts, channel_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Concatenate all features into a single vector."""
        all_features = []
        feature_names = []
        
        for feature_dict in feature_dicts:
            for feature_name, feature_values in feature_dict.items():
                if feature_values.ndim == 1:
                    # Channel-wise features
                    if len(feature_values) == len(channel_names):
                        for ch_idx, ch_name in enumerate(channel_names):
                            all_features.append(feature_values[ch_idx])
                            feature_names.append(f"{feature_name}_{ch_name}")
                    else:
                        # Global features
                        all_features.extend(feature_values)
                        if len(feature_values) == 1:
                            feature_names.append(feature_name)
                        else:
                            for i in range(len(feature_values)):
                                feature_names.append(f"{feature_name}_{i}")
                else:
                    # Multi-dimensional features (flatten)
                    all_features.extend(feature_values.flatten())
                    for i in range(feature_values.size):
                        feature_names.append(f"{feature_name}_{i}")
        
        return np.array(all_features), feature_names
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using specified method."""
        method = self.config['normalization_method']
        
        if method == 'standard':
            # Z-score normalization
            mean = np.mean(features)
            std = np.std(features)
            if std > 0:
                return (features - mean) / std
            else:
                return features - mean
        
        elif method == 'minmax':
            # Min-max normalization
            min_val = np.min(features)
            max_val = np.max(features)
            if max_val > min_val:
                return (features - min_val) / (max_val - min_val)
            else:
                return features - min_val
        
        elif method == 'robust':
            # Robust normalization (using median and MAD)
            median = np.median(features)
            mad = np.median(np.abs(features - median))
            if mad > 0:
                return (features - median) / (1.4826 * mad)
            else:
                return features - median
        
        else:
            return features


class FeatureSelector:
    """Feature selection for EEG classification."""
    
    def __init__(self, method: str = 'mutual_info', n_features: int = 50):
        self.method = method
        self.n_features = n_features
        self.selector = None
        self.selected_indices = None
        
    def fit_select(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit selector and transform features."""
        if self.method == 'mutual_info':
            self.selector = SelectKBest(
                score_func=mutual_info_classif,
                k=min(self.n_features, features.shape[1])
            )
        elif self.method == 'f_test':
            self.selector = SelectKBest(
                score_func=f_classif,
                k=min(self.n_features, features.shape[1])
            )
        elif self.method == 'pca':
            self.selector = PCA(n_components=min(self.n_features, features.shape[1]))
        else:
            raise ValueError(f"Unknown selection method: {self.method}")
        
        selected_features = self.selector.fit_transform(features, labels)
        
        # Store selected indices for feature names
        if hasattr(self.selector, 'get_support'):
            self.selected_indices = self.selector.get_support(indices=True)
        
        return selected_features
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted selector."""
        if self.selector is None:
            raise ValueError("Selector not fitted yet")
        
        return self.selector.transform(features)
    
    def get_selected_feature_names(self, all_feature_names: List[str]) -> List[str]:
        """Get names of selected features."""
        if self.selected_indices is not None:
            return [all_feature_names[i] for i in self.selected_indices]
        else:
            # For PCA, create component names
            return [f"PC_{i+1}" for i in range(self.selector.n_components_)]


def extract_features(data: np.ndarray,
                    sampling_rate: float = 200.0,
                    channel_names: Optional[List[str]] = None,
                    config: Optional[Dict] = None) -> FeatureSet:
    """
    Convenience function to extract features.
    
    Args:
        data: EEG data of shape (n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        channel_names: Optional channel names
        config: Optional configuration
        
    Returns:
        FeatureSet with extracted features
    """
    extractor = EEGFeatureExtractor(sampling_rate=sampling_rate, config=config)
    return extractor.extract_features(data, channel_names) 