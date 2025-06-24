#!/usr/bin/env python3
"""
Advanced signal filtering and cleaning for EEG data.

This module provides:
- Multi-stage bandpass filtering with linear phase response
- Independent Component Analysis (ICA) for artifact removal
- Wavelet-based denoising
- Adaptive filtering
- Notch filtering for line noise
- Common average referencing
- Spatial filtering
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Callable
import numpy as np
from scipy import signal, stats
from scipy.ndimage import median_filter
import pywt
import mne
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from sklearn.decomposition import FastICA
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class EEGFilter:
    """Advanced EEG signal filtering and cleaning."""
    
    def __init__(self, sampling_rate: float = 200.0, config: Optional[Dict] = None):
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
        
        # Default configuration
        self.config = {
            # Bandpass filter settings
            'bandpass': {
                'lowcut': 0.5,      # Hz
                'highcut': 50.0,    # Hz
                'order': 4,
                'method': 'butterworth',  # 'butterworth', 'chebyshev', 'elliptic'
                'phase': 'zero'     # 'zero' for zero-phase, 'minimum' for causal
            },
            
            # Notch filter settings
            'notch': {
                'freqs': [50.0, 60.0],  # Hz (EU and US power line)
                'quality': 30,          # Quality factor
                'enabled': True
            },
            
            # ICA settings
            'ica': {
                'n_components': 20,
                'method': 'fastica',
                'max_iter': 1000,
                'random_state': 42,
                'artifact_threshold': 3.0  # Z-score threshold
            },
            
            # Wavelet denoising
            'wavelet': {
                'wavelet': 'db4',
                'level': 5,
                'threshold_method': 'soft',  # 'soft' or 'hard'
                'threshold_scale': 'universal',  # 'universal' or 'bayes'
                'noise_estimate': 'mad'  # 'mad' or 'std'
            },
            
            # Adaptive filter
            'adaptive': {
                'enabled': True,
                'mu': 0.01,  # Learning rate
                'order': 10  # Filter order
            },
            
            # Reference settings
            'reference': {
                'type': 'average',  # 'average', 'bipolar', 'laplacian', 'none'
                'exclude_bad': True
            }
        }
        
        if config:
            self.config.update(config)
    
    def apply_filters(self, data: np.ndarray,
                     bad_channels: Optional[List[int]] = None) -> np.ndarray:
        """
        Apply comprehensive filtering pipeline.
        
        Args:
            data: EEG data of shape (n_channels, n_samples)
            bad_channels: List of bad channel indices to exclude
            
        Returns:
            Filtered data of same shape
        """
        logger.info("Applying comprehensive filtering pipeline")
        
        # Copy data to avoid modifying original
        filtered_data = data.copy()
        
        # 1. Remove DC offset
        filtered_data = self._remove_dc_offset(filtered_data)
        
        # 2. Apply notch filter for line noise
        if self.config['notch']['enabled']:
            filtered_data = self._apply_notch_filter(filtered_data)
        
        # 3. Apply bandpass filter
        filtered_data = self._apply_bandpass_filter(filtered_data)
        
        # 4. Apply common average reference (before ICA)
        if self.config['reference']['type'] == 'average':
            filtered_data = self._apply_car(filtered_data, bad_channels)
        
        # 5. Apply ICA for artifact removal
        if data.shape[1] > 1000:  # Only if enough samples
            filtered_data = self._apply_ica(filtered_data, bad_channels)
        
        # 6. Apply wavelet denoising
        filtered_data = self._apply_wavelet_denoising(filtered_data)
        
        # 7. Apply adaptive filtering if enabled
        if self.config['adaptive']['enabled']:
            filtered_data = self._apply_adaptive_filter(filtered_data)
        
        return filtered_data
    
    def _remove_dc_offset(self, data: np.ndarray) -> np.ndarray:
        """Remove DC offset from each channel."""
        return data - np.mean(data, axis=1, keepdims=True)
    
    def _apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter with specified parameters."""
        bp_config = self.config['bandpass']
        
        # Design filter
        if bp_config['method'] == 'butterworth':
            sos = signal.butter(
                bp_config['order'],
                [bp_config['lowcut'], bp_config['highcut']],
                btype='band',
                fs=self.sampling_rate,
                output='sos'
            )
        elif bp_config['method'] == 'chebyshev':
            sos = signal.cheby1(
                bp_config['order'],
                0.5,  # 0.5 dB ripple
                [bp_config['lowcut'], bp_config['highcut']],
                btype='band',
                fs=self.sampling_rate,
                output='sos'
            )
        elif bp_config['method'] == 'elliptic':
            sos = signal.ellip(
                bp_config['order'],
                0.5,  # 0.5 dB ripple
                40,   # 40 dB stopband attenuation
                [bp_config['lowcut'], bp_config['highcut']],
                btype='band',
                fs=self.sampling_rate,
                output='sos'
            )
        else:
            raise ValueError(f"Unknown filter method: {bp_config['method']}")
        
        # Apply filter
        if bp_config['phase'] == 'zero':
            # Zero-phase filtering (non-causal)
            filtered = signal.sosfiltfilt(sos, data, axis=1)
        else:
            # Minimum-phase filtering (causal)
            filtered = signal.sosfilt(sos, data, axis=1)
        
        logger.info(f"Applied {bp_config['method']} bandpass filter "
                   f"({bp_config['lowcut']}-{bp_config['highcut']} Hz)")
        
        return filtered
    
    def _apply_notch_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply notch filter for line noise removal."""
        notch_config = self.config['notch']
        filtered = data.copy()
        
        for freq in notch_config['freqs']:
            # Skip if frequency is above Nyquist
            if freq >= self.nyquist:
                continue
            
            # Design notch filter
            b, a = signal.iirnotch(freq, notch_config['quality'], self.sampling_rate)
            
            # Apply filter (zero-phase)
            filtered = signal.filtfilt(b, a, filtered, axis=1)
            
            logger.info(f"Applied notch filter at {freq} Hz")
        
        return filtered
    
    def _apply_car(self, data: np.ndarray, 
                   bad_channels: Optional[List[int]] = None) -> np.ndarray:
        """Apply Common Average Reference."""
        if bad_channels is None:
            bad_channels = []
        
        # Calculate average of good channels
        good_channels = [i for i in range(data.shape[0]) if i not in bad_channels]
        
        if len(good_channels) < 2:
            logger.warning("Too few good channels for CAR")
            return data
        
        # Calculate and subtract average
        avg_signal = np.mean(data[good_channels, :], axis=0)
        referenced_data = data - avg_signal[np.newaxis, :]
        
        logger.info(f"Applied CAR using {len(good_channels)} channels")
        
        return referenced_data
    
    def _apply_ica(self, data: np.ndarray,
                   bad_channels: Optional[List[int]] = None) -> np.ndarray:
        """Apply ICA for artifact removal."""
        ica_config = self.config['ica']
        
        if bad_channels is None:
            bad_channels = []
        
        # Use only good channels for ICA
        good_channels = [i for i in range(data.shape[0]) if i not in bad_channels]
        
        if len(good_channels) < 10:
            logger.warning("Too few channels for reliable ICA")
            return data
        
        n_components = min(ica_config['n_components'], len(good_channels) - 1)
        
        try:
            # Perform ICA
            ica = FastICA(
                n_components=n_components,
                max_iter=ica_config['max_iter'],
                random_state=ica_config['random_state']
            )
            
            # Fit ICA on good channels
            sources = ica.fit_transform(data[good_channels, :].T).T
            
            # Identify artifact components
            artifact_components = self._identify_artifact_components(
                sources, ica_config['artifact_threshold']
            )
            
            if artifact_components:
                # Zero out artifact components
                sources[artifact_components, :] = 0
                
                # Reconstruct signal
                cleaned_data = ica.inverse_transform(sources.T).T
                
                # Put back into full data array
                data_cleaned = data.copy()
                data_cleaned[good_channels, :] = cleaned_data
                
                logger.info(f"Removed {len(artifact_components)} ICA components")
                
                return data_cleaned
            else:
                logger.info("No artifact components identified")
                return data
                
        except Exception as e:
            logger.error(f"ICA failed: {str(e)}")
            return data
    
    def _identify_artifact_components(self, sources: np.ndarray,
                                    threshold: float) -> List[int]:
        """Identify artifact components using various heuristics."""
        artifact_components = []
        
        for i in range(sources.shape[0]):
            component = sources[i, :]
            
            # Check for various artifact patterns
            # 1. High kurtosis (peaky distributions)
            kurtosis = stats.kurtosis(component)
            if abs(kurtosis) > threshold * 2:
                artifact_components.append(i)
                continue
            
            # 2. Low autocorrelation (high frequency)
            autocorr = np.corrcoef(component[:-1], component[1:])[0, 1]
            if autocorr < 0.5:
                artifact_components.append(i)
                continue
            
            # 3. Extreme values
            z_scores = np.abs(stats.zscore(component))
            if np.max(z_scores) > threshold * 3:
                artifact_components.append(i)
        
        return artifact_components
    
    def _apply_wavelet_denoising(self, data: np.ndarray) -> np.ndarray:
        """Apply wavelet-based denoising."""
        wavelet_config = self.config['wavelet']
        denoised_data = np.zeros_like(data)
        
        for ch in range(data.shape[0]):
            # Wavelet decomposition
            coeffs = pywt.wavedec(
                data[ch, :],
                wavelet_config['wavelet'],
                level=wavelet_config['level']
            )
            
            # Estimate noise level
            if wavelet_config['noise_estimate'] == 'mad':
                # Median Absolute Deviation of finest scale coefficients
                sigma = 1.4826 * np.median(np.abs(coeffs[-1]))
            else:
                # Standard deviation of finest scale coefficients
                sigma = np.std(coeffs[-1])
            
            # Calculate threshold
            if wavelet_config['threshold_scale'] == 'universal':
                # Universal threshold
                threshold = sigma * np.sqrt(2 * np.log(len(data[ch, :])))
            else:
                # Bayes threshold (simplified)
                threshold = sigma * 2
            
            # Apply thresholding to detail coefficients
            if wavelet_config['threshold_method'] == 'soft':
                coeffs[1:] = [pywt.threshold(c, threshold, 'soft') for c in coeffs[1:]]
            else:
                coeffs[1:] = [pywt.threshold(c, threshold, 'hard') for c in coeffs[1:]]
            
            # Reconstruct signal
            denoised_data[ch, :] = pywt.waverec(coeffs, wavelet_config['wavelet'])
            
            # Handle length mismatch from wavelet reconstruction
            if denoised_data[ch, :].shape[0] > data.shape[1]:
                denoised_data[ch, :] = denoised_data[ch, :data.shape[1]]
        
        logger.info("Applied wavelet denoising")
        
        return denoised_data
    
    def _apply_adaptive_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply adaptive filter for non-stationary noise."""
        adaptive_config = self.config['adaptive']
        filtered_data = np.zeros_like(data)
        
        for ch in range(data.shape[0]):
            # Simple LMS adaptive filter
            filtered_data[ch, :] = self._lms_filter(
                data[ch, :],
                mu=adaptive_config['mu'],
                order=adaptive_config['order']
            )
        
        logger.info("Applied adaptive filtering")
        
        return filtered_data
    
    def _lms_filter(self, signal: np.ndarray, mu: float, order: int) -> np.ndarray:
        """Least Mean Squares adaptive filter."""
        n = len(signal)
        filtered = np.zeros(n)
        weights = np.zeros(order)
        
        for i in range(order, n):
            # Get input vector
            x = signal[i-order:i][::-1]
            
            # Filter output
            y = np.dot(weights, x)
            filtered[i] = y
            
            # Error
            e = signal[i] - y
            
            # Update weights
            weights += mu * e * x
        
        # Fill in initial samples
        filtered[:order] = signal[:order]
        
        return filtered
    
    def apply_spatial_filter(self, data: np.ndarray,
                           filter_type: str = 'laplacian') -> np.ndarray:
        """
        Apply spatial filtering.
        
        Args:
            data: EEG data of shape (n_channels, n_samples)
            filter_type: Type of spatial filter ('laplacian', 'car', 'bipolar')
            
        Returns:
            Spatially filtered data
        """
        if filter_type == 'laplacian':
            return self._apply_laplacian_filter(data)
        elif filter_type == 'car':
            return self._apply_car(data)
        elif filter_type == 'bipolar':
            return self._apply_bipolar_filter(data)
        else:
            raise ValueError(f"Unknown spatial filter type: {filter_type}")
    
    def _apply_laplacian_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply surface Laplacian spatial filter."""
        # Simplified Laplacian - each channel minus average of neighbors
        # In practice, would use actual electrode positions
        n_channels = data.shape[0]
        filtered = np.zeros_like(data)
        
        for ch in range(n_channels):
            # Define neighbors (simplified - would use montage in practice)
            if ch == 0:
                neighbors = [1]
            elif ch == n_channels - 1:
                neighbors = [n_channels - 2]
            else:
                neighbors = [ch - 1, ch + 1]
            
            # Laplacian: channel - average of neighbors
            neighbor_avg = np.mean(data[neighbors, :], axis=0)
            filtered[ch, :] = data[ch, :] - neighbor_avg
        
        return filtered
    
    def _apply_bipolar_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bipolar montage."""
        # Sequential bipolar montage
        n_channels = data.shape[0]
        if n_channels < 2:
            return data
        
        filtered = np.zeros((n_channels - 1, data.shape[1]))
        
        for i in range(n_channels - 1):
            filtered[i, :] = data[i, :] - data[i + 1, :]
        
        return filtered
    
    def segment_signal(self, data: np.ndarray,
                      segment_length: float,
                      overlap: float = 0.0) -> List[np.ndarray]:
        """
        Segment signal into fixed-length windows.
        
        Args:
            data: EEG data of shape (n_channels, n_samples)
            segment_length: Length of each segment in seconds
            overlap: Overlap ratio (0-1)
            
        Returns:
            List of segments
        """
        segment_samples = int(segment_length * self.sampling_rate)
        step_samples = int(segment_samples * (1 - overlap))
        
        segments = []
        start = 0
        
        while start + segment_samples <= data.shape[1]:
            segment = data[:, start:start + segment_samples]
            segments.append(segment)
            start += step_samples
        
        logger.info(f"Created {len(segments)} segments of {segment_length}s "
                   f"with {overlap*100}% overlap")
        
        return segments


class AdvancedDenoiser:
    """Advanced denoising techniques for EEG signals."""
    
    def __init__(self, sampling_rate: float = 200.0):
        self.sampling_rate = sampling_rate
    
    def denoise_emd(self, signal: np.ndarray, num_imfs_remove: int = 2) -> np.ndarray:
        """
        Empirical Mode Decomposition denoising.
        
        Note: Requires PyEMD package for full implementation.
        This is a simplified version.
        """
        # Simplified - just remove high-frequency components
        fft_signal = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/self.sampling_rate)
        
        # Remove high frequencies
        cutoff = 40  # Hz
        fft_signal[np.abs(freqs) > cutoff] = 0
        
        return np.real(np.fft.ifft(fft_signal))
    
    def denoise_svd(self, data: np.ndarray, rank: Optional[int] = None) -> np.ndarray:
        """
        Singular Value Decomposition denoising.
        
        Args:
            data: Multi-channel EEG data
            rank: Number of components to keep
            
        Returns:
            Denoised data
        """
        # SVD decomposition
        U, s, Vt = np.linalg.svd(data, full_matrices=False)
        
        # Determine rank if not specified
        if rank is None:
            # Keep components explaining 95% variance
            cumsum = np.cumsum(s**2) / np.sum(s**2)
            rank = np.argmax(cumsum >= 0.95) + 1
        
        # Reconstruct with reduced rank
        s_reduced = s.copy()
        s_reduced[rank:] = 0
        S_reduced = np.diag(s_reduced)
        
        return U @ S_reduced @ Vt
    
    def denoise_kalman(self, signal: np.ndarray,
                      process_variance: float = 1e-5,
                      measurement_variance: float = 1e-1) -> np.ndarray:
        """
        Kalman filter denoising.
        
        Simple 1D Kalman filter for signal denoising.
        """
        n = len(signal)
        filtered = np.zeros(n)
        
        # Initial estimates
        x_est = signal[0]
        p_est = 1.0
        
        for i in range(n):
            # Prediction
            x_pred = x_est
            p_pred = p_est + process_variance
            
            # Update
            k_gain = p_pred / (p_pred + measurement_variance)
            x_est = x_pred + k_gain * (signal[i] - x_pred)
            p_est = (1 - k_gain) * p_pred
            
            filtered[i] = x_est
        
        return filtered


def create_eeg_filter(sampling_rate: float = 200.0,
                     config: Optional[Dict] = None) -> EEGFilter:
    """
    Create configured EEG filter.
    
    Args:
        sampling_rate: Sampling rate in Hz
        config: Optional configuration dictionary
        
    Returns:
        Configured EEGFilter instance
    """
    return EEGFilter(sampling_rate=sampling_rate, config=config) 