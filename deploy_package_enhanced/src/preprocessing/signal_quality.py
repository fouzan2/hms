#!/usr/bin/env python3
"""
Signal quality assessment for EEG data.

This module provides comprehensive quality assessment including:
- Artifact detection (eye blinks, muscle, electrode pops)
- Noise identification (line noise, EMI)
- Bad channel detection
- Signal quality metrics
"""

import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import mne
from mne.preprocessing import find_eog_events, find_ecg_events
import antropy as ant
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Container for signal quality metrics."""
    overall_quality_score: float  # 0-1, higher is better
    snr: float  # Signal-to-noise ratio in dB
    artifact_ratio: float  # Proportion of data with artifacts
    noise_levels: Dict[str, float]  # Noise levels by type
    bad_channels: List[str]  # List of bad channel names
    bad_segments: List[Tuple[float, float]]  # List of (start, end) times
    quality_per_channel: Dict[str, float]  # Quality score per channel


class SignalQualityAssessor:
    """Comprehensive EEG signal quality assessment."""
    
    def __init__(self, sampling_rate: float = 200.0, config: Optional[Dict] = None):
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate / 2
        
        # Default configuration
        self.config = {
            # Artifact detection thresholds
            'amplitude_threshold': 150e-6,  # 150 µV
            'gradient_threshold': 50e-6,    # 50 µV/sample
            'eye_blink_threshold': 100e-6,  # 100 µV
            'muscle_freq_range': (20, 40),  # Hz
            'muscle_threshold_factor': 3.0,  # STD multiplier
            
            # Bad channel detection
            'correlation_threshold': 0.1,    # Min correlation with others
            'std_threshold_factor': 5.0,     # STD multiplier
            'flatline_duration': 1.0,        # seconds
            'noise_threshold_factor': 4.0,   # STD multiplier
            
            # Line noise detection
            'line_freqs': [50, 60],         # Hz (both EU and US)
            'line_noise_threshold': 0.1,     # Relative power
            
            # Quality scoring weights
            'weights': {
                'amplitude': 0.2,
                'noise': 0.3,
                'artifacts': 0.3,
                'connectivity': 0.2
            }
        }
        
        if config:
            self.config.update(config)
    
    def assess_quality(self, data: np.ndarray, 
                      channel_names: Optional[List[str]] = None) -> QualityMetrics:
        """
        Perform comprehensive quality assessment.
        
        Args:
            data: EEG data array of shape (n_channels, n_samples)
            channel_names: List of channel names
            
        Returns:
            QualityMetrics object with assessment results
        """
        n_channels, n_samples = data.shape
        
        if channel_names is None:
            channel_names = [f'ch_{i}' for i in range(n_channels)]
        
        logger.info(f"Assessing signal quality for {n_channels} channels, {n_samples} samples")
        
        # Detect bad channels
        bad_channels, channel_scores = self._detect_bad_channels(data, channel_names)
        
        # Detect artifacts
        artifact_mask = self._detect_artifacts(data)
        artifact_ratio = np.mean(artifact_mask)
        
        # Assess noise levels
        noise_levels = self._assess_noise_levels(data)
        
        # Calculate SNR
        snr = self._calculate_snr(data, artifact_mask)
        
        # Find bad segments
        bad_segments = self._find_bad_segments(artifact_mask)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            data, artifact_ratio, noise_levels, bad_channels, channel_scores
        )
        
        return QualityMetrics(
            overall_quality_score=quality_score,
            snr=snr,
            artifact_ratio=artifact_ratio,
            noise_levels=noise_levels,
            bad_channels=bad_channels,
            bad_segments=bad_segments,
            quality_per_channel=dict(zip(channel_names, channel_scores))
        )
    
    def _detect_bad_channels(self, data: np.ndarray, 
                            channel_names: List[str]) -> Tuple[List[str], np.ndarray]:
        """Detect bad channels using multiple criteria."""
        n_channels = data.shape[0]
        channel_scores = np.ones(n_channels)
        bad_channels = []
        
        # 1. Flat/dead channels
        flat_channels = self._detect_flat_channels(data)
        
        # 2. High amplitude/noisy channels
        noisy_channels = self._detect_noisy_channels(data)
        
        # 3. Low correlation with other channels
        low_corr_channels = self._detect_low_correlation_channels(data)
        
        # 4. High impedance (high frequency noise)
        high_impedance_channels = self._detect_high_impedance_channels(data)
        
        # Combine detections
        for ch_idx in range(n_channels):
            issues = 0
            if ch_idx in flat_channels:
                issues += 2  # Flat is severe
            if ch_idx in noisy_channels:
                issues += 1
            if ch_idx in low_corr_channels:
                issues += 1
            if ch_idx in high_impedance_channels:
                issues += 1
            
            # Score based on issues (0-1, higher is better)
            channel_scores[ch_idx] = max(0, 1 - issues * 0.25)
            
            if issues >= 2:  # Bad if 2+ issues
                bad_channels.append(channel_names[ch_idx])
        
        logger.info(f"Detected {len(bad_channels)} bad channels: {bad_channels}")
        
        return bad_channels, channel_scores
    
    def _detect_flat_channels(self, data: np.ndarray) -> List[int]:
        """Detect flat/dead channels."""
        flat_channels = []
        flatline_samples = int(self.config['flatline_duration'] * self.sampling_rate)
        
        for ch_idx in range(data.shape[0]):
            channel_data = data[ch_idx, :]
            
            # Check for extended flat segments
            diff = np.abs(np.diff(channel_data))
            flat_mask = diff < 1e-10  # Nearly zero change
            
            # Find consecutive flat samples
            flat_lengths = []
            current_length = 0
            for is_flat in flat_mask:
                if is_flat:
                    current_length += 1
                else:
                    if current_length > 0:
                        flat_lengths.append(current_length)
                    current_length = 0
            
            # Check if any flat segment is too long
            if flat_lengths and max(flat_lengths) > flatline_samples:
                flat_channels.append(ch_idx)
        
        return flat_channels
    
    def _detect_noisy_channels(self, data: np.ndarray) -> List[int]:
        """Detect channels with excessive noise."""
        noisy_channels = []
        
        # Calculate robust statistics
        channel_stds = np.std(data, axis=1)
        median_std = np.median(channel_stds)
        mad_std = stats.median_abs_deviation(channel_stds)
        
        # Threshold based on robust statistics
        threshold = median_std + self.config['std_threshold_factor'] * mad_std
        
        for ch_idx in range(data.shape[0]):
            if channel_stds[ch_idx] > threshold:
                noisy_channels.append(ch_idx)
        
        return noisy_channels
    
    def _detect_low_correlation_channels(self, data: np.ndarray) -> List[int]:
        """Detect channels with low correlation to others."""
        low_corr_channels = []
        n_channels = data.shape[0]
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(data)
        
        for ch_idx in range(n_channels):
            # Get correlations with other channels
            correlations = np.abs(corr_matrix[ch_idx, :])
            correlations[ch_idx] = 0  # Exclude self-correlation
            
            # Check if median correlation is too low
            median_corr = np.median(correlations)
            if median_corr < self.config['correlation_threshold']:
                low_corr_channels.append(ch_idx)
        
        return low_corr_channels
    
    def _detect_high_impedance_channels(self, data: np.ndarray) -> List[int]:
        """Detect high impedance channels by high-frequency noise."""
        high_impedance_channels = []
        
        for ch_idx in range(data.shape[0]):
            # Calculate power spectral density
            freqs, psd = signal.welch(data[ch_idx, :], 
                                     fs=self.sampling_rate, 
                                     nperseg=min(256, data.shape[1]//4))
            
            # Check high frequency power (>40 Hz)
            high_freq_mask = freqs > 40
            if np.any(high_freq_mask):
                high_freq_power = np.mean(psd[high_freq_mask])
                total_power = np.mean(psd)
                
                # High impedance if high frequency power is significant
                if high_freq_power / total_power > 0.3:
                    high_impedance_channels.append(ch_idx)
        
        return high_impedance_channels
    
    def _detect_artifacts(self, data: np.ndarray) -> np.ndarray:
        """Detect various types of artifacts."""
        n_channels, n_samples = data.shape
        artifact_mask = np.zeros(n_samples, dtype=bool)
        
        # 1. Amplitude artifacts (clipping, electrode pops)
        amplitude_artifacts = self._detect_amplitude_artifacts(data)
        artifact_mask |= amplitude_artifacts
        
        # 2. Gradient artifacts (sudden changes)
        gradient_artifacts = self._detect_gradient_artifacts(data)
        artifact_mask |= gradient_artifacts
        
        # 3. Eye blink artifacts (if frontal channels available)
        eye_artifacts = self._detect_eye_artifacts(data)
        artifact_mask |= eye_artifacts
        
        # 4. Muscle artifacts
        muscle_artifacts = self._detect_muscle_artifacts(data)
        artifact_mask |= muscle_artifacts
        
        return artifact_mask
    
    def _detect_amplitude_artifacts(self, data: np.ndarray) -> np.ndarray:
        """Detect amplitude-based artifacts."""
        # Check if any channel exceeds threshold
        max_amplitudes = np.max(np.abs(data), axis=0)
        return max_amplitudes > self.config['amplitude_threshold']
    
    def _detect_gradient_artifacts(self, data: np.ndarray) -> np.ndarray:
        """Detect gradient-based artifacts (sudden changes)."""
        # Calculate gradients
        gradients = np.abs(np.diff(data, axis=1))
        max_gradients = np.max(gradients, axis=0)
        
        # Pad to match original length
        artifact_mask = np.zeros(data.shape[1], dtype=bool)
        artifact_mask[1:] = max_gradients > self.config['gradient_threshold']
        
        return artifact_mask
    
    def _detect_eye_artifacts(self, data: np.ndarray) -> np.ndarray:
        """Detect eye blink and movement artifacts."""
        artifact_mask = np.zeros(data.shape[1], dtype=bool)
        
        # Look for frontal channels (typically contain eye artifacts)
        # This is simplified - in practice, would use proper EOG channels
        frontal_data = data[:4, :] if data.shape[0] >= 4 else data
        
        # Detect large amplitude deflections in frontal channels
        frontal_mean = np.mean(frontal_data, axis=0)
        eye_threshold = self.config['eye_blink_threshold']
        
        # Find peaks that could be blinks
        peaks, _ = signal.find_peaks(np.abs(frontal_mean), 
                                    height=eye_threshold,
                                    distance=int(0.2 * self.sampling_rate))
        
        # Mark regions around peaks as artifacts
        for peak in peaks:
            start = max(0, peak - int(0.1 * self.sampling_rate))
            end = min(data.shape[1], peak + int(0.2 * self.sampling_rate))
            artifact_mask[start:end] = True
        
        return artifact_mask
    
    def _detect_muscle_artifacts(self, data: np.ndarray) -> np.ndarray:
        """Detect muscle artifacts using high-frequency power."""
        artifact_mask = np.zeros(data.shape[1], dtype=bool)
        
        # Use sliding window for muscle detection
        window_size = int(0.5 * self.sampling_rate)  # 0.5 second windows
        step_size = window_size // 2
        
        for start in range(0, data.shape[1] - window_size, step_size):
            end = start + window_size
            window_data = data[:, start:end]
            
            # Calculate high-frequency power
            muscle_power = self._calculate_muscle_power(window_data)
            
            # Threshold based on overall power
            total_power = np.mean(np.var(window_data, axis=1))
            if muscle_power / total_power > 0.4:  # High proportion of HF power
                artifact_mask[start:end] = True
        
        return artifact_mask
    
    def _calculate_muscle_power(self, data: np.ndarray) -> float:
        """Calculate power in muscle frequency band."""
        freqs, psd = signal.welch(data, fs=self.sampling_rate, 
                                 nperseg=min(128, data.shape[1]),
                                 axis=1)
        
        # Find muscle frequency band
        muscle_low, muscle_high = self.config['muscle_freq_range']
        muscle_mask = (freqs >= muscle_low) & (freqs <= muscle_high)
        
        if np.any(muscle_mask):
            muscle_power = np.mean(psd[:, muscle_mask])
        else:
            muscle_power = 0.0
        
        return muscle_power
    
    def _assess_noise_levels(self, data: np.ndarray) -> Dict[str, float]:
        """Assess different types of noise in the signal."""
        noise_levels = {}
        
        # Line noise (50/60 Hz)
        for line_freq in self.config['line_freqs']:
            noise_levels[f'line_noise_{line_freq}Hz'] = self._assess_line_noise(data, line_freq)
        
        # White noise (high frequency)
        noise_levels['white_noise'] = self._assess_white_noise(data)
        
        # 1/f noise (pink noise)
        noise_levels['pink_noise'] = self._assess_pink_noise(data)
        
        # Environmental noise (very low frequency)
        noise_levels['environmental_noise'] = self._assess_environmental_noise(data)
        
        return noise_levels
    
    def _assess_line_noise(self, data: np.ndarray, line_freq: float) -> float:
        """Assess line noise at specific frequency."""
        # Calculate average PSD across channels
        freqs, psd = signal.welch(data, fs=self.sampling_rate, 
                                 nperseg=min(512, data.shape[1]//2),
                                 axis=1)
        avg_psd = np.mean(psd, axis=0)
        
        # Find power at line frequency
        freq_idx = np.argmin(np.abs(freqs - line_freq))
        line_power = avg_psd[freq_idx]
        
        # Compare to surrounding frequencies
        surrounding_idxs = [i for i in range(len(freqs)) 
                           if abs(i - freq_idx) > 2 and abs(i - freq_idx) < 10]
        
        if surrounding_idxs:
            surrounding_power = np.mean(avg_psd[surrounding_idxs])
            relative_power = line_power / (surrounding_power + 1e-10)
        else:
            relative_power = 0.0
        
        return relative_power
    
    def _assess_white_noise(self, data: np.ndarray) -> float:
        """Assess white noise level (flat spectrum)."""
        # Calculate PSD
        freqs, psd = signal.welch(data, fs=self.sampling_rate,
                                 nperseg=min(256, data.shape[1]//4),
                                 axis=1)
        avg_psd = np.mean(psd, axis=0)
        
        # Check flatness of spectrum in high frequencies
        high_freq_mask = freqs > 30
        if np.any(high_freq_mask):
            high_freq_psd = avg_psd[high_freq_mask]
            # Coefficient of variation as flatness measure
            cv = np.std(high_freq_psd) / (np.mean(high_freq_psd) + 1e-10)
            white_noise_level = 1.0 - cv  # More flat = more white noise
        else:
            white_noise_level = 0.0
        
        return np.clip(white_noise_level, 0, 1)
    
    def _assess_pink_noise(self, data: np.ndarray) -> float:
        """Assess 1/f (pink) noise level."""
        # Calculate PSD
        freqs, psd = signal.welch(data, fs=self.sampling_rate,
                                 nperseg=min(256, data.shape[1]//4),
                                 axis=1)
        avg_psd = np.mean(psd, axis=0)
        
        # Fit 1/f curve in log-log space
        valid_mask = (freqs > 1) & (freqs < 30)  # Avoid DC and high freq
        if np.any(valid_mask):
            log_freqs = np.log10(freqs[valid_mask])
            log_psd = np.log10(avg_psd[valid_mask] + 1e-10)
            
            # Linear fit in log-log space
            slope, intercept = np.polyfit(log_freqs, log_psd, 1)
            
            # Pink noise has slope around -1
            pink_noise_level = 1.0 - abs(slope + 1.0) / 2.0
        else:
            pink_noise_level = 0.0
        
        return np.clip(pink_noise_level, 0, 1)
    
    def _assess_environmental_noise(self, data: np.ndarray) -> float:
        """Assess very low frequency environmental noise."""
        # High-pass filter to remove very low frequencies
        sos = signal.butter(4, 0.1, 'high', fs=self.sampling_rate, output='sos')
        filtered_data = signal.sosfiltfilt(sos, data, axis=1)
        
        # Compare power before and after filtering
        orig_power = np.mean(np.var(data, axis=1))
        filtered_power = np.mean(np.var(filtered_data, axis=1))
        
        # Environmental noise is the removed power
        env_noise_ratio = (orig_power - filtered_power) / (orig_power + 1e-10)
        
        return np.clip(env_noise_ratio, 0, 1)
    
    def _calculate_snr(self, data: np.ndarray, artifact_mask: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        # Use artifact-free segments as signal
        clean_mask = ~artifact_mask
        
        if np.sum(clean_mask) < 100:  # Too few clean samples
            return 0.0
        
        # Estimate signal power from clean segments
        signal_power = np.mean(np.var(data[:, clean_mask], axis=1))
        
        # Estimate noise using high-pass filtered data
        sos = signal.butter(4, 30, 'high', fs=self.sampling_rate, output='sos')
        noise_data = signal.sosfiltfilt(sos, data, axis=1)
        noise_power = np.mean(np.var(noise_data[:, clean_mask], axis=1))
        
        # Calculate SNR in dB
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        return snr_db
    
    def _find_bad_segments(self, artifact_mask: np.ndarray) -> List[Tuple[float, float]]:
        """Find continuous bad segments in the data."""
        bad_segments = []
        
        # Find transitions
        diff = np.diff(np.concatenate([[False], artifact_mask, [False]]).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        # Convert to time
        for start, end in zip(starts, ends):
            start_time = start / self.sampling_rate
            end_time = end / self.sampling_rate
            bad_segments.append((start_time, end_time))
        
        return bad_segments
    
    def _calculate_quality_score(self, data: np.ndarray, 
                                artifact_ratio: float,
                                noise_levels: Dict[str, float],
                                bad_channels: List[str],
                                channel_scores: np.ndarray) -> float:
        """Calculate overall quality score (0-1, higher is better)."""
        weights = self.config['weights']
        scores = {}
        
        # Amplitude score (penalize extreme values)
        max_amp = np.max(np.abs(data))
        scores['amplitude'] = 1.0 - min(1.0, max_amp / (200e-6))  # 200µV as max reasonable
        
        # Noise score (average of noise metrics)
        noise_values = list(noise_levels.values())
        avg_noise = np.mean(noise_values) if noise_values else 0
        scores['noise'] = 1.0 - min(1.0, avg_noise)
        
        # Artifact score
        scores['artifacts'] = 1.0 - artifact_ratio
        
        # Channel connectivity score
        scores['connectivity'] = np.mean(channel_scores)
        
        # Weighted average
        total_score = sum(weights[key] * scores[key] for key in weights)
        
        return np.clip(total_score, 0, 1)


def assess_signal_quality(data: np.ndarray,
                         sampling_rate: float = 200.0,
                         channel_names: Optional[List[str]] = None,
                         config: Optional[Dict] = None) -> QualityMetrics:
    """
    Convenience function to assess signal quality.
    
    Args:
        data: EEG data array of shape (n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        channel_names: List of channel names
        config: Optional configuration dictionary
        
    Returns:
        QualityMetrics object with assessment results
    """
    assessor = SignalQualityAssessor(sampling_rate=sampling_rate, config=config)
    return assessor.assess_quality(data, channel_names) 