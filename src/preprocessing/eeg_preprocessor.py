"""
Comprehensive EEG preprocessing pipeline for HMS Brain Activity Classification.
Includes artifact removal, filtering, denoising, and normalization.
Now with adaptive preprocessing support for automatic parameter optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import mne
from mne.preprocessing import ICA
from scipy import signal
from scipy.stats import zscore
import pywt
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import logging
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

from .adaptive_preprocessor import AdaptivePreprocessor

logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """Comprehensive EEG preprocessing pipeline with adaptive optimization support."""
    
    def __init__(self, config_path: Union[str, Dict] = "config/config.yaml", use_adaptive: bool = False):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config_path: Path to configuration file or configuration dictionary
            use_adaptive: Whether to use adaptive preprocessing
        """
        if isinstance(config_path, dict):
            # Config passed as dictionary
            self.config = config_path
        else:
            # Config passed as file path
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
        self.preprocessing_config = self.config['preprocessing']
        self.eeg_config = self.config['eeg']
        self.sampling_rate = self.config['dataset']['eeg_sampling_rate']
        
        # Initialize scalers
        self.scaler = self._init_scaler()
        
        # Initialize adaptive preprocessor if enabled
        self.use_adaptive = use_adaptive
        if use_adaptive:
            self.adaptive_preprocessor = AdaptivePreprocessor(self.config)
            logger.info("Adaptive preprocessing enabled")
        else:
            self.adaptive_preprocessor = None
            
    def _init_scaler(self):
        """Initialize the appropriate scaler based on configuration."""
        method = self.preprocessing_config['normalization']['method']
        
        if method == 'robust':
            return RobustScaler()
        elif method == 'standard':
            return StandardScaler()
        elif method == 'minmax':
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
            
    def create_mne_raw(self, eeg_data: np.ndarray, channel_names: List[str]) -> mne.io.RawArray:
        """Create MNE Raw object from numpy array."""
        if eeg_data.shape[0] != len(channel_names):
            eeg_data = eeg_data.T
            
        # Create info object
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=self.sampling_rate,
            ch_types='eeg'
        )
        
        # Create Raw object
        raw = mne.io.RawArray(eeg_data, info)
        
        # Set standard montage (handle gracefully if missing electrode positions)
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, on_missing='ignore')
        except Exception as e:
            logger.warning(f"Could not set montage: {e}. Continuing without electrode positions.")
        
        return raw
        
    def apply_bandpass_filter(self, raw: mne.io.RawArray) -> mne.io.RawArray:
        """Apply bandpass filter to remove low and high frequency noise."""
        filter_config = self.preprocessing_config['filter']
        
        logger.info(f"Applying bandpass filter: {filter_config['lowcut']}-{filter_config['highcut']} Hz")
        
        raw_filtered = raw.copy().filter(
            l_freq=filter_config['lowcut'],
            h_freq=filter_config['highcut'],
            fir_design='firwin',
            skip_by_annotation='edge'
        )
        
        return raw_filtered
        
    def apply_notch_filter(self, raw: mne.io.RawArray, freqs: List[float] = [50, 60]) -> mne.io.RawArray:
        """Apply notch filter to remove power line interference."""
        logger.info(f"Applying notch filter at {freqs} Hz")
        
        raw_notched = raw.copy()
        for freq in freqs:
            raw_notched = raw_notched.notch_filter(
                freqs=freq,
                picks='eeg',
                fir_design='firwin',
                verbose=False
            )
            
        return raw_notched
        
    def detect_bad_channels(self, raw: mne.io.RawArray) -> List[str]:
        """Automatically detect bad channels using various criteria."""
        data = raw.get_data()
        bad_channels = []
        
        # Criterion 1: Flat channels (very low variance)
        channel_vars = np.var(data, axis=1)
        flat_threshold = np.percentile(channel_vars, 5)
        flat_channels = np.where(channel_vars < flat_threshold)[0]
        
        # Criterion 2: Noisy channels (very high variance)
        noisy_threshold = np.percentile(channel_vars, 95)
        noisy_channels = np.where(channel_vars > noisy_threshold)[0]
        
        # Criterion 3: Channels with excessive high-frequency noise
        high_freq_power = []
        for ch_idx in range(data.shape[0]):
            freqs, psd = signal.welch(data[ch_idx], fs=self.sampling_rate)
            high_freq_mask = freqs > 40
            high_freq_power.append(np.mean(psd[high_freq_mask]))
            
        high_freq_threshold = np.percentile(high_freq_power, 95)
        high_freq_channels = np.where(np.array(high_freq_power) > high_freq_threshold)[0]
        
        # Combine all bad channels
        bad_idx = np.unique(np.concatenate([flat_channels, noisy_channels, high_freq_channels]))
        bad_channels = [raw.ch_names[idx] for idx in bad_idx]
        
        logger.info(f"Detected {len(bad_channels)} bad channels: {bad_channels}")
        
        return bad_channels
        
    def interpolate_bad_channels(self, raw: mne.io.RawArray, bad_channels: List[str]) -> mne.io.RawArray:
        """Interpolate bad channels using spherical splines."""
        if not bad_channels:
            return raw
            
        raw_interp = raw.copy()
        raw_interp.info['bads'] = bad_channels
        
        logger.info(f"Interpolating {len(bad_channels)} bad channels")
        
        try:
            raw_interp.interpolate_bads(reset_bads=True)
        except Exception as e:
            logger.warning(f"Could not interpolate bad channels: {e}. Skipping interpolation.")
            # Remove bad channels from bads list if interpolation fails
            raw_interp.info['bads'] = []
        
        return raw_interp
        
    def remove_artifacts_ica(self, raw: mne.io.RawArray) -> mne.io.RawArray:
        """Remove artifacts using Independent Component Analysis."""
        if not self.preprocessing_config['artifact_removal']['use_ica']:
            return raw
            
        logger.info("Applying ICA for artifact removal")
        
        # Configure ICA
        n_components = self.preprocessing_config['artifact_removal']['n_components']
        method = self.preprocessing_config['artifact_removal']['ica_method']
        
        ica = ICA(
            n_components=n_components,
            method=method,
            random_state=42,
            max_iter=500
        )
        
        # Fit ICA
        ica.fit(raw, picks='eeg')
        
        # Automatically detect EOG components
        eog_channels = self.preprocessing_config['artifact_removal']['eog_channels']
        eog_indices = []
        
        for eog_ch in eog_channels:
            if eog_ch in raw.ch_names:
                eog_inds, _ = ica.find_bads_eog(raw, ch_name=eog_ch)
                eog_indices.extend(eog_inds)
                
        # Detect muscle artifacts
        muscle_idx, _ = ica.find_bads_muscle(raw)
        
        # Combine all artifact components
        exclude_idx = list(set(eog_indices + muscle_idx))
        logger.info(f"Excluding {len(exclude_idx)} ICA components")
        
        # Apply ICA
        ica.exclude = exclude_idx
        raw_clean = ica.apply(raw.copy())
        
        return raw_clean
        
    def apply_wavelet_denoising(self, data: np.ndarray) -> np.ndarray:
        """Apply wavelet denoising to EEG signals."""
        if not self.preprocessing_config['denoising']['use_wavelet']:
            return data
            
        logger.info("Applying wavelet denoising")
        
        wavelet = self.preprocessing_config['denoising']['wavelet_type']
        level = self.preprocessing_config['denoising']['wavelet_level']
        
        denoised_data = np.zeros_like(data)
        
        for ch_idx in range(data.shape[0]):
            # Decompose signal
            coeffs = pywt.wavedec(data[ch_idx], wavelet, level=level)
            
            # Estimate noise level using MAD
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            
            # Universal threshold
            threshold = sigma * np.sqrt(2 * np.log(len(data[ch_idx])))
            
            # Soft thresholding
            coeffs_thresh = list(coeffs)
            for i in range(1, len(coeffs)):
                coeffs_thresh[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
                
            # Reconstruct signal
            denoised_data[ch_idx] = pywt.waverec(coeffs_thresh, wavelet)[:len(data[ch_idx])]
            
        return denoised_data
        
    def remove_baseline_drift(self, data: np.ndarray) -> np.ndarray:
        """Remove baseline drift using high-pass filtering."""
        logger.info("Removing baseline drift")
        
        # Design high-pass filter
        b, a = signal.butter(
            4,
            0.5 / (self.sampling_rate / 2),
            btype='high'
        )
        
        # Apply filter
        filtered_data = np.zeros_like(data)
        for ch_idx in range(data.shape[0]):
            filtered_data[ch_idx] = signal.filtfilt(b, a, data[ch_idx])
            
        return filtered_data
        
    def normalize_data(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize EEG data using configured method."""
        logger.info(f"Normalizing data using {self.preprocessing_config['normalization']['method']} method")
        
        # Reshape for scaler
        n_channels, n_samples = data.shape
        data_reshaped = data.T
        
        if fit:
            normalized = self.scaler.fit_transform(data_reshaped)
        else:
            normalized = self.scaler.transform(data_reshaped)
            
        return normalized.T
        
    def detect_and_remove_spikes(self, data: np.ndarray, threshold: float = 5.0) -> np.ndarray:
        """Detect and remove spike artifacts."""
        logger.info("Detecting and removing spike artifacts")
        
        cleaned_data = data.copy()
        
        for ch_idx in range(data.shape[0]):
            channel_data = data[ch_idx]
            
            # Calculate z-scores
            z_scores = np.abs(zscore(channel_data))
            
            # Detect spikes
            spike_mask = z_scores > threshold
            
            if np.any(spike_mask):
                # Interpolate spike regions
                spike_indices = np.where(spike_mask)[0]
                
                # Group consecutive spikes
                spike_groups = []
                current_group = [spike_indices[0]]
                
                for i in range(1, len(spike_indices)):
                    if spike_indices[i] - spike_indices[i-1] == 1:
                        current_group.append(spike_indices[i])
                    else:
                        spike_groups.append(current_group)
                        current_group = [spike_indices[i]]
                spike_groups.append(current_group)
                
                # Interpolate each spike group
                for group in spike_groups:
                    if len(group) < len(channel_data) * 0.1:  # Only if spike is less than 10% of signal
                        start_idx = max(0, group[0] - 10)
                        end_idx = min(len(channel_data), group[-1] + 10)
                        
                        # Linear interpolation
                        x = np.arange(len(channel_data))
                        mask = np.ones_like(channel_data, dtype=bool)
                        mask[group] = False
                        
                        cleaned_data[ch_idx, group] = np.interp(
                            group,
                            x[mask],
                            channel_data[mask]
                        )
                        
        return cleaned_data
        
    def preprocess_eeg(self, 
                      eeg_data: np.ndarray,
                      channel_names: List[str],
                      apply_ica: bool = True,
                      remove_bad_channels: bool = True,
                      use_adaptive: Optional[bool] = None) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Complete preprocessing pipeline for EEG data.
        
        Args:
            eeg_data: Raw EEG data array
            channel_names: List of channel names
            apply_ica: Whether to apply ICA
            remove_bad_channels: Whether to remove bad channels
            use_adaptive: Override adaptive preprocessing setting
            
        Returns:
            Tuple of (preprocessed_data, preprocessing_info)
        """
        # Determine whether to use adaptive preprocessing
        use_adaptive_here = use_adaptive if use_adaptive is not None else self.use_adaptive
        
        if use_adaptive_here and self.adaptive_preprocessor is not None:
            logger.info("Using adaptive preprocessing")
            preprocessed_data, processing_info = self.adaptive_preprocessor.preprocess(
                eeg_data, channel_names
            )
            return preprocessed_data, processing_info
        else:
            # Standard preprocessing pipeline
            logger.info("Starting standard EEG preprocessing pipeline")
            
            # Create MNE Raw object
            raw = self.create_mne_raw(eeg_data, channel_names)
            
            # 1. Detect and interpolate bad channels
            if remove_bad_channels:
                bad_channels = self.detect_bad_channels(raw)
                raw = self.interpolate_bad_channels(raw, bad_channels)
                
            # 2. Apply bandpass filter
            raw = self.apply_bandpass_filter(raw)
            
            # 3. Apply notch filter
            raw = self.apply_notch_filter(raw)
            
            # 4. Remove artifacts with ICA
            if apply_ica:
                raw = self.remove_artifacts_ica(raw)
                
            # Get data array
            data = raw.get_data()
            
            # 5. Apply wavelet denoising
            data = self.apply_wavelet_denoising(data)
            
            # 6. Remove baseline drift
            data = self.remove_baseline_drift(data)
            
            # 7. Detect and remove spikes
            data = self.detect_and_remove_spikes(data)
            
            # 8. Normalize data
            data = self.normalize_data(data)
            
            logger.info("EEG preprocessing completed")
            
            # Create processing info
            processing_info = {
                'method': 'standard',
                'bad_channels': bad_channels if remove_bad_channels else [],
                'applied_ica': apply_ica,
                'normalization_method': self.preprocessing_config['normalization']['method']
            }
            
            return data, processing_info
            
    def preprocess(self, eeg_data: np.ndarray, 
                  channel_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Backward compatible preprocessing method.
        
        Args:
            eeg_data: Raw EEG data
            channel_names: Optional channel names
            
        Returns:
            Preprocessed data
        """
        if channel_names is None:
            channel_names = [f'CH_{i}' for i in range(eeg_data.shape[0])]
            
        preprocessed, _ = self.preprocess_eeg(eeg_data, channel_names)
        return preprocessed
        
    def preprocess_batch(self, 
                        eeg_batch: List[np.ndarray],
                        channel_names: List[str],
                        n_jobs: int = -1,
                        use_adaptive: Optional[bool] = None) -> List[Tuple[np.ndarray, Dict]]:
        """
        Preprocess a batch of EEG recordings in parallel.
        
        Args:
            eeg_batch: List of EEG recordings
            channel_names: Channel names
            n_jobs: Number of parallel jobs
            use_adaptive: Override adaptive preprocessing setting
            
        Returns:
            List of (preprocessed_data, processing_info) tuples
        """
        from joblib import Parallel, delayed
        
        logger.info(f"Preprocessing batch of {len(eeg_batch)} recordings")
        
        # Process in parallel
        preprocessed = Parallel(n_jobs=n_jobs)(
            delayed(self.preprocess_eeg)(eeg, channel_names, use_adaptive=use_adaptive) 
            for eeg in eeg_batch
        )
        
        return preprocessed
        
    def get_adaptive_metrics(self) -> Optional[Dict[str, any]]:
        """Get metrics from adaptive preprocessing if available."""
        if self.adaptive_preprocessor is not None:
            return self.adaptive_preprocessor.get_metrics()
        return None
        
    def save_adaptive_optimizer(self, path: Path):
        """Save adaptive preprocessing optimizer if available."""
        if self.adaptive_preprocessor is not None:
            self.adaptive_preprocessor.save_optimizer(path)
        else:
            logger.warning("No adaptive preprocessor to save")
            
    def load_adaptive_optimizer(self, path: Path):
        """Load adaptive preprocessing optimizer if available."""
        if self.adaptive_preprocessor is not None:
            self.adaptive_preprocessor.load_optimizer(path)
        else:
            logger.warning("No adaptive preprocessor to load into") 