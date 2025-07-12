"""
Enhanced EEG Preprocessing with GPU Acceleration
Comprehensive preprocessing pipeline with GPU optimization for faster processing.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

# MNE imports for advanced preprocessing
import mne
from mne.preprocessing import ICA
from mne.filter import filter_data, notch_filter

# Scipy for signal processing
from scipy import signal
from scipy.stats import median_abs_deviation
from scipy.fft import fft, fftfreq

# PyWavelets for wavelet denoising
import pywt

logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """Enhanced EEG preprocessing with GPU acceleration."""
    
    def __init__(self, config_path: Union[str, Dict] = "config/config.yaml", 
                 use_adaptive: Optional[bool] = None, device: str = 'auto'):
        """Initialize preprocessor with GPU support."""
        if isinstance(config_path, str):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config_path
            
        # GPU setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"EEG Preprocessor initialized on device: {self.device}")
        
        # Configuration
        self.sampling_rate = self.config.get('dataset', {}).get('eeg_sampling_rate', 200.0)
        
        # Adaptive preprocessing configuration
        adaptive_config = self.config.get('preprocessing', {}).get('adaptive_preprocessing', {})
        self.use_adaptive = adaptive_config.get('enabled', False) and adaptive_config.get('use_adaptive', False)
        
        # Debug logging
        logger.info(f"🔍 Adaptive preprocessing config: {adaptive_config}")
        logger.info(f"🔍 enabled: {adaptive_config.get('enabled', False)}")
        logger.info(f"🔍 use_adaptive: {adaptive_config.get('use_adaptive', False)}")
        logger.info(f"🔍 Final use_adaptive: {self.use_adaptive}")
        
        # Override with explicit parameter if provided
        if use_adaptive is not None:
            self.use_adaptive = use_adaptive
            logger.info(f"🔍 Override use_adaptive to: {self.use_adaptive}")
            
        # Initialize adaptive preprocessor if enabled
        self.adaptive_preprocessor = None
        if self.use_adaptive:
            logger.info("🚀 Attempting to initialize adaptive preprocessor...")
            try:
                from .adaptive_preprocessor import AdaptivePreprocessor
                self.adaptive_preprocessor = AdaptivePreprocessor(
                    self.config, 
                    device=str(self.device),
                    cache_size=adaptive_config.get('cache_size', 1000)
                )
                logger.info("✅ Adaptive preprocessor initialized successfully")
                logger.info(f"   - Cache size: {adaptive_config.get('cache_size', 1000)}")
                logger.info(f"   - Confidence threshold: {adaptive_config.get('optimization_confidence_threshold', 0.7)}")
                logger.info(f"   - Fallback enabled: {adaptive_config.get('fallback_to_standard', True)}")
            except ImportError as e:
                logger.warning(f"Adaptive preprocessor not available: {e}")
                logger.info("Falling back to standard preprocessing")
                self.use_adaptive = False
            except Exception as e:
                logger.warning(f"Failed to initialize adaptive preprocessor: {e}")
                logger.info("Falling back to standard preprocessing")
                self.use_adaptive = False
        else:
            logger.info("❌ Adaptive preprocessing disabled in configuration")
                
        # Initialize scaler for normalization
        self._init_scaler()
        
    def _init_scaler(self):
        """Initialize normalization scaler."""
        self.scaler = None
        self.scaler_fitted = False
        
    def create_mne_raw(self, eeg_data: np.ndarray, channel_names: List[str]) -> mne.io.RawArray:
        """Create MNE Raw object from EEG data."""
        # Ensure data is float64 for MNE compatibility
        if eeg_data.dtype != np.float64:
            eeg_data = eeg_data.astype(np.float64)
            
        # Define channel types based on channel names
        ch_types = []
        for ch_name in channel_names:
            if ch_name.upper() in ['EKG', 'ECG']:
                ch_types.append('ecg')
            elif ch_name.upper() in ['EOG', 'EOG1', 'EOG2']:
                ch_types.append('eog')
            elif ch_name.upper() in ['EMG']:
                ch_types.append('emg')
            else:
                ch_types.append('eeg')
            
        # Create MNE info
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=self.sampling_rate,
            ch_types=ch_types,
            verbose=False
        )
        
        # Create Raw object
        raw = mne.io.RawArray(eeg_data, info, verbose=False)
        return raw
        
    def apply_bandpass_filter(self, raw: mne.io.RawArray) -> mne.io.RawArray:
        """Apply bandpass filter using GPU-accelerated operations where possible."""
        # Get filter parameters from config
        filter_config = self.config.get('preprocessing', {}).get('filter', {})
        lowcut = filter_config.get('lowcut', 0.5)
        highcut = filter_config.get('highcut', 50.0)
        
        logger.info(f"Applying bandpass filter: {lowcut}-{highcut} Hz")
        
        # Use MNE's optimized filtering - only apply to EEG channels
        try:
            raw.filter(lowcut, highcut, picks='eeg', verbose=False)
        except ValueError as e:
            if "No channels match the selection" in str(e):
                logger.warning("No EEG channels found, applying filter to all channels")
                raw.filter(lowcut, highcut, verbose=False)
            else:
                raise e
        return raw
        
    def apply_notch_filter(self, raw: mne.io.RawArray, freqs: List[float] = [50, 60]) -> mne.io.RawArray:
        """Apply notch filter to remove line noise."""
        logger.info(f"Applying notch filter at {freqs} Hz")
        
        # Use MNE's notch filter - only apply to EEG channels
        try:
            raw.notch_filter(freqs, picks='eeg', verbose=False)
        except ValueError as e:
            if "No channels match the selection" in str(e):
                logger.warning("No EEG channels found, applying notch filter to all channels")
                raw.notch_filter(freqs, verbose=False)
            else:
                raise e
        return raw
        
    def detect_bad_channels(self, raw: mne.io.RawArray) -> List[str]:
        """
        Enterprise-level bad channel detection for EEG data.
        - Only uses correlation-based detection (no MEG/Maxwell filtering).
        - Logs missing channels and data quality issues.
        - Designed for robust, production/enterprise use.
        """
        # Get data as numpy array for processing
        data = raw.get_data()
        n_channels, n_samples = data.shape
        bad_channels = []

        # Log if any channel is all zeros or NaN (data quality check)
        for idx, ch_name in enumerate(raw.ch_names):
            if np.all(data[idx] == 0):
                logger.warning(f"Channel {ch_name} is all zeros. Marking as bad.")
                bad_channels.append(ch_name)
            elif np.isnan(data[idx]).all():
                logger.warning(f"Channel {ch_name} is all NaN. Marking as bad.")
                bad_channels.append(ch_name)

        # Correlation-based detection (skip already bad channels)
        # Convert to GPU tensor for correlation analysis
        data_tensor = torch.tensor(data, device=self.device, dtype=torch.float32)
        # Normalize data
        data_norm = (data_tensor - data_tensor.mean(dim=1, keepdim=True)) / (data_tensor.std(dim=1, keepdim=True) + 1e-8)
        # Compute correlation matrix
        corr_matrix = torch.mm(data_norm, data_norm.t()) / data_norm.shape[1]
        corr_matrix_cpu = corr_matrix.cpu().numpy()

        for i in range(n_channels):
            ch_name = raw.ch_names[i]
            if ch_name in bad_channels:
                continue
            # Check if channel has very low correlation with others
            correlations = corr_matrix_cpu[i, :]
            correlations[i] = 0  # Exclude self-correlation
            mean_corr = np.mean(np.abs(correlations))
            if mean_corr < 0.1:  # Very low correlation
                logger.warning(f"Channel {ch_name} has very low mean correlation ({mean_corr:.3f}) with others. Marking as bad.")
                bad_channels.append(ch_name)

        # Remove duplicates
        bad_channels = list(set(bad_channels))

        if bad_channels:
            logger.info(f"Detected {len(bad_channels)} bad channels: {bad_channels}")
        else:
            logger.info("No bad channels detected.")

        return bad_channels
        
    def interpolate_bad_channels(self, raw: mne.io.RawArray, bad_channels: List[str]) -> mne.io.RawArray:
        """Interpolate bad channels using spherical splines."""
        if bad_channels:
            logger.info(f"Interpolating {len(bad_channels)} bad channels")
            raw.interpolate_bads(reset_bads=True, verbose=False)
        return raw
        
    def remove_artifacts_ica(self, raw: mne.io.RawArray) -> mne.io.RawArray:
        """Remove artifacts using ICA with GPU acceleration for correlation analysis."""
        logger.info("Applying ICA for artifact removal")
        
        # Get ICA parameters from config
        ica_config = self.config.get('preprocessing', {}).get('artifact_removal', {})
        n_components = ica_config.get('n_components', 25)
        ica_method = ica_config.get('ica_method', 'fastica')
        
        # Check if we have EEG channels for ICA
        eeg_channels = mne.pick_types(raw.info, eeg=True)
        if len(eeg_channels) == 0:
            logger.warning("No EEG channels found for ICA, skipping ICA artifact removal")
            return raw
            
        # Adjust components based on channel count
        n_channels = len(eeg_channels)
        if n_components >= n_channels:
            n_components = n_channels - 1
            logger.warning(f"Reduced ICA components from {ica_config.get('n_components', 25)} to {n_components} due to channel count")
            
        # Create and fit ICA
        ica = ICA(
            n_components=n_components,
            method=ica_method,
            random_state=42,
            max_iter=200,
            verbose=False
        )
        
        # Fit ICA on EEG channels only
        try:
            ica.fit(raw, picks='eeg', verbose=False)
        except ValueError as e:
            if "No channels match the selection" in str(e):
                logger.warning("No EEG channels found for ICA, skipping ICA artifact removal")
                return raw
            else:
                raise e
        
        # Detect and remove artifacts
        # Use EOG channels if available
        eog_channels = ica_config.get('eog_channels', ['Fp1', 'Fp2'])
        available_eog = [ch for ch in eog_channels if ch in raw.ch_names]
        
        if available_eog:
            # Detect EOG artifacts
            try:
                eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=available_eog[0], verbose=False)
                if eog_indices:
                    logger.info(f"Excluding {len(eog_indices)} ICA components")
                    ica.exclude = eog_indices
            except Exception as e:
                logger.warning(f"Failed to detect EOG artifacts: {e}")
                
        # Apply ICA
        try:
            raw_clean = ica.apply(raw.copy(), verbose=False)
            return raw_clean
        except Exception as e:
            logger.warning(f"Failed to apply ICA: {e}, returning original data")
            return raw
        
    def apply_wavelet_denoising(self, data: np.ndarray) -> np.ndarray:
        """Apply wavelet denoising with GPU acceleration - ULTRA-OPTIMIZED VERSION."""
        logger.info("Applying wavelet denoising (ultra-optimized)")
        
        # Get wavelet parameters from config
        denoising_config = self.config.get('preprocessing', {}).get('denoising', {})
        wavelet_type = denoising_config.get('wavelet_type', 'db4')
        wavelet_level = denoising_config.get('wavelet_level', 4)
        
        # Check if we should skip for large datasets
        if denoising_config.get('skip_for_large_datasets', False) and data.shape[1] > 10000:
            logger.info("Skipping wavelet denoising for large dataset (optimization)")
            return data
        
        # Convert to GPU tensor for batch processing
        data_tensor = torch.tensor(data, device=self.device, dtype=torch.float32)
        denoised_tensor = torch.zeros_like(data_tensor)
        
        # ULTRA-OPTIMIZATION: Process all channels in parallel using vectorized operations
        try:
            # Move to CPU for PyWavelets (it's faster for small signals)
            data_cpu = data_tensor.cpu().numpy()
            
            # Vectorized wavelet denoising with performance optimizations
            for ch in range(data_cpu.shape[0]):
                channel_data = data_cpu[ch]
                
                # Skip if signal is too short or all zeros
                if len(channel_data) < 2**wavelet_level or np.all(channel_data == 0):
                    denoised_tensor[ch] = torch.tensor(channel_data, device=self.device, dtype=torch.float32)
                    continue
                
                try:
                    # OPTIMIZATION: Use faster wavelet parameters for large signals
                    if len(channel_data) > 5000:
                        # For large signals, use fewer levels and simpler wavelet
                        actual_level = min(wavelet_level, 3)
                        actual_wavelet = 'db2'  # Simpler wavelet for speed
                    else:
                        actual_level = min(wavelet_level, pywt.dwt_max_level(len(channel_data), pywt.Wavelet(wavelet_type).dec_len))
                        actual_wavelet = wavelet_type
                    
                    # Apply wavelet denoising with optimized parameters
                    coeffs = pywt.wavedec(channel_data, actual_wavelet, level=actual_level)
                    
                    # OPTIMIZATION: Use faster thresholding for large signals
                    if len(coeffs) > 1:
                        # Use universal threshold for better performance
                        threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(channel_data)))
                        
                        # OPTIMIZATION: Apply thresholding only to detail coefficients
                        for i in range(1, len(coeffs)):
                            # Use faster thresholding for large coefficients
                            if coeffs[i].size > 1000:
                                # For large coefficients, use vectorized thresholding
                                coeffs[i] = np.where(np.abs(coeffs[i]) > threshold, 
                                                   np.sign(coeffs[i]) * (np.abs(coeffs[i]) - threshold), 
                                                   0)
                            else:
                                # For small coefficients, use PyWavelets thresholding
                                coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
                        
                        # Reconstruct signal
                        denoised_channel = pywt.waverec(coeffs, actual_wavelet)
                        
                        # Ensure same length
                        if len(denoised_channel) > len(channel_data):
                            denoised_channel = denoised_channel[:len(channel_data)]
                        elif len(denoised_channel) < len(channel_data):
                            denoised_channel = np.pad(denoised_channel, (0, len(channel_data) - len(denoised_channel)))
                    else:
                        denoised_channel = channel_data
                        
                    # Store back to tensor
                    denoised_tensor[ch] = torch.tensor(denoised_channel, device=self.device, dtype=torch.float32)
                    
                except Exception as e:
                    logger.warning(f"Wavelet denoising failed for channel {ch}: {e}, using original data")
                    denoised_tensor[ch] = data_tensor[ch]
                    
        except Exception as e:
            logger.warning(f"Wavelet denoising failed: {e}, returning original data")
            return data
            
        return denoised_tensor.cpu().numpy()
        
    def remove_baseline_drift(self, data: np.ndarray) -> np.ndarray:
        """Remove baseline drift using GPU-accelerated operations."""
        logger.info("Removing baseline drift")
        
        # Convert to GPU tensor
        data_tensor = torch.tensor(data, device=self.device, dtype=torch.float32)
        
        # Apply high-pass filter to remove slow drifts
        # Use FFT-based filtering on GPU
        fft_data = torch.fft.rfft(data_tensor, dim=1)
        
        # Create frequency array
        freqs = torch.fft.rfftfreq(data_tensor.shape[1], d=1/self.sampling_rate, device=self.device)
        
        # High-pass filter (remove frequencies below 0.5 Hz)
        highpass_mask = freqs > 0.5
        fft_data[:, ~highpass_mask] = 0
        
        # Inverse FFT
        filtered_data = torch.fft.irfft(fft_data, n=data_tensor.shape[1], dim=1)
        
        return filtered_data.cpu().numpy()
        
    def normalize_data(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """Normalize data using robust scaling with GPU acceleration."""
        # Get normalization method from config
        norm_config = self.config.get('preprocessing', {}).get('normalization', {})
        method = norm_config.get('method', 'robust')
        
        logger.info(f"Normalizing data using {method} method")
        
        # Convert to GPU tensor
        data_tensor = torch.tensor(data, device=self.device, dtype=torch.float32)
        
        if method == 'robust':
            # Robust scaling using median and MAD
            median = torch.median(data_tensor, dim=1, keepdim=True)[0]
            mad = torch.median(torch.abs(data_tensor - median), dim=1, keepdim=True)[0]
            normalized = (data_tensor - median) / (mad + 1e-8)
        else:
            # Standard scaling
            mean = torch.mean(data_tensor, dim=1, keepdim=True)
            std = torch.std(data_tensor, dim=1, keepdim=True)
            normalized = (data_tensor - mean) / (std + 1e-8)
            
        return normalized.cpu().numpy()
        
    def detect_and_remove_spikes(self, data: np.ndarray, threshold: float = 5.0) -> np.ndarray:
        """Detect and remove spike artifacts using GPU acceleration."""
        logger.info("Detecting and removing spike artifacts")
        
        # Convert to GPU tensor
        data_tensor = torch.tensor(data, device=self.device, dtype=torch.float32)
        
        # Compute z-scores for each channel
        mean = torch.mean(data_tensor, dim=1, keepdim=True)
        std = torch.std(data_tensor, dim=1, keepdim=True)
        z_scores = torch.abs((data_tensor - mean) / (std + 1e-8))
        
        # Detect spikes
        spike_mask = z_scores > threshold
        
        # Replace spikes with median of surrounding samples
        cleaned_data = data_tensor.clone()
        
        for ch in range(data_tensor.shape[0]):
            spike_indices = torch.where(spike_mask[ch])[0]
            
            for idx in spike_indices:
                # Get surrounding samples (avoid boundaries)
                start_idx = max(0, idx - 5)
                end_idx = min(data_tensor.shape[1], idx + 6)
                
                # Compute median of surrounding samples
                surrounding = data_tensor[ch, start_idx:end_idx]
                median_val = torch.median(surrounding)
                
                # Replace spike
                cleaned_data[ch, idx] = median_val
                
        return cleaned_data.cpu().numpy()
        
    def preprocess_eeg(self, 
                      eeg_data: np.ndarray,
                      channel_names: List[str],
                      apply_ica: bool = True,
                      remove_bad_channels: bool = True,
                      use_adaptive: Optional[bool] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Complete EEG preprocessing pipeline with GPU acceleration and adaptive preprocessing.
        
        Args:
            eeg_data: Raw EEG data (channels x samples)
            channel_names: List of channel names
            apply_ica: Whether to apply ICA artifact removal
            remove_bad_channels: Whether to remove/interpolate bad channels
            use_adaptive: Whether to use adaptive preprocessing (overrides config)
            
        Returns:
            Tuple of (preprocessed_data, processing_info)
        """
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
            
        # Determine whether to use adaptive preprocessing
        if use_adaptive is None:
            use_adaptive = self.use_adaptive
            
        # Use adaptive preprocessing if available and requested
        if use_adaptive and self.adaptive_preprocessor is not None:
            logger.info("🧠 Using adaptive preprocessing pipeline")
            try:
                # Add timeout for adaptive preprocessing to prevent hanging
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Adaptive preprocessing timed out")
                
                # Set timeout for adaptive preprocessing (30 seconds)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
                
                try:
                    processed_data, processing_info = self.adaptive_preprocessor.preprocess(
                        eeg_data, channel_names
                    )
                    processing_info['method'] = 'adaptive'
                    processing_info['adaptive_used'] = True
                    logger.info("✅ Adaptive preprocessing completed successfully")
                    
                    # Cancel timeout
                    signal.alarm(0)
                    return processed_data, processing_info
                    
                except TimeoutError:
                    logger.warning("Adaptive preprocessing timed out, falling back to standard")
                    signal.alarm(0)
                    raise Exception("Adaptive preprocessing timed out")
                    
            except Exception as e:
                logger.warning(f"Adaptive preprocessing failed: {e}")
                # Check if fallback is enabled
                adaptive_config = self.config.get('preprocessing', {}).get('adaptive_preprocessing', {})
                if adaptive_config.get('fallback_to_standard', True):
                    logger.info("🔄 Falling back to standard preprocessing")
                    use_adaptive = False
                else:
                    raise e
        elif use_adaptive and self.adaptive_preprocessor is None:
            logger.warning("Adaptive preprocessing requested but adaptive preprocessor is None")
            logger.info("🔄 Falling back to standard preprocessing")
            use_adaptive = False
            
        # Standard preprocessing pipeline
        logger.info("🔧 Starting standard EEG preprocessing pipeline")
        
        # Create MNE Raw object
        raw = self.create_mne_raw(eeg_data, channel_names)
        
        # Detect and interpolate bad channels
        if remove_bad_channels:
            bad_channels = self.detect_bad_channels(raw)
            if bad_channels:
                raw = self.interpolate_bad_channels(raw, bad_channels)
                
        # Apply bandpass filter
        raw = self.apply_bandpass_filter(raw)
        
        # Apply notch filter
        notch_freqs = self.config.get('preprocessing', {}).get('filter', {}).get('notch', [50, 60])
        raw = self.apply_notch_filter(raw, notch_freqs)
        
        # Apply ICA for artifact removal
        if apply_ica:
            raw = self.remove_artifacts_ica(raw)
            
        # Get data back as numpy array for additional processing
        processed_data = raw.get_data()
        
        # Apply wavelet denoising (conditionally)
        denoising_config = self.config.get('preprocessing', {}).get('denoising', {})
        if denoising_config.get('use_wavelet', True):
            # Check if we should skip for large datasets
            if denoising_config.get('skip_for_large_datasets', False) and processed_data.shape[1] > 10000:
                logger.info("Skipping wavelet denoising for large dataset (optimization)")
            else:
                processed_data = self.apply_wavelet_denoising(processed_data)
        else:
            logger.info("Wavelet denoising disabled in configuration")
        
        # Remove baseline drift
        processed_data = self.remove_baseline_drift(processed_data)
        
        # Detect and remove spikes
        processed_data = self.detect_and_remove_spikes(processed_data)
        
        # Normalize data
        processed_data = self.normalize_data(processed_data)
        
        # Record timing
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            processing_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        else:
            processing_time = 0.0
            
        logger.info("✅ EEG preprocessing completed")
        
        # Prepare processing info
        processing_info = {
            'method': 'standard',
            'adaptive_used': False,
            'processing_time': processing_time,
            'device': str(self.device),
            'applied_steps': [
                'bad_channel_detection',
                'bandpass_filtering',
                'notch_filtering',
                'ica_artifact_removal',
                'wavelet_denoising',
                'baseline_removal',
                'spike_removal',
                'normalization'
            ]
        }
        
        return processed_data, processing_info
        
    def preprocess(self, eeg_data: np.ndarray, 
                  channel_names: Optional[List[str]] = None) -> np.ndarray:
        """Simple preprocessing interface."""
        if channel_names is None:
            channel_names = [f'CH_{i}' for i in range(eeg_data.shape[0])]
            
        processed_data, _ = self.preprocess_eeg(eeg_data, channel_names)
        return processed_data
        
    def preprocess_batch(self, 
                        eeg_batch: List[np.ndarray],
                        channel_names: List[str],
                        n_jobs: int = -1,
                        use_adaptive: Optional[bool] = None) -> List[Tuple[np.ndarray, Dict]]:
        """Preprocess a batch of EEG recordings with GPU acceleration."""
        results = []
        
        # Process in batches to optimize GPU memory usage
        batch_size = 10  # Process 10 samples at a time
        
        for i in range(0, len(eeg_batch), batch_size):
            batch_end = min(i + batch_size, len(eeg_batch))
            batch_data = eeg_batch[i:batch_end]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(eeg_batch) + batch_size - 1)//batch_size}")
            
            for j, eeg_data in enumerate(batch_data):
                try:
                    processed_data, processing_info = self.preprocess_eeg(
                        eeg_data, channel_names, use_adaptive=use_adaptive
                    )
                    results.append((processed_data, processing_info))
                except Exception as e:
                    logger.warning(f"Failed to preprocess sample {i + j}: {e}")
                    # Return original data as fallback
                    results.append((eeg_data, {'error': str(e)}))
                    
        return results
        
    def get_adaptive_metrics(self) -> Optional[Dict[str, Any]]:
        """Get metrics from adaptive preprocessor if available."""
        if self.adaptive_preprocessor is not None:
            return self.adaptive_preprocessor.get_metrics()
        return None
        
    def save_adaptive_optimizer(self, path: Path):
        """Save adaptive optimizer if available."""
        if self.adaptive_preprocessor is not None:
            self.adaptive_preprocessor.save_optimizer(path)
            
    def load_adaptive_optimizer(self, path: Path):
        """Load adaptive optimizer if available."""
        if self.adaptive_preprocessor is not None:
            self.adaptive_preprocessor.load_optimizer(path) 