"""
Spectrogram generation for EEG signals with GPU acceleration.
Converts time-domain EEG signals to time-frequency representations.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from scipy import signal
from scipy.ndimage import gaussian_filter
import librosa
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import logging
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class SpectrogramGenerator:
    """Generate spectrograms from EEG signals for frequency domain analysis with GPU acceleration."""
    
    def __init__(self, config_path: Union[str, Dict] = "config/novita_enhanced_config.yaml", device: str = 'auto'):
        """Initialize generator with configuration and GPU support."""
        if isinstance(config_path, str):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # config_path is already a dict
            self.config = config_path
            
        # GPU setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Spectrogram Generator initialized on device: {self.device}")
            
        self.spectrogram_config = self.config['preprocessing']['spectrogram']
        self.sampling_rate = self.config['dataset']['eeg_sampling_rate']
        
        # Spectrogram parameters
        self.window_size = self.spectrogram_config['window_size']
        self.overlap = self.spectrogram_config['overlap']
        self.nfft = self.spectrogram_config['nfft']
        self.freq_min = self.spectrogram_config['freq_min']
        self.freq_max = self.spectrogram_config['freq_max']
        
    def generate_spectrogram(self, 
                           eeg_signal: np.ndarray,
                           method: str = 'stft') -> np.ndarray:
        """Generate spectrogram from single channel EEG signal with GPU acceleration."""
        
        if method == 'stft':
            return self._generate_stft_spectrogram(eeg_signal)
        elif method == 'multitaper':
            return self._generate_multitaper_spectrogram(eeg_signal)
        elif method == 'wavelet':
            return self._generate_wavelet_spectrogram(eeg_signal)
        elif method == 'mel':
            return self._generate_mel_spectrogram(eeg_signal)
        else:
            raise ValueError(f"Unknown spectrogram method: {method}")
            
    def _generate_stft_spectrogram(self, eeg_signal: np.ndarray) -> np.ndarray:
        """Generate STFT spectrogram with GPU acceleration."""
        
        try:
            # Check for valid signal
            if len(eeg_signal) < 100:  # Minimum signal length
                logger.warning("Signal too short for spectrogram generation")
                return np.zeros((64, 64))
            
            # Check for NaN or inf values and replace with zeros
            if np.any(np.isnan(eeg_signal)) or np.any(np.isinf(eeg_signal)):
                logger.warning("Signal contains NaN or inf values, replacing with zeros")
                eeg_signal = np.nan_to_num(eeg_signal, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ensure signal is finite
            if not np.all(np.isfinite(eeg_signal)):
                logger.warning("Signal contains non-finite values, using zeros")
                return np.zeros((64, 64))
            
            # Convert to GPU tensor
            signal_tensor = torch.tensor(eeg_signal, device=self.device, dtype=torch.float32)
            
            # Generate spectrogram with fixed parameters for consistency using GPU FFT
            # Use PyTorch's STFT implementation
            window = torch.hann_window(self.window_size, device=self.device)
            
            # Compute STFT
            stft_result = torch.stft(
                signal_tensor,
                n_fft=self.nfft,
                hop_length=self.window_size - self.overlap,
                win_length=self.window_size,
                window=window,
                return_complex=True,
                center=True,
                normalized=False
            )
            
            # Compute power spectrum
            power_spectrum = torch.abs(stft_result) ** 2
            
            # Convert to frequency domain
            freqs = torch.fft.rfftfreq(self.nfft, d=1/self.sampling_rate, device=self.device)
            
            # Filter frequencies
            freq_mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
            power_spectrum = power_spectrum[freq_mask, :]
            
            # Convert to dB scale
            power_spectrum_db = 10 * torch.log10(power_spectrum + 1e-10)
            
            # Ensure output is finite
            if not torch.all(torch.isfinite(power_spectrum_db)):
                logger.warning("Spectrogram contains non-finite values, using zeros")
                return np.zeros((64, 64))
            
            return power_spectrum_db.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Failed to generate STFT spectrogram: {e}")
            return np.zeros((64, 64))
        
    def _generate_multitaper_spectrogram(self, eeg_signal: np.ndarray) -> np.ndarray:
        """Generate spectrogram using multitaper method for better frequency resolution."""
        
        # Parameters for multitaper
        time_bandwidth = 4
        num_tapers = 7
        
        # Segment the signal
        segment_length = self.window_size
        step = segment_length - self.overlap
        num_segments = (len(eeg_signal) - segment_length) // step + 1
        
        # Initialize spectrogram
        freqs = np.fft.rfftfreq(self.nfft, 1/self.sampling_rate)
        freq_mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        freqs = freqs[freq_mask]
        
        spectrogram = np.zeros((len(freqs), num_segments))
        
        # Convert to GPU tensor for processing
        signal_tensor = torch.tensor(eeg_signal, device=self.device, dtype=torch.float32)
        
        # Compute multitaper spectrum for each segment
        for i in range(num_segments):
            start = i * step
            end = start + segment_length
            segment = signal_tensor[start:end]
            
            # Apply multitaper method
            from scipy.signal.windows import dpss
            tapers = dpss(segment_length, time_bandwidth, num_tapers)
            
            # Compute spectrum for each taper using GPU
            spectra = []
            for taper in tapers:
                taper_tensor = torch.tensor(taper, device=self.device, dtype=torch.float32)
                windowed = segment * taper_tensor
                
                # Use PyTorch FFT
                spectrum = torch.fft.rfft(windowed, n=self.nfft)
                spectrum_power = torch.abs(spectrum) ** 2
                spectra.append(spectrum_power[freq_mask].cpu().numpy())
                
            # Average across tapers
            spectrogram[:, i] = np.mean(spectra, axis=0)
            
        # Convert to dB scale
        spectrogram_db = 10 * np.log10(spectrogram + 1e-10)
        
        return spectrogram_db
        
    def _generate_wavelet_spectrogram(self, eeg_signal: np.ndarray) -> np.ndarray:
        """Generate spectrogram using continuous wavelet transform."""
        
        # Define scales to match desired frequency range
        frequencies = np.logspace(
            np.log10(self.freq_min),
            np.log10(self.freq_max),
            128
        )
        scales = self.sampling_rate / (2 * frequencies * np.pi)
        
        # Compute CWT
        cwt_matrix = signal.cwt(eeg_signal, signal.morlet2, scales, w=5)
        power = np.abs(cwt_matrix)**2
        
        # Convert to dB scale
        power_db = 10 * np.log10(power + 1e-10)
        
        return power_db
        
    def _generate_mel_spectrogram(self, eeg_signal: np.ndarray) -> np.ndarray:
        """Generate mel-scaled spectrogram for perceptually relevant features."""
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=eeg_signal,
            sr=self.sampling_rate,
            n_fft=self.nfft,
            hop_length=self.window_size - self.overlap,
            win_length=self.window_size,
            n_mels=128,
            fmin=self.freq_min,
            fmax=self.freq_max
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
        
    def generate_multichannel_spectrogram(self,
                                        eeg_data: np.ndarray,
                                        method: str = 'stft') -> np.ndarray:
        """Generate spectrograms for all EEG channels with GPU acceleration."""
        
        n_channels = eeg_data.shape[0]
        
        try:
            # Check for NaN or inf values in the entire dataset
            if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
                logger.warning("EEG data contains NaN or inf values, replacing with zeros")
                eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Convert to GPU tensor for batch processing
            eeg_tensor = torch.tensor(eeg_data, device=self.device, dtype=torch.float32)
            
            # Generate spectrogram for first channel to get dimensions
            first_spec = self.generate_spectrogram(eeg_data[0], method)
            
            # Ensure first spectrogram has valid shape
            if first_spec.shape[0] == 0 or first_spec.shape[1] == 0:
                logger.warning("First spectrogram has invalid shape, using default")
                first_spec = np.zeros((64, 64))
            
            n_freqs, n_times = first_spec.shape
            
            # Initialize multichannel spectrogram
            spectrograms = np.zeros((n_channels, n_freqs, n_times))
            
            # Generate spectrogram for each channel
            for ch_idx in range(n_channels):
                try:
                    channel_spec = self.generate_spectrogram(eeg_data[ch_idx], method)
                    
                    # Ensure the spectrogram has the same shape as the first one
                    if channel_spec.shape != (n_freqs, n_times):
                        logger.warning(f"Channel {ch_idx} spectrogram shape {channel_spec.shape} doesn't match expected {(n_freqs, n_times)}, resizing")
                        # Resize to match first spectrogram
                        if channel_spec.shape[0] > 0 and channel_spec.shape[1] > 0:
                            from scipy.ndimage import zoom
                            zoom_factors = (n_freqs / channel_spec.shape[0], n_times / channel_spec.shape[1])
                            channel_spec = zoom(channel_spec, zoom_factors, order=1)
                        else:
                            channel_spec = np.zeros((n_freqs, n_times))
                    
                    spectrograms[ch_idx] = channel_spec
                    
                except Exception as e:
                    logger.warning(f"Failed to generate spectrogram for channel {ch_idx}: {e}")
                    # Use zeros for failed channel
                    spectrograms[ch_idx] = np.zeros((n_freqs, n_times))
                    
            return spectrograms
            
        except Exception as e:
            logger.error(f"Failed to generate multichannel spectrogram: {e}")
            # Return a default spectrogram with consistent shape
            default_shape = (n_channels, 64, 64)  # Default size
            return np.zeros(default_shape)
        
    def generate_batch(self, eeg_batch: List[np.ndarray], method: str = 'stft') -> List[np.ndarray]:
        """
        Generate spectrograms for a batch of EEG recordings with GPU acceleration.
        
        Args:
            eeg_batch: List of EEG recordings (each of shape channels x time)
            method: Spectrogram generation method
            
        Returns:
            List of spectrograms with consistent shapes
        """
        logger.info(f"Generating spectrograms for batch of {len(eeg_batch)} recordings on {self.device}")
        
        # Define standard spectrogram shape
        standard_shape = (224, 224, 3)  # Standard CNN input shape
        
        spectrograms = []
        for i, eeg_data in enumerate(eeg_batch):
            try:
                # Check for valid EEG data
                if eeg_data is None or eeg_data.size == 0:
                    logger.warning(f"Sample {i} has empty EEG data, using default")
                    default_spec = np.zeros(standard_shape, dtype=np.uint8)
                    spectrograms.append(default_spec)
                    continue
                
                # Check for NaN or inf values
                if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
                    logger.warning(f"Sample {i} contains NaN or inf values, replacing with zeros")
                    eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Generate multichannel spectrogram
                spec = self.generate_multichannel_spectrogram(eeg_data, method)
                
                # Ensure spec has valid shape
                if spec is None or spec.size == 0:
                    logger.warning(f"Sample {i} generated empty spectrogram, using default")
                    default_spec = np.zeros(standard_shape, dtype=np.uint8)
                    spectrograms.append(default_spec)
                    continue
                
                # Create 3D representation for CNN input with consistent shape
                spec_3d = self.create_3d_spectrogram_representation(spec, standard_shape)
                
                # Ensure final output has correct shape and type
                if spec_3d.shape != standard_shape:
                    logger.warning(f"Sample {i} spectrogram shape {spec_3d.shape} doesn't match {standard_shape}, resizing")
                    # Resize to standard shape
                    if spec_3d.shape[0] > 0 and spec_3d.shape[1] > 0:
                        from scipy.ndimage import zoom
                        zoom_factors = (standard_shape[0] / spec_3d.shape[0], 
                                      standard_shape[1] / spec_3d.shape[1])
                        spec_3d = zoom(spec_3d, zoom_factors, order=1)
                        # Ensure 3 channels
                        if spec_3d.ndim == 2:
                            spec_3d = np.stack([spec_3d] * 3, axis=2)
                        elif spec_3d.shape[2] != 3:
                            spec_3d = spec_3d[:, :, :3]  # Take first 3 channels
                    else:
                        spec_3d = np.zeros(standard_shape, dtype=np.uint8)
                
                # Ensure correct data type
                if spec_3d.dtype != np.uint8:
                    spec_3d = spec_3d.astype(np.uint8)
                
                spectrograms.append(spec_3d)
                
            except Exception as e:
                logger.warning(f"Failed to generate spectrogram for sample {i}: {e}")
                # Create a default spectrogram with consistent shape
                default_spec = np.zeros(standard_shape, dtype=np.uint8)
                spectrograms.append(default_spec)
        
        # Verify all spectrograms have the same shape
        shapes = [spec.shape for spec in spectrograms]
        if len(set(shapes)) > 1:
            logger.warning(f"Inconsistent spectrogram shapes in batch: {set(shapes)}")
            # Resize all to the most common shape
            most_common_shape = max(set(shapes), key=shapes.count)
            for i, spec in enumerate(spectrograms):
                if spec.shape != most_common_shape:
                    logger.warning(f"Resizing spectrogram {i} from {spec.shape} to {most_common_shape}")
                    if spec.shape[0] > 0 and spec.shape[1] > 0:
                        from scipy.ndimage import zoom
                        zoom_factors = (most_common_shape[0] / spec.shape[0], 
                                      most_common_shape[1] / spec.shape[1])
                        spec_resized = zoom(spec, zoom_factors, order=1)
                        if spec_resized.shape != most_common_shape:
                            spec_resized = np.zeros(most_common_shape, dtype=np.uint8)
                        spectrograms[i] = spec_resized
                    else:
                        spectrograms[i] = np.zeros(most_common_shape, dtype=np.uint8)
        
        return spectrograms
        
    def extract_frequency_bands(self, spectrogram: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract classical EEG frequency bands from spectrogram."""
        
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        # Get frequency array
        freqs = np.linspace(self.freq_min, self.freq_max, spectrogram.shape[0])
        
        # Extract band powers
        band_powers = {}
        for band_name, (fmin, fmax) in bands.items():
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            band_powers[band_name] = np.mean(spectrogram[freq_mask, :], axis=0)
            
        return band_powers
        
    def compute_spectral_features(self, spectrogram: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute spectral features from spectrogram with GPU acceleration."""
        
        features = {}
        
        # Convert to GPU tensor
        spec_tensor = torch.tensor(spectrogram, device=self.device, dtype=torch.float32)
        
        # Get frequency array
        freqs = torch.linspace(self.freq_min, self.freq_max, spectrogram.shape[0], device=self.device)
        
        # Convert from dB to power
        power = 10 ** (spec_tensor / 10)
        
        # Spectral centroid
        features['spectral_centroid'] = torch.sum(freqs[:, None] * power, dim=0) / torch.sum(power, dim=0)
        
        # Spectral bandwidth
        centroid = features['spectral_centroid']
        features['spectral_bandwidth'] = torch.sqrt(
            torch.sum(((freqs[:, None] - centroid) ** 2) * power, dim=0) / 
            torch.sum(power, dim=0)
        )
        
        # Spectral rolloff
        cumsum = torch.cumsum(power, dim=0)
        threshold = 0.85 * cumsum[-1, :]
        features['spectral_rolloff'] = freqs[torch.argmax(cumsum >= threshold, dim=0)]
        
        # Spectral flux
        features['spectral_flux'] = torch.sum(torch.diff(power, dim=1) ** 2, dim=0)
        features['spectral_flux'] = F.pad(features['spectral_flux'], (1, 0), mode='constant')
        
        # Spectral entropy
        normalized_power = power / torch.sum(power, dim=0, keepdim=True)
        features['spectral_entropy'] = -torch.sum(
            normalized_power * torch.log2(normalized_power + 1e-10), 
            dim=0
        )
        
        # Convert back to numpy
        return {k: v.cpu().numpy() for k, v in features.items()}
        
    def apply_denoising(self, spectrogram: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian smoothing for denoising with GPU acceleration."""
        
        # Convert to GPU tensor
        spec_tensor = torch.tensor(spectrogram, device=self.device, dtype=torch.float32)
        
        # Apply Gaussian smoothing using PyTorch
        # Use 2D convolution with Gaussian kernel
        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel_2d(kernel_size, sigma)
        kernel = kernel.to(self.device)
        
        # Apply convolution
        if spec_tensor.ndim == 2:
            spec_tensor = spec_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            denoised = F.conv2d(spec_tensor, kernel.unsqueeze(0).unsqueeze(0), padding=kernel_size//2)
            denoised = denoised.squeeze(0).squeeze(0)  # Remove batch and channel dims
        else:
            denoised = F.conv2d(spec_tensor, kernel.unsqueeze(0).unsqueeze(0), padding=kernel_size//2)
            
        return denoised.cpu().numpy()
        
    def _create_gaussian_kernel_2d(self, kernel_size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel for denoising."""
        kernel = torch.zeros(kernel_size, kernel_size)
        center = kernel_size // 2
        
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
                
        return kernel / kernel.sum()
        
    def enhance_contrast(self, spectrogram: np.ndarray, method: str = 'histogram') -> np.ndarray:
        """Enhance spectrogram contrast for better feature visibility."""
        
        if method == 'histogram':
            # Histogram equalization
            from skimage import exposure
            enhanced = exposure.equalize_hist(spectrogram)
            
        elif method == 'adaptive':
            # Adaptive histogram equalization
            from skimage import exposure
            enhanced = exposure.equalize_adapthist(spectrogram)
            
        elif method == 'gamma':
            # Gamma correction
            normalized = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
            enhanced = np.power(normalized, 0.5)  # gamma = 0.5
            enhanced = enhanced * (spectrogram.max() - spectrogram.min()) + spectrogram.min()
            
        else:
            enhanced = spectrogram
            
        return enhanced
        
    def create_3d_spectrogram_representation(self,
                                           spectrograms: np.ndarray,
                                           output_shape: Tuple[int, int, int] = (224, 224, 3)) -> np.ndarray:
        """Create 3-channel representation for CNN input with GPU acceleration."""
        
        try:
            n_channels, n_freqs, n_times = spectrograms.shape
            
            # Handle edge cases
            if n_channels == 0 or n_freqs == 0 or n_times == 0:
                logger.warning("Empty spectrogram input, returning zeros")
                return np.zeros(output_shape, dtype=np.uint8)
            
            # Convert to GPU tensor
            spec_tensor = torch.tensor(spectrograms, device=self.device, dtype=torch.float32)
            
            # Method 1: Use 3 different spectrogram types
            if n_channels >= 1:
                # Channel 1: Standard STFT
                channel1 = self._resize_spectrogram(spec_tensor[0], output_shape[:2])
                
                # Channel 2: Delta (difference) spectrogram
                if n_channels >= 2:
                    delta = torch.diff(spec_tensor[0], dim=1, prepend=spec_tensor[0][:, [0]])
                    channel2 = self._resize_spectrogram(delta, output_shape[:2])
                else:
                    channel2 = channel1
                    
                # Channel 3: Acceleration (second difference) spectrogram  
                if n_channels >= 3:
                    acceleration = torch.diff(delta, dim=1, prepend=delta[:, [0]])
                    channel3 = self._resize_spectrogram(acceleration, output_shape[:2])
                else:
                    channel3 = channel1
                    
            # Stack channels
            rgb_spectrogram = torch.stack([channel1, channel2, channel3], dim=2)
            
            # Normalize to [0, 255]
            rgb_spectrogram = self._normalize_to_uint8(rgb_spectrogram)
            
            # Ensure correct shape
            if rgb_spectrogram.shape != output_shape:
                logger.warning(f"RGB spectrogram shape {rgb_spectrogram.shape} doesn't match {output_shape}, resizing")
                if rgb_spectrogram.shape[0] > 0 and rgb_spectrogram.shape[1] > 0:
                    # Use PyTorch interpolation for GPU acceleration
                    rgb_spectrogram = F.interpolate(
                        rgb_spectrogram.unsqueeze(0).permute(0, 3, 1, 2),  # Add batch dim and rearrange
                        size=output_shape[:2],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)  # Remove batch dim and rearrange back
                    
                    # Ensure 3 channels
                    if rgb_spectrogram.shape[2] != 3:
                        rgb_spectrogram = rgb_spectrogram[:, :, :3]
                else:
                    rgb_spectrogram = torch.zeros(output_shape, device=self.device, dtype=torch.uint8)
            
            return rgb_spectrogram.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Failed to create 3D spectrogram representation: {e}")
            return np.zeros(output_shape, dtype=np.uint8)
        
    def _resize_spectrogram(self, spectrogram: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        """Resize spectrogram to target shape using GPU acceleration."""
        
        try:
            # Handle edge cases
            if spectrogram.numel() == 0:
                logger.warning("Empty spectrogram input, returning zeros")
                return torch.zeros(target_shape, device=self.device)
            
            if spectrogram.shape[0] == 0 or spectrogram.shape[1] == 0:
                logger.warning("Spectrogram has zero dimensions, returning zeros")
                return torch.zeros(target_shape, device=self.device)
            
            # Check if resize is needed
            if spectrogram.shape == target_shape:
                return spectrogram
            
            # Use PyTorch interpolation for GPU acceleration
            resized = F.interpolate(
                spectrogram.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
                size=target_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)  # Remove batch and channel dims
            
            return resized
            
        except Exception as e:
            logger.error(f"Failed to resize spectrogram: {e}")
            return torch.zeros(target_shape, device=self.device)
        
    def _normalize_to_uint8(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data to uint8 range [0, 255] with GPU acceleration."""
        
        try:
            # Handle edge cases
            if data.numel() == 0:
                logger.warning("Empty data input, returning zeros")
                return torch.zeros((224, 224, 3), device=self.device, dtype=torch.uint8)
            
            # Ensure data is finite
            if not torch.all(torch.isfinite(data)):
                logger.warning("Data contains non-finite values, replacing with zeros")
                data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize each channel independently
            normalized = torch.zeros_like(data)
            
            for i in range(data.shape[2]):
                channel = data[:, :, i]
                
                # Handle edge cases for channel
                if torch.all(channel == 0) or torch.all(torch.isnan(channel)):
                    normalized[:, :, i] = 0
                    continue
                
                # Calculate percentiles for robust normalization
                channel_min = torch.quantile(channel, 0.01)
                channel_max = torch.quantile(channel, 0.99)
                
                # Handle case where min and max are the same
                if channel_max <= channel_min:
                    normalized[:, :, i] = 128  # Middle gray
                else:
                    normalized[:, :, i] = torch.clamp(
                        255 * (channel - channel_min) / (channel_max - channel_min + 1e-10),
                        0, 255
                    )
            
            return normalized.to(torch.uint8)
            
        except Exception as e:
            logger.error(f"Failed to normalize data: {e}")
            return torch.zeros((224, 224, 3), device=self.device, dtype=torch.uint8)
        
    def save_spectrogram_image(self, 
                             spectrogram: np.ndarray,
                             output_path: Path,
                             cmap: str = 'viridis'):
        """Save spectrogram as image file."""
        
        plt.figure(figsize=(10, 6))
        
        # Create time and frequency arrays
        times = np.linspace(0, spectrogram.shape[1] / self.sampling_rate, spectrogram.shape[1])
        freqs = np.linspace(self.freq_min, self.freq_max, spectrogram.shape[0])
        
        # Plot spectrogram
        plt.pcolormesh(times, freqs, spectrogram, cmap=cmap, shading='auto')
        plt.colorbar(label='Power (dB)')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('EEG Spectrogram')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved spectrogram to {output_path}") 