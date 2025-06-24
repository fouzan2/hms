"""
Spectrogram generation for EEG signals.
Converts time-domain EEG signals to time-frequency representations.
"""

import numpy as np
import pandas as pd
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
    """Generate spectrograms from EEG signals for frequency domain analysis."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize generator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
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
        """Generate spectrogram from single channel EEG signal."""
        
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
        """Generate spectrogram using Short-Time Fourier Transform."""
        
        # Compute STFT
        frequencies, times, Sxx = signal.spectrogram(
            eeg_signal,
            fs=self.sampling_rate,
            window='hann',
            nperseg=self.window_size,
            noverlap=self.overlap,
            nfft=self.nfft,
            scaling='density',
            mode='psd'
        )
        
        # Filter frequencies
        freq_mask = (frequencies >= self.freq_min) & (frequencies <= self.freq_max)
        frequencies = frequencies[freq_mask]
        Sxx = Sxx[freq_mask, :]
        
        # Convert to dB scale
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        return Sxx_db
        
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
        
        # Compute multitaper spectrum for each segment
        for i in range(num_segments):
            start = i * step
            end = start + segment_length
            segment = eeg_signal[start:end]
            
            # Apply multitaper method
            from scipy.signal.windows import dpss
            tapers = dpss(segment_length, time_bandwidth, num_tapers)
            
            # Compute spectrum for each taper
            spectra = []
            for taper in tapers:
                windowed = segment * taper
                spectrum = np.abs(np.fft.rfft(windowed, n=self.nfft))**2
                spectra.append(spectrum[freq_mask])
                
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
        """Generate spectrograms for all EEG channels."""
        
        n_channels = eeg_data.shape[0]
        
        # Generate spectrogram for first channel to get dimensions
        first_spec = self.generate_spectrogram(eeg_data[0], method)
        n_freqs, n_times = first_spec.shape
        
        # Initialize multichannel spectrogram
        spectrograms = np.zeros((n_channels, n_freqs, n_times))
        
        # Generate spectrogram for each channel
        for ch_idx in range(n_channels):
            spectrograms[ch_idx] = self.generate_spectrogram(eeg_data[ch_idx], method)
            
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
        """Compute spectral features from spectrogram."""
        
        features = {}
        
        # Get frequency array
        freqs = np.linspace(self.freq_min, self.freq_max, spectrogram.shape[0])
        
        # Convert from dB to power
        power = 10 ** (spectrogram / 10)
        
        # Spectral centroid
        features['spectral_centroid'] = np.sum(freqs[:, np.newaxis] * power, axis=0) / np.sum(power, axis=0)
        
        # Spectral bandwidth
        features['spectral_bandwidth'] = np.sqrt(
            np.sum(((freqs[:, np.newaxis] - features['spectral_centroid']) ** 2) * power, axis=0) / 
            np.sum(power, axis=0)
        )
        
        # Spectral rolloff
        cumsum = np.cumsum(power, axis=0)
        threshold = 0.85 * cumsum[-1, :]
        features['spectral_rolloff'] = freqs[np.argmax(cumsum >= threshold, axis=0)]
        
        # Spectral flux
        features['spectral_flux'] = np.sum(np.diff(power, axis=1) ** 2, axis=0)
        features['spectral_flux'] = np.pad(features['spectral_flux'], (1, 0), mode='constant')
        
        # Spectral entropy
        normalized_power = power / np.sum(power, axis=0, keepdims=True)
        features['spectral_entropy'] = -np.sum(
            normalized_power * np.log2(normalized_power + 1e-10), 
            axis=0
        )
        
        return features
        
    def apply_denoising(self, spectrogram: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian smoothing for denoising."""
        
        # Apply Gaussian filter
        denoised = gaussian_filter(spectrogram, sigma=sigma)
        
        return denoised
        
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
        """Create 3-channel representation for CNN input."""
        
        n_channels, n_freqs, n_times = spectrograms.shape
        
        # Method 1: Use 3 different spectrogram types
        if n_channels >= 1:
            # Channel 1: Standard STFT
            channel1 = self._resize_spectrogram(spectrograms[0], output_shape[:2])
            
            # Channel 2: Delta (difference) spectrogram
            if n_channels >= 2:
                delta = np.diff(spectrograms[0], axis=1, prepend=spectrograms[0][:, [0]])
                channel2 = self._resize_spectrogram(delta, output_shape[:2])
            else:
                channel2 = channel1
                
            # Channel 3: Acceleration (second difference) spectrogram  
            if n_channels >= 3:
                acceleration = np.diff(delta, axis=1, prepend=delta[:, [0]])
                channel3 = self._resize_spectrogram(acceleration, output_shape[:2])
            else:
                channel3 = channel1
                
        # Stack channels
        rgb_spectrogram = np.stack([channel1, channel2, channel3], axis=2)
        
        # Normalize to [0, 255]
        rgb_spectrogram = self._normalize_to_uint8(rgb_spectrogram)
        
        return rgb_spectrogram
        
    def _resize_spectrogram(self, spectrogram: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize spectrogram to target shape."""
        
        from scipy.ndimage import zoom
        
        zoom_factors = (
            target_shape[0] / spectrogram.shape[0],
            target_shape[1] / spectrogram.shape[1]
        )
        
        resized = zoom(spectrogram, zoom_factors, order=1)
        
        return resized
        
    def _normalize_to_uint8(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to uint8 range [0, 255]."""
        
        # Normalize each channel independently
        normalized = np.zeros_like(data)
        
        for i in range(data.shape[2]):
            channel = data[:, :, i]
            channel_min = np.percentile(channel, 1)
            channel_max = np.percentile(channel, 99)
            
            normalized[:, :, i] = np.clip(
                255 * (channel - channel_min) / (channel_max - channel_min + 1e-10),
                0, 255
            )
            
        return normalized.astype(np.uint8)
        
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