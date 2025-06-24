"""Unit tests for preprocessing modules."""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import yaml

from src.preprocessing import EEGPreprocessor, SpectrogramGenerator


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'preprocessing': {
            'num_channels': 19,
            'sampling_rate': 200,
            'lowcut': 0.5,
            'highcut': 50,
            'notch_freq': 60,
            'notch_width': 2,
            'reference': 'average',
            'standardize': True,
            'remove_artifacts': True,
            'ica_components': 0.95,
            'interpolate_bad_channels': True,
            'wavelet_denoising': True,
            'wavelet': 'db4',
            'wavelet_level': 4
        },
        'spectrogram': {
            'method': 'multitaper',
            'window_size': 4.0,
            'overlap': 0.5,
            'freq_bins': 128,
            'time_bins': 256,
            'normalize': True,
            'log_scale': True,
            'freq_bands': {
                'delta': [0.5, 4],
                'theta': [4, 8],
                'alpha': [8, 13],
                'beta': [13, 30],
                'gamma': [30, 50]
            }
        }
    }


@pytest.fixture
def sample_eeg_data():
    """Generate sample EEG data."""
    # 19 channels, 10000 samples (50 seconds at 200 Hz)
    time = np.linspace(0, 50, 10000)
    channels = []
    
    # Generate synthetic EEG signals
    for i in range(19):
        # Mix of different frequency components
        signal = (
            0.5 * np.sin(2 * np.pi * 10 * time) +  # Alpha
            0.3 * np.sin(2 * np.pi * 20 * time) +  # Beta
            0.2 * np.sin(2 * np.pi * 5 * time) +   # Theta
            0.1 * np.random.randn(len(time))       # Noise
        )
        channels.append(signal)
    
    # Create DataFrame with HMS format
    data = np.array(channels).T
    columns = [f'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
               'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    
    df = pd.DataFrame(data, columns=columns)
    return df


class TestEEGPreprocessor:
    """Test EEG preprocessing functionality."""
    
    def test_init(self, sample_config):
        """Test preprocessor initialization."""
        preprocessor = EEGPreprocessor(sample_config)
        assert preprocessor.config == sample_config
        assert preprocessor.preprocessing_config == sample_config['preprocessing']
        assert preprocessor.num_channels == 19
        assert preprocessor.sampling_rate == 200
    
    def test_bandpass_filter(self, sample_config, sample_eeg_data):
        """Test bandpass filtering."""
        preprocessor = EEGPreprocessor(sample_config)
        filtered = preprocessor._bandpass_filter(sample_eeg_data.values)
        
        # Check shape is preserved
        assert filtered.shape == sample_eeg_data.values.shape
        
        # Check that high frequencies are attenuated
        fft_original = np.fft.fft(sample_eeg_data.values[:, 0])
        fft_filtered = np.fft.fft(filtered[:, 0])
        freqs = np.fft.fftfreq(len(fft_original), 1/200)
        
        # Power should be reduced above 50 Hz
        high_freq_mask = np.abs(freqs) > 50
        assert np.mean(np.abs(fft_filtered[high_freq_mask])) < np.mean(np.abs(fft_original[high_freq_mask]))
    
    def test_notch_filter(self, sample_config):
        """Test notch filtering."""
        preprocessor = EEGPreprocessor(sample_config)
        
        # Create signal with 60 Hz component
        time = np.linspace(0, 1, 200)
        signal = np.sin(2 * np.pi * 60 * time) + np.sin(2 * np.pi * 10 * time)
        signal_2d = signal.reshape(-1, 1)
        
        filtered = preprocessor._notch_filter(signal_2d)
        
        # Check 60 Hz component is reduced
        fft_original = np.fft.fft(signal)
        fft_filtered = np.fft.fft(filtered[:, 0])
        freqs = np.fft.fftfreq(len(signal), 1/200)
        
        # Find 60 Hz bin
        idx_60hz = np.argmin(np.abs(freqs - 60))
        assert np.abs(fft_filtered[idx_60hz]) < np.abs(fft_original[idx_60hz]) * 0.1
    
    def test_standardize(self, sample_config, sample_eeg_data):
        """Test standardization."""
        preprocessor = EEGPreprocessor(sample_config)
        standardized = preprocessor._standardize(sample_eeg_data.values)
        
        # Check mean is close to 0 and std is close to 1
        assert np.allclose(np.mean(standardized, axis=0), 0, atol=0.01)
        assert np.allclose(np.std(standardized, axis=0), 1, atol=0.01)
    
    def test_preprocess_single(self, sample_config, sample_eeg_data):
        """Test full preprocessing pipeline."""
        preprocessor = EEGPreprocessor(sample_config)
        preprocessed = preprocessor.preprocess_single(sample_eeg_data)
        
        # Check output shape
        assert preprocessed.shape == sample_eeg_data.values.shape
        
        # Check standardization
        if sample_config['preprocessing']['standardize']:
            assert np.allclose(np.mean(preprocessed, axis=0), 0, atol=0.1)
            assert np.allclose(np.std(preprocessed, axis=0), 1, atol=0.1)


class TestSpectrogramGenerator:
    """Test spectrogram generation functionality."""
    
    def test_init(self, sample_config):
        """Test generator initialization."""
        generator = SpectrogramGenerator(sample_config)
        assert generator.config == sample_config
        assert generator.spectrogram_config == sample_config['spectrogram']
        assert generator.method == 'multitaper'
    
    def test_generate_stft(self, sample_config):
        """Test STFT spectrogram generation."""
        generator = SpectrogramGenerator(sample_config)
        
        # Create simple test signal
        time = np.linspace(0, 10, 2000)
        signal = np.sin(2 * np.pi * 10 * time)
        
        spec = generator._generate_stft(signal, 200)
        
        # Check output shape
        assert len(spec.shape) == 2
        assert spec.shape[0] <= sample_config['spectrogram']['freq_bins']
    
    def test_generate_multitaper(self, sample_config):
        """Test multitaper spectrogram generation."""
        generator = SpectrogramGenerator(sample_config)
        
        # Create test signal
        time = np.linspace(0, 10, 2000)
        signal = np.sin(2 * np.pi * 10 * time) + np.sin(2 * np.pi * 20 * time)
        
        spec = generator._generate_multitaper(signal, 200)
        
        # Check output shape
        assert len(spec.shape) == 2
        
        # Check that 10 Hz and 20 Hz components are present
        freqs = np.linspace(0, 100, spec.shape[0])
        idx_10hz = np.argmin(np.abs(freqs - 10))
        idx_20hz = np.argmin(np.abs(freqs - 20))
        
        # Average power across time
        avg_power = np.mean(spec, axis=1)
        assert avg_power[idx_10hz] > np.median(avg_power)
        assert avg_power[idx_20hz] > np.median(avg_power)
    
    def test_extract_frequency_bands(self, sample_config):
        """Test frequency band extraction."""
        generator = SpectrogramGenerator(sample_config)
        
        # Create mock spectrogram
        freqs = np.linspace(0, 50, 100)
        times = np.linspace(0, 10, 200)
        spec = np.random.rand(100, 200)
        
        bands = generator._extract_frequency_bands(spec, freqs)
        
        # Check all bands are present
        expected_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        assert all(band in bands for band in expected_bands)
        
        # Check band shapes
        for band_spec in bands.values():
            assert band_spec.shape[1] == spec.shape[1]  # Same time dimension
    
    def test_generate_from_eeg(self, sample_config):
        """Test full spectrogram generation from EEG."""
        generator = SpectrogramGenerator(sample_config)
        
        # Create multi-channel EEG data
        eeg_data = np.random.randn(10000, 19)  # 50s at 200Hz, 19 channels
        
        spectrograms = generator.generate_from_eeg(eeg_data)
        
        # Check output is dictionary
        assert isinstance(spectrograms, dict)
        
        # Check required keys
        assert 'full_spectrum' in spectrograms
        assert 'frequency_bands' in spectrograms
        
        # Check shapes
        full_spec = spectrograms['full_spectrum']
        assert len(full_spec.shape) == 3  # channels x freq x time
        assert full_spec.shape[0] == 19  # Number of channels
    
    def test_create_3d_representation(self, sample_config):
        """Test 3D representation creation."""
        generator = SpectrogramGenerator(sample_config)
        
        # Create mock spectrograms
        spectrograms = {
            'delta': np.random.rand(19, 50),
            'theta': np.random.rand(19, 50),
            'alpha': np.random.rand(19, 50)
        }
        
        repr_3d = generator.create_3d_representation(spectrograms)
        
        # Check shape
        assert len(repr_3d.shape) == 3
        assert repr_3d.shape[0] == 3  # RGB channels
        
        # Check values are normalized
        assert repr_3d.min() >= 0
        assert repr_3d.max() <= 1


@pytest.mark.integration
class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline."""
    
    def test_full_pipeline(self, sample_config, sample_eeg_data):
        """Test full preprocessing and spectrogram generation pipeline."""
        # Initialize components
        eeg_preprocessor = EEGPreprocessor(sample_config)
        spectrogram_generator = SpectrogramGenerator(sample_config)
        
        # Process EEG
        preprocessed_eeg = eeg_preprocessor.preprocess_single(sample_eeg_data)
        
        # Generate spectrograms
        spectrograms = spectrogram_generator.generate_from_eeg(preprocessed_eeg)
        
        # Validate outputs
        assert preprocessed_eeg.shape == sample_eeg_data.values.shape
        assert 'full_spectrum' in spectrograms
        assert 'frequency_bands' in spectrograms
        
        # Check that data is finite
        assert np.all(np.isfinite(preprocessed_eeg))
        assert np.all(np.isfinite(spectrograms['full_spectrum'])) 