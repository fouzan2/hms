#!/usr/bin/env python3
"""
Google Colab GPU-Accelerated Preprocessing Script for HMS Brain Activity Classification
This script is optimized to run on Google Colab with GPU acceleration.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import yaml
from tqdm import tqdm
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# GPU-specific imports
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import cupy as cp  # GPU-accelerated numpy
import cupyx.scipy.signal as cp_signal
import cupyx.scipy.ndimage as cp_ndimage
from numba import cuda, jit, prange
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Google Colab specific setup
def setup_colab_environment():
    """Setup Google Colab environment with necessary dependencies."""
    
    print("🚀 Setting up Google Colab environment...")
    
    # Install required packages
    os.system('pip install -q kaggle cupy-cuda11x numba torch torchvision torchaudio')
    os.system('pip install -q librosa scipy scikit-learn pandas numpy pywt')
    os.system('pip install -q h5py pyarrow fastparquet')
    
    # Setup Kaggle API
    os.system('mkdir -p ~/.kaggle')
    
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("❌ No GPU detected! This script requires GPU.")
        
    # Setup CUDA environment
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    return torch.cuda.is_available()

# GPU-accelerated signal processing functions
@cuda.jit
def gpu_bandpass_filter_kernel(signal, filtered_signal, b, a, n_samples):
    """CUDA kernel for bandpass filtering."""
    idx = cuda.grid(1)
    if idx < n_samples - len(b):
        # Simple IIR filter implementation
        y = 0.0
        for i in range(len(b)):
            y += b[i] * signal[idx + i]
        for i in range(1, len(a)):
            if idx >= i:
                y -= a[i] * filtered_signal[idx - i]
        filtered_signal[idx] = y / a[0]

class GPUSpectrogramGenerator:
    """GPU-accelerated spectrogram generation using PyTorch and CuPy."""
    
    def __init__(self, sampling_rate=200, device='cuda'):
        self.sampling_rate = sampling_rate
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Spectrogram parameters
        self.window_size = 256
        self.hop_length = 128
        self.n_fft = 512
        self.freq_min = 0.5
        self.freq_max = 50.0
        
    def generate_spectrogram_batch(self, signals_batch):
        """Generate spectrograms for a batch of signals on GPU."""
        # Convert to torch tensor
        if isinstance(signals_batch, np.ndarray):
            signals_batch = torch.from_numpy(signals_batch).float().to(self.device)
        
        batch_size, n_channels, signal_length = signals_batch.shape
        
        # Prepare for STFT
        window = torch.hann_window(self.window_size).to(self.device)
        
        # Batch STFT computation
        spectrograms = []
        for i in range(batch_size):
            channel_specs = []
            for ch in range(n_channels):
                # Compute STFT
                spec = torch.stft(
                    signals_batch[i, ch],
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.window_size,
                    window=window,
                    return_complex=True
                )
                
                # Convert to power and dB scale
                power_spec = torch.abs(spec) ** 2
                db_spec = 10 * torch.log10(power_spec + 1e-10)
                
                # Frequency filtering
                freqs = torch.linspace(0, self.sampling_rate/2, spec.shape[0]).to(self.device)
                freq_mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
                db_spec = db_spec[freq_mask, :]
                
                channel_specs.append(db_spec)
            
            spectrograms.append(torch.stack(channel_specs))
        
        return torch.stack(spectrograms)

class GPUEEGProcessor:
    """GPU-accelerated EEG preprocessing pipeline."""
    
    def __init__(self, config_path="config/config.yaml", device='cuda'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sampling_rate = self.config['dataset']['eeg_sampling_rate']
        
        # Initialize GPU components
        self.spectrogram_generator = GPUSpectrogramGenerator(
            sampling_rate=self.sampling_rate,
            device=device
        )
        
        # Filter parameters
        self.lowcut = self.config['preprocessing']['filter']['lowcut']
        self.highcut = self.config['preprocessing']['filter']['highcut']
        self.notch_freq = self.config['preprocessing']['filter']['notch_freq']
        
    def process_batch_gpu(self, eeg_batch, batch_metadata):
        """Process a batch of EEG signals on GPU."""
        
        # Convert to GPU tensors
        if isinstance(eeg_batch, np.ndarray):
            eeg_batch = torch.from_numpy(eeg_batch).float().to(self.device)
        
        batch_size = eeg_batch.shape[0]
        
        # 1. Apply filters on GPU
        filtered_batch = self.batch_filter_gpu(eeg_batch)
        
        # 2. Quality assessment on GPU
        quality_scores = self.batch_quality_assessment_gpu(filtered_batch)
        
        # 3. Artifact removal on GPU
        cleaned_batch = self.batch_artifact_removal_gpu(filtered_batch)
        
        # 4. Generate spectrograms on GPU
        spectrograms = self.spectrogram_generator.generate_spectrogram_batch(cleaned_batch)
        
        # 5. Extract features on GPU
        features = self.batch_feature_extraction_gpu(cleaned_batch, spectrograms)
        
        # Convert back to CPU for saving
        results = {
            'filtered_signals': filtered_batch.cpu().numpy(),
            'spectrograms': spectrograms.cpu().numpy(),
            'features': features.cpu().numpy(),
            'quality_scores': quality_scores.cpu().numpy()
        }
        
        return results
    
    def batch_filter_gpu(self, signals):
        """Apply bandpass and notch filters on GPU."""
        # Use torch.fft for filtering
        batch_size, n_channels, signal_length = signals.shape
        
        # Create frequency domain filter
        freqs = torch.fft.fftfreq(signal_length, 1/self.sampling_rate).to(self.device)
        
        # Bandpass filter mask
        bandpass_mask = (torch.abs(freqs) >= self.lowcut) & (torch.abs(freqs) <= self.highcut)
        
        # Notch filter mask
        notch_width = 2.0  # Hz
        notch_mask = ~((torch.abs(freqs - self.notch_freq) < notch_width/2) | 
                      (torch.abs(freqs + self.notch_freq) < notch_width/2))
        
        # Combined filter
        filter_mask = bandpass_mask & notch_mask
        
        # Apply filter in frequency domain
        filtered_signals = torch.zeros_like(signals)
        for i in range(batch_size):
            for ch in range(n_channels):
                # FFT
                signal_fft = torch.fft.fft(signals[i, ch])
                # Apply filter
                signal_fft *= filter_mask
                # Inverse FFT
                filtered_signals[i, ch] = torch.fft.ifft(signal_fft).real
        
        return filtered_signals
    
    def batch_quality_assessment_gpu(self, signals):
        """Assess signal quality on GPU."""
        batch_size, n_channels, _ = signals.shape
        
        quality_scores = torch.zeros(batch_size, n_channels).to(self.device)
        
        # Calculate SNR and other metrics on GPU
        for i in range(batch_size):
            for ch in range(n_channels):
                signal = signals[i, ch]
                
                # Simple SNR calculation
                signal_power = torch.mean(signal ** 2)
                noise_power = torch.var(signal - torch.mean(signal))
                snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
                
                # Artifact ratio (high frequency power ratio)
                fft = torch.fft.fft(signal)
                freqs = torch.fft.fftfreq(len(signal), 1/self.sampling_rate).to(self.device)
                high_freq_mask = torch.abs(freqs) > 30
                artifact_ratio = torch.sum(torch.abs(fft[high_freq_mask]) ** 2) / torch.sum(torch.abs(fft) ** 2)
                
                # Combined quality score
                quality_scores[i, ch] = torch.sigmoid(snr / 10 - artifact_ratio * 5)
        
        return quality_scores
    
    def batch_artifact_removal_gpu(self, signals):
        """Remove artifacts using GPU-accelerated methods."""
        # Simple threshold-based artifact removal
        # In practice, you might want to use more sophisticated methods
        
        # Calculate statistics
        mean = torch.mean(signals, dim=2, keepdim=True)
        std = torch.std(signals, dim=2, keepdim=True)
        
        # Threshold clipping
        threshold = 5.0
        cleaned = torch.clamp(signals, 
                            mean - threshold * std, 
                            mean + threshold * std)
        
        return cleaned
    
    def batch_feature_extraction_gpu(self, signals, spectrograms):
        """Extract features on GPU."""
        batch_size, n_channels, _ = signals.shape
        
        features_list = []
        
        # Time domain features
        time_features = self._extract_time_features_gpu(signals)
        
        # Frequency domain features
        freq_features = self._extract_frequency_features_gpu(spectrograms)
        
        # Combine features
        features = torch.cat([time_features, freq_features], dim=1)
        
        return features
    
    def _extract_time_features_gpu(self, signals):
        """Extract time domain features on GPU."""
        batch_size, n_channels, signal_length = signals.shape
        
        features = []
        
        # Statistical features
        features.append(torch.mean(signals, dim=2))  # Mean
        features.append(torch.std(signals, dim=2))   # Std
        features.append(torch.var(signals, dim=2))   # Variance
        features.append(signals.min(dim=2)[0])       # Min
        features.append(signals.max(dim=2)[0])       # Max
        
        # Higher order statistics
        features.append(torch.mean(signals ** 3, dim=2))  # Skewness component
        features.append(torch.mean(signals ** 4, dim=2))  # Kurtosis component
        
        # Zero crossing rate
        zero_crossings = torch.sum(torch.diff(torch.sign(signals), dim=2) != 0, dim=2).float()
        features.append(zero_crossings / signal_length)
        
        # Combine all features
        return torch.cat([f.view(batch_size, -1) for f in features], dim=1)
    
    def _extract_frequency_features_gpu(self, spectrograms):
        """Extract frequency domain features on GPU."""
        batch_size, n_channels, n_freqs, n_times = spectrograms.shape
        
        features = []
        
        # Band powers
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        freqs = torch.linspace(self.freq_min, self.freq_max, n_freqs).to(self.device)
        
        for band_name, (fmin, fmax) in bands.items():
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            band_power = torch.mean(spectrograms[:, :, freq_mask, :], dim=(2, 3))
            features.append(band_power)
        
        # Spectral centroid
        power = 10 ** (spectrograms / 10)
        freq_weights = freqs.view(1, 1, -1, 1).expand_as(spectrograms)
        centroid = torch.sum(freq_weights * power, dim=2) / (torch.sum(power, dim=2) + 1e-10)
        features.append(torch.mean(centroid, dim=2))
        
        # Combine all features
        return torch.cat([f.view(batch_size, -1) for f in features], dim=1)

class OptimizedDataLoader:
    """Optimized data loader for parallel processing."""
    
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def load_batch_parallel(self, file_ids):
        """Load a batch of EEG files in parallel."""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for file_id in file_ids:
                file_path = self.data_dir / f"{file_id}.parquet"
                future = executor.submit(self._load_single_file, file_path)
                futures.append(future)
            
            # Collect results
            batch_data = []
            for future in futures:
                data = future.result()
                if data is not None:
                    batch_data.append(data)
                    
        return batch_data
    
    def _load_single_file(self, file_path):
        """Load a single EEG file."""
        try:
            df = pd.read_parquet(file_path)
            # Convert to numpy array
            eeg_data = df.values.T  # Transpose to (channels, time)
            return eeg_data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

def process_data_on_colab_gpu(data_path, output_path, max_samples=None):
    """Main function to process data on Google Colab with GPU."""
    
    # Setup environment
    gpu_available = setup_colab_environment()
    if not gpu_available:
        raise RuntimeError("GPU is required for this script!")
    
    # Initialize GPU processor
    processor = GPUEEGProcessor(device='cuda')
    
    # Load metadata
    metadata = pd.read_csv(os.path.join(data_path, 'train.csv'))
    if max_samples:
        metadata = metadata.head(max_samples)
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'spectrograms'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'features'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'filtered'), exist_ok=True)
    
    # Process in batches
    batch_size = 32  # Adjust based on GPU memory
    data_loader = OptimizedDataLoader(
        os.path.join(data_path, 'train_eegs'),
        batch_size=batch_size
    )
    
    # Progress tracking
    total_batches = len(metadata) // batch_size + (1 if len(metadata) % batch_size else 0)
    
    print(f"\n📊 Processing {len(metadata)} samples in {total_batches} batches...")
    print(f"   Batch size: {batch_size}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    results_summary = []
    
    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        # Get batch file IDs
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(metadata))
        batch_metadata = metadata.iloc[start_idx:end_idx]
        
        # Load batch data in parallel
        file_ids = batch_metadata['eeg_id'].tolist()
        batch_data = data_loader.load_batch_parallel(file_ids)
        
        if not batch_data:
            continue
        
        # Stack into batch tensor
        # Pad sequences to same length if needed
        max_length = max(data.shape[1] for data in batch_data)
        padded_batch = []
        for data in batch_data:
            if data.shape[1] < max_length:
                pad_width = max_length - data.shape[1]
                data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
            padded_batch.append(data)
        
        batch_tensor = np.stack(padded_batch)
        
        # Process batch on GPU
        with torch.cuda.amp.autocast():  # Mixed precision for faster processing
            results = processor.process_batch_gpu(batch_tensor, batch_metadata)
        
        # Save results
        for i, (idx, row) in enumerate(batch_metadata.iterrows()):
            file_id = row['eeg_id']
            
            # Save spectrograms
            np.save(
                os.path.join(output_path, 'spectrograms', f'{file_id}_spec.npy'),
                results['spectrograms'][i]
            )
            
            # Save features
            np.save(
                os.path.join(output_path, 'features', f'{file_id}_feat.npy'),
                results['features'][i]
            )
            
            # Save filtered signals (optional, takes more space)
            np.save(
                os.path.join(output_path, 'filtered', f'{file_id}_filt.npy'),
                results['filtered_signals'][i]
            )
            
            # Track results
            results_summary.append({
                'file_id': file_id,
                'label': row['expert_consensus'],
                'quality_score': float(torch.mean(results['quality_scores'][i]).cpu())
            })
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(os.path.join(output_path, 'processing_summary.csv'), index=False)
    
    print(f"\n✅ Processing completed!")
    print(f"   Processed samples: {len(summary_df)}")
    print(f"   Average quality score: {summary_df['quality_score'].mean():.3f}")
    print(f"   Output saved to: {output_path}")

if __name__ == "__main__":
    # Configuration
    DATA_PATH = "/content/drive/MyDrive/hms-data"  # Adjust to your data path
    OUTPUT_PATH = "/content/drive/MyDrive/hms-processed"  # Adjust to your output path
    MAX_SAMPLES = None  # Set to small number for testing, None for all data
    
    # Run processing
    process_data_on_colab_gpu(DATA_PATH, OUTPUT_PATH, MAX_SAMPLES) 