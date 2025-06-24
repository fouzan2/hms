#!/usr/bin/env python3
"""
Test script for the comprehensive preprocessing pipeline.
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from preprocessing import (
    MultiFormatEEGReader,
    SignalQualityAssessor,
    EEGFilter,
    EEGFeatureExtractor,
    extract_features,
    assess_signal_quality,
    create_eeg_filter
)

def generate_test_eeg_data(n_channels=19, n_samples=10000, sampling_rate=200.0):
    """Generate synthetic EEG data for testing."""
    # Generate base EEG signals
    time = np.arange(n_samples) / sampling_rate
    data = np.zeros((n_channels, n_samples))
    
    # Add different frequency components to different channels
    for ch in range(n_channels):
        # Alpha rhythm (8-13 Hz)
        alpha_freq = 10 + np.random.rand() * 3
        data[ch, :] += 20e-6 * np.sin(2 * np.pi * alpha_freq * time)
        
        # Beta rhythm (13-30 Hz)
        beta_freq = 20 + np.random.rand() * 10
        data[ch, :] += 10e-6 * np.sin(2 * np.pi * beta_freq * time)
        
        # Add some noise
        data[ch, :] += 5e-6 * np.random.randn(n_samples)
    
    # Add artifacts to some channels
    # Eye blink on frontal channels
    blink_time = int(2 * sampling_rate)
    data[0:2, blink_time:blink_time+50] += 100e-6 * np.exp(-np.arange(50)/10)
    
    # Muscle artifact
    muscle_start = int(5 * sampling_rate)
    muscle_end = int(5.5 * sampling_rate)
    data[10, muscle_start:muscle_end] += 30e-6 * np.random.randn(muscle_end - muscle_start)
    
    # Bad channel (flatline)
    data[15, :] = 1e-9 * np.random.randn(n_samples)
    
    # Line noise on one channel
    data[8, :] += 20e-6 * np.sin(2 * np.pi * 50 * time)
    
    return data

def test_signal_quality_assessment():
    """Test signal quality assessment."""
    print("\n=== Testing Signal Quality Assessment ===")
    
    # Generate test data
    data = generate_test_eeg_data()
    channel_names = [f'ch_{i}' for i in range(data.shape[0])]
    
    # Assess quality
    quality_metrics = assess_signal_quality(data, channel_names=channel_names)
    
    print(f"Overall quality score: {quality_metrics.overall_quality_score:.3f}")
    print(f"SNR: {quality_metrics.snr:.1f} dB")
    print(f"Artifact ratio: {quality_metrics.artifact_ratio:.3f}")
    print(f"Bad channels: {quality_metrics.bad_channels}")
    print(f"Number of bad segments: {len(quality_metrics.bad_segments)}")
    
    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Plot a few channels
    time = np.arange(data.shape[1]) / 200.0
    for ch in [0, 8, 15]:  # Good, noisy, bad
        axes[0].plot(time[:1000], data[ch, :1000] * 1e6, label=f'Channel {ch}')
    axes[0].set_ylabel('Amplitude (µV)')
    axes[0].set_title('Raw EEG Signals')
    axes[0].legend()
    
    # Quality scores per channel
    channel_scores = [quality_metrics.quality_per_channel.get(ch, 0) 
                     for ch in channel_names]
    axes[1].bar(range(len(channel_scores)), channel_scores)
    axes[1].set_xlabel('Channel')
    axes[1].set_ylabel('Quality Score')
    axes[1].set_title('Channel Quality Scores')
    
    # Noise levels
    noise_types = list(quality_metrics.noise_levels.keys())
    noise_values = list(quality_metrics.noise_levels.values())
    axes[2].bar(range(len(noise_types)), noise_values)
    axes[2].set_xticks(range(len(noise_types)))
    axes[2].set_xticklabels(noise_types, rotation=45)
    axes[2].set_ylabel('Noise Level')
    axes[2].set_title('Detected Noise Levels')
    
    plt.tight_layout()
    plt.savefig('test_quality_assessment.png')
    plt.close()
    
    return quality_metrics

def test_signal_filtering():
    """Test signal filtering."""
    print("\n=== Testing Signal Filtering ===")
    
    # Generate test data
    data = generate_test_eeg_data()
    
    # Create filter
    eeg_filter = create_eeg_filter()
    
    # Apply filters
    filtered_data = eeg_filter.apply_filters(data, bad_channels=[15])
    
    # Compare before and after
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Time domain comparison
    time = np.arange(1000) / 200.0
    axes[0, 0].plot(time, data[0, :1000] * 1e6, label='Original')
    axes[0, 0].plot(time, filtered_data[0, :1000] * 1e6, label='Filtered', alpha=0.7)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude (µV)')
    axes[0, 0].set_title('Time Domain - Channel 0')
    axes[0, 0].legend()
    
    # Frequency domain comparison
    from scipy import signal
    freqs_orig, psd_orig = signal.welch(data[8, :], fs=200, nperseg=512)
    freqs_filt, psd_filt = signal.welch(filtered_data[8, :], fs=200, nperseg=512)
    
    axes[0, 1].semilogy(freqs_orig, psd_orig, label='Original')
    axes[0, 1].semilogy(freqs_filt, psd_filt, label='Filtered', alpha=0.7)
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('PSD')
    axes[0, 1].set_title('Frequency Domain - Channel 8 (with line noise)')
    axes[0, 1].legend()
    axes[0, 1].set_xlim([0, 100])
    
    # Segment signal
    segments = eeg_filter.segment_signal(filtered_data, segment_length=2.0, overlap=0.5)
    print(f"Created {len(segments)} segments")
    
    # Plot segments
    for i, seg in enumerate(segments[:3]):
        axes[1, 0].plot(seg[0, :] * 1e6 + i * 50, label=f'Segment {i}')
    axes[1, 0].set_xlabel('Samples')
    axes[1, 0].set_ylabel('Amplitude (µV)')
    axes[1, 0].set_title('Segmented Signals')
    axes[1, 0].legend()
    
    # Spatial filtering comparison
    car_filtered = eeg_filter.apply_spatial_filter(filtered_data, 'car')
    axes[1, 1].plot(data[0, :500] * 1e6, label='Original', alpha=0.5)
    axes[1, 1].plot(filtered_data[0, :500] * 1e6, label='Filtered', alpha=0.5)
    axes[1, 1].plot(car_filtered[0, :500] * 1e6, label='CAR', alpha=0.7)
    axes[1, 1].set_xlabel('Samples')
    axes[1, 1].set_ylabel('Amplitude (µV)')
    axes[1, 1].set_title('Spatial Filtering Effects')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('test_signal_filtering.png')
    plt.close()
    
    return filtered_data

def test_feature_extraction():
    """Test feature extraction."""
    print("\n=== Testing Feature Extraction ===")
    
    # Generate test data
    data = generate_test_eeg_data(n_samples=2000)  # Shorter for faster processing
    
    # Extract features
    feature_set = extract_features(data)
    
    print(f"Extracted {len(feature_set.feature_vector)} features")
    print(f"Feature vector shape: {feature_set.feature_vector.shape}")
    
    # Display some features
    print("\nTime domain features:")
    for feat_name, feat_values in list(feature_set.time_features.items())[:3]:
        print(f"  {feat_name}: mean={np.mean(feat_values):.3f}, "
              f"std={np.std(feat_values):.3f}")
    
    print("\nFrequency domain features:")
    for feat_name, feat_values in list(feature_set.frequency_features.items())[:3]:
        if feat_values.size > 0:
            print(f"  {feat_name}: mean={np.mean(feat_values):.3f}, "
                  f"std={np.std(feat_values):.3f}")
    
    # Visualize features
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Band powers
    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    band_powers = []
    for band in band_names:
        key = f'band_power_{band}'
        if key in feature_set.frequency_features:
            band_powers.append(np.mean(feature_set.frequency_features[key]))
        else:
            band_powers.append(0)
    
    axes[0, 0].bar(band_names, band_powers)
    axes[0, 0].set_xlabel('Frequency Band')
    axes[0, 0].set_ylabel('Average Power')
    axes[0, 0].set_title('Band Power Distribution')
    
    # Feature distribution
    axes[0, 1].hist(feature_set.feature_vector, bins=50, edgecolor='black')
    axes[0, 1].set_xlabel('Feature Value')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Feature Value Distribution')
    
    # Connectivity features
    if 'correlation' in feature_set.connectivity_features:
        corr_values = feature_set.connectivity_features['correlation']
        axes[1, 0].hist(corr_values, bins=30, edgecolor='black')
        axes[1, 0].set_xlabel('Correlation')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Channel Correlation Distribution')
    
    # Clinical features
    if 'spike_rate' in feature_set.clinical_features:
        spike_rates = feature_set.clinical_features['spike_rate']
        axes[1, 1].bar(range(len(spike_rates)), spike_rates)
        axes[1, 1].set_xlabel('Channel')
        axes[1, 1].set_ylabel('Spike Rate (Hz)')
        axes[1, 1].set_title('Detected Spike Rates')
    
    plt.tight_layout()
    plt.savefig('test_feature_extraction.png')
    plt.close()
    
    return feature_set

def main():
    """Run all tests."""
    print("Testing Comprehensive Preprocessing Pipeline")
    print("=" * 50)
    
    # Test 1: Signal Quality Assessment
    quality_metrics = test_signal_quality_assessment()
    
    # Test 2: Signal Filtering
    filtered_data = test_signal_filtering()
    
    # Test 3: Feature Extraction
    feature_set = test_feature_extraction()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print(f"Generated test outputs:")
    print("  - test_quality_assessment.png")
    print("  - test_signal_filtering.png")
    print("  - test_feature_extraction.png")

if __name__ == "__main__":
    main() 