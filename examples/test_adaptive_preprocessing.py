"""
Demonstration of Adaptive Preprocessing for HMS Brain Activity Classification
Shows how adaptive preprocessing automatically optimizes parameters based on data quality
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import (
    EEGPreprocessor, 
    AdaptivePreprocessor,
    SignalQualityAssessor
)
from src.preprocessing.adaptive_preprocessor import DataProfiler

def generate_synthetic_eeg(n_channels=19, n_samples=10000, quality='good'):
    """Generate synthetic EEG data with different quality levels."""
    
    # Base signal - mix of different frequency bands
    t = np.linspace(0, n_samples/200, n_samples)  # 200 Hz sampling rate
    
    # Initialize data
    eeg_data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Alpha rhythm (8-13 Hz)
        alpha = 0.5 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
        
        # Beta rhythm (13-30 Hz)
        beta = 0.3 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
        
        # Theta rhythm (4-8 Hz)
        theta = 0.4 * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi)
        
        # Combine rhythms
        eeg_data[ch] = alpha + beta + theta
        
        # Add quality-specific artifacts
        if quality == 'poor':
            # Add strong line noise
            line_noise = 2.0 * np.sin(2 * np.pi * 60 * t)
            eeg_data[ch] += line_noise
            
            # Add muscle artifacts (high frequency)
            muscle = 0.8 * np.random.randn(n_samples) * (np.random.rand(n_samples) > 0.7)
            eeg_data[ch] += muscle
            
            # Add movement artifacts (low frequency, high amplitude)
            if ch < 5 and np.random.rand() > 0.5:
                movement = 5.0 * np.sin(2 * np.pi * 0.5 * t)
                eeg_data[ch] += movement
                
            # Add random spikes
            spike_times = np.random.choice(n_samples, size=20, replace=False)
            eeg_data[ch, spike_times] += np.random.randn(20) * 10
            
        elif quality == 'medium':
            # Add moderate line noise
            line_noise = 0.5 * np.sin(2 * np.pi * 50 * t)
            eeg_data[ch] += line_noise
            
            # Add some white noise
            white_noise = 0.2 * np.random.randn(n_samples)
            eeg_data[ch] += white_noise
            
        elif quality == 'good':
            # Just add minimal white noise
            white_noise = 0.05 * np.random.randn(n_samples)
            eeg_data[ch] += white_noise
            
    # Make some channels bad
    if quality == 'poor':
        # Dead channel
        eeg_data[15] = np.random.randn(n_samples) * 0.001
        
        # Extremely noisy channel
        eeg_data[10] = np.random.randn(n_samples) * 20
        
    return eeg_data


def visualize_preprocessing_comparison(original, standard_preprocessed, adaptive_preprocessed, 
                                     channel_idx=0, save_path=None):
    """Visualize the comparison between preprocessing methods."""
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Time axis (assuming 200 Hz sampling rate)
    time = np.arange(original.shape[1]) / 200
    
    # Plot original signal
    axes[0].plot(time[:1000], original[channel_idx, :1000], 'b-', linewidth=0.5)
    axes[0].set_title('Original Signal', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude (μV)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot standard preprocessed
    axes[1].plot(time[:1000], standard_preprocessed[channel_idx, :1000], 'g-', linewidth=0.5)
    axes[1].set_title('Standard Preprocessing', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Amplitude (normalized)')
    axes[1].grid(True, alpha=0.3)
    
    # Plot adaptive preprocessed
    axes[2].plot(time[:1000], adaptive_preprocessed[channel_idx, :1000], 'r-', linewidth=0.5)
    axes[2].set_title('Adaptive Preprocessing', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Amplitude (normalized)')
    axes[2].grid(True, alpha=0.3)
    
    # Plot spectral comparison
    from scipy import signal
    
    # Compute PSDs
    freqs_orig, psd_orig = signal.welch(original[channel_idx], fs=200, nperseg=512)
    freqs_std, psd_std = signal.welch(standard_preprocessed[channel_idx], fs=200, nperseg=512)
    freqs_adp, psd_adp = signal.welch(adaptive_preprocessed[channel_idx], fs=200, nperseg=512)
    
    axes[3].semilogy(freqs_orig, psd_orig, 'b-', label='Original', alpha=0.7)
    axes[3].semilogy(freqs_std, psd_std, 'g-', label='Standard', alpha=0.7)
    axes[3].semilogy(freqs_adp, psd_adp, 'r-', label='Adaptive', alpha=0.7)
    axes[3].set_xlabel('Frequency (Hz)')
    axes[3].set_ylabel('PSD')
    axes[3].set_title('Power Spectral Density Comparison', fontsize=14, fontweight='bold')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim([0, 100])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_adaptive_preprocessing():
    """Main demonstration of adaptive preprocessing capabilities."""
    
    print("=" * 80)
    print("HMS Brain Activity Classification - Adaptive Preprocessing Demonstration")
    print("=" * 80)
    
    # Generate synthetic data with different quality levels
    channel_names = [f'CH{i+1}' for i in range(19)]
    
    # Test 1: High-quality data
    print("\nTest 1: High-Quality EEG Data")
    print("-" * 40)
    
    good_eeg = generate_synthetic_eeg(quality='good')
    
    # Initialize preprocessors
    standard_preprocessor = EEGPreprocessor(use_adaptive=False)
    adaptive_preprocessor = EEGPreprocessor(use_adaptive=True)
    
    # Process with both methods
    std_processed_good, std_info = standard_preprocessor.preprocess_eeg(good_eeg, channel_names)
    adp_processed_good, adp_info = adaptive_preprocessor.preprocess_eeg(good_eeg, channel_names)
    
    print(f"Standard preprocessing: {std_info['method']}")
    print(f"Adaptive preprocessing: {adp_info['method']}")
    print(f"Adaptive decision: Used {adp_info['method']} due to high initial quality")
    
    # Test 2: Poor-quality data
    print("\nTest 2: Poor-Quality EEG Data")
    print("-" * 40)
    
    poor_eeg = generate_synthetic_eeg(quality='poor')
    
    # Process with both methods
    std_processed_poor, std_info = standard_preprocessor.preprocess_eeg(poor_eeg, channel_names)
    adp_processed_poor, adp_info = adaptive_preprocessor.preprocess_eeg(poor_eeg, channel_names)
    
    print(f"Standard preprocessing: {std_info['method']}")
    print(f"Adaptive preprocessing: {adp_info['method']}")
    print(f"Initial quality score: {adp_info.get('initial_quality', 'N/A'):.3f}")
    print(f"Final quality score: {adp_info.get('final_quality', 'N/A'):.3f}")
    print(f"Quality improvement: {adp_info.get('quality_improvement', 0):.3f}")
    
    # Display optimized parameters
    if 'parameters' in adp_info and isinstance(adp_info['parameters'], dict):
        print("\nOptimized Parameters:")
        params = adp_info['parameters']
        print(f"  - Highpass filter: {params.get('highpass_freq', 'N/A'):.2f} Hz")
        print(f"  - Lowpass filter: {params.get('lowpass_freq', 'N/A'):.2f} Hz")
        print(f"  - Notch filter: {params.get('notch_freq', 'None')} Hz")
        print(f"  - ICA components: {params.get('ica_components', 'N/A')}")
        print(f"  - Artifact threshold: {params.get('artifact_threshold', 'N/A'):.2f}")
        print(f"  - Optimization confidence: {params.get('optimization_confidence', 'N/A'):.3f}")
    
    # Visualize comparison
    print("\nGenerating visualization...")
    visualize_preprocessing_comparison(
        poor_eeg, std_processed_poor, adp_processed_poor,
        channel_idx=0,
        save_path='adaptive_preprocessing_comparison.png'
    )
    
    # Test 3: Data profiling demonstration
    print("\nTest 3: Data Profiling Analysis")
    print("-" * 40)
    
    profiler = DataProfiler(sampling_rate=200)
    profile = profiler.profile_data(poor_eeg, channel_names)
    
    print("Data Profile Summary:")
    print(f"  - Dominant frequency: {profile['frequency_profile']['dominant_frequency']:.2f} Hz")
    print(f"  - Primary frequency band: ", end="")
    band_powers = profile['frequency_profile']['band_powers']
    primary_band = max(band_powers.items(), key=lambda x: x[1])
    print(f"{primary_band[0]} ({primary_band[1]:.2%})")
    
    print("\nNoise Analysis:")
    noise_profile = profile['noise_profile']
    print(f"  - Line noise (50Hz): {noise_profile.get('line_noise_50hz', 0):.2f}")
    print(f"  - Line noise (60Hz): {noise_profile.get('line_noise_60hz', 0):.2f}")
    print(f"  - High-frequency noise: {noise_profile['high_freq_noise']:.3f}")
    print(f"  - Low-frequency drift: {noise_profile['low_freq_drift']:.3f}")
    
    print("\nArtifact Detection:")
    artifact_profile = profile['artifact_profile']
    print(f"  - Eye blinks: {artifact_profile['eye_blink_score']:.2f} per second")
    print(f"  - Movement score: {artifact_profile['movement_score']:.3f}")
    print(f"  - Muscle artifacts: {artifact_profile['muscle_artifact_score']:.3f}")
    
    print("\nChannel Quality:")
    channel_quality = profile['channel_quality']
    print(f"  - Dead channels: {channel_quality['dead_channels']}")
    print(f"  - Noisy channels: {channel_quality['noisy_channels']}")
    print(f"  - Mean correlation: {channel_quality['mean_correlation']:.3f}")
    
    # Test 4: Performance metrics
    print("\nTest 4: Performance Metrics")
    print("-" * 40)
    
    # Process multiple recordings to gather metrics
    n_recordings = 10
    qualities = ['good', 'medium', 'poor']
    
    for i in range(n_recordings):
        quality = qualities[i % 3]
        test_eeg = generate_synthetic_eeg(quality=quality)
        _, _ = adaptive_preprocessor.preprocess_eeg(test_eeg, channel_names)
    
    # Get metrics
    metrics = adaptive_preprocessor.get_adaptive_metrics()
    if metrics:
        print(f"Total recordings processed: {metrics['total_processed']}")
        if 'avg_quality_improvement' in metrics:
            print(f"Average quality improvement: {metrics['avg_quality_improvement']:.3f}")
            print(f"Quality improvement std: {metrics['quality_improvement_std']:.3f}")
        if 'avg_processing_time' in metrics:
            print(f"Average processing time: {metrics['avg_processing_time']:.3f}s")
        print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
    
    # Quality assessment comparison
    print("\nTest 5: Quality Assessment Comparison")
    print("-" * 40)
    
    quality_assessor = SignalQualityAssessor(sampling_rate=200)
    
    # Assess original poor quality data
    orig_quality = quality_assessor.assess_quality(poor_eeg, channel_names)
    print(f"Original data quality score: {orig_quality.overall_quality_score:.3f}")
    print(f"  - SNR: {orig_quality.snr:.2f} dB")
    print(f"  - Artifact ratio: {orig_quality.artifact_ratio:.2%}")
    
    # Assess after standard preprocessing
    std_quality = quality_assessor.assess_quality(std_processed_poor, channel_names)
    print(f"\nStandard preprocessing quality score: {std_quality.overall_quality_score:.3f}")
    print(f"  - SNR: {std_quality.snr:.2f} dB")
    print(f"  - Artifact ratio: {std_quality.artifact_ratio:.2%}")
    
    # Assess after adaptive preprocessing
    adp_quality = quality_assessor.assess_quality(adp_processed_poor, channel_names)
    print(f"\nAdaptive preprocessing quality score: {adp_quality.overall_quality_score:.3f}")
    print(f"  - SNR: {adp_quality.snr:.2f} dB")
    print(f"  - Artifact ratio: {adp_quality.artifact_ratio:.2%}")
    
    print("\n" + "=" * 80)
    print("Demonstration Complete!")
    print("=" * 80)
    
    # Create a summary plot
    create_summary_plot(orig_quality, std_quality, adp_quality)


def create_summary_plot(orig_quality, std_quality, adp_quality):
    """Create a summary plot comparing quality metrics."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Quality scores comparison
    methods = ['Original', 'Standard', 'Adaptive']
    scores = [
        orig_quality.overall_quality_score,
        std_quality.overall_quality_score,
        adp_quality.overall_quality_score
    ]
    colors = ['red', 'yellow', 'green']
    
    bars1 = ax1.bar(methods, scores, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Quality Score', fontsize=12)
    ax1.set_title('Overall Quality Score Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars1, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom')
    
    # SNR and artifact ratio comparison
    snrs = [orig_quality.snr, std_quality.snr, adp_quality.snr]
    artifact_ratios = [
        orig_quality.artifact_ratio * 100,
        std_quality.artifact_ratio * 100,
        adp_quality.artifact_ratio * 100
    ]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, snrs, width, label='SNR (dB)', alpha=0.7)
    bars3 = ax2.bar(x + width/2, artifact_ratios, width, label='Artifact Ratio (%)', alpha=0.7)
    
    ax2.set_xlabel('Method')
    ax2.set_title('SNR and Artifact Ratio Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('adaptive_preprocessing_summary.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    demonstrate_adaptive_preprocessing() 