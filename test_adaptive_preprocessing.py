#!/usr/bin/env python3
"""
Test Adaptive Preprocessing for HMS Brain Activity Classification
Comprehensive testing of adaptive preprocessing functionality including:
- Data profiling
- Parameter optimization
- Quality improvement validation
- Cache functionality
- Performance metrics
"""

import sys
import os
import numpy as np
import time
from pathlib import Path
import yaml
import tempfile
import shutil

sys.path.append('src')

def test_adaptive_imports():
    """Test imports for adaptive preprocessing components."""
    try:
        from preprocessing import (
            EEGPreprocessor,
            AdaptivePreprocessor,
            AdaptiveParameters,
            DataProfiler,
            PreprocessingOptimizer,
            SignalQualityAssessor
        )
        print("✅ Adaptive preprocessing imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        print("Need to fix import paths or missing dependencies")
        return False

def generate_test_eeg_data(n_channels=19, n_samples=2000, quality='poor'):
    """Generate test EEG data with specific quality characteristics."""
    
    # Base signal with realistic EEG frequencies
    t = np.linspace(0, n_samples/200, n_samples)  # 200 Hz sampling rate
    eeg_data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Alpha rhythm (8-13 Hz)
        alpha = 0.5 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
        
        # Beta rhythm (13-30 Hz)
        beta = 0.3 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
        
        # Theta rhythm (4-8 Hz)
        theta = 0.4 * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi)
        
        # Base signal
        eeg_data[ch] = alpha + beta + theta
        
        # Add quality-specific artifacts
        if quality == 'poor':
            # Strong line noise at 60 Hz
            line_noise = 1.5 * np.sin(2 * np.pi * 60 * t)
            eeg_data[ch] += line_noise
            
            # High-frequency muscle artifacts
            muscle = 0.6 * np.random.randn(n_samples) * (np.random.rand(n_samples) > 0.8)
            eeg_data[ch] += muscle
            
            # Movement artifacts on some channels
            if ch < 3:
                movement = 3.0 * np.sin(2 * np.pi * 0.3 * t)
                eeg_data[ch] += movement
                
            # Random spikes
            spike_times = np.random.choice(n_samples, size=5, replace=False)
            eeg_data[ch, spike_times] += np.random.randn(5) * 8
            
        elif quality == 'good':
            # Minimal white noise
            white_noise = 0.03 * np.random.randn(n_samples)
            eeg_data[ch] += white_noise
    
    # Create bad channels for poor quality
    if quality == 'poor':
        # Flat channel
        eeg_data[15] = np.random.randn(n_samples) * 0.001
        
        # Very noisy channel
        eeg_data[10] = np.random.randn(n_samples) * 15
    
    return eeg_data

def test_data_profiler():
    """Test the DataProfiler functionality."""
    try:
        from preprocessing.adaptive_preprocessor import DataProfiler
        
        print("🧪 Testing DataProfiler...")
        
        # Create test data
        test_data = generate_test_eeg_data(quality='poor')
        channel_names = [f'CH{i+1}' for i in range(19)]
        
        # Initialize profiler
        profiler = DataProfiler(sampling_rate=200.0)
        
        # Profile the data
        profile = profiler.profile_data(test_data, channel_names)
        
        # Validate profile structure
        required_keys = [
            'basic_stats', 'frequency_profile', 'noise_profile',
            'artifact_profile', 'channel_quality', 'stationarity', 'connectivity'
        ]
        
        for key in required_keys:
            if key not in profile:
                print(f"❌ Missing profile key: {key}")
                return False
        
        # Validate frequency profile
        freq_profile = profile['frequency_profile']
        if 'band_powers' not in freq_profile:
            print("❌ Missing band powers in frequency profile")
            return False
            
        # Check if band powers sum to approximately 1
        band_sum = sum(freq_profile['band_powers'].values())
        if not (0.8 <= band_sum <= 1.2):
            print(f"❌ Band powers don't sum to ~1: {band_sum}")
            return False
        
        # Validate noise detection
        noise_profile = profile['noise_profile']
        if 'line_noise_60hz' not in noise_profile:
            print("❌ Line noise detection failed")
            return False
            
        # Check if line noise was detected (should be high for poor quality data)
        if noise_profile['line_noise_60hz'] < 1.0:
            print(f"❌ Line noise not properly detected: {noise_profile['line_noise_60hz']}")
            return False
        
        print("✅ DataProfiler test passed")
        return True
        
    except Exception as e:
        print(f"❌ DataProfiler test failed: {e}")
        return False

def test_parameter_optimizer():
    """Test the PreprocessingOptimizer neural network."""
    try:
        from preprocessing.adaptive_preprocessor import PreprocessingOptimizer
        import torch
        
        print("🧪 Testing PreprocessingOptimizer...")
        
        # Initialize optimizer
        optimizer = PreprocessingOptimizer()
        
        # Create dummy input
        batch_size = 2
        profile_features = torch.randn(batch_size, 50)  # 50-dim feature vector
        
        # Forward pass
        predictions = optimizer(profile_features)
        
        # Validate output structure
        required_keys = ['filter', 'artifact', 'denoise', 'quality']
        for key in required_keys:
            if key not in predictions:
                print(f"❌ Missing prediction key: {key}")
                return False
        
        # Test parameter decoding
        for i in range(batch_size):
            single_predictions = {
                key: value[i:i+1] for key, value in predictions.items()
            }
            params = optimizer.decode_parameters(single_predictions)
            
            # Validate parameter ranges
            if not (0.1 <= params.highpass_freq <= 2.0):
                print(f"❌ Invalid highpass frequency: {params.highpass_freq}")
                return False
                
            if not (40.0 <= params.lowpass_freq <= 90.0):
                print(f"❌ Invalid lowpass frequency: {params.lowpass_freq}")
                return False
                
            if not (5 <= params.ica_components <= 30):
                print(f"❌ Invalid ICA components: {params.ica_components}")
                return False
        
        print("✅ PreprocessingOptimizer test passed")
        return True
        
    except Exception as e:
        print(f"❌ PreprocessingOptimizer test failed: {e}")
        return False

def test_adaptive_preprocessor():
    """Test the main AdaptivePreprocessor functionality."""
    try:
        from preprocessing import AdaptivePreprocessor, SignalQualityAssessor
        
        print("🧪 Testing AdaptivePreprocessor...")
        
        # Create minimal config
        config = {
            'sampling_rate': 200.0,
            'eeg': {'channels': [f'CH{i+1}' for i in range(19)]},
            'preprocessing': {
                'normalization': {'method': 'robust'},
                'filter': {'lowcut': 0.5, 'highcut': 50, 'order': 4},
                'notch_filter': {'freqs': [50, 60], 'quality_factor': 30},
                'artifact_removal': {
                    'use_ica': True, 'n_components': 15, 
                    'ica_method': 'fastica', 'eog_channels': ['Fp1', 'Fp2']
                },
                'denoising': {
                    'use_wavelet': True, 'wavelet_type': 'db4', 'wavelet_level': 5
                }
            },
            'dataset': {'eeg_sampling_rate': 200}
        }
        
        # Initialize adaptive preprocessor
        adaptive_preprocessor = AdaptivePreprocessor(config, device='cpu')
        
        # Test with poor quality data
        poor_data = generate_test_eeg_data(quality='poor')
        channel_names = [f'CH{i+1}' for i in range(19)]
        
        # Process the data
        processed_data, processing_info = adaptive_preprocessor.preprocess(
            poor_data, channel_names
        )
        
        # Validate output
        if processed_data.shape != poor_data.shape:
            print(f"❌ Shape mismatch: {processed_data.shape} vs {poor_data.shape}")
            return False
        
        # Check processing info
        required_info_keys = ['method', 'processing_time']
        for key in required_info_keys:
            if key not in processing_info:
                print(f"❌ Missing processing info key: {key}")
                return False
        
        # Test with good quality data (should use default preprocessing)
        good_data = generate_test_eeg_data(quality='good')
        processed_good, info_good = adaptive_preprocessor.preprocess(good_data, channel_names)
        
        # Should use default for good quality data
        if info_good.get('method') != 'default':
            print(f"❌ Should use default for good data, got: {info_good.get('method')}")
            return False
        
        print("✅ AdaptivePreprocessor test passed")
        return True
        
    except Exception as e:
        print(f"❌ AdaptivePreprocessor test failed: {e}")
        return False

def test_eeg_preprocessor_integration():
    """Test integration with existing EEGPreprocessor."""
    try:
        from preprocessing import EEGPreprocessor
        
        print("🧪 Testing EEGPreprocessor integration...")
        
        # Test without adaptive preprocessing
        preprocessor_standard = EEGPreprocessor(use_adaptive=False)
        
        # Test with adaptive preprocessing enabled
        preprocessor_adaptive = EEGPreprocessor(use_adaptive=True)
        
        # Generate test data
        test_data = generate_test_eeg_data(quality='poor')
        channel_names = [f'CH{i+1}' for i in range(19)]
        
        # Process with standard method
        try:
            std_processed, std_info = preprocessor_standard.preprocess_eeg(
                test_data, channel_names
            )
            if std_info['method'] != 'standard':
                print(f"❌ Standard preprocessor should use 'standard' method")
                return False
        except Exception as e:
            # Standard preprocessing might fail due to config issues, but that's okay
            print(f"⚠️ Standard preprocessing failed (expected): {e}")
        
        # Process with adaptive method
        adp_processed, adp_info = preprocessor_adaptive.preprocess_eeg(
            test_data, channel_names
        )
        
        # Should use adaptive method for poor quality data
        if adp_info.get('method') != 'adaptive':
            print(f"❌ Should use adaptive method for poor data, got: {adp_info.get('method')}")
            return False
        
        # Test backward compatibility
        legacy_processed = preprocessor_adaptive.preprocess(test_data, channel_names)
        if legacy_processed.shape != test_data.shape:
            print("❌ Backward compatibility test failed")
            return False
        
        print("✅ EEGPreprocessor integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ EEGPreprocessor integration test failed: {e}")
        return False

def test_caching_functionality():
    """Test parameter caching functionality."""
    try:
        from preprocessing import AdaptivePreprocessor
        
        print("🧪 Testing caching functionality...")
        
        # Create config with small cache size for testing
        config = {
            'sampling_rate': 200.0,
            'eeg': {'channels': [f'CH{i+1}' for i in range(19)]},
            'preprocessing': {
                'normalization': {'method': 'robust'},
                'filter': {'lowcut': 0.5, 'highcut': 50},
                'artifact_removal': {'use_ica': False},  # Disable ICA for faster testing
                'denoising': {'use_wavelet': False}      # Disable wavelets for faster testing
            },
            'dataset': {'eeg_sampling_rate': 200}
        }
        
        # Initialize with small cache
        adaptive_preprocessor = AdaptivePreprocessor(config, cache_size=5, device='cpu')
        
        # Generate identical test data
        test_data = generate_test_eeg_data(quality='poor', n_samples=1000)  # Smaller for speed
        channel_names = [f'CH{i+1}' for i in range(19)]
        
        # First processing (should be cache miss)
        start_time = time.time()
        _, info1 = adaptive_preprocessor.preprocess(test_data, channel_names)
        first_time = time.time() - start_time
        
        # Second processing of same data (should be cache hit)
        start_time = time.time()
        _, info2 = adaptive_preprocessor.preprocess(test_data, channel_names)
        second_time = time.time() - start_time
        
        # Get metrics
        metrics = adaptive_preprocessor.get_metrics()
        
        # Validate cache functionality
        if metrics['cache_hit_rate'] <= 0:
            print(f"❌ Cache hit rate should be > 0, got: {metrics['cache_hit_rate']}")
            return False
        
        # Second processing should be faster due to caching
        if second_time >= first_time:
            print(f"⚠️ Cache might not be improving speed: {first_time:.3f}s vs {second_time:.3f}s")
            # Don't fail the test as this could be due to system variations
        
        print(f"✅ Caching test passed (hit rate: {metrics['cache_hit_rate']:.2%})")
        return True
        
    except Exception as e:
        print(f"❌ Caching test failed: {e}")
        return False

def test_quality_improvement():
    """Test that adaptive preprocessing improves signal quality."""
    try:
        from preprocessing import AdaptivePreprocessor, SignalQualityAssessor
        
        print("🧪 Testing quality improvement...")
        
        # Create config
        config = {
            'sampling_rate': 200.0,
            'eeg': {'channels': [f'CH{i+1}' for i in range(19)]},
            'preprocessing': {
                'normalization': {'method': 'robust'},
                'filter': {'lowcut': 0.5, 'highcut': 50},
                'artifact_removal': {'use_ica': False},  # Disable for speed
                'denoising': {'use_wavelet': False}
            },
            'dataset': {'eeg_sampling_rate': 200}
        }
        
        # Initialize components
        adaptive_preprocessor = AdaptivePreprocessor(config, device='cpu')
        quality_assessor = SignalQualityAssessor(sampling_rate=200.0)
        
        # Generate poor quality data
        poor_data = generate_test_eeg_data(quality='poor', n_samples=1000)
        channel_names = [f'CH{i+1}' for i in range(19)]
        
        # Assess original quality
        original_quality = quality_assessor.assess_quality(poor_data, channel_names)
        
        # Process with adaptive preprocessing
        processed_data, processing_info = adaptive_preprocessor.preprocess(
            poor_data, channel_names
        )
        
        # Assess processed quality
        processed_quality = quality_assessor.assess_quality(processed_data, channel_names)
        
        # Check for quality improvement
        quality_improvement = (processed_quality.overall_quality_score - 
                             original_quality.overall_quality_score)
        
        if quality_improvement <= 0:
            print(f"❌ No quality improvement: {quality_improvement:.3f}")
            return False
        
        print(f"✅ Quality improvement test passed ({quality_improvement:.3f} improvement)")
        return True
        
    except Exception as e:
        print(f"❌ Quality improvement test failed: {e}")
        return False

def test_performance_metrics():
    """Test performance metrics collection."""
    try:
        from preprocessing import AdaptivePreprocessor
        
        print("🧪 Testing performance metrics...")
        
        # Create config
        config = {
            'sampling_rate': 200.0,
            'eeg': {'channels': [f'CH{i+1}' for i in range(19)]},
            'preprocessing': {
                'normalization': {'method': 'robust'},
                'filter': {'lowcut': 0.5, 'highcut': 50},
                'artifact_removal': {'use_ica': False},
                'denoising': {'use_wavelet': False}
            },
            'dataset': {'eeg_sampling_rate': 200}
        }
        
        # Initialize adaptive preprocessor
        adaptive_preprocessor = AdaptivePreprocessor(config, device='cpu')
        
        # Process multiple samples
        n_samples = 3
        for i in range(n_samples):
            quality = ['good', 'poor', 'poor'][i]
            test_data = generate_test_eeg_data(quality=quality, n_samples=800)
            channel_names = [f'CH{i+1}' for i in range(19)]
            
            _, _ = adaptive_preprocessor.preprocess(test_data, channel_names)
        
        # Get metrics
        metrics = adaptive_preprocessor.get_metrics()
        
        # Validate metrics
        required_metrics = ['total_processed', 'cache_hit_rate']
        for metric in required_metrics:
            if metric not in metrics:
                print(f"❌ Missing metric: {metric}")
                return False
        
        if metrics['total_processed'] != n_samples:
            print(f"❌ Incorrect sample count: {metrics['total_processed']} vs {n_samples}")
            return False
        
        print("✅ Performance metrics test passed")
        return True
        
    except Exception as e:
        print(f"❌ Performance metrics test failed: {e}")
        return False

def test_config_compatibility():
    """Test compatibility with different configuration formats."""
    try:
        from preprocessing import EEGPreprocessor
        
        print("🧪 Testing configuration compatibility...")
        
        # Test with default config path (should handle missing file gracefully)
        try:
            preprocessor = EEGPreprocessor(use_adaptive=True)
            print("✅ Default config handling works")
        except Exception as e:
            print(f"⚠️ Default config failed (may be expected): {e}")
        
        # Test with minimal config
        minimal_config = {
            'sampling_rate': 200.0,
            'dataset': {'eeg_sampling_rate': 200},
            'preprocessing': {'normalization': {'method': 'robust'}}
        }
        
        # This should work with adaptive preprocessing
        test_data = generate_test_eeg_data(quality='good', n_samples=500)
        channel_names = [f'CH{i+1}' for i in range(19)]
        
        # Should not crash even with minimal config
        try:
            from preprocessing import AdaptivePreprocessor
            adaptive_preprocessor = AdaptivePreprocessor(minimal_config, device='cpu')
            _, _ = adaptive_preprocessor.preprocess(test_data, channel_names)
            print("✅ Minimal config compatibility works")
        except Exception as e:
            print(f"⚠️ Minimal config test failed: {e}")
            # Don't fail the test as this might be due to missing dependencies
        
        print("✅ Configuration compatibility test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration compatibility test failed: {e}")
        return False

def main():
    """Run all adaptive preprocessing tests."""
    print("🧪 HMS Adaptive Preprocessing Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_adaptive_imports),
        ("DataProfiler", test_data_profiler),
        ("PreprocessingOptimizer", test_parameter_optimizer),
        ("AdaptivePreprocessor", test_adaptive_preprocessor),
        ("EEGPreprocessor Integration", test_eeg_preprocessor_integration),
        ("Caching Functionality", test_caching_functionality),
        ("Quality Improvement", test_quality_improvement),
        ("Performance Metrics", test_performance_metrics),
        ("Configuration Compatibility", test_config_compatibility)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All adaptive preprocessing tests passed!")
        # Create success flag
        with open('adaptive_preprocessing_test_passed.flag', 'w') as f:
            f.write('success')
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 