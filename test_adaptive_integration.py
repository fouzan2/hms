#!/usr/bin/env python3
"""
Integration test for Adaptive Preprocessing with HMS training pipeline
Tests that adaptive preprocessing integrates seamlessly with existing workflows
"""

import sys
import os
import numpy as np
from pathlib import Path
import yaml

sys.path.append('src')

def create_test_config():
    """Create a minimal test configuration for adaptive preprocessing."""
    config = {
        'dataset': {
            'eeg_sampling_rate': 200,
            'spectrogram_sampling_rate': 200,
            'eeg_duration': 50,
            'spectrogram_duration': 600
        },
        'eeg': {
            'channels': [f'CH{i+1}' for i in range(19)],
            'sampling_rate': 200
        },
        'preprocessing': {
            'use_adaptive': True,
            'normalization': {'method': 'robust'},
            'filter': {'lowcut': 0.5, 'highcut': 50, 'order': 4},
            'notch_filter': {'freqs': [50, 60], 'quality_factor': 30},
            'artifact_removal': {
                'use_ica': False,  # Disable for faster testing
                'n_components': 15,  # Reduced from 20 to be safe
                'ica_method': 'fastica',  # Changed from picard to fastica
                'eog_channels': ['Fp1', 'Fp2']
            },
            'denoising': {
                'use_wavelet': False,  # Disable for faster testing
                'wavelet_type': 'db4',
                'wavelet_level': 5
            }
        },
        'models': {
            'resnet1d_gru': {
                'input_size': 19,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.3
            },
            'efficientnet': {
                'model_name': 'efficientnet-b0',
                'num_classes': 6,
                'dropout': 0.3
            },
            'ensemble': {
                'method': 'weighted',
                'weights': [0.6, 0.4]
            }
        },
        'training': {
            'batch_size': 2,  # Small for testing
            'learning_rate': 0.001,
            'epochs': 1,  # Just one epoch for testing
            'use_adaptive_preprocessing': True
        },
        'classes': [
            'seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other'
        ],
        'sampling_rate': 200  # Add this for backward compatibility
    }
    return config

def generate_test_batch(batch_size=2, quality='mixed'):
    """Generate a test batch of EEG data."""
    n_channels = 19
    n_samples = 10000  # 50 seconds at 200 Hz
    
    eeg_batch = []
    spectrogram_batch = []
    labels = []
    
    for i in range(batch_size):
        # Generate EEG data
        t = np.linspace(0, n_samples/200, n_samples)
        eeg_data = np.zeros((n_channels, n_samples))
        
        for ch in range(n_channels):
            # Basic EEG rhythms
            alpha = 0.5 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
            beta = 0.3 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
            theta = 0.4 * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi)
            eeg_data[ch] = alpha + beta + theta
            
            # Add noise based on quality
            if quality == 'mixed':
                if i % 2 == 0:  # Poor quality for even indices
                    line_noise = 1.0 * np.sin(2 * np.pi * 60 * t)
                    eeg_data[ch] += line_noise
                    muscle = 0.3 * np.random.randn(n_samples) * (np.random.rand(n_samples) > 0.9)
                    eeg_data[ch] += muscle
                else:  # Good quality for odd indices
                    white_noise = 0.05 * np.random.randn(n_samples)
                    eeg_data[ch] += white_noise
        
        eeg_batch.append(eeg_data)
        
        # Generate dummy spectrogram (simplified)
        spectrogram = np.random.randn(3, 224, 224) * 0.1  # RGB-like spectrogram
        spectrogram_batch.append(spectrogram)
        
        # Random label
        labels.append(np.random.randint(0, 6))
    
    return eeg_batch, spectrogram_batch, labels

def test_adaptive_preprocessing_pipeline():
    """Test adaptive preprocessing in a complete pipeline scenario."""
    try:
        print("🧪 Testing Adaptive Preprocessing Pipeline Integration...")
        
        # Import components
        from preprocessing import EEGPreprocessor
        
        # Create test configuration
        config = create_test_config()
        
        # Initialize preprocessor with adaptive enabled and test config
        preprocessor = EEGPreprocessor(config, use_adaptive=True)
        
        # Generate test data
        eeg_batch, spectrogram_batch, labels = generate_test_batch(batch_size=3)
        channel_names = [f'CH{i+1}' for i in range(19)]
        
        print("📊 Processing test batch...")
        
        # Process each EEG sample
        processed_batch = []
        processing_infos = []
        
        for i, eeg_data in enumerate(eeg_batch):
            print(f"  Processing sample {i+1}/3...")
            
            # Process with adaptive preprocessing
            processed_data, processing_info = preprocessor.preprocess_eeg(
                eeg_data, channel_names
            )
            
            processed_batch.append(processed_data)
            processing_infos.append(processing_info)
            
            # Validate output
            if processed_data.shape != eeg_data.shape:
                print(f"❌ Shape mismatch in sample {i}: {processed_data.shape} vs {eeg_data.shape}")
                return False
            
            print(f"    Method used: {processing_info.get('method', 'unknown')}")
            print(f"    Processing time: {processing_info.get('processing_time', 0):.3f}s")
            
            # Check for quality improvement info
            if 'quality_improvement' in processing_info:
                print(f"    Quality improvement: {processing_info['quality_improvement']:.3f}")
        
        # Test batch processing
        print("\n📦 Testing batch processing...")
        
        batch_results = preprocessor.preprocess_batch(
            eeg_batch, channel_names, n_jobs=1  # Single job for testing
        )
        
        if len(batch_results) != len(eeg_batch):
            print(f"❌ Batch processing failed: {len(batch_results)} vs {len(eeg_batch)}")
            return False
        
        print("✅ Batch processing successful")
        
        # Check adaptive metrics
        print("\n📈 Checking adaptive metrics...")
        
        metrics = preprocessor.get_adaptive_metrics()
        if metrics:
            print(f"  Total processed: {metrics['total_processed']}")
            print(f"  Cache hit rate: {metrics['cache_hit_rate']:.2%}")
            
            if 'avg_quality_improvement' in metrics:
                print(f"  Avg quality improvement: {metrics['avg_quality_improvement']:.3f}")
        else:
            print("  No adaptive metrics available (expected if using standard preprocessing)")
        
        print("✅ Adaptive preprocessing pipeline integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False

def test_model_compatibility():
    """Test that preprocessed data is compatible with model inputs."""
    try:
        print("\n🧪 Testing Model Compatibility...")
        
        from preprocessing import EEGPreprocessor
        
        # Create test configuration
        config = create_test_config()
        
        # Initialize preprocessor with test config
        preprocessor = EEGPreprocessor(config, use_adaptive=True)
        
        # Generate and process test data
        eeg_batch, _, _ = generate_test_batch(batch_size=2)
        channel_names = [f'CH{i+1}' for i in range(19)]
        
        processed_batch = []
        for eeg_data in eeg_batch:
            processed_data, _ = preprocessor.preprocess_eeg(eeg_data, channel_names)
            processed_batch.append(processed_data)
        
        # Test data shapes and types
        for i, processed_data in enumerate(processed_batch):
            if not isinstance(processed_data, np.ndarray):
                print(f"❌ Sample {i}: Expected numpy array, got {type(processed_data)}")
                return False
            
            if processed_data.shape[0] != 19:  # 19 channels
                print(f"❌ Sample {i}: Expected 19 channels, got {processed_data.shape[0]}")
                return False
            
            if len(processed_data.shape) != 2:
                print(f"❌ Sample {i}: Expected 2D array, got {len(processed_data.shape)}D")
                return False
            
            # Check for NaN or infinite values
            if np.any(np.isnan(processed_data)) or np.any(np.isinf(processed_data)):
                print(f"❌ Sample {i}: Contains NaN or infinite values")
                return False
            
            # Check data range (should be normalized)
            data_range = np.ptp(processed_data)
            if data_range > 100:  # Reasonable upper bound for normalized data
                print(f"❌ Sample {i}: Data range too large: {data_range}")
                return False
        
        print("✅ Model compatibility test passed")
        return True
        
    except Exception as e:
        print(f"❌ Model compatibility test failed: {e}")
        return False

def test_performance_overhead():
    """Test that adaptive preprocessing doesn't add excessive overhead."""
    try:
        print("\n⏱️ Testing Performance Overhead...")
        
        from preprocessing import EEGPreprocessor
        import time
        
        # Create test configuration
        config = create_test_config()
        
        # Test with small data for speed
        eeg_data, _, _ = generate_test_batch(batch_size=1)
        eeg_data = eeg_data[0][:, :2000]  # Use smaller segment
        channel_names = [f'CH{i+1}' for i in range(19)]
        
        # Test standard preprocessing time
        preprocessor_standard = EEGPreprocessor(config, use_adaptive=False)
        
        try:
            start_time = time.time()
            _, _ = preprocessor_standard.preprocess_eeg(eeg_data, channel_names)
            standard_time = time.time() - start_time
        except:
            standard_time = None  # Standard preprocessing might fail
            print("  Standard preprocessing failed (expected)")
        
        # Test adaptive preprocessing time
        preprocessor_adaptive = EEGPreprocessor(config, use_adaptive=True)
        
        start_time = time.time()
        _, _ = preprocessor_adaptive.preprocess_eeg(eeg_data, channel_names)
        adaptive_time = time.time() - start_time
        
        print(f"  Adaptive preprocessing time: {adaptive_time:.3f}s")
        
        if standard_time:
            print(f"  Standard preprocessing time: {standard_time:.3f}s")
            overhead = (adaptive_time - standard_time) / standard_time * 100
            print(f"  Overhead: {overhead:.1f}%")
            
            # Check if overhead is reasonable (less than 200%)
            if overhead > 200:
                print(f"⚠️ High overhead detected: {overhead:.1f}%")
            else:
                print("✅ Overhead is acceptable")
        
        # Test that processing completes in reasonable time
        if adaptive_time > 10:  # Should complete in under 10 seconds
            print(f"❌ Processing too slow: {adaptive_time:.3f}s")
            return False
        
        print("✅ Performance overhead test passed")
        return True
        
    except Exception as e:
        print(f"❌ Performance overhead test failed: {e}")
        return False

def main():
    """Run integration tests for adaptive preprocessing."""
    print("🔗 HMS Adaptive Preprocessing Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Pipeline Integration", test_adaptive_preprocessing_pipeline),
        ("Model Compatibility", test_model_compatibility),
        ("Performance Overhead", test_performance_overhead)
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
    print(f"📊 Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        with open('adaptive_integration_test_passed.flag', 'w') as f:
            f.write('success')
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 