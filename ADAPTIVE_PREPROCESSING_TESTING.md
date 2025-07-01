# Adaptive Preprocessing Testing Guide

## Overview

The HMS Brain Activity Classification project now includes comprehensive **Adaptive Preprocessing** functionality with a complete testing suite. This document explains the new testing features and how to use them.

## New Test Files

### 1. `test_adaptive_preprocessing.py`
**Comprehensive unit tests for adaptive preprocessing components**

**What it tests:**
- ✅ Import functionality for all adaptive preprocessing components
- ✅ DataProfiler: Analyzes EEG signal characteristics
- ✅ PreprocessingOptimizer: Neural network parameter optimization
- ✅ AdaptivePreprocessor: Main adaptive preprocessing functionality
- ✅ EEGPreprocessor integration: Backward compatibility
- ✅ Caching functionality: Parameter caching for performance
- ✅ Quality improvement: Validates signal quality enhancement
- ✅ Performance metrics: Metrics collection and reporting
- ✅ Configuration compatibility: Various config formats

**Key Features Tested:**
- Automatic data quality assessment
- Parameter optimization based on signal characteristics
- Intelligent caching for similar data patterns
- Quality score improvement validation
- Processing time and overhead measurement

### 2. `test_adaptive_integration.py`
**Integration tests for adaptive preprocessing with existing pipeline**

**What it tests:**
- 🔗 Pipeline integration with existing HMS components
- 🔗 Model compatibility with preprocessed data
- 🔗 Performance overhead measurement
- 🔗 Batch processing functionality
- 🔗 End-to-end workflow validation

**Key Scenarios:**
- Mixed quality data processing
- Batch vs. single sample processing
- Memory and time performance
- Data format compatibility

## Running the Tests

### Individual Test Execution

```bash
# Run adaptive preprocessing unit tests
python test_adaptive_preprocessing.py

# Run integration tests
python test_adaptive_integration.py
```

### Full Test Suite

```bash
# Run complete HMS test suite (includes adaptive preprocessing)
./run_all_tests.sh
```

## Test Results and Validation

### Success Indicators

The tests create flag files upon successful completion:
- ✅ `adaptive_preprocessing_test_passed.flag` - Unit tests passed
- ✅ `adaptive_integration_test_passed.flag` - Integration tests passed

### Validation Checklist

The updated `validation_checklist.sh` now includes:
- Preprocessing tests status
- Adaptive preprocessing tests status
- Feature availability checks
- Performance validation

## What Gets Tested

### 1. Data Profiling Accuracy
```python
# Tests signal analysis capabilities
profile = profiler.profile_data(eeg_data, channel_names)

# Validates:
- Frequency band analysis (delta, theta, alpha, beta, gamma)
- Noise detection (line noise, muscle artifacts, etc.)
- Channel quality assessment
- Artifact detection (eye blinks, movement, spikes)
- Connectivity analysis
```

### 2. Parameter Optimization
```python
# Tests neural network parameter prediction
optimizer = PreprocessingOptimizer()
predictions = optimizer(profile_features)
parameters = optimizer.decode_parameters(predictions)

# Validates:
- Filter parameter ranges (0.1-2 Hz highpass, 40-90 Hz lowpass)
- ICA component selection (5-30 components)
- Artifact thresholds (0.5-4.0 range)
- Method selection (wavelets, normalization)
```

### 3. Quality Improvement
```python
# Tests actual quality enhancement
original_quality = quality_assessor.assess_quality(raw_data)
processed_data, info = adaptive_preprocessor.preprocess(raw_data)
final_quality = quality_assessor.assess_quality(processed_data)

# Validates:
- Overall quality score improvement
- SNR enhancement
- Artifact reduction
- Processing time within limits
```

### 4. Performance Metrics
```python
# Tests caching and performance tracking
metrics = adaptive_preprocessor.get_metrics()

# Validates:
- Cache hit rate improvement
- Processing time tracking
- Quality improvement statistics
- Total samples processed
```

## Expected Performance

### Quality Improvements
- **Poor Quality Data**: 20-40% quality score improvement
- **Good Quality Data**: Uses default preprocessing (no overhead)
- **Mixed Data**: Adaptive selection based on initial assessment

### Performance Characteristics
- **Cache Hit Rate**: 60-80% for similar data patterns
- **Processing Time**: <10 seconds per 50-second EEG segment
- **Memory Usage**: <500MB cache by default
- **Overhead**: <200% compared to standard preprocessing

### Accuracy Targets
- **Parameter Selection**: >90% appropriate parameter choices
- **Quality Assessment**: >95% accurate quality scoring
- **Artifact Detection**: >90% sensitivity for major artifacts

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Fix: Check Python path and dependencies
   pip install -r requirements.txt
   export PYTHONPATH="${PYTHONPATH}:src"
   ```

2. **Memory Issues**
   ```bash
   # Fix: Reduce cache size or use CPU-only mode
   # In config: cache_size: 100, device: 'cpu'
   ```

3. **Slow Performance**
   ```bash
   # Fix: Disable heavy components for testing
   # Set: use_ica: false, use_wavelet: false
   ```

### Debug Mode

Run tests with debug information:
```bash
# Enable verbose logging
export HMS_LOG_LEVEL=DEBUG
python test_adaptive_preprocessing.py
```

## Integration with Existing Code

### Backward Compatibility
The adaptive preprocessing maintains full backward compatibility:

```python
# Old way (still works)
preprocessor = EEGPreprocessor()
processed_data = preprocessor.preprocess(eeg_data)

# New way (with adaptive features)
preprocessor = EEGPreprocessor(use_adaptive=True)
processed_data, info = preprocessor.preprocess_eeg(eeg_data, channel_names)

# Check what method was used
print(f"Method: {info['method']}")  # 'adaptive' or 'default'
print(f"Quality improvement: {info.get('quality_improvement', 0):.3f}")
```

### Training Pipeline Integration
```python
# Enable adaptive preprocessing in training
config['training']['use_adaptive_preprocessing'] = True

# The training pipeline will automatically use adaptive preprocessing
# for data with quality scores below the threshold (default: 0.9)
```

## Continuous Integration

The tests are automatically run as part of the CI/CD pipeline:
1. Unit tests validate individual components
2. Integration tests validate pipeline compatibility
3. Performance tests ensure acceptable overhead
4. Quality tests validate signal improvement

## Success Criteria

For the tests to pass, the following criteria must be met:
- ✅ All imports successful
- ✅ Data profiling accuracy >95%
- ✅ Parameter optimization within valid ranges
- ✅ Quality improvement on poor-quality data
- ✅ Cache functionality working
- ✅ Processing time <10 seconds per segment
- ✅ No data corruption or NaN values
- ✅ Backward compatibility maintained

## Future Enhancements

The testing framework is designed to be extensible for future features:
- Multi-modal preprocessing tests
- Real-time streaming tests
- Hardware-specific optimization tests
- Clinical validation tests with expert annotations

---

## Quick Start

To test the adaptive preprocessing immediately:

```bash
# 1. Run the tests
./run_all_tests.sh

# 2. Check results
cat adaptive_preprocessing_test_passed.flag
cat adaptive_integration_test_passed.flag

# 3. View validation
./validation_checklist.sh
```

The adaptive preprocessing is now fully integrated into your HMS testing pipeline and ready for production use! 