# HMS EEG Classification System - Technical Overview

## Project Purpose
The HMS EEG Classification System classifies six types of brain activities from EEG recordings:
- **Seizures**: Abnormal, excessive neuronal activity
- **LPD (Lateralized Periodic Discharges)**: Unilateral periodic patterns
- **GPD (Generalized Periodic Discharges)**: Bilateral periodic patterns  
- **LRDA (Lateralized Rhythmic Delta Activity)**: Unilateral rhythmic slow waves
- **GRDA (Generalized Rhythmic Delta Activity)**: Bilateral rhythmic slow waves
- **Other**: Background activity or artifacts

## 1. EEG Preprocessing Pipeline

### Signal Processing Steps
1. **Filtering**
   - Bandpass filter: 0.5-50 Hz to remove low/high frequency noise
   - Notch filter: 60 Hz to remove power line interference

2. **Artifact Removal**
   - ICA (Independent Component Analysis) to remove EOG/EMG artifacts
   - Bad channel detection and interpolation
   - Spike artifact detection and removal

3. **Signal Enhancement**
   - Wavelet denoising using multi-level decomposition
   - Baseline drift correction
   - Robust normalization (removes outliers)

4. **Quality Assessment**
   - Signal-to-noise ratio calculation
   - Artifact ratio measurement
   - Channel quality scoring

5. **Data Segmentation**
   - 50-second overlapping windows for training
   - Time-frequency analysis using spectrograms

### Feature Extraction
- **Time Domain**: Mean, standard deviation, skewness, kurtosis, zero crossings
- **Frequency Domain**: Power spectral density, band powers (delta, theta, alpha, beta, gamma)
- **Time-Frequency**: Wavelet coefficients, spectral entropy
- **Clinical Features**: Spike detection, sharp wave analysis

## 2. Training Pipeline

### Model Architectures

#### ResNet1D-GRU Model
```python
# Architecture for raw EEG signals
- ResNet1D backbone (conv1d layers with residual connections)
- Squeeze-and-excitation blocks for feature refinement
- Bidirectional GRU layers for temporal modeling
- Multi-head attention mechanism
- Classification head with dropout
```

#### EfficientNet Model
```python
# Architecture for spectrograms
- Pre-trained EfficientNet-B3 backbone
- Custom classification head for 6 classes
- Progressive training with unfreezing
- Mixup and CutMix augmentation
```

#### Ensemble Model
```python
# Final combined model
- Stacking ensemble of ResNet1D-GRU + EfficientNet
- Neural meta-learner for combination
- Uncertainty quantification
- Temperature scaling for calibration
```

### Training Strategy
1. **Data Preparation**
   - Patient-grouped train/validation split (80/20)
   - Stratified sampling to maintain class balance
   - Data augmentation (time-domain and frequency-domain)

2. **Training Configuration**
   - Mixed precision training (FP16) for memory efficiency
   - Gradient checkpointing for large models
   - Cosine annealing learning rate schedule
   - Early stopping with patience

3. **Loss Functions**
   - Focal loss to handle class imbalance
   - Label smoothing for better calibration
   - Weighted cross-entropy for critical classes

4. **Optimization**
   - Adam optimizer with weight decay
   - Gradient clipping for stability
   - Cross-validation with 5 folds

## 3. Visualization System

### Training Monitoring
- **Real-time Progress**: Loss curves, accuracy plots, learning rate tracking
- **Resource Usage**: GPU/CPU utilization, memory consumption
- **Training Metrics**: Per-class performance, confusion matrices

### Model Performance Analysis
- **Confusion Matrix**: Clinical-aware visualization highlighting critical misclassifications
- **ROC Curves**: Per-class receiver operating characteristic curves
- **Feature Importance**: SHAP analysis for model interpretability

### Clinical Dashboard
- **EEG Signal Viewer**: Multi-channel waveform display with annotations
- **Spectrogram Viewer**: Time-frequency visualization with interactive controls
- **Alert Timeline**: Real-time seizure detection and clinical alerts
- **Patient Reports**: Automated clinical summary generation

### Interactive Features
- Real-time dashboard built with Dash and Plotly
- Prometheus metrics integration for monitoring
- Grafana dashboards for system health tracking

## 4. Model Optimization

### Compression Techniques
1. **Quantization**
   - INT8 quantization for 4x smaller models
   - Post-training quantization with calibration dataset
   - Maintains 99%+ accuracy while reducing size

2. **Pruning**
   - Structured pruning removing entire channels/layers
   - Unstructured pruning removing individual weights
   - 30-50% parameter reduction with minimal accuracy loss

3. **Knowledge Distillation**
   - Teacher-student training to compress ensemble
   - Single model with ensemble-level performance
   - Faster inference for production deployment

### Export Formats
- **ONNX**: Cross-platform model deployment
- **TensorRT**: GPU-optimized inference for NVIDIA hardware
- **Optimized PyTorch**: Quantized models for CPU inference

### Performance Improvements
- **Inference Speed**: 3-5x faster than original models
- **Memory Usage**: 70% reduction in RAM requirements
- **Model Size**: 75% smaller disk footprint

## 5. Final Model Architecture

### Ensemble Model Composition
The final production model is an **ensemble** that combines:

1. **ResNet1D-GRU Branch** (for raw EEG)
   - Input: 19-channel EEG signals (50 seconds, 200Hz)
   - ResNet1D feature extraction with residual connections
   - Bidirectional GRU for temporal sequence modeling
   - Multi-head attention for important pattern focus

2. **EfficientNet Branch** (for spectrograms)
   - Input: Multi-channel spectrograms (frequency-time representations)
   - Pre-trained EfficientNet-B3 for image feature extraction
   - Custom head adapted for EEG spectrogram patterns

3. **Meta-Learner Fusion**
   - Neural network that combines predictions from both branches
   - Learns optimal weighting based on input characteristics
   - Provides uncertainty estimates and confidence scores

### Model Capabilities
The final model performs:

#### Primary Function
- **Multi-class Classification**: Predicts one of 6 brain activity types
- **Confidence Scoring**: Provides prediction confidence (0-1 scale)
- **Uncertainty Quantification**: Estimates prediction uncertainty using Monte Carlo dropout

#### Output Format
```python
{
    'predicted_class': 'Seizure',
    'class_probabilities': {
        'Seizure': 0.95,
        'LPD': 0.02,
        'GPD': 0.01,
        'LRDA': 0.01,
        'GRDA': 0.01,
        'Other': 0.00
    },
    'confidence': 0.95,
    'uncertainty': 0.05,
    'clinical_recommendation': 'URGENT: Immediate clinical intervention required'
}
```