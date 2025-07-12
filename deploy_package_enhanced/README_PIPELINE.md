# HMS Brain Activity Classification - Complete Pipeline Guide

## 🚀 Quick Start

### 1. **Environment Setup**
```bash
# Activate conda environment
conda activate hms

# Install dependencies (if not already installed)
pip install -r requirements.txt
```

### 2. **Configuration**
The pipeline uses `config/novita_enhanced_config.yaml` for all settings. Key configurations:

- **Dataset**: Raw data path, processing parameters
- **Models**: Architecture configurations for each model type
- **Training**: Hyperparameters, batch sizes, learning rates
- **Logging**: Wandb project, MLflow tracking URI

### 3. **Complete Pipeline Execution**

#### **Option A: Run All Stages (Recommended)**
```bash
python run_novita_training_enhanced.py --stage all
```

#### **Option B: Run Individual Stages**
```bash
# Stage 1: Data Download and Setup
python run_novita_training_enhanced.py --stage data

# Stage 2: Preprocessing
python run_novita_training_enhanced.py --stage preprocessing

# Stage 3: Foundation Model Pre-training
python run_novita_training_enhanced.py --stage foundation

# Stage 4: Individual Model Training
python run_novita_training_enhanced.py --stage models

# Stage 5: Ensemble Training
python run_novita_training_enhanced.py --stage ensemble

# Stage 6: Model Export and Deployment
python run_novita_training_enhanced.py --stage export
```

#### **Option C: Resume from Specific Stage**
```bash
# Resume from preprocessing stage
python run_novita_training_enhanced.py --stage preprocessing --resume

# Resume from model training
python run_novita_training_enhanced.py --stage models --resume
```

## 📊 Pipeline Stages Overview

### **Stage 1: Data Download and Setup**
- Downloads HMS dataset from Kaggle
- Extracts and validates data
- Sets up directory structure

### **Stage 2: Advanced Preprocessing**
- **EEG Signal Processing**:
  - Bandpass filtering (0.5-50 Hz)
  - Artifact removal
  - Normalization (robust scaling)
  - Segmentation into 10-second windows
  
- **Spectrogram Generation**:
  - STFT spectrograms
  - Multitaper spectrograms
  - Wavelet spectrograms
  - Mel-scaled spectrograms
  
- **Feature Extraction**:
  - Frequency band powers
  - Spectral features (centroid, bandwidth, rolloff)
  - Temporal features

### **Stage 3: Foundation Model Pre-training**
- Self-supervised learning on unlabeled EEG data
- Masked autoencoding
- Contrastive learning
- Transfer learning preparation

### **Stage 4: Individual Model Training**
Trains multiple model architectures:

1. **ResNet1D-GRU**: 1D CNN + GRU for temporal patterns
2. **EfficientNet**: 2D CNN for spectrogram analysis
3. **EEG Foundation Model**: Pre-trained transformer
4. **Ensemble Components**: Individual specialized models

### **Stage 5: Ensemble Training**
- Combines predictions from all models
- Learns optimal weights for each model
- Cross-validation for robustness

### **Stage 6: Model Export and Deployment**
- Model optimization (ONNX, TensorRT)
- API deployment preparation
- Performance benchmarking

## 🔧 Configuration Details

### **Dataset Configuration**
```yaml
dataset:
  raw_data_path: "data/raw"
  processed_data_path: "data/processed"
  eeg_sampling_rate: 200
  processing_batch_size: 100
  kaggle_competition: "hms-harmful-brain-activity-classification"
```

### **Preprocessing Configuration**
```yaml
preprocessing:
  eeg:
    filter_low: 0.5
    filter_high: 50
    notch_freq: 60
    segment_length: 2000  # 10 seconds at 200 Hz
    
  spectrogram:
    window_size: 256
    overlap: 128
    nfft: 512
    freq_min: 0.5
    freq_max: 50
```

### **Model Configurations**
```yaml
models:
  resnet1d_gru:
    enabled: true
    initial_filters: 64
    num_blocks: [2, 2, 2, 2]
    dropout: 0.3
    se_block: true
    
  efficientnet:
    enabled: true
    model_name: "efficientnet_b3"
    pretrained: true
    num_classes: 6
    
  eeg_foundation:
    enabled: true
    model_size: "base"
    pretrained: true
```

## 📈 Monitoring and Logging

### **Wandb Integration**
- Automatic experiment tracking
- Model performance metrics
- Hyperparameter logging
- Artifact management

### **MLflow Integration (Optional)**
- Model versioning
- Experiment tracking
- Model registry

### **Local Logging**
- Training logs in `logs/` directory
- Model checkpoints in `models/checkpoints/`
- Performance metrics in `logs/metrics/`

## 🎯 Expected Performance

### **Model Accuracies (Expected)**
- **ResNet1D-GRU**: ~85-88%
- **EfficientNet**: ~87-90%
- **EEG Foundation**: ~88-92%
- **Ensemble**: ~90-94%

### **Training Time Estimates**
- **Preprocessing**: 2-4 hours
- **Foundation Pre-training**: 4-8 hours
- **Individual Models**: 2-4 hours each
- **Ensemble Training**: 1-2 hours
- **Total Pipeline**: 12-24 hours

## 🚨 Troubleshooting

### **Common Issues**

1. **Kaggle API Error**
   ```bash
   # Set up Kaggle credentials
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   ```

2. **Memory Issues**
   - Reduce `processing_batch_size` in config
   - Use gradient checkpointing
   - Enable mixed precision training

3. **Wandb Connection Issues**
   - Check internet connection
   - Verify API key: `wandb login`
   - Run offline: `wandb offline`

4. **MLflow Connection Issues**
   - MLflow is optional, can be disabled
   - Set `mlflow_tracking_uri: null` in config

### **Resume Training**
The pipeline automatically saves training state and can resume from any stage:
```bash
# Check current status
python run_novita_training_enhanced.py --status

# Resume from last completed stage
python run_novita_training_enhanced.py --resume
```

## 📁 Output Structure

```
deploy_package_enhanced/
├── data/
│   ├── raw/                 # Downloaded dataset
│   └── processed/           # Preprocessed data
├── models/
│   ├── checkpoints/         # Training checkpoints
│   ├── final/              # Final trained models
│   └── deployments/        # Optimized models
├── logs/
│   ├── training/           # Training logs
│   └── metrics/            # Performance metrics
├── wandb/                  # Wandb experiment data
└── training_state.json     # Pipeline state
```

## 🎉 Success Indicators

- ✅ All stages complete without errors
- ✅ Model accuracies > 85%
- ✅ Ensemble accuracy > 90%
- ✅ Models exported successfully
- ✅ API deployment ready

## 🔄 Continuous Improvement

- Monitor model performance in production
- Retrain with new data periodically
- Update hyperparameters based on validation results
- Add new model architectures as needed 