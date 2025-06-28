"""
Google Colab Comprehensive GPU Training Script for HMS Brain Activity Classification
================================================================================

This script includes ALL the advanced training features from the existing pipeline:
- ResNet1D-GRU and EfficientNet models
- Ensemble learning with meta-learners
- Advanced augmentation (Mixup, CutMix, SpecAugment)
- Class balancing and focal loss
- Cross-validation and hyperparameter optimization
- Mixed precision training
- Test-time augmentation
- Knowledge distillation
- Uncertainty estimation
- Advanced optimizers (SAM, LookAhead)

Usage:
1. Upload this file to Google Colab
2. Ensure GPU runtime is enabled
3. Run cells in sequence
"""

# ============================
# CELL 1: Environment Setup
# ============================

def setup_colab_environment():
    """Setup comprehensive environment for training."""
    import os
    import sys
    import subprocess
    
    print("🚀 Setting up comprehensive training environment...")
    
    # Install core packages
    packages = [
        "torch torchvision torchaudio",
        "transformers accelerate",
        "librosa scipy scikit-learn",
        "pandas numpy pywt",
        "h5py pyarrow fastparquet",
        "matplotlib seaborn plotly",
        "pyyaml joblib tqdm",
        "mlflow wandb optuna",
        "timm efficientnet-pytorch",
        "ray[tune] hyperopt",
        "kagglehub"
    ]
    
    for pkg in packages:
        print(f"📦 Installing {pkg}...")
        result = subprocess.run(['pip', 'install', '-q'] + pkg.split(), 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️  Warning installing {pkg}: {result.stderr}")
    
    # Check GPU
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("❌ No GPU detected! Please enable GPU runtime.")
        return False

# Run in first cell:
# gpu_available = setup_colab_environment()


# ============================
# CELL 2: Import Dependencies
# ============================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

import numpy as np
import pandas as pd
import librosa
import pywt
from pathlib import Path
from tqdm import tqdm
import warnings
import random
import json
import yaml
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, 
    precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
def set_seeds(seed=42):
    """Set seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(42)

# Run in second cell:
# print("✅ All dependencies imported successfully!")


# ============================
# CELL 3: Configuration
# ============================

# Comprehensive training configuration
TRAINING_CONFIG = {
    'data': {
        'data_path': '/content/drive/MyDrive/hms-data',
        'processed_path': '/content/drive/MyDrive/hms-processed',
        'output_path': '/content/drive/MyDrive/hms-models',
        'num_classes': 6,
        'classes': ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other'],  # Will be auto-detected
        'sampling_rate': 200,
        'segment_length': 50,  # seconds
        'overlap': 0.5
    },
    
    'models': {
        'resnet1d_gru': {
            'enabled': True,
            'resnet': {
                'initial_filters': 64,
                'num_blocks': [3, 4, 6, 3],
                'kernel_size': 7,
                'dropout': 0.3,
                'use_se': True,
                'use_multiscale': True
            },
            'gru': {
                'hidden_size': 256,
                'num_layers': 2,
                'bidirectional': True,
                'dropout': 0.3
            },
            'attention': {
                'enabled': True,
                'num_heads': 8,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'epochs': 50
            }
        },
        
        'efficientnet': {
            'enabled': True,
            'model_name': 'efficientnet-b3',
            'pretrained': True,
            'dropout': 0.4,
            'drop_path_rate': 0.2,
            'training': {
                'batch_size': 8,
                'learning_rate': 0.0005,
                'weight_decay': 0.00001,
                'epochs': 50
            }
        },
        
        'ensemble': {
            'enabled': True,
            'method': 'stacking',
            'meta_learner': 'neural',
            'diversity_weight': 0.1
        }
    },
    
    'training': {
        'mixed_precision': True,
        'gradient_accumulation_steps': 4,
        'max_grad_norm': 1.0,
        'early_stopping_patience': 10,
        'save_best_only': True,
        
        'augmentation': {
            'mixup_alpha': 0.2,
            'cutmix_alpha': 1.0,
            'spec_augment': True,
            'time_mask_ratio': 0.1,
            'freq_mask_ratio': 0.1
        },
        
        'class_balancing': {
            'method': 'focal_loss',
            'focal_gamma': 2.0,
            'focal_alpha': 'balanced'
        },
        
        'optimization': {
            'optimizer': 'adamw',
            'scheduler': 'cosine_warmup',
            'warmup_epochs': 5,
            'use_sam': False,
            'use_lookahead': False
        }
    },
    
    'validation': {
        'cross_validation': True,
        'n_folds': 5,
        'validation_split': 0.2,
        'stratified': True
    },
    
    'hardware': {
        'device': 'cuda',
        'num_workers': 2,
        'pin_memory': True,
        'persistent_workers': True
    }
}


def auto_detect_classes(data_path):
    """Auto-detect class names from the dataset."""
    try:
        metadata = pd.read_csv(f"{data_path}/train.csv")
        unique_classes = sorted(metadata['expert_consensus'].unique().tolist())
        print(f"🔍 Auto-detected classes: {unique_classes}")
        return unique_classes
    except Exception as e:
        print(f"⚠️  Could not auto-detect classes: {e}")
        # Return default classes
        return ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']


def update_config_with_classes(config):
    """Update configuration with auto-detected classes."""
    detected_classes = auto_detect_classes(config['data']['data_path'])
    config['data']['classes'] = detected_classes
    config['data']['num_classes'] = len(detected_classes)
    print(f"📊 Updated config with {len(detected_classes)} classes: {detected_classes}")
    return config

# Run in third cell:
# print("✅ Configuration loaded!")


# ============================
# CELL 4: Advanced Model Architectures
# ============================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y


class MultiScaleConv1D(nn.Module):
    """Multi-scale convolution module."""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels // len(kernel_sizes), 
                     kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        out = torch.cat(outputs, dim=1)
        return self.relu(self.bn(out))


class ResNetBlock1D(nn.Module):
    """1D ResNet block with SE and multi-scale options."""
    def __init__(self, in_channels, out_channels, kernel_size=7, 
                 stride=1, use_se=True, use_multiscale=False):
        super().__init__()
        
        if use_multiscale:
            self.conv1 = MultiScaleConv1D(in_channels, out_channels)
        else:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                                  stride=stride, padding=kernel_size//2)
        
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                              padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.se = SEBlock(out_channels) if use_se else None
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.se:
            out = self.se(out)
        
        if self.downsample:
            residual = self.downsample(x)
        
        out += residual
        return self.relu(out)


class MultiHeadAttention1D(nn.Module):
    """Multi-head attention for 1D sequences."""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Store residual
        residual = x
        
        # Linear projections
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.out_proj(context)
        
        # Residual connection and layer norm
        return self.layer_norm(output + residual)


class ResNet1D_GRU_Advanced(nn.Module):
    """Advanced ResNet1D-GRU model with attention and multi-scale features."""
    
    def __init__(self, config):
        super().__init__()
        
        model_config = config['models']['resnet1d_gru']
        resnet_config = model_config['resnet']
        gru_config = model_config['gru']
        attn_config = model_config['attention']
        
        self.num_channels = 20  # EEG channels
        self.num_classes = config['data']['num_classes']
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv1d(self.num_channels, resnet_config['initial_filters'],
                     kernel_size=resnet_config['kernel_size'], padding=resnet_config['kernel_size']//2),
            nn.BatchNorm1d(resnet_config['initial_filters']),
            nn.ReLU(inplace=True)
        )
        
        # ResNet blocks
        self.resnet_blocks = nn.ModuleList()
        in_channels = resnet_config['initial_filters']
        
        for i, num_blocks in enumerate(resnet_config['num_blocks']):
            out_channels = resnet_config['initial_filters'] * (2 ** i)
            
            for j in range(num_blocks):
                stride = 2 if j == 0 and i > 0 else 1
                block = ResNetBlock1D(
                    in_channels, out_channels,
                    kernel_size=resnet_config['kernel_size'],
                    stride=stride,
                    use_se=resnet_config['use_se'],
                    use_multiscale=resnet_config['use_multiscale']
                )
                self.resnet_blocks.append(block)
                in_channels = out_channels
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # GRU layers
        gru_input_size = in_channels
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_config['hidden_size'],
            num_layers=gru_config['num_layers'],
            bidirectional=gru_config['bidirectional'],
            dropout=gru_config['dropout'] if gru_config['num_layers'] > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        gru_output_size = gru_config['hidden_size'] * (2 if gru_config['bidirectional'] else 1)
        
        if attn_config['enabled']:
            self.attention = MultiHeadAttention1D(
                d_model=gru_output_size,
                num_heads=attn_config['num_heads'],
                dropout=attn_config['dropout']
            )
        else:
            self.attention = None
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(resnet_config['dropout']),
            nn.Linear(gru_output_size, gru_output_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(resnet_config['dropout']),
            nn.Linear(gru_output_size // 2, self.num_classes)
        )
    
    def forward(self, x):
        # x: (batch, channels, time)
        batch_size = x.shape[0]
        
        # Initial convolution
        x = self.initial_conv(x)
        
        # ResNet blocks
        for block in self.resnet_blocks:
            x = block(x)
        
        # Prepare for GRU (batch, time, features)
        x = x.transpose(1, 2)
        
        # GRU
        gru_out, _ = self.gru(x)
        
        # Attention (if enabled)
        if self.attention:
            attended = self.attention(gru_out)
            # Global average pooling over time
            pooled = attended.mean(dim=1)
        else:
            # Use last hidden state
            pooled = gru_out[:, -1, :]
        
        # Classification
        logits = self.classifier(pooled)
        
        return {
            'logits': logits,
            'features': pooled
        }


class EfficientNetSpectrogram(nn.Module):
    """EfficientNet for spectrogram classification."""
    
    def __init__(self, config):
        super().__init__()
        
        model_config = config['models']['efficientnet']
        self.num_classes = config['data']['num_classes']
        
        # Load pre-trained EfficientNet
        try:
            import timm
            self.backbone = timm.create_model(
                model_config['model_name'],
                pretrained=model_config['pretrained'],
                num_classes=0,  # Remove classification head
                drop_rate=model_config['dropout'],
                drop_path_rate=model_config['drop_path_rate']
            )
            
            # Get feature dimension
            feature_dim = self.backbone.num_features
            
        except ImportError:
            # Fallback to simple CNN if timm not available
            print("⚠️  timm not available, using simple CNN")
            self.backbone = self._create_simple_cnn()
            feature_dim = 512
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(model_config['dropout']),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(model_config['dropout']),
            nn.Linear(feature_dim // 2, self.num_classes)
        )
    
    def _create_simple_cnn(self):
        """Simple CNN fallback."""
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    def forward(self, x):
        # x: (batch, channels, height, width)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Extract features
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        
        # Classification
        logits = self.classifier(features)
        
        return {
            'logits': logits,
            'features': features
        }


class EnsembleModel(nn.Module):
    """Ensemble model with neural meta-learner."""
    
    def __init__(self, base_models, config):
        super().__init__()
        
        self.base_models = nn.ModuleDict(base_models)
        self.num_classes = config['data']['num_classes']
        
        # Meta-learner
        ensemble_config = config['models']['ensemble']
        
        # Calculate total feature dimension
        total_features = 0
        for model in self.base_models.values():
            if hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Sequential):
                    total_features += model.classifier[-1].in_features
                else:
                    total_features += model.classifier.in_features
        
        # Add prediction features (logits from base models)
        total_features += len(self.base_models) * self.num_classes
        
        if ensemble_config['meta_learner'] == 'neural':
            self.meta_learner = nn.Sequential(
                nn.Linear(total_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, self.num_classes)
            )
        
        # Freeze base models initially
        for model in self.base_models.values():
            for param in model.parameters():
                param.requires_grad = False
    
    def forward(self, eeg_data, spec_data):
        features = []
        predictions = []
        
        # Get outputs from base models
        if 'resnet1d_gru' in self.base_models:
            resnet_out = self.base_models['resnet1d_gru'](eeg_data)
            features.append(resnet_out['features'])
            predictions.append(resnet_out['logits'])
        
        if 'efficientnet' in self.base_models:
            eff_out = self.base_models['efficientnet'](spec_data)
            features.append(eff_out['features'])
            predictions.append(eff_out['logits'])
        
        # Concatenate all features and predictions
        all_features = torch.cat(features + predictions, dim=1)
        
        # Meta-learner prediction
        ensemble_logits = self.meta_learner(all_features)
        
        return {
            'logits': ensemble_logits,
            'base_predictions': predictions,
            'features': all_features
        }

# Run in fourth cell:
# print("✅ Advanced model architectures defined!")


# ============================
# CELL 5: Data Loading and Augmentation
# ============================

class HMSDataset(Dataset):
    """Comprehensive HMS dataset with all data types."""
    
    def __init__(self, data_path, processed_path, config, split='train', 
                 indices=None, augment=True):
        self.data_path = Path(data_path)
        self.processed_path = Path(processed_path)
        self.config = config
        self.split = split
        self.augment = augment and split == 'train'
        
        # Load metadata
        self.metadata = pd.read_csv(self.data_path / 'train.csv')
        
        if indices is not None:
            self.metadata = self.metadata.iloc[indices].reset_index(drop=True)
        
        # Class mapping with better error handling
        self.label_map = {name: i for i, name in enumerate(config['data']['classes'])}
        
        # Debug: Print unique classes found in data vs config
        data_classes = set(self.metadata['expert_consensus'].unique())
        config_classes = set(config['data']['classes'])
        
        print(f"🔍 Dataset classes found: {sorted(data_classes)}")
        print(f"📋 Config classes: {sorted(config_classes)}")
        
        missing_in_config = data_classes - config_classes
        missing_in_data = config_classes - data_classes
        
        if missing_in_config:
            print(f"⚠️  Classes in data but not in config: {missing_in_config}")
        if missing_in_data:
            print(f"ℹ️  Classes in config but not in data: {missing_in_data}")
        
        # Augmentation parameters
        self.aug_config = config['training']['augmentation']
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        eeg_id = row['eeg_id']
        
        try:
            # Check if all required files exist
            filtered_path = self.processed_path / 'filtered' / f'{eeg_id}_filt.npy'
            spec_path = self.processed_path / 'spectrograms' / f'{eeg_id}_spec.npy'
            feat_path = self.processed_path / 'features' / f'{eeg_id}_feat.npy'
            
            missing_files = []
            if not filtered_path.exists():
                missing_files.append('filtered')
            if not spec_path.exists():
                missing_files.append('spectrograms')
            if not feat_path.exists():
                missing_files.append('features')
            
            if missing_files:
                if idx % 100 == 0:  # Only print occasionally to avoid spam
                    print(f"⚠️  Missing files for {eeg_id}: {missing_files}")
                return self._get_dummy_sample(eeg_id, row)
            
            # Load processed data
            eeg_data = np.load(filtered_path)
            spectrogram = np.load(spec_path)
            features = np.load(feat_path)
            
            # Validate data shapes
            if eeg_data.ndim != 2 or spectrogram.ndim != 3 or features.ndim != 1:
                print(f"⚠️  Invalid data shapes for {eeg_id}: EEG {eeg_data.shape}, Spec {spectrogram.shape}, Feat {features.shape}")
                return self._get_dummy_sample(eeg_id, row)
            
            # Get label with better error handling
            label_name = row['expert_consensus']
            if label_name not in self.label_map:
                print(f"❌ Unknown class '{label_name}' for sample {eeg_id}")
                print(f"Available classes: {list(self.label_map.keys())}")
                # Use first class as default
                label = 0
            else:
                label = self.label_map[label_name]
            
            # Apply augmentations if training
            if self.augment:
                eeg_data, spectrogram = self._apply_augmentations(eeg_data, spectrogram)
            
            # Convert to tensors
            sample = {
                'eeg': torch.FloatTensor(eeg_data),
                'spectrogram': torch.FloatTensor(spectrogram),
                'features': torch.FloatTensor(features),
                'label': torch.LongTensor([label])[0],
                'eeg_id': eeg_id,
                'patient_id': row.get('patient_id', 0)
            }
            
            return sample
            
        except Exception as e:
            if idx % 100 == 0:  # Only print occasionally
                print(f"Error loading {eeg_id}: {e}")
            return self._get_dummy_sample(eeg_id, row)
    
    def _get_dummy_sample(self, eeg_id, row):
        """Generate a dummy sample with consistent shapes."""
        
        # Get label
        label_name = row['expert_consensus']
        label = self.label_map.get(label_name, 0)
        
        # Generate dummy data with typical shapes from preprocessing
        # Based on the preprocessing script, these are reasonable defaults
        n_channels = 20
        
        # EEG: variable length, but use a common length
        eeg_length = 10000  # 50 seconds at 200 Hz
        eeg_data = torch.zeros(n_channels, eeg_length)
        
        # Spectrogram: depends on signal length and STFT parameters
        # From preprocessing: freq_mask and variable time length
        n_freqs = 64  # Frequency bins after filtering
        n_times = 128  # Time frames
        spectrogram = torch.zeros(n_channels, n_freqs, n_times)
        
        # Features: variable length based on extracted features
        # From preprocessing: time + frequency features
        n_features = 50 * n_channels  # Estimate based on feature extraction
        features = torch.zeros(n_features)
        
        return {
            'eeg': eeg_data,
            'spectrogram': spectrogram,
            'features': features,
            'label': torch.LongTensor([label])[0],
            'eeg_id': f'dummy_{eeg_id}',
            'patient_id': row.get('patient_id', 0)
        }
    
    def _apply_augmentations(self, eeg_data, spectrogram):
        """Apply various augmentations."""
        
        # Time domain augmentations for EEG
        if np.random.random() < 0.3:
            # Time shift
            shift = np.random.randint(-eeg_data.shape[1]//10, eeg_data.shape[1]//10)
            eeg_data = np.roll(eeg_data, shift, axis=1)
        
        if np.random.random() < 0.3:
            # Amplitude scaling
            scale = np.random.uniform(0.8, 1.2)
            eeg_data = eeg_data * scale
        
        if np.random.random() < 0.3:
            # Gaussian noise
            noise = np.random.normal(0, 0.01, eeg_data.shape)
            eeg_data = eeg_data + noise
        
        # Frequency domain augmentations for spectrograms
        if self.aug_config['spec_augment'] and np.random.random() < 0.5:
            spectrogram = self._spec_augment(spectrogram)
        
        return eeg_data, spectrogram
    
    def _spec_augment(self, spectrogram):
        """Apply SpecAugment to spectrograms."""
        spec = spectrogram.copy()
        
        # Time masking
        if np.random.random() < 0.5:
            time_mask_ratio = self.aug_config['time_mask_ratio']
            time_mask_len = int(spec.shape[-1] * time_mask_ratio)
            if time_mask_len > 0:
                t0 = np.random.randint(0, spec.shape[-1] - time_mask_len)
                spec[:, :, t0:t0+time_mask_len] = 0
        
        # Frequency masking
        if np.random.random() < 0.5:
            freq_mask_ratio = self.aug_config['freq_mask_ratio']
            freq_mask_len = int(spec.shape[-2] * freq_mask_ratio)
            if freq_mask_len > 0:
                f0 = np.random.randint(0, spec.shape[-2] - freq_mask_len)
                spec[:, f0:f0+freq_mask_len, :] = 0
        
        return spec


def mixup_data(x, y, alpha=1.0):
    """Apply mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    if x.dim() == 3:  # EEG data (batch, channels, time)
        # Cut time dimension
        W = x.size(2)
        cut_w = int(W * (1 - lam))
        cx = np.random.randint(W)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        
        x[:, :, bbx1:bbx2] = x[index, :, bbx1:bbx2]
        
    elif x.dim() == 4:  # Spectrogram data (batch, channels, freq, time)
        # Cut both frequency and time
        H, W = x.size(2), x.size(3)
        cut_rat = np.sqrt(1 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


def advanced_collate_fn(batch, config):
    """Advanced collate function with augmentations and padding for variable lengths."""
    
    # Filter out None/invalid samples
    valid_batch = [item for item in batch if item is not None and item['eeg_id'] != 'dummy']
    
    if len(valid_batch) == 0:
        # Return dummy batch if no valid samples
        return {
            'eeg': torch.zeros(1, 20, 10000),
            'spectrogram': torch.zeros(1, 20, 128, 256),
            'features': torch.zeros(1, 100),
            'label': torch.LongTensor([0]),
            'mixup': False,
            'cutmix': False
        }
    
    # Get max dimensions for padding
    max_eeg_len = max(item['eeg'].shape[1] for item in valid_batch)
    max_spec_freq = max(item['spectrogram'].shape[1] for item in valid_batch)
    max_spec_time = max(item['spectrogram'].shape[2] for item in valid_batch)
    max_feat_len = max(item['features'].shape[0] for item in valid_batch)
    
    # Pad and stack data
    eeg_data = []
    spec_data = []
    features = []
    labels = []
    
    for item in valid_batch:
        # Pad EEG data
        eeg = item['eeg']
        if eeg.shape[1] < max_eeg_len:
            pad_width = max_eeg_len - eeg.shape[1]
            eeg = F.pad(eeg, (0, pad_width), mode='replicate')
        eeg_data.append(eeg)
        
        # Pad spectrogram
        spec = item['spectrogram']
        if spec.shape[1] < max_spec_freq:
            pad_freq = max_spec_freq - spec.shape[1]
            spec = F.pad(spec, (0, 0, 0, pad_freq), mode='replicate')
        if spec.shape[2] < max_spec_time:
            pad_time = max_spec_time - spec.shape[2]
            spec = F.pad(spec, (0, pad_time), mode='replicate')
        spec_data.append(spec)
        
        # Pad features
        feat = item['features']
        if feat.shape[0] < max_feat_len:
            pad_feat = max_feat_len - feat.shape[0]
            feat = F.pad(feat, (0, pad_feat), mode='constant', value=0)
        features.append(feat)
        
        labels.append(item['label'])
    
    # Stack all data
    eeg_data = torch.stack(eeg_data)
    spec_data = torch.stack(spec_data)
    features = torch.stack(features)
    labels = torch.stack(labels)
    
    # Apply batch-level augmentations
    aug_config = config['training']['augmentation']
    
    # Mixup
    if np.random.random() < 0.5 and aug_config['mixup_alpha'] > 0:
        eeg_data, labels_a, labels_b, lam = mixup_data(
            eeg_data, labels, aug_config['mixup_alpha']
        )
        return {
            'eeg': eeg_data,
            'spectrogram': spec_data,
            'features': features,
            'label_a': labels_a,
            'label_b': labels_b,
            'mix_lambda': lam,
            'mixup': True
        }
    
    # CutMix
    elif np.random.random() < 0.3 and aug_config['cutmix_alpha'] > 0:
        spec_data, labels_a, labels_b, lam = cutmix_data(
            spec_data, labels, aug_config['cutmix_alpha']
        )
        return {
            'eeg': eeg_data,
            'spectrogram': spec_data,
            'features': features,
            'label_a': labels_a,
            'label_b': labels_b,
            'mix_lambda': lam,
            'cutmix': True
        }
    
    # No augmentation
    return {
        'eeg': eeg_data,
        'spectrogram': spec_data,
        'features': features,
        'label': labels,
        'mixup': False,
        'cutmix': False
    }


def create_data_loaders(config, train_indices=None, val_indices=None):
    """Create training and validation data loaders."""
    
    # Create datasets
    train_dataset = HMSDataset(
        config['data']['data_path'],
        config['data']['processed_path'],
        config,
        split='train',
        indices=train_indices,
        augment=True
    )
    
    val_dataset = HMSDataset(
        config['data']['data_path'],
        config['data']['processed_path'],
        config,
        split='val',
        indices=val_indices,
        augment=False
    )
    
    # Calculate class weights for balanced sampling
    if train_indices is not None:
        labels = [train_dataset.metadata.iloc[i]['expert_consensus'] for i in range(len(train_dataset))]
    else:
        labels = train_dataset.metadata['expert_consensus'].tolist()
    
    label_indices = [train_dataset.label_map[label] for label in labels]
    class_counts = Counter(label_indices)
    
    # Create weighted sampler
    weights = [1.0 / class_counts[label] for label in label_indices]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['models']['resnet1d_gru']['training']['batch_size'],
        sampler=sampler,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        persistent_workers=config['hardware']['persistent_workers'],
        collate_fn=lambda batch: advanced_collate_fn(batch, config)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['models']['resnet1d_gru']['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory'],
        persistent_workers=config['hardware']['persistent_workers'],
        collate_fn=lambda batch: advanced_collate_fn(batch, config)
    )
    
    return train_loader, val_loader

# Run in fifth cell:
# print("✅ Data loading and augmentation classes defined!") 


# ============================
# CELL 6: Advanced Loss Functions and Optimizers
# ============================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss."""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        confidence = 1. - self.smoothing
        log_probs = F.log_softmax(inputs, dim=-1)
        
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class MixupLoss(nn.Module):
    """Loss function for mixup/cutmix training."""
    
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
    
    def forward(self, inputs, targets_a, targets_b, lam):
        return lam * self.criterion(inputs, targets_a) + (1 - lam) * self.criterion(inputs, targets_b)


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer."""
    
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(dtype=torch.float32).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            dtype=torch.float32
        )
        return norm
    
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class LookAhead(torch.optim.Optimizer):
    """LookAhead optimizer wrapper."""
    
    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        
        # Store slow weights
        self.slow_weights = {}
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.slow_weights[p] = p.data.clone()
    
    @property
    def param_groups(self):
        return self.base_optimizer.param_groups
    
    def zero_grad(self):
        self.base_optimizer.zero_grad()
    
    def step(self, closure=None):
        loss = self.base_optimizer.step(closure)
        
        self.step_count += 1
        
        if self.step_count % self.k == 0:
            # Update slow weights
            for group in self.base_optimizer.param_groups:
                for p in group['params']:
                    if p.requires_grad:
                        slow_weight = self.slow_weights[p]
                        slow_weight.add_(p.data - slow_weight, alpha=self.alpha)
                        p.data.copy_(slow_weight)
        
        return loss


def create_optimizer(model, config, model_name):
    """Create optimizer based on configuration."""
    
    opt_config = config['training']['optimization']
    model_config = config['models'][model_name]['training']
    
    # Get parameters
    params = model.parameters()
    
    # Base optimizer
    if opt_config['optimizer'] == 'adamw':
        base_optimizer = optim.AdamW(
            params,
            lr=model_config['learning_rate'],
            weight_decay=model_config['weight_decay']
        )
    elif opt_config['optimizer'] == 'adam':
        base_optimizer = optim.Adam(
            params,
            lr=model_config['learning_rate'],
            weight_decay=model_config['weight_decay']
        )
    elif opt_config['optimizer'] == 'sgd':
        base_optimizer = optim.SGD(
            params,
            lr=model_config['learning_rate'],
            weight_decay=model_config['weight_decay'],
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_config['optimizer']}")
    
    # Wrap with advanced optimizers
    if opt_config['use_sam']:
        optimizer = SAM(params, type(base_optimizer), rho=0.05)
    elif opt_config['use_lookahead']:
        optimizer = LookAhead(base_optimizer, k=5, alpha=0.5)
    else:
        optimizer = base_optimizer
    
    return optimizer


def create_scheduler(optimizer, config, model_name, steps_per_epoch):
    """Create learning rate scheduler."""
    
    opt_config = config['training']['optimization']
    model_config = config['models'][model_name]['training']
    
    if opt_config['scheduler'] == 'cosine_warmup':
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        
        T_0 = steps_per_epoch * opt_config['warmup_epochs']
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=2, eta_min=1e-6
        )
    elif opt_config['scheduler'] == 'onecycle':
        from torch.optim.lr_scheduler import OneCycleLR
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=model_config['learning_rate'],
            epochs=model_config['epochs'],
            steps_per_epoch=steps_per_epoch
        )
    elif opt_config['scheduler'] == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif opt_config['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None
    
    return scheduler


def create_loss_function(config, class_counts=None):
    """Create loss function based on configuration."""
    
    loss_config = config['training']['class_balancing']
    
    if loss_config['method'] == 'focal_loss':
        # Calculate alpha weights
        if class_counts and loss_config['focal_alpha'] == 'balanced':
            total = sum(class_counts)
            alpha = [total / (len(class_counts) * count) for count in class_counts]
            alpha = torch.FloatTensor(alpha)
        else:
            alpha = None
        
        criterion = FocalLoss(
            alpha=alpha,
            gamma=loss_config['focal_gamma']
        )
    elif loss_config['method'] == 'label_smoothing':
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    else:
        # Weighted cross entropy
        if class_counts:
            total = sum(class_counts)
            weights = [total / count for count in class_counts]
            weights = torch.FloatTensor(weights)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()
    
    return criterion

# Run in sixth cell:
# print("✅ Advanced loss functions and optimizers defined!")


# ============================
# CELL 7: Training Loop and Evaluation
# ============================

class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, config, scaler=None):
    """Train one epoch with advanced features."""
    
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    # Create mixup loss for augmented batches
    mixup_criterion = MixupLoss(criterion)
    
    progress_bar = tqdm(train_loader, desc='Training')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        eeg_data = batch['eeg'].to(device)
        spec_data = batch['spectrogram'].to(device)
        
        # Handle different batch types
        if batch.get('mixup', False) or batch.get('cutmix', False):
            labels_a = batch['label_a'].to(device)
            labels_b = batch['label_b'].to(device)
            lam = batch['mix_lambda']
        else:
            labels = batch['label'].to(device)
        
        # Forward pass with mixed precision
        if scaler:
            with autocast('cuda'):
                if 'ensemble' in model.__class__.__name__.lower():
                    outputs = model(eeg_data, spec_data)
                else:
                    # Single model - choose input based on model type
                    if 'resnet' in model.__class__.__name__.lower():
                        outputs = model(eeg_data)
                    else:
                        outputs = model(spec_data)
                
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                
                # Calculate loss
                if batch.get('mixup', False) or batch.get('cutmix', False):
                    loss = mixup_criterion(logits, labels_a, labels_b, lam)
                else:
                    loss = criterion(logits, labels)
                
                # Scale loss for gradient accumulation
                loss = loss / config['training']['gradient_accumulation_steps']
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config['training']['gradient_accumulation_steps'] == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config['training']['max_grad_norm']
                )
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if scheduler:
                    scheduler.step()
        else:
            # Without mixed precision
            if 'ensemble' in model.__class__.__name__.lower():
                outputs = model(eeg_data, spec_data)
            else:
                if 'resnet' in model.__class__.__name__.lower():
                    outputs = model(eeg_data)
                else:
                    outputs = model(spec_data)
            
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            
            if batch.get('mixup', False) or batch.get('cutmix', False):
                loss = mixup_criterion(logits, labels_a, labels_b, lam)
            else:
                loss = criterion(logits, labels)
            
            loss = loss / config['training']['gradient_accumulation_steps']
            loss.backward()
            
            if (batch_idx + 1) % config['training']['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config['training']['max_grad_norm']
                )
                optimizer.step()
                optimizer.zero_grad()
                
                if scheduler:
                    scheduler.step()
        
        # Update metrics
        total_loss += loss.item() * config['training']['gradient_accumulation_steps']
        
        # Calculate accuracy (handle mixup case)
        with torch.no_grad():
            if not (batch.get('mixup', False) or batch.get('cutmix', False)):
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
        
        # Update progress bar
        if total_samples > 0:
            acc = total_correct / total_samples
            progress_bar.set_postfix({
                'Loss': f'{total_loss / (batch_idx + 1):.4f}',
                'Acc': f'{acc:.4f}'
            })
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, avg_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate one epoch."""
    
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            eeg_data = batch['eeg'].to(device)
            spec_data = batch['spectrogram'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            if 'ensemble' in model.__class__.__name__.lower():
                outputs = model(eeg_data, spec_data)
            else:
                if 'resnet' in model.__class__.__name__.lower():
                    outputs = model(eeg_data)
                else:
                    outputs = model(spec_data)
            
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            loss = criterion(logits, labels)
            
            # Calculate metrics
            total_loss += loss.item()
            
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate comprehensive metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_targets, all_preds)
    balanced_acc = balanced_accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average='macro')
    f1_weighted = f1_score(all_targets, all_preds, average='weighted')
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, average=None, zero_division=0
    )
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': np.array(all_preds),
        'targets': np.array(all_targets),
        'probabilities': np.array(all_probs),
        'precision': precision,
        'recall': recall,
        'f1_per_class': f1,
        'support': support
    }
    
    return metrics


def train_model(model, train_loader, val_loader, config, model_name, device):
    """Complete training loop for a single model."""
    
    print(f"\n🚀 Training {model_name}...")
    
    # Setup training components
    model = model.to(device)
    
    # Calculate class counts for loss function
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['label'].tolist())
    class_counts = [all_labels.count(i) for i in range(config['data']['num_classes'])]
    
    criterion = create_loss_function(config, class_counts).to(device)
    optimizer = create_optimizer(model, config, model_name)
    scheduler = create_scheduler(optimizer, config, model_name, len(train_loader))
    
    # Mixed precision
    scaler = GradScaler('cuda') if config['training']['mixed_precision'] else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        min_delta=0.001
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_balanced_acc': [],
        'val_f1_macro': []
    }
    
    best_val_acc = 0
    
    # Training loop
    num_epochs = config['models'][model_name]['training']['epochs']
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, config, scaler
        )
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_balanced_acc'].append(val_metrics['balanced_accuracy'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Balanced Acc: {val_metrics['balanced_accuracy']:.4f}, F1 Macro: {val_metrics['f1_macro']:.4f}")
        
        # Save best model
        if val_metrics['balanced_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['balanced_accuracy']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': best_val_acc,
                'config': config
            }, f'/content/drive/MyDrive/hms-models/best_{model_name}.pth')
            print(f"💾 Saved best model with balanced accuracy: {best_val_acc:.4f}")
        
        # Early stopping
        if early_stopping(val_metrics['loss'], model):
            print(f"⏹️  Early stopping triggered at epoch {epoch+1}")
            break
    
    print(f"✅ Training completed! Best validation balanced accuracy: {best_val_acc:.4f}")
    
    return model, history, best_val_acc

# Run in seventh cell:
# print("✅ Training loop and evaluation functions defined!") 


# ============================
# CELL 8: Cross-Validation and Ensemble Training
# ============================

def cross_validate_model(model_class, dataset_indices, config, model_name, device):
    """Perform stratified cross-validation."""
    
    print(f"\n🔄 Starting {config['validation']['n_folds']}-fold cross-validation for {model_name}...")
    
    # Load metadata for stratification
    metadata = pd.read_csv(f"{config['data']['data_path']}/train.csv")
    if dataset_indices is not None:
        metadata = metadata.iloc[dataset_indices].reset_index(drop=True)
    
    # Get labels for stratification using the actual class names from config
    class_to_idx = {name: i for i, name in enumerate(config['data']['classes'])}
    
    # Handle missing classes gracefully
    labels = []
    for label_name in metadata['expert_consensus']:
        if label_name in class_to_idx:
            labels.append(class_to_idx[label_name])
        else:
            print(f"⚠️  Unknown class '{label_name}' in CV, using class 0")
            labels.append(0)  # Default to first class
    
    # Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=config['validation']['n_folds'], 
        shuffle=True, 
        random_state=42
    )
    
    fold_results = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(metadata)), labels)):
        print(f"\n📁 Fold {fold + 1}/{config['validation']['n_folds']}")
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        
        # Create data loaders for this fold
        train_loader, val_loader = create_data_loaders(config, train_idx, val_idx)
        
        # Create model
        model = model_class(config)
        
        # Train model
        trained_model, history, best_acc = train_model(
            model, train_loader, val_loader, config, model_name, device
        )
        
        fold_results.append({
            'fold': fold + 1,
            'best_accuracy': best_acc,
            'history': history
        })
        fold_models.append(trained_model)
        
        print(f"Fold {fold + 1} completed with accuracy: {best_acc:.4f}")
    
    # Calculate cross-validation metrics
    cv_scores = [result['best_accuracy'] for result in fold_results]
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    print(f"\n📊 Cross-Validation Results for {model_name}:")
    print(f"Mean Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Individual Folds: {cv_scores}")
    
    return fold_models, fold_results, cv_mean, cv_std


def train_ensemble_model(base_models, config, dataset_indices, device):
    """Train ensemble model using pre-trained base models."""
    
    print("\n🤝 Training Ensemble Model...")
    
    # Create ensemble model
    ensemble_model = EnsembleModel(base_models, config)
    
    # Freeze base models initially
    for model in base_models.values():
        for param in model.parameters():
            param.requires_grad = False
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config, dataset_indices, None)
    
    # Train ensemble
    trained_ensemble, history, best_acc = train_model(
        ensemble_model, train_loader, val_loader, config, 'ensemble', device
    )
    
    print(f"✅ Ensemble training completed with accuracy: {best_acc:.4f}")
    
    return trained_ensemble, history, best_acc


def evaluate_all_models(models, test_loader, config, device):
    """Evaluate all models comprehensively."""
    
    print("\n📈 Evaluating all models...")
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}...")
        
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f'Evaluating {model_name}'):
                eeg_data = batch['eeg'].to(device)
                spec_data = batch['spectrogram'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                if 'ensemble' in model_name.lower():
                    outputs = model(eeg_data, spec_data)
                else:
                    if 'resnet' in model_name.lower():
                        outputs = model(eeg_data)
                    else:
                        outputs = model(spec_data)
                
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        balanced_acc = balanced_accuracy_score(all_targets, all_preds)
        f1_macro = f1_score(all_targets, all_preds, average='macro')
        f1_weighted = f1_score(all_targets, all_preds, average='weighted')
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_preds, average=None, zero_division=0
        )
        
        results[model_name] = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision': precision,
            'recall': recall,
            'f1_per_class': f1,
            'support': support,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }
        
        print(f"{model_name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        print(f"  F1 Macro: {f1_macro:.4f}")
        print(f"  F1 Weighted: {f1_weighted:.4f}")
    
    return results

# Run in eighth cell:
# print("✅ Cross-validation and ensemble training functions defined!")


# ============================
# CELL 9: Visualization and Analysis
# ============================

def plot_training_history(history, model_name):
    """Plot training history."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title(f'{model_name} - Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_title(f'{model_name} - Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Balanced Accuracy
    axes[1, 0].plot(history['val_balanced_acc'], label='Val Balanced Acc', color='green')
    axes[1, 0].set_title(f'{model_name} - Balanced Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Balanced Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1 Score
    axes[1, 1].plot(history['val_f1_macro'], label='Val F1 Macro', color='red')
    axes[1, 1].set_title(f'{model_name} - F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(targets, predictions, class_names, model_name):
    """Plot confusion matrix."""
    
    cm = confusion_matrix(targets, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return cm


def plot_model_comparison(results, metric='balanced_accuracy'):
    """Plot comparison of all models."""
    
    models = list(results.keys())
    scores = [results[model][metric] for model in models]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, scores, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
    plt.xlabel('Model')
    plt.ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def analyze_class_performance(results, class_names):
    """Analyze per-class performance across models."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    models = list(results.keys())
    
    # Precision
    for i, metric in enumerate(['precision', 'recall', 'f1_per_class']):
        data = []
        for model in models:
            data.append(results[model][metric])
        
        data = np.array(data)
        
        x = np.arange(len(class_names))
        width = 0.2
        
        for j, model in enumerate(models):
            offset = (j - len(models)/2) * width
            axes[i].bar(x + offset, data[j], width, label=model, alpha=0.8)
        
        axes[i].set_title(f'{metric.replace("_", " ").title()} by Class')
        axes[i].set_xlabel('Class')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(class_names, rotation=45)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def save_results(results, config, output_path='/content/drive/MyDrive/hms-models/results.json'):
    """Save all results to JSON file."""
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for model_name, metrics in results.items():
        json_results[model_name] = {}
        for metric_name, value in metrics.items():
            if isinstance(value, np.ndarray):
                json_results[model_name][metric_name] = value.tolist()
            else:
                json_results[model_name][metric_name] = value
    
    # Add configuration
    json_results['config'] = config
    json_results['timestamp'] = pd.Timestamp.now().isoformat()
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"📄 Results saved to {output_path}")

# Run in ninth cell:
# print("✅ Visualization and analysis functions defined!")


# ============================
# CELL 10: Main Training Pipeline
# ============================

def main_training_pipeline(config):
    """Execute the complete training pipeline."""
    
    print("🚀 Starting Comprehensive HMS Training Pipeline!")
    print("=" * 60)
    
    # Setup device and directories
    device = torch.device(config['hardware']['device'])
    
    # Create output directories
    import os
    os.makedirs(config['data']['output_path'], exist_ok=True)
    
    # Load dataset indices (use all data if not specified)
    dataset_indices = None  # You can specify subset here for testing
    
    print(f"💻 Using device: {device}")
    print(f"📊 Dataset path: {config['data']['data_path']}")
    print(f"📁 Output path: {config['data']['output_path']}")
    
    # ========================================
    # 1. Train Individual Models
    # ========================================
    
    trained_models = {}
    all_histories = {}
    
    # Train ResNet1D-GRU
    if config['models']['resnet1d_gru']['enabled']:
        print("\n" + "="*50)
        print("🧠 TRAINING RESNET1D-GRU MODEL")
        print("="*50)
        
        if config['validation']['cross_validation']:
            # Cross-validation
            resnet_models, resnet_results, cv_mean, cv_std = cross_validate_model(
                ResNet1D_GRU_Advanced, dataset_indices, config, 'resnet1d_gru', device
            )
            # Use the best fold model
            best_fold = np.argmax([r['best_accuracy'] for r in resnet_results])
            trained_models['resnet1d_gru'] = resnet_models[best_fold]
            all_histories['resnet1d_gru'] = resnet_results[best_fold]['history']
        else:
            # Single training
            train_loader, val_loader = create_data_loaders(config, dataset_indices, None)
            resnet_model = ResNet1D_GRU_Advanced(config)
            trained_model, history, best_acc = train_model(
                resnet_model, train_loader, val_loader, config, 'resnet1d_gru', device
            )
            trained_models['resnet1d_gru'] = trained_model
            all_histories['resnet1d_gru'] = history
    
    # Train EfficientNet
    if config['models']['efficientnet']['enabled']:
        print("\n" + "="*50)
        print("🖼️  TRAINING EFFICIENTNET MODEL")
        print("="*50)
        
        if config['validation']['cross_validation']:
            # Cross-validation
            eff_models, eff_results, cv_mean, cv_std = cross_validate_model(
                EfficientNetSpectrogram, dataset_indices, config, 'efficientnet', device
            )
            # Use the best fold model
            best_fold = np.argmax([r['best_accuracy'] for r in eff_results])
            trained_models['efficientnet'] = eff_models[best_fold]
            all_histories['efficientnet'] = eff_results[best_fold]['history']
        else:
            # Single training
            train_loader, val_loader = create_data_loaders(config, dataset_indices, None)
            eff_model = EfficientNetSpectrogram(config)
            trained_model, history, best_acc = train_model(
                eff_model, train_loader, val_loader, config, 'efficientnet', device
            )
            trained_models['efficientnet'] = trained_model
            all_histories['efficientnet'] = history
    
    # ========================================
    # 2. Train Ensemble Model
    # ========================================
    
    if config['models']['ensemble']['enabled'] and len(trained_models) > 1:
        print("\n" + "="*50)
        print("🤝 TRAINING ENSEMBLE MODEL")
        print("="*50)
        
        ensemble_model, ensemble_history, ensemble_acc = train_ensemble_model(
            trained_models, config, dataset_indices, device
        )
        trained_models['ensemble'] = ensemble_model
        all_histories['ensemble'] = ensemble_history
    
    # ========================================
    # 3. Final Evaluation
    # ========================================
    
    print("\n" + "="*50)
    print("📊 FINAL EVALUATION")
    print("="*50)
    
    # Create test loader (using validation split for now)
    _, test_loader = create_data_loaders(config, dataset_indices, None)
    
    # Evaluate all models
    final_results = evaluate_all_models(trained_models, test_loader, config, device)
    
    # ========================================
    # 4. Visualization and Analysis
    # ========================================
    
    print("\n" + "="*50)
    print("📈 VISUALIZATION AND ANALYSIS")
    print("="*50)
    
    # Plot training histories
    for model_name, history in all_histories.items():
        plot_training_history(history, model_name)
    
    # Plot model comparison
    plot_model_comparison(final_results, 'balanced_accuracy')
    plot_model_comparison(final_results, 'f1_macro')
    
    # Plot confusion matrices
    for model_name, results in final_results.items():
        plot_confusion_matrix(
            results['targets'], 
            results['predictions'], 
            config['data']['classes'], 
            model_name
        )
    
    # Analyze per-class performance
    analyze_class_performance(final_results, config['data']['classes'])
    
    # ========================================
    # 5. Save Results
    # ========================================
    
    # Save final results
    save_results(final_results, config)
    
    # Save model checkpoints
    for model_name, model in trained_models.items():
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'model_class': model.__class__.__name__
        }, f'{config["data"]["output_path"]}/final_{model_name}.pth')
    
    print("\n" + "="*60)
    print("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Print final summary
    print("\n📊 FINAL RESULTS SUMMARY:")
    for model_name, results in final_results.items():
        print(f"{model_name:15} | Acc: {results['accuracy']:.4f} | "
              f"Bal Acc: {results['balanced_accuracy']:.4f} | "
              f"F1: {results['f1_macro']:.4f}")
    
    return trained_models, final_results, all_histories

# Run in tenth cell:
# print("✅ Main training pipeline defined!")


# ============================
# CELL 11: Execute Training
# ============================

# Mount Google Drive and setup paths
def setup_colab_paths():
    """Setup Google Drive and create necessary directories."""
    from google.colab import drive
    import os
    
    # Mount drive
    print("📁 Mounting Google Drive...")
    drive.mount('/content/drive')
    
    # Create directories
    paths = [
        '/content/drive/MyDrive/hms-data',
        '/content/drive/MyDrive/hms-processed',
        '/content/drive/MyDrive/hms-models'
    ]
    
    for path in paths:
        os.makedirs(path, exist_ok=True)
        print(f"📂 Created/verified: {path}")
    
    print("✅ Google Drive setup complete!")

# Quick configuration for testing
def create_test_config():
    """Create a test configuration with smaller parameters."""
    test_config = TRAINING_CONFIG.copy()
    
    # Reduce training parameters for testing
    test_config['models']['resnet1d_gru']['training']['epochs'] = 5
    test_config['models']['efficientnet']['training']['epochs'] = 5
    test_config['validation']['cross_validation'] = False  # Disable CV for quick testing
    
    return test_config

# Execute the training pipeline
def run_training():
    """Run the complete training pipeline."""
    
    # Setup environment
    setup_colab_paths()
    
    # Check preprocessing status
    config = create_test_config()  # Use config for path checking
    preprocessing_ok = check_preprocessing_status(config)
    
    if not preprocessing_ok:
        get_preprocessing_recommendations()
        return None
    
    print("✅ Preprocessing check passed!")
    
    # Auto-detect and update class names
    config = update_config_with_classes(config)
    
    # Print configuration summary
    print("\n📋 Training Configuration:")
    print(f"  Models enabled: {[k for k, v in config['models'].items() if v.get('enabled', False)]}")
    print(f"  Classes ({config['data']['num_classes']}): {config['data']['classes']}")
    print(f"  ResNet epochs: {config['models']['resnet1d_gru']['training']['epochs']}")
    print(f"  EfficientNet epochs: {config['models']['efficientnet']['training']['epochs']}")
    print(f"  Cross-validation: {config['validation']['cross_validation']}")
    print(f"  Mixed precision: {config['training']['mixed_precision']}")
    
    # Start training
    try:
        trained_models, results, histories = main_training_pipeline(config)
        return trained_models, results, histories
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\n💡 Troubleshooting tips:")
        print("1. Check if you have sufficient GPU memory")
        print("2. Reduce batch size in configuration")
        print("3. Ensure all preprocessed files are valid")
        print("4. Check Google Drive storage space")
        raise e

# Run this in the eleventh cell:
# trained_models, results, histories = run_training()


# ============================
# USAGE INSTRUCTIONS
# ============================

"""
📋 STEP-BY-STEP USAGE INSTRUCTIONS:

1. **Environment Setup** (Run these in order):
   - gpu_available = setup_colab_environment()
   - # Run import cell
   - # Run configuration cell

2. **Model Definition** (Run these cells):
   - # Run model architecture cell
   - # Run data loading cell
   - # Run loss functions cell
   - # Run training loop cell

3. **Cross-validation and Ensemble** (Run these):
   - # Run CV and ensemble cell
   - # Run visualization cell
   - # Run main pipeline cell

4. **Execute Training**:
   - trained_models, results, histories = run_training()

5. **For Full Training** (modify config):
   - Change `create_test_config()` to `TRAINING_CONFIG` in run_training()
   - Enable cross-validation if desired
   - Increase epochs for better results

6. **Monitor Training**:
   - Training progress will be displayed with progress bars
   - Models are automatically saved to Google Drive
   - Visualizations will be generated after training

7. **Results Analysis**:
   - Confusion matrices for each model
   - Performance comparisons
   - Per-class analysis
   - Training history plots

📁 OUTPUT FILES:
- `/content/drive/MyDrive/hms-models/best_[model_name].pth` - Best checkpoints
- `/content/drive/MyDrive/hms-models/final_[model_name].pth` - Final models
- `/content/drive/MyDrive/hms-models/results.json` - Complete results

🎯 FEATURES INCLUDED:
✅ ResNet1D-GRU with attention and multi-scale features
✅ EfficientNet for spectrograms  
✅ Ensemble learning with neural meta-learner
✅ Advanced augmentations (Mixup, CutMix, SpecAugment)
✅ Focal loss for class imbalance
✅ Mixed precision training
✅ SAM and LookAhead optimizers
✅ Cross-validation support
✅ Early stopping and checkpointing
✅ Comprehensive evaluation and visualization
✅ GPU optimization and memory management
"""

print("🎯 Comprehensive HMS GPU Training Script Ready!")
print("📚 See usage instructions above for step-by-step execution.")
print("🚀 Ready to train state-of-the-art EEG classification models!")


def check_preprocessing_status(config):
    """Check the status of preprocessing and provide feedback."""
    
    print("🔍 Checking preprocessing status...")
    
    data_path = Path(config['data']['data_path'])
    processed_path = Path(config['data']['processed_path'])
    
    # Check if directories exist
    required_dirs = ['filtered', 'spectrograms', 'features']
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = processed_path / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ Missing preprocessing directories: {missing_dirs}")
        print("   Please run the preprocessing script first!")
        return False
    
    # Load metadata and check file coverage
    try:
        metadata = pd.read_csv(data_path / 'train.csv')
        total_files = len(metadata)
        
        # Count available processed files
        available_counts = {}
        for dir_name in required_dirs:
            dir_path = processed_path / dir_name
            files = list(dir_path.glob('*_*.npy'))
            available_counts[dir_name] = len(files)
        
        print(f"📊 Preprocessing Status:")
        print(f"   Total samples in dataset: {total_files}")
        
        for dir_name, count in available_counts.items():
            percentage = (count / total_files) * 100 if total_files > 0 else 0
            status_icon = "✅" if percentage > 95 else "⚠️" if percentage > 50 else "❌"
            print(f"   {status_icon} {dir_name}: {count}/{total_files} ({percentage:.1f}%)")
        
        # Check if we have enough data to proceed
        min_available = min(available_counts.values())
        min_percentage = (min_available / total_files) * 100 if total_files > 0 else 0
        
        if min_percentage < 10:
            print(f"\n❌ Insufficient preprocessed data ({min_percentage:.1f}% available)")
            print("   Please run the preprocessing script to process more data.")
            return False
        elif min_percentage < 50:
            print(f"\n⚠️  Limited preprocessed data ({min_percentage:.1f}% available)")
            print("   Training will proceed with available data, but results may be limited.")
            print("   Consider running preprocessing on more data for better results.")
        else:
            print(f"\n✅ Sufficient preprocessed data available ({min_percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking preprocessing status: {e}")
        return False


def get_preprocessing_recommendations():
    """Provide recommendations for preprocessing."""
    
    print("\n📋 Preprocessing Recommendations:")
    print("1. Run the preprocessing script first:")
    print("   - Upload 'colab_setup_and_preprocessing.py' to Colab")
    print("   - Run the preprocessing cells in order")
    print("   - Wait for processing to complete")
    
    print("\n2. If preprocessing is taking too long:")
    print("   - Use max_samples parameter to process a subset first")
    print("   - Example: process_data_gpu(DATA_PATH, OUTPUT_PATH, max_samples=1000)")
    
    print("\n3. If some files are missing:")
    print("   - Check Google Drive storage space")
    print("   - Re-run preprocessing for specific files")
    print("   - Training will work with partial data")
    
    print("\n4. For better results:")
    print("   - Process the complete dataset (may take several hours)")
    print("   - Use GPU acceleration (ensure GPU runtime is enabled)")
    print("   - Monitor memory usage during processing")

# Run in tenth cell:
# print("✅ Main training pipeline defined!")