"""
EfficientNet model adapted for EEG spectrogram classification.
Optimized for time-frequency representations of brain activity.
Enhanced with squeeze-and-excitation, depth-wise separable convolutions, and progressive resizing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from efficientnet_pytorch import EfficientNet as BaseEfficientNet
from torchvision import transforms
import timm
import logging
from einops import rearrange
import kornia

logger = logging.getLogger(__name__)


class ChannelSEBlock(nn.Module):
    """Enhanced Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channels: int, reduction: int = 16, activation: str = 'relu'):
        super(ChannelSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        
        # Dual pooling for better representation
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        combined = torch.cat([avg_out, max_out], dim=1)
        
        channel_weights = self.fc(combined).view(b, c, 1, 1)
        return x * channel_weights.expand_as(x)


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise separable convolution for efficiency."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super(DepthwiseSeparableConv2d, self).__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU6(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class SpectrogramAttention(nn.Module):
    """Enhanced attention mechanism for spectrograms with frequency awareness."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super(SpectrogramAttention, self).__init__()
        
        # Channel attention with SE block
        self.channel_attention = ChannelSEBlock(in_channels, reduction)
        
        # Spatial attention with frequency awareness
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv2d(8, 8, kernel_size=3),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Frequency-specific attention
        self.freq_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=(7, 1), padding=(3, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=(7, 1), padding=(3, 0)),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        x = self.channel_attention(x)
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        x = x * spatial_att
        
        # Frequency attention
        freq_att = self.freq_attention(x)
        x = x * freq_att
        
        return x


class FrequencyAwarePooling(nn.Module):
    """Enhanced pooling that preserves clinical frequency bands."""
    
    def __init__(self, freq_bins: List[int] = None):
        super(FrequencyAwarePooling, self).__init__()
        # Clinical frequency bands: delta, theta, alpha, beta, gamma
        self.freq_bins = freq_bins or [4, 8, 13, 30, 50]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, freq, time)
        batch, channels, freq, time = x.shape
        
        pooled_features = []
        prev_bin = 0
        
        for freq_boundary in self.freq_bins:
            # Map frequency boundary to spectrogram bin
            bin_idx = min(int(freq_boundary * freq / 50), freq)  # Assuming 50Hz max
            
            if bin_idx > prev_bin:
                # Pool this frequency band
                band_features = x[:, :, prev_bin:bin_idx, :]
                
                # Adaptive pooling to preserve temporal information
                pooled = F.adaptive_avg_pool2d(band_features, (1, time))
                pooled_features.append(pooled)
                
                prev_bin = bin_idx
        
        # Handle remaining frequencies
        if prev_bin < freq:
            band_features = x[:, :, prev_bin:, :]
            pooled = F.adaptive_avg_pool2d(band_features, (1, time))
            pooled_features.append(pooled)
        
        # Concatenate along frequency dimension
        output = torch.cat(pooled_features, dim=2)
        
        return output


class ProgressiveResizing(nn.Module):
    """Progressive resizing for better feature learning."""
    
    def __init__(self, initial_size: Tuple[int, int], final_size: Tuple[int, int],
                 epochs_to_final: int = 50):
        super(ProgressiveResizing, self).__init__()
        self.initial_size = initial_size
        self.final_size = final_size
        self.epochs_to_final = epochs_to_final
        self.current_epoch = 0
        
    def set_epoch(self, epoch: int):
        """Update current epoch for progressive resizing."""
        self.current_epoch = epoch
        
    def get_current_size(self) -> Tuple[int, int]:
        """Get current target size based on training progress."""
        if self.current_epoch >= self.epochs_to_final:
            return self.final_size
            
        # Linear interpolation
        progress = self.current_epoch / self.epochs_to_final
        h = int(self.initial_size[0] + (self.final_size[0] - self.initial_size[0]) * progress)
        w = int(self.initial_size[1] + (self.final_size[1] - self.initial_size[1]) * progress)
        
        return (h, w)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Resize input to current target size."""
        current_size = self.get_current_size()
        
        if x.shape[-2:] != current_size:
            x = F.interpolate(x, size=current_size, mode='bilinear', align_corners=False)
            
        return x


class EfficientNetSpectrogram(nn.Module):
    """Enhanced EfficientNet for EEG spectrogram classification."""
    
    def __init__(self, config: Dict, pretrained: bool = True):
        super(EfficientNetSpectrogram, self).__init__()
        
        # Extract configuration
        model_config = config['models']['efficientnet']
        self.n_classes = len(config['classes'])
        self.dropout_rate = model_config['dropout']
        
        # Progressive resizing
        self.progressive_resize = ProgressiveResizing(
            initial_size=(128, 128),
            final_size=(224, 224),
            epochs_to_final=model_config.get('progressive_epochs', 50)
        )
        
        # Load pretrained EfficientNet with compound scaling
        if pretrained:
            self.backbone = timm.create_model(
                model_config['model_name'],
                pretrained=True,
                num_classes=0,  # Remove classification head
                global_pool='',  # Remove global pooling
                drop_rate=self.dropout_rate,
                drop_path_rate=model_config.get('drop_path_rate', 0.2)
            )
        else:
            self.backbone = BaseEfficientNet.from_name(
                model_config['model_name'],
                num_classes=self.n_classes
            )
            
        # Get feature dimensions
        if hasattr(self.backbone, 'num_features'):
            self.feat_dim = self.backbone.num_features
        else:
            # For EfficientNet-B3
            self.feat_dim = 1536
            
        # Enhanced spectrogram-specific components
        self.spectrogram_attention = SpectrogramAttention(self.feat_dim)
        self.frequency_pool = FrequencyAwarePooling()
        
        # Multi-scale feature fusion
        self.multi_scale_fusion = nn.ModuleList([
            DepthwiseSeparableConv2d(self.feat_dim, self.feat_dim // 4, kernel_size=1),
            DepthwiseSeparableConv2d(self.feat_dim, self.feat_dim // 4, kernel_size=3),
            DepthwiseSeparableConv2d(self.feat_dim, self.feat_dim // 4, kernel_size=5, padding=2),
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.feat_dim, self.feat_dim // 4, kernel_size=1),
                nn.ReLU(inplace=True)
            )
        ])
        
        self.fusion_conv = nn.Conv2d(self.feat_dim, self.feat_dim, kernel_size=1)
        
        # Global pooling with attention
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim // 2),
            nn.LayerNorm(self.feat_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feat_dim // 2, self.feat_dim // 4),
            nn.LayerNorm(self.feat_dim // 4),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.feat_dim // 4, self.n_classes)
        )
        
        # Multi-task heads
        self.seizure_detector = nn.Sequential(
            nn.Linear(self.feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 1)
        )
        
        self.artifact_detector = nn.Sequential(
            nn.Linear(self.feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(128, 1)
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for new layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features with multi-scale fusion."""
        # Progressive resizing
        x = self.progressive_resize(x)
        
        # Pass through backbone
        features = self.backbone(x)
        
        # Multi-scale feature extraction
        multi_scale_features = []
        for fusion_layer in self.multi_scale_fusion[:-1]:
            multi_scale_features.append(fusion_layer(features))
            
        # Global feature
        global_feat = self.multi_scale_fusion[-1](features)
        global_feat = F.interpolate(
            global_feat, size=features.shape[-2:], 
            mode='bilinear', align_corners=False
        )
        multi_scale_features.append(global_feat)
        
        # Concatenate and fuse
        fused_features = torch.cat(multi_scale_features, dim=1)
        features = self.fusion_conv(fused_features)
        
        # Apply spectrogram-specific attention
        features = self.spectrogram_attention(features)
        
        return features
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with uncertainty estimation."""
        # Extract features
        features = self.extract_features(x)
        
        # Global pooling
        pooled_features = self.global_pool(features)
        
        # Main classification
        logits = self.classifier(pooled_features)
        
        # Uncertainty estimation
        log_variance = self.uncertainty_head(pooled_features)
        
        # Multi-task outputs
        seizure_prob = torch.sigmoid(self.seizure_detector(pooled_features))
        artifact_prob = torch.sigmoid(self.artifact_detector(pooled_features))
        
        return {
            'logits': logits,
            'log_variance': log_variance,
            'seizure_probability': seizure_prob,
            'artifact_probability': artifact_prob,
            'features': pooled_features
        }


class SpectrogramAugmentation:
    """Data augmentation specifically for EEG spectrograms."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.augmentation_config = config['training']['augmentation']['frequency_domain']
        
    def get_train_transform(self) -> transforms.Compose:
        """Get training augmentation pipeline."""
        transform_list = [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),  # Time flip
            transforms.RandomVerticalFlip(p=0.3),    # Frequency flip (with care)
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ]
        
        # Add frequency masking
        if self.augmentation_config.get('freq_mask', 0) > 0:
            transform_list.append(
                FrequencyMasking(max_mask_size=self.augmentation_config['freq_mask'])
            )
            
        # Add time masking
        if self.augmentation_config.get('time_mask', 0) > 0:
            transform_list.append(
                TimeMasking(max_mask_size=self.augmentation_config['time_mask'])
            )
            
        return transforms.Compose(transform_list)
        
    def get_val_transform(self) -> transforms.Compose:
        """Get validation/test augmentation pipeline."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])


class FrequencyMasking(nn.Module):
    """Frequency masking augmentation for spectrograms."""
    
    def __init__(self, max_mask_size: float = 0.1):
        super().__init__()
        self.max_mask_size = max_mask_size
        
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply frequency masking."""
        _, freq_bins, _ = spectrogram.shape
        
        mask_size = int(freq_bins * np.random.uniform(0, self.max_mask_size))
        if mask_size > 0:
            start = np.random.randint(0, freq_bins - mask_size)
            spectrogram[:, start:start + mask_size, :] = 0
            
        return spectrogram


class TimeMasking(nn.Module):
    """Time masking augmentation for spectrograms."""
    
    def __init__(self, max_mask_size: float = 0.1):
        super().__init__()
        self.max_mask_size = max_mask_size
        
    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply time masking."""
        _, _, time_bins = spectrogram.shape
        
        mask_size = int(time_bins * np.random.uniform(0, self.max_mask_size))
        if mask_size > 0:
            start = np.random.randint(0, time_bins - mask_size)
            spectrogram[:, :, start:start + mask_size] = 0
            
        return spectrogram


class EfficientNetTrainer:
    """Training utilities for EfficientNet spectrogram model."""
    
    def __init__(self, model: EfficientNetSpectrogram, config: Dict, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.seizure_loss = nn.BCELoss()
        self.artifact_loss = nn.BCELoss()
        
        # Loss weights for multi-task learning
        self.loss_weights = {
            'classification': 1.0,
            'seizure': 0.3,
            'artifact': 0.2
        }
        
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with different learning rates for different parts."""
        training_config = self.config['models']['efficientnet']['training']
        
        # Different learning rates for pretrained and new layers
        params = [
            {'params': self.model.backbone.parameters(), 
             'lr': training_config['learning_rate'] * 0.1},
            {'params': self.model.classifier.parameters(),
             'lr': training_config['learning_rate']},
            {'params': self.model.spectrogram_attention.parameters(),
             'lr': training_config['learning_rate']},
        ]
        
        optimizer = torch.optim.AdamW(
            params,
            weight_decay=training_config['weight_decay']
        )
        
        return optimizer
        
    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        training_config = self.config['models']['efficientnet']['training']
        
        # Use OneCycleLR for better convergence
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=training_config['learning_rate'],
            epochs=training_config['epochs'],
            steps_per_epoch=1000,  # Will be updated based on actual dataloader
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        return scheduler
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with multi-task learning."""
        self.model.train()
        
        # Move data to device
        spectrograms = batch['spectrogram'].to(self.device)
        labels = batch['label'].to(self.device)
        seizure_labels = batch.get('seizure_label', torch.zeros(len(labels))).to(self.device)
        artifact_labels = batch.get('artifact_label', torch.zeros(len(labels))).to(self.device)
        
        # Forward pass
        outputs = self.model(spectrograms)
        
        # Calculate losses
        class_loss = self.classification_loss(outputs['logits'], labels)
        seizure_loss = self.seizure_loss(outputs['seizure_probability'].squeeze(), seizure_labels.float())
        artifact_loss = self.artifact_loss(outputs['artifact_probability'].squeeze(), artifact_labels.float())
        
        # Combined loss
        total_loss = (self.loss_weights['classification'] * class_loss +
                     self.loss_weights['seizure'] * seizure_loss +
                     self.loss_weights['artifact'] * artifact_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Calculate metrics
        with torch.no_grad():
            preds = torch.argmax(outputs['logits'], dim=1)
            accuracy = (preds == labels).float().mean()
            
        return {
            'loss': total_loss.item(),
            'class_loss': class_loss.item(),
            'seizure_loss': seizure_loss.item(),
            'artifact_loss': artifact_loss.item(),
            'accuracy': accuracy.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        } 