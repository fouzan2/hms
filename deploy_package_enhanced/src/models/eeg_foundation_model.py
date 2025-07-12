"""
EEG Foundation Model for HMS Brain Activity Classification
A transformer-based foundation model with self-supervised pre-training
for robust EEG representation learning and transfer learning capabilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import math
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class EEGFoundationConfig:
    """Configuration for EEG Foundation Model."""
    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    d_ff: int = 2048
    dropout: float = 0.1
    
    # EEG-specific parameters
    n_channels: int = 19
    max_seq_length: int = 10000  # Maximum sequence length (50s at 200Hz)
    patch_size: int = 200  # Patch size for temporal patches (1s at 200Hz)
    overlap: float = 0.5  # Overlap between patches
    
    # Pre-training parameters
    mask_ratio: float = 0.15  # Ratio of patches to mask
    contrastive_temperature: float = 0.07
    reconstruction_weight: float = 1.0
    contrastive_weight: float = 0.5
    
    # Positional encoding
    max_position_embeddings: int = 1000
    use_learned_positional: bool = True
    
    # Channel attention
    use_channel_attention: bool = True
    channel_attention_heads: int = 4
    
    # Multi-scale modeling
    use_multi_scale: bool = True
    scale_factors: List[int] = None
    
    def __post_init__(self):
        if self.scale_factors is None:
            self.scale_factors = [1, 2, 4, 8]  # Different temporal scales


class MultiScaleTemporalEncoder(nn.Module):
    """Multi-scale temporal encoding for EEG signals."""
    
    def __init__(self, config: EEGFoundationConfig):
        super().__init__()
        self.config = config
        self.scale_factors = config.scale_factors
        
        # Convolutional encoders for different scales
        self.scale_encoders = nn.ModuleList()
        for scale in self.scale_factors:
            encoder = nn.Sequential(
                nn.Conv1d(
                    config.n_channels, 
                    config.d_model // len(self.scale_factors),
                    kernel_size=config.patch_size // scale,
                    stride=config.patch_size // scale // 2,
                    padding=config.patch_size // scale // 4
                ),
                nn.BatchNorm1d(config.d_model // len(self.scale_factors)),
                nn.ReLU(),
                nn.Conv1d(
                    config.d_model // len(self.scale_factors),
                    config.d_model // len(self.scale_factors),
                    kernel_size=3,
                    padding=1
                ),
                nn.BatchNorm1d(config.d_model // len(self.scale_factors)),
                nn.ReLU()
            )
            self.scale_encoders.append(encoder)
        
        # Fusion layer
        self.fusion = nn.Linear(config.d_model, config.d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-scale encoding.
        
        Args:
            x: Input tensor (batch_size, n_channels, seq_length)
            
        Returns:
            Multi-scale encoded features (batch_size, n_patches, d_model)
        """
        batch_size = x.size(0)
        scale_features = []
        
        for i, (scale, encoder) in enumerate(zip(self.scale_factors, self.scale_encoders)):
            # Downsample for different scales
            if scale > 1:
                x_scale = F.avg_pool1d(x, kernel_size=scale, stride=scale)
            else:
                x_scale = x
                
            # Apply scale-specific encoder
            features = encoder(x_scale)  # (batch_size, d_model//n_scales, n_patches)
            features = features.transpose(1, 2)  # (batch_size, n_patches, d_model//n_scales)
            scale_features.append(features)
        
        # Concatenate features from all scales
        # Pad to same length if necessary
        max_patches = max(f.size(1) for f in scale_features)
        padded_features = []
        
        for features in scale_features:
            if features.size(1) < max_patches:
                padding = max_patches - features.size(1)
                features = F.pad(features, (0, 0, 0, padding))
            elif features.size(1) > max_patches:
                features = features[:, :max_patches, :]
            padded_features.append(features)
        
        # Concatenate along feature dimension
        multi_scale_features = torch.cat(padded_features, dim=2)  # (batch_size, n_patches, d_model)
        
        # Apply fusion
        fused_features = self.fusion(multi_scale_features)
        
        return fused_features


class ChannelAttention(nn.Module):
    """Cross-channel attention mechanism for EEG."""
    
    def __init__(self, config: EEGFoundationConfig):
        super().__init__()
        self.config = config
        self.n_channels = config.n_channels
        self.n_heads = config.channel_attention_heads
        self.head_dim = config.d_model // self.n_heads
        
        self.query = nn.Linear(config.d_model, config.d_model)
        self.key = nn.Linear(config.d_model, config.d_model)
        self.value = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.d_model)
        
    def forward(self, x: torch.Tensor, channel_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention.
        
        Args:
            x: Input features (batch_size, n_patches, d_model)
            channel_embeddings: Channel embeddings (n_channels, d_model)
            
        Returns:
            Channel-attended features (batch_size, n_patches, d_model)
        """
        batch_size, n_patches, d_model = x.shape
        
        # Expand channel embeddings for each patch
        channel_emb_expanded = channel_embeddings.unsqueeze(0).unsqueeze(0)  # (1, 1, n_channels, d_model)
        channel_emb_expanded = channel_emb_expanded.expand(batch_size, n_patches, -1, -1)  # (batch_size, n_patches, n_channels, d_model)
        
        # Compute attention
        q = self.query(x).view(batch_size, n_patches, self.n_heads, self.head_dim)
        k = self.key(channel_emb_expanded).view(batch_size, n_patches, self.n_channels, self.n_heads, self.head_dim)
        v = self.value(channel_emb_expanded).view(batch_size, n_patches, self.n_channels, self.n_heads, self.head_dim)
        
        # Compute attention scores
        q = q.unsqueeze(2)  # (batch_size, n_patches, 1, n_heads, head_dim)
        attention_scores = torch.sum(q * k, dim=-1) / math.sqrt(self.head_dim)  # (batch_size, n_patches, n_channels, n_heads)
        attention_weights = F.softmax(attention_scores, dim=2)  # Attention over channels
        
        # Apply attention
        attended = torch.sum(attention_weights.unsqueeze(-1) * v, dim=2)  # (batch_size, n_patches, n_heads, head_dim)
        attended = attended.view(batch_size, n_patches, d_model)
        
        # Output projection and residual connection
        output = self.output(attended)
        output = self.dropout(output)
        output = self.layer_norm(x + output)
        
        return output


class EEGTransformerBlock(nn.Module):
    """Transformer block optimized for EEG data."""
    
    def __init__(self, config: EEGFoundationConfig):
        super().__init__()
        self.config = config
        
        # Self-attention
        self.self_attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        
        # Channel attention (optional)
        if config.use_channel_attention:
            self.channel_attention = ChannelAttention(config)
        else:
            self.channel_attention = None
            
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                channel_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor (batch_size, seq_length, d_model)
            attention_mask: Attention mask for self-attention
            channel_embeddings: Channel embeddings for channel attention
            
        Returns:
            Output tensor (batch_size, seq_length, d_model)
        """
        # Self-attention with residual connection
        attended, _ = self.self_attention(x, x, x, attn_mask=attention_mask)
        x = self.ln1(x + attended)
        
        # Channel attention (if enabled)
        if self.channel_attention is not None and channel_embeddings is not None:
            x = self.channel_attention(x, channel_embeddings)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.ln2(x + ff_output)
        
        return x


class EEGFoundationModel(nn.Module):
    """
    EEG Foundation Model with self-supervised pre-training capabilities.
    
    This model uses a transformer architecture with EEG-specific modifications:
    - Multi-scale temporal encoding
    - Channel attention mechanisms
    - Masked EEG modeling for pre-training
    - Contrastive learning for robust representations
    """
    
    def __init__(self, config: EEGFoundationConfig):
        super().__init__()
        self.config = config
        
        # Multi-scale temporal encoder
        if config.use_multi_scale:
            self.temporal_encoder = MultiScaleTemporalEncoder(config)
        else:
            # Simple patch embedding
            self.patch_embedding = nn.Conv1d(
                config.n_channels,
                config.d_model,
                kernel_size=config.patch_size,
                stride=int(config.patch_size * (1 - config.overlap))
            )
            
        # Channel embeddings
        self.channel_embeddings = nn.Embedding(config.n_channels, config.d_model)
        
        # Positional embeddings
        if config.use_learned_positional:
            self.positional_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        else:
            self.register_buffer('positional_embeddings', self._create_sinusoidal_embeddings())
            
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            EEGTransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # Pre-training heads
        self.masked_eeg_head = nn.Linear(config.d_model, config.patch_size * config.n_channels)
        self.contrastive_projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model // 2)
        )
        
        # Classification head (for fine-tuning)
        self.classification_head = None
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _create_sinusoidal_embeddings(self) -> torch.Tensor:
        """Create sinusoidal positional embeddings."""
        position = torch.arange(self.config.max_position_embeddings).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.config.d_model, 2).float() *
                           -(math.log(10000.0) / self.config.d_model))
        
        pe = torch.zeros(self.config.max_position_embeddings, self.config.d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
        
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
    def create_attention_mask(self, seq_length: int, mask_ratio: float = 0.0) -> torch.Tensor:
        """Create attention mask for masked modeling."""
        if mask_ratio == 0.0:
            return None
            
        mask = torch.ones(seq_length, dtype=torch.bool)
        num_masked = int(seq_length * mask_ratio)
        masked_indices = torch.randperm(seq_length)[:num_masked]
        mask[masked_indices] = False
        
        return mask
        
    def forward(self, 
                x: torch.Tensor,
                mask_ratio: float = 0.0,
                return_hidden_states: bool = False,
                return_attentions: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the foundation model.
        
        Args:
            x: Input EEG data (batch_size, n_channels, seq_length)
            mask_ratio: Ratio of patches to mask (for pre-training)
            return_hidden_states: Whether to return hidden states
            return_attentions: Whether to return attention weights
            
        Returns:
            Dictionary containing model outputs
        """
        batch_size = x.size(0)
        
        # Temporal encoding
        if self.config.use_multi_scale:
            x = self.temporal_encoder(x)  # (batch_size, n_patches, d_model)
        else:
            x = self.patch_embedding(x)  # (batch_size, d_model, n_patches)
            x = x.transpose(1, 2)  # (batch_size, n_patches, d_model)
            
        seq_length = x.size(1)
        
        # Add positional embeddings
        if self.config.use_learned_positional:
            position_ids = torch.arange(seq_length, device=x.device).unsqueeze(0)
            position_embeddings = self.positional_embeddings(position_ids)
            x = x + position_embeddings
        else:
            x = x + self.positional_embeddings[:seq_length].unsqueeze(0)
            
        # Create attention mask for masked modeling
        attention_mask = None
        masked_positions = None
        if mask_ratio > 0.0:
            attention_mask = self.create_attention_mask(seq_length, mask_ratio)
            masked_positions = ~attention_mask
            
        # Get channel embeddings
        channel_ids = torch.arange(self.config.n_channels, device=x.device)
        channel_embeddings = self.channel_embeddings(channel_ids)
        
        # Forward through transformer blocks
        hidden_states = []
        attentions = []
        
        for block in self.transformer_blocks:
            x = block(x, attention_mask, channel_embeddings)
            
            if return_hidden_states:
                hidden_states.append(x)
                
        # Final layer normalization
        x = self.layer_norm(x)
        
        # Prepare outputs
        outputs = {
            'last_hidden_state': x,
            'pooled_output': torch.mean(x, dim=1)  # Global pooling
        }
        
        if return_hidden_states:
            outputs['hidden_states'] = hidden_states
            
        if return_attentions:
            outputs['attentions'] = attentions
            
        # Pre-training heads
        if mask_ratio > 0.0 and masked_positions is not None:
            # Masked EEG modeling
            masked_tokens = x[masked_positions.unsqueeze(0).expand(batch_size, -1)]
            reconstruction = self.masked_eeg_head(masked_tokens)
            outputs['reconstruction'] = reconstruction
            outputs['masked_positions'] = masked_positions
            
        # Contrastive learning projection
        contrastive_features = self.contrastive_projection(outputs['pooled_output'])
        outputs['contrastive_features'] = contrastive_features
        
        return outputs
        
    def add_classification_head(self, num_classes: int, dropout: float = 0.1):
        """Add classification head for fine-tuning."""
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.config.d_model, num_classes)
        )
        
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification task."""
        if self.classification_head is None:
            raise ValueError("Classification head not initialized. Call add_classification_head() first.")
            
        outputs = self.forward(x)
        logits = self.classification_head(outputs['pooled_output'])
        return logits
        
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings for downstream tasks."""
        outputs = self.forward(x)
        return outputs['pooled_output']
        
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning."""
        for name, param in self.named_parameters():
            if 'classification_head' not in name:
                param.requires_grad = False
                
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.parameters():
            param.requires_grad = True
            
    def save_pretrained(self, save_directory: Path):
        """Save pre-trained model."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), save_directory / 'pytorch_model.bin')
        
        # Save config
        config_dict = {
            'd_model': self.config.d_model,
            'n_heads': self.config.n_heads,
            'n_layers': self.config.n_layers,
            'd_ff': self.config.d_ff,
            'dropout': self.config.dropout,
            'n_channels': self.config.n_channels,
            'max_seq_length': self.config.max_seq_length,
            'patch_size': self.config.patch_size,
            'overlap': self.config.overlap,
            'mask_ratio': self.config.mask_ratio,
            'contrastive_temperature': self.config.contrastive_temperature,
            'reconstruction_weight': self.config.reconstruction_weight,
            'contrastive_weight': self.config.contrastive_weight,
            'max_position_embeddings': self.config.max_position_embeddings,
            'use_learned_positional': self.config.use_learned_positional,
            'use_channel_attention': self.config.use_channel_attention,
            'channel_attention_heads': self.config.channel_attention_heads,
            'use_multi_scale': self.config.use_multi_scale,
            'scale_factors': self.config.scale_factors
        }
        
        with open(save_directory / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
            
        logger.info(f"Model saved to {save_directory}")
        
    @classmethod
    def from_pretrained(cls, model_directory: Path):
        """Load pre-trained model."""
        model_directory = Path(model_directory)
        
        # Load config
        with open(model_directory / 'config.json', 'r') as f:
            config_dict = json.load(f)
            
        config = EEGFoundationConfig(**config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights
        state_dict = torch.load(model_directory / 'pytorch_model.bin', map_location='cpu')
        model.load_state_dict(state_dict)
        
        logger.info(f"Model loaded from {model_directory}")
        return model


class EEGFoundationPreTrainer:
    """Pre-trainer for EEG Foundation Model with self-supervised learning."""
    
    def __init__(self, 
                 model: EEGFoundationModel,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.config = model.config
        
    def compute_masked_eeg_loss(self, 
                               predictions: torch.Tensor,
                               targets: torch.Tensor,
                               masked_positions: torch.Tensor) -> torch.Tensor:
        """Compute masked EEG modeling loss."""
        # Flatten predictions and targets
        predictions = predictions.view(-1, predictions.size(-1))
        targets = targets.view(-1, targets.size(-1))
        
        # Apply mask
        masked_predictions = predictions[masked_positions.view(-1)]
        masked_targets = targets[masked_positions.view(-1)]
        
        # Compute MSE loss
        loss = F.mse_loss(masked_predictions, masked_targets)
        return loss
        
    def compute_contrastive_loss(self, 
                               features: torch.Tensor,
                               temperature: float = 0.07) -> torch.Tensor:
        """Compute contrastive learning loss."""
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / temperature
        
        # Create labels (positive pairs are (i, i))
        labels = torch.arange(batch_size, device=features.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss
        
    def pretrain_step(self, 
                     eeg_data: torch.Tensor,
                     mask_ratio: float = None) -> Dict[str, float]:
        """Single pre-training step."""
        if mask_ratio is None:
            mask_ratio = self.config.mask_ratio
            
        eeg_data = eeg_data.to(self.device)
        batch_size = eeg_data.size(0)
        
        # Forward pass
        outputs = self.model(eeg_data, mask_ratio=mask_ratio)
        
        total_loss = 0.0
        losses = {}
        
        # Masked EEG modeling loss
        if 'reconstruction' in outputs and 'masked_positions' in outputs:
            # Create target patches
            # This is a simplified version - in practice, you'd extract the actual masked patches
            target_patches = torch.randn_like(outputs['reconstruction'])  # Placeholder
            
            masked_loss = self.compute_masked_eeg_loss(
                outputs['reconstruction'],
                target_patches,
                outputs['masked_positions']
            )
            
            total_loss += self.config.reconstruction_weight * masked_loss
            losses['masked_eeg_loss'] = masked_loss.item()
            
        # Contrastive learning loss
        if 'contrastive_features' in outputs:
            contrastive_loss = self.compute_contrastive_loss(
                outputs['contrastive_features'],
                self.config.contrastive_temperature
            )
            
            total_loss += self.config.contrastive_weight * contrastive_loss
            losses['contrastive_loss'] = contrastive_loss.item()
            
        losses['total_loss'] = total_loss.item()
        
        return total_loss, losses 