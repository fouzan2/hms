"""
ResNet1D-GRU hybrid model for EEG time series classification.
Combines spatial feature extraction with temporal sequence modeling.
Enhanced with multi-scale features, dilated convolutions, and advanced attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from einops import rearrange, repeat
import logging
import math

logger = logging.getLogger(__name__)


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, channel: int, reduction: int = 16):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MultiScaleBlock1D(nn.Module):
    """Multi-scale feature extraction with dilated convolutions."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 dilations: List[int] = [1, 2, 4, 8]):
        super(MultiScaleBlock1D, self).__init__()
        
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels // len(dilations),
                             kernel_size=3, padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(out_channels // len(dilations)),
                    nn.ReLU(inplace=True)
                )
            )
            
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for branch in self.branches:
            outputs.append(branch(x))
        x = torch.cat(outputs, dim=1)
        return self.fusion(x)


class BasicBlock1D(nn.Module):
    """Enhanced ResNet block with SE attention and multi-scale features."""
    
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, 
                 stride: int = 1, downsample: Optional[nn.Module] = None,
                 dropout: float = 0.0, use_se: bool = True, 
                 use_multiscale: bool = False):
        super(BasicBlock1D, self).__init__()
        
        # First convolution
        if use_multiscale and stride == 1:
            self.conv1 = MultiScaleBlock1D(in_channels, out_channels)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3,
                         stride=stride, padding=1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        # Second convolution
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # SE block
        self.se = SEBlock1D(out_channels) if use_se else None
        
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        
        if self.dropout:
            out = self.dropout(out)
            
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.se:
            out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


class ResNet1D(nn.Module):
    """Enhanced ResNet for 1D EEG signals with clinical signal preservation."""
    
    def __init__(self, block: nn.Module, layers: List[int], 
                 in_channels: int = 20, initial_filters: int = 64,
                 dropout: float = 0.3, use_se: bool = True,
                 use_multiscale: bool = True):
        super(ResNet1D, self).__init__()
        
        self.in_channels = initial_filters
        self.dropout = dropout
        self.use_se = use_se
        self.use_multiscale = use_multiscale
        
        # Initial convolution with smaller kernel for better temporal resolution
        self.conv1 = nn.Conv1d(in_channels, initial_filters, kernel_size=15,
                               stride=1, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(initial_filters)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection from raw input
        self.input_skip = nn.Conv1d(in_channels, initial_filters, kernel_size=1)
        
        # ResNet layers with progressive feature extraction
        self.layer1 = self._make_layer(block, initial_filters, layers[0])
        self.layer2 = self._make_layer(block, initial_filters * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, initial_filters * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, initial_filters * 8, layers[3], stride=2)
        
        # Global pooling options
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, block: nn.Module, out_channels: int, 
                    blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )
            
        layers = []
        # First block with stride
        layers.append(block(self.in_channels, out_channels, stride, 
                           downsample, self.dropout, self.use_se, False))
        self.in_channels = out_channels * block.expansion
        
        # Remaining blocks with multi-scale option
        for i in range(1, blocks):
            use_multiscale = self.use_multiscale and (i == blocks // 2)
            layers.append(block(self.in_channels, out_channels, 
                               dropout=self.dropout, use_se=self.use_se,
                               use_multiscale=use_multiscale))
            
        return nn.Sequential(*layers)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x shape: (batch, channels, time)
        
        # Store intermediate features for skip connections
        features = {}
        
        # Initial processing
        x_init = self.conv1(x)
        x_init = self.bn1(x_init)
        x_init = self.relu(x_init)
        
        # Input skip for clinical signal preservation
        input_features = F.avg_pool1d(self.input_skip(x), kernel_size=8, stride=8)
        features['input_skip'] = input_features
        
        # Progressive feature extraction
        x1 = self.layer1(x_init)
        features['layer1'] = x1
        
        x2 = self.layer2(x1)
        features['layer2'] = x2
        
        x3 = self.layer3(x2)
        features['layer3'] = x3
        
        x4 = self.layer4(x3)
        features['layer4'] = x4
        
        return x4, features


class MultiHeadTemporalAttention(nn.Module):
    """Multi-head attention for temporal sequence modeling."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadTemporalAttention, self).__init__()
        
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections in batch from d_model => h x d_k
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear layer
        output = self.W_o(context)
        
        # Add & Norm
        output = self.layer_norm(output + x)
        
        # Average attention weights across heads
        attention_weights = attention_weights.mean(dim=1)
        
        return output, attention_weights


class AttentionGRU(nn.Module):
    """Enhanced GRU with multi-head attention and temporal modeling."""
    
    def __init__(self, input_size: int, hidden_size: int, 
                 num_layers: int = 2, bidirectional: bool = True,
                 dropout: float = 0.3, n_heads: int = 8):
        super(AttentionGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Multi-head temporal attention
        self.attention_dim = hidden_size * self.num_directions
        self.temporal_attention = MultiHeadTemporalAttention(
            self.attention_dim, n_heads=n_heads, dropout=dropout
        )
        
        # Context vector for attention
        self.context_vector = nn.Parameter(torch.randn(self.attention_dim))
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.attention_dim, self.attention_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.attention_dim // 2, self.attention_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch, time, features)
        
        # GRU forward pass
        gru_out, hidden = self.gru(x)
        # gru_out shape: (batch, time, hidden_size * num_directions)
        
        # Apply temporal attention
        attended_out, attention_weights = self.temporal_attention(gru_out)
        
        # Context-based attention pooling
        context = self.context_vector.expand(attended_out.size(0), 1, -1)
        scores = torch.bmm(attended_out, context.transpose(1, 2)).squeeze(2)
        attention_scores = F.softmax(scores, dim=1)
        
        # Weighted pooling
        output = torch.bmm(attention_scores.unsqueeze(1), attended_out).squeeze(1)
        
        # Output projection
        output = self.output_projection(output)
        
        return output, attention_scores


class ResNet1D_GRU(nn.Module):
    """Complete ResNet1D-GRU model for EEG classification."""
    
    def __init__(self, config: Dict):
        super(ResNet1D_GRU, self).__init__()
        
        # Extract configuration
        resnet_config = config['models']['resnet1d_gru']['resnet']
        gru_config = config['models']['resnet1d_gru']['gru']
        
        n_channels = len(config['eeg']['channels'])
        n_classes = len(config['classes'])
        
        # Build ResNet backbone
        self.resnet = ResNet1D(
            block=BasicBlock1D,
            layers=resnet_config['num_blocks'],
            in_channels=n_channels,
            initial_filters=resnet_config['initial_filters'],
            dropout=resnet_config['dropout'],
            use_se=resnet_config.get('use_se', resnet_config.get('se_block', True)),
            use_multiscale=resnet_config.get('use_multiscale', False)
        )
        
        # Calculate ResNet output size
        resnet_out_channels = resnet_config['initial_filters'] * 8
        
        # Build GRU
        self.gru = AttentionGRU(
            input_size=resnet_out_channels,
            hidden_size=gru_config['hidden_size'],
            num_layers=gru_config['num_layers'],
            bidirectional=gru_config['bidirectional'],
            dropout=gru_config['dropout'],
            n_heads=gru_config['n_heads']
        )
        
        # Classification head
        gru_out_size = gru_config['hidden_size'] * (2 if gru_config['bidirectional'] else 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_size, gru_out_size // 2),
            nn.ReLU(),
            nn.Dropout(resnet_config['dropout']),
            nn.Linear(gru_out_size // 2, n_classes)
        )
        
        # Additional outputs for interpretability
        self.features = {}
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x shape: (batch, channels, time)
        
        # ResNet feature extraction
        resnet_features, skip_features = self.resnet(x)
        # resnet_features shape: (batch, channels, compressed_time)
        
        # Store for interpretability
        self.features['resnet_output'] = resnet_features.detach()
        self.features['skip_features'] = {k: v.detach() for k, v in skip_features.items()}
        
        # Prepare for GRU: transpose to (batch, time, channels)
        gru_input = rearrange(resnet_features, 'b c t -> b t c')
        
        # GRU temporal modeling
        gru_output, attention_weights = self.gru(gru_input)
        
        # Store attention weights
        self.features['attention_weights'] = attention_weights.detach()
        
        # Classification
        logits = self.classifier(gru_output)
        
        return {
            'logits': logits,
            'attention_weights': attention_weights,
            'features': self.features
        }
        
    def get_attention_maps(self) -> Optional[torch.Tensor]:
        """Get attention weights from last forward pass."""
        return self.features.get('attention_weights', None)
        
    def get_feature_maps(self) -> Optional[torch.Tensor]:
        """Get ResNet feature maps from last forward pass."""
        return self.features.get('resnet_output', None)


class FocalLoss(nn.Module):
    """Focal Loss for handling extreme class imbalance."""
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            focal_loss = self.alpha[targets] * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TemporalConsistencyLoss(nn.Module):
    """Loss that encourages temporal consistency in predictions."""
    
    def __init__(self, weight: float = 0.1):
        super(TemporalConsistencyLoss, self).__init__()
        self.weight = weight
        
    def forward(self, attention_weights: torch.Tensor) -> torch.Tensor:
        # Encourage smooth attention weights over time
        if attention_weights.size(1) > 1:
            diff = attention_weights[:, 1:] - attention_weights[:, :-1]
            loss = torch.mean(torch.abs(diff))
            return self.weight * loss
        return torch.tensor(0.0, device=attention_weights.device)


class CombinedLoss(nn.Module):
    """Combined loss function with focal loss and temporal consistency."""
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None, 
                 gamma: float = 2.0, temporal_weight: float = 0.1):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=gamma)
        self.temporal_loss = TemporalConsistencyLoss(weight=temporal_weight)
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Focal loss for classification
        classification_loss = self.focal_loss(outputs['logits'], targets)
        
        # Temporal consistency loss
        temporal_loss = self.temporal_loss(outputs['attention_weights'])
        
        # Total loss
        total_loss = classification_loss + temporal_loss
        
        return {
            'loss': total_loss,
            'classification_loss': classification_loss,
            'temporal_loss': temporal_loss
        }


class ResNet1D_GRU_Trainer:
    """Training utilities for ResNet1D-GRU model."""
    
    def __init__(self, model: ResNet1D_GRU, config: Dict, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Loss function with class weights
        class_weights = self._calculate_class_weights()
        self.criterion = CombinedLoss(
            class_weights=class_weights,
            gamma=config['models']['resnet1d_gru']['training'].get('focal_gamma', 2.0),
            temporal_weight=config['models']['resnet1d_gru']['training'].get('temporal_weight', 0.1)
        )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config['models']['resnet1d_gru']['training'].get('mixed_precision', True) else None
        
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights based on dataset distribution."""
        # This should be calculated from actual dataset
        # Placeholder for now
        n_classes = len(self.config['classes'])
        return torch.ones(n_classes)
        
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with differential learning rates."""
        training_config = self.config['models']['resnet1d_gru']['training']
        
        # Different learning rates for different parts
        param_groups = [
            {'params': self.model.resnet.parameters(), 
             'lr': training_config['learning_rate'] * 0.1},
            {'params': self.model.gru.parameters(), 
             'lr': training_config['learning_rate']},
            {'params': self.model.classifier.parameters(), 
             'lr': training_config['learning_rate']}
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=training_config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
        
    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler with warm-up."""
        training_config = self.config['models']['resnet1d_gru']['training']
        
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=training_config.get('T_0', 10),
            T_mult=training_config.get('T_mult', 2),
            eta_min=1e-6
        )
        
        return scheduler
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with mixed precision."""
        self.model.train()
        
        # Move data to device
        inputs = batch['eeg'].to(self.device)
        targets = batch['label'].to(self.device)
        
        # Mixed precision training
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                losses = self.criterion(outputs, targets)
        else:
            outputs = self.model(inputs)
            losses = self.criterion(outputs, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.scaler:
            self.scaler.scale(losses['loss']).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            preds = torch.argmax(outputs['logits'], dim=1)
            accuracy = (preds == targets).float().mean()
            
        return {
            'loss': losses['loss'].item(),
            'classification_loss': losses['classification_loss'].item(),
            'temporal_loss': losses['temporal_loss'].item(),
            'accuracy': accuracy.item()
        }
        
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate model on validation/test set."""
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['eeg'].to(self.device)
                targets = batch['label'].to(self.device)
                
                # Use mixed precision for evaluation too
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        losses = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    losses = self.criterion(outputs, targets)
                
                probs = F.softmax(outputs['logits'], dim=1)
                preds = torch.argmax(outputs['logits'], dim=1)
                
                total_loss += losses['loss'].item() * inputs.size(0)
                total_correct += (preds == targets).sum().item()
                total_samples += inputs.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        metrics = {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples,
            'predictions': np.array(all_preds),
            'targets': np.array(all_targets),
            'probabilities': np.array(all_probs)
        }
        
        return metrics 