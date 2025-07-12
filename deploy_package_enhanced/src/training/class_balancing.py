"""
Class balancing strategies for handling imbalanced EEG datasets.
Includes various sampling and loss weighting techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
from sklearn.utils import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
import logging

logger = logging.getLogger(__name__)


class ClassWeightCalculator:
    """Calculate class weights for imbalanced datasets."""
    
    @staticmethod
    def calculate_weights(labels: np.ndarray, 
                         method: str = 'balanced') -> torch.Tensor:
        """
        Calculate class weights.
        
        Args:
            labels: Array of labels
            method: Weight calculation method
            
        Returns:
            Tensor of class weights
        """
        unique_classes = np.unique(labels)
        n_classes = len(unique_classes)
        
        if method == 'balanced':
            # Sklearn balanced weights
            weights = compute_class_weight(
                class_weight='balanced',
                classes=unique_classes,
                y=labels
            )
        elif method == 'effective_samples':
            # Effective number of samples
            weights = ClassWeightCalculator._effective_samples_weights(labels)
        elif method == 'inverse_frequency':
            # Inverse frequency
            class_counts = Counter(labels)
            total_samples = len(labels)
            weights = np.array([
                total_samples / (n_classes * class_counts[i])
                for i in range(n_classes)
            ])
        elif method == 'sqrt_inverse':
            # Square root of inverse frequency
            class_counts = Counter(labels)
            weights = np.array([
                np.sqrt(len(labels) / (n_classes * class_counts[i]))
                for i in range(n_classes)
            ])
        else:
            # Equal weights
            weights = np.ones(n_classes)
            
        # Normalize weights
        weights = weights / weights.sum() * n_classes
        
        return torch.tensor(weights, dtype=torch.float32)
        
    @staticmethod
    def _effective_samples_weights(labels: np.ndarray, 
                                  beta: float = 0.999) -> np.ndarray:
        """Calculate weights using effective number of samples."""
        class_counts = Counter(labels)
        n_classes = len(class_counts)
        
        effective_num = []
        for class_id in range(n_classes):
            n_samples = class_counts[class_id]
            effective = (1 - beta ** n_samples) / (1 - beta)
            effective_num.append(effective)
            
        weights = np.array([
            1.0 / eff if eff > 0 else 1.0 
            for eff in effective_num
        ])
        
        return weights


class BalancedBatchSampler:
    """Sampler that ensures balanced classes in each batch."""
    
    def __init__(self, labels: np.ndarray, batch_size: int,
                 drop_last: bool = False):
        self.labels = labels
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group indices by class
        self.class_indices = self._group_by_class(labels)
        self.n_classes = len(self.class_indices)
        self.samples_per_class = batch_size // self.n_classes
        
        # Calculate number of batches
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        self.n_batches = min_class_size // self.samples_per_class
        
    def _group_by_class(self, labels: np.ndarray) -> Dict[int, List[int]]:
        """Group sample indices by class."""
        class_indices = {}
        for idx, label in enumerate(labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices
        
    def __iter__(self):
        # Shuffle indices within each class
        for indices in self.class_indices.values():
            np.random.shuffle(indices)
            
        # Generate balanced batches
        for batch_idx in range(self.n_batches):
            batch = []
            
            # Sample from each class
            for class_id, indices in self.class_indices.items():
                start = batch_idx * self.samples_per_class
                end = start + self.samples_per_class
                batch.extend(indices[start:end])
                
            # Shuffle batch
            np.random.shuffle(batch)
            yield batch
            
    def __len__(self):
        return self.n_batches


class WeightedSampler:
    """Create weighted sampler for imbalanced datasets."""
    
    @staticmethod
    def create_sampler(labels: np.ndarray, 
                      method: str = 'balanced') -> WeightedRandomSampler:
        """
        Create weighted random sampler.
        
        Args:
            labels: Array of labels
            method: Sampling method
            
        Returns:
            WeightedRandomSampler
        """
        # Calculate class weights
        class_weights = ClassWeightCalculator.calculate_weights(labels, method)
        
        # Assign weight to each sample
        sample_weights = torch.tensor([
            class_weights[label] for label in labels
        ])
        
        # Create sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(labels),
            replacement=True
        )
        
        return sampler


class SMOTEBalancer:
    """SMOTE-based balancing for feature vectors."""
    
    def __init__(self, method: str = 'regular', 
                 sampling_strategy: Union[str, Dict] = 'auto',
                 k_neighbors: int = 5):
        self.method = method
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        
        # Select SMOTE variant
        if method == 'regular':
            self.sampler = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=42
            )
        elif method == 'borderline':
            self.sampler = BorderlineSMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=42
            )
        elif method == 'adasyn':
            self.sampler = ADASYN(
                sampling_strategy=sampling_strategy,
                n_neighbors=k_neighbors,
                random_state=42
            )
        elif method == 'smote_tomek':
            self.sampler = SMOTETomek(
                random_state=42
            )
        elif method == 'smote_enn':
            self.sampler = SMOTEENN(
                random_state=42
            )
        else:
            raise ValueError(f"Unknown SMOTE method: {method}")
            
    def balance_features(self, features: np.ndarray, 
                        labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance features using SMOTE.
        
        Args:
            features: Feature array (n_samples, n_features)
            labels: Label array
            
        Returns:
            Balanced features and labels
        """
        # Check if balancing is needed
        class_counts = Counter(labels)
        if len(set(class_counts.values())) == 1:
            return features, labels
            
        # Apply SMOTE
        try:
            features_balanced, labels_balanced = self.sampler.fit_resample(
                features, labels
            )
            
            logger.info(f"SMOTE balancing: {len(features)} -> {len(features_balanced)} samples")
            logger.info(f"Class distribution: {Counter(labels_balanced)}")
            
            return features_balanced, labels_balanced
            
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}. Returning original data.")
            return features, labels


class FocalLoss(nn.Module):
    """Focal Loss for addressing extreme class imbalance."""
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, 
                 gamma: float = 2.0, reduction: str = 'mean',
                 label_smoothing: float = 0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.
        
        Args:
            inputs: Model predictions (batch, n_classes)
            targets: Ground truth labels (batch,)
            
        Returns:
            Focal loss value
        """
        n_classes = inputs.size(1)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_one_hot = F.one_hot(targets, n_classes).float()
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + \
                            self.label_smoothing / n_classes
            ce_loss = -targets_one_hot * F.log_softmax(inputs, dim=1)
            ce_loss = ce_loss.sum(dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            
        # Calculate focal term
        p_t = torch.exp(-ce_loss)
        focal_term = (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss
            
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """Class-balanced loss based on effective number of samples."""
    
    def __init__(self, class_counts: List[int], beta: float = 0.999,
                 loss_type: str = 'cross_entropy'):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.loss_type = loss_type
        
        # Calculate effective number of samples
        effective_num = [(1 - beta ** n) / (1 - beta) for n in class_counts]
        weights = [1.0 / num for num in effective_num]
        
        # Normalize weights
        weights = torch.tensor(weights) / sum(weights) * len(weights)
        self.weights = weights
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate class-balanced loss."""
        if self.weights.device != inputs.device:
            self.weights = self.weights.to(inputs.device)
            
        if self.loss_type == 'cross_entropy':
            return F.cross_entropy(inputs, targets, weight=self.weights)
        elif self.loss_type == 'focal':
            focal = FocalLoss(alpha=self.weights, gamma=2.0)
            return focal(inputs, targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin Loss."""
    
    def __init__(self, class_counts: List[int], max_margin: float = 0.5,
                 scale: float = 30.0):
        super(LDAMLoss, self).__init__()
        
        # Calculate margins based on class frequencies
        margins = []
        for count in class_counts:
            margin = max_margin * (1 - np.sqrt(count / max(class_counts)))
            margins.append(margin)
            
        self.margins = torch.tensor(margins)
        self.scale = scale
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate LDAM loss."""
        if self.margins.device != inputs.device:
            self.margins = self.margins.to(inputs.device)
            
        # Get batch margins
        batch_margins = self.margins[targets]
        
        # Modify logits with margins
        one_hot = F.one_hot(targets, inputs.size(1)).float()
        margin_logits = inputs - one_hot * batch_margins.unsqueeze(1)
        
        # Scale and calculate loss
        return F.cross_entropy(self.scale * margin_logits, targets)


class MixupLoss(nn.Module):
    """Loss function for mixup augmentation."""
    
    def __init__(self, base_loss: nn.Module):
        super(MixupLoss, self).__init__()
        self.base_loss = base_loss
        
    def forward(self, inputs: torch.Tensor, targets_a: torch.Tensor,
                targets_b: torch.Tensor, lam: float) -> torch.Tensor:
        """Calculate mixup loss."""
        loss_a = self.base_loss(inputs, targets_a)
        loss_b = self.base_loss(inputs, targets_b)
        return lam * loss_a + (1 - lam) * loss_b


class BalancedDataLoader:
    """Create balanced data loaders with various strategies."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.balancing_config = config['training']['class_balancing']
        
    def create_loader(self, dataset: Dataset, is_training: bool = True) -> DataLoader:
        """Create data loader with appropriate balancing strategy."""
        # Extract labels
        labels = np.array([dataset[i]['label'] for i in range(len(dataset))])
        
        # Log class distribution
        class_counts = Counter(labels)
        logger.info(f"Class distribution: {dict(class_counts)}")
        
        if not is_training:
            # No balancing for validation/test
            return DataLoader(
                dataset,
                batch_size=self.config['inference']['batch_size'],
                shuffle=False,
                num_workers=self.config['hardware']['num_workers'],
                pin_memory=self.config['hardware']['pin_memory']
            )
            
        # Select balancing strategy
        method = self.balancing_config['method']
        
        if method == 'weighted_sampling':
            sampler = WeightedSampler.create_sampler(
                labels, 
                method=self.balancing_config.get('weighting_method', 'balanced')
            )
            return DataLoader(
                dataset,
                batch_size=self.config['models']['resnet1d_gru']['training']['batch_size'],
                sampler=sampler,
                num_workers=self.config['hardware']['num_workers'],
                pin_memory=self.config['hardware']['pin_memory']
            )
            
        elif method == 'balanced_batch':
            sampler = BalancedBatchSampler(
                labels,
                batch_size=self.config['models']['resnet1d_gru']['training']['batch_size'],
                drop_last=True
            )
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=self.config['hardware']['num_workers'],
                pin_memory=self.config['hardware']['pin_memory']
            )
            
        else:
            # Standard random sampling
            return DataLoader(
                dataset,
                batch_size=self.config['models']['resnet1d_gru']['training']['batch_size'],
                shuffle=True,
                num_workers=self.config['hardware']['num_workers'],
                pin_memory=self.config['hardware']['pin_memory']
            )
            
    def get_loss_function(self, class_counts: Optional[List[int]] = None) -> nn.Module:
        """Get appropriate loss function based on configuration."""
        method = self.balancing_config['method']
        
        if method == 'focal_loss':
            # Calculate class weights if needed
            if self.balancing_config.get('class_weights') == 'balanced' and class_counts:
                labels = []
                for class_id, count in enumerate(class_counts):
                    labels.extend([class_id] * count)
                weights = ClassWeightCalculator.calculate_weights(
                    np.array(labels), 'balanced'
                )
            else:
                weights = None
                
            return FocalLoss(
                alpha=weights,
                gamma=self.balancing_config.get('focal_gamma', 2.0),
                label_smoothing=self.config['models']['efficientnet']['training'].get(
                    'label_smoothing', 0.0
                )
            )
            
        elif method == 'class_balanced' and class_counts:
            return ClassBalancedLoss(
                class_counts,
                beta=self.balancing_config.get('cb_beta', 0.999)
            )
            
        elif method == 'ldam' and class_counts:
            return LDAMLoss(
                class_counts,
                max_margin=self.balancing_config.get('ldam_margin', 0.5)
            )
            
        else:
            # Standard cross-entropy with optional class weights
            if self.balancing_config.get('class_weights') == 'balanced' and class_counts:
                labels = []
                for class_id, count in enumerate(class_counts):
                    labels.extend([class_id] * count)
                weights = ClassWeightCalculator.calculate_weights(
                    np.array(labels), 'balanced'
                )
                return nn.CrossEntropyLoss(weight=weights)
            else:
                return nn.CrossEntropyLoss()


class HardExampleMining:
    """Hard example mining for focusing on difficult samples."""
    
    def __init__(self, mining_type: str = 'hard', percentile: float = 0.3):
        self.mining_type = mining_type
        self.percentile = percentile
        
    def mine_hard_examples(self, losses: torch.Tensor, 
                          labels: torch.Tensor) -> torch.Tensor:
        """
        Mine hard examples based on loss values.
        
        Args:
            losses: Loss values for each sample
            labels: Ground truth labels
            
        Returns:
            Mask for selected samples
        """
        batch_size = losses.size(0)
        
        if self.mining_type == 'hard':
            # Select top percentile of losses
            k = int(batch_size * self.percentile)
            _, indices = torch.topk(losses, k)
            mask = torch.zeros_like(losses, dtype=torch.bool)
            mask[indices] = True
            
        elif self.mining_type == 'semi-hard':
            # Select samples with moderate loss
            sorted_losses, _ = torch.sort(losses)
            low_threshold = sorted_losses[int(batch_size * 0.3)]
            high_threshold = sorted_losses[int(batch_size * 0.7)]
            mask = (losses > low_threshold) & (losses < high_threshold)
            
        elif self.mining_type == 'class_aware':
            # Mine hard examples per class
            mask = torch.zeros_like(losses, dtype=torch.bool)
            unique_labels = torch.unique(labels)
            
            for label in unique_labels:
                label_mask = labels == label
                label_losses = losses[label_mask]
                
                if len(label_losses) > 0:
                    k = max(1, int(len(label_losses) * self.percentile))
                    _, indices = torch.topk(label_losses, k)
                    label_indices = torch.where(label_mask)[0][indices]
                    mask[label_indices] = True
                    
        else:
            # Use all samples
            mask = torch.ones_like(losses, dtype=torch.bool)
            
        return mask 