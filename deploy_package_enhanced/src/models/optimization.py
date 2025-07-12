"""
Advanced optimization techniques for EEG classification models.
Includes regularization, compression, and optimization strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
import math
import logging

logger = logging.getLogger(__name__)


class DropConnect(nn.Module):
    """DropConnect regularization - drops individual weights."""
    
    def __init__(self, drop_rate: float = 0.5):
        super(DropConnect, self).__init__()
        self.drop_rate = drop_rate
        
    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return F.linear(x, weight)
            
        # Create binary mask for weights
        mask = torch.bernoulli(torch.ones_like(weight) * (1 - self.drop_rate))
        dropped_weight = weight * mask
        
        # Scale to maintain expected value
        return F.linear(x, dropped_weight * (1 / (1 - self.drop_rate)))


class Cutout(nn.Module):
    """Cutout augmentation for spectrograms."""
    
    def __init__(self, n_holes: int = 1, length: int = 16):
        super(Cutout, self).__init__()
        self.n_holes = n_holes
        self.length = length
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
            
        h, w = x.size(-2), x.size(-1)
        mask = torch.ones_like(x)
        
        for _ in range(self.n_holes):
            y = torch.randint(h, (1,)).item()
            x_coord = torch.randint(w, (1,)).item()
            
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x_coord - self.length // 2)
            x2 = min(w, x_coord + self.length // 2)
            
            mask[:, :, y1:y2, x1:x2] = 0
            
        return x * mask


class Mixup(nn.Module):
    """Mixup augmentation for EEG data."""
    
    def __init__(self, alpha: float = 1.0):
        super(Mixup, self).__init__()
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup augmentation.
        
        Returns:
            mixed_x: Mixed inputs
            y_a: Original labels
            y_b: Shuffled labels
            lam: Mixing coefficient
        """
        if not self.training:
            return x, y, y, 1.0
            
        batch_size = x.size(0)
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        
        # Random shuffle
        index = torch.randperm(batch_size).to(x.device)
        
        # Mix inputs
        mixed_x = lam * x + (1 - lam) * x[index]
        
        return mixed_x, y, y[index], lam


class CosineAnnealingWarmRestarts(_LRScheduler):
    """Cosine annealing with warm restarts and linear warmup."""
    
    def __init__(self, optimizer: Optimizer, T_0: int, T_mult: int = 1,
                 eta_min: float = 0, warmup_steps: int = 0, 
                 warmup_start_lr: float = 1e-6):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.T_cur = 0
        self.T_i = T_0
        self.cycle = 0
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, -1)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_steps
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * warmup_factor
                    for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            return [self.eta_min + (base_lr - self.eta_min) * 
                    (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                    for base_lr in self.base_lrs]
            
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            
        self.last_epoch = epoch
        self.T_cur = self.T_cur + 1
        
        if self.T_cur >= self.T_i:
            self.cycle += 1
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
            
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class LookAheadOptimizer(Optimizer):
    """LookAhead optimizer wrapper for better convergence."""
    
    def __init__(self, base_optimizer: Optimizer, k: int = 5, alpha: float = 0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        
        # Store slow weights
        self.slow_weights = [[p.clone().detach() for p in group['params']]
                            for group in base_optimizer.param_groups]
        
        # Initialize state
        defaults = dict(k=k, alpha=alpha)
        super(LookAheadOptimizer, self).__init__(base_optimizer.param_groups, defaults)
        
    def step(self, closure: Optional[Callable] = None):
        loss = self.base_optimizer.step(closure)
        self.step_count += 1
        
        if self.step_count % self.k == 0:
            # Update slow weights
            for group_idx, group in enumerate(self.base_optimizer.param_groups):
                for p_idx, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    # Update slow weight
                    self.slow_weights[group_idx][p_idx].mul_(1 - self.alpha).add_(
                        p.data, alpha=self.alpha
                    )
                    # Update fast weight
                    p.data.copy_(self.slow_weights[group_idx][p_idx])
                    
        return loss


class SAM(Optimizer):
    """Sharpness Aware Minimization (SAM) optimizer."""
    
    def __init__(self, base_optimizer: Optimizer, rho: float = 0.05):
        self.base_optimizer = base_optimizer
        self.rho = rho
        super(SAM, self).__init__(base_optimizer.param_groups, base_optimizer.defaults)
        
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)  # Climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w
                
        if zero_grad:
            self.zero_grad()
            
    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # Get back to "w" from "w + e(w)"
                
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
            
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


class ModelCompression:
    """Model compression techniques including pruning and quantization."""
    
    @staticmethod
    def magnitude_pruning(model: nn.Module, sparsity: float = 0.5) -> nn.Module:
        """Prune weights based on magnitude."""
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                weight = module.weight.data
                threshold = torch.quantile(torch.abs(weight), sparsity)
                mask = torch.abs(weight) > threshold
                module.weight.data = weight * mask
                
                # Store mask for inference
                module.register_buffer('weight_mask', mask)
                
        return model
    
    @staticmethod
    def structured_pruning(model: nn.Module, pruning_rate: float = 0.3) -> nn.Module:
        """Structured pruning - remove entire channels/filters."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Calculate importance scores for each filter
                weight = module.weight.data
                importance = torch.norm(weight, p=2, dim=(1, 2, 3))
                
                # Keep top (1-pruning_rate) filters
                n_keep = int(weight.shape[0] * (1 - pruning_rate))
                keep_indices = torch.topk(importance, n_keep).indices
                
                # Create new pruned layer
                new_conv = nn.Conv2d(
                    module.in_channels,
                    n_keep,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                    module.bias is not None
                )
                
                # Copy weights
                new_conv.weight.data = weight[keep_indices]
                if module.bias is not None:
                    new_conv.bias.data = module.bias.data[keep_indices]
                    
                # Replace module
                parent_module = model
                name_parts = name.split('.')
                for part in name_parts[:-1]:
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, name_parts[-1], new_conv)
                
        return model
    
    @staticmethod
    def quantize_model(model: nn.Module, backend: str = 'qnnpack') -> nn.Module:
        """Quantize model to INT8."""
        torch.backends.quantized.engine = backend
        
        # Prepare model for quantization
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        model_prepared = torch.quantization.prepare(model)
        
        # Calibrate with representative data (would need actual data in practice)
        # model_prepared(calibration_data)
        
        # Convert to quantized model
        model_quantized = torch.quantization.convert(model_prepared)
        
        return model_quantized


class KnowledgeDistillation(nn.Module):
    """Knowledge distillation for model compression."""
    
    def __init__(self, teacher_model: nn.Module, temperature: float = 3.0,
                 alpha: float = 0.7):
        super(KnowledgeDistillation, self).__init__()
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
    def forward(self, student_logits: torch.Tensor, teacher_input: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate distillation loss.
        
        Args:
            student_logits: Logits from student model
            teacher_input: Input to teacher model
            labels: Ground truth labels
            
        Returns:
            Combined loss
        """
        # Get teacher predictions
        with torch.no_grad():
            teacher_logits = self.teacher_model(teacher_input)
            if isinstance(teacher_logits, dict):
                teacher_logits = teacher_logits['logits']
                
        # Soft targets loss
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean')
        distillation_loss *= self.temperature ** 2
        
        # Hard targets loss
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        return self.alpha * distillation_loss + (1 - self.alpha) * hard_loss


class GradientClipping:
    """Advanced gradient clipping strategies."""
    
    @staticmethod
    def adaptive_clip_grad_norm_(parameters, max_norm: float = 1.0, 
                                norm_type: float = 2.0,
                                adaptive_rate: float = 0.1) -> float:
        """Adaptive gradient clipping based on gradient history."""
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        
        if len(parameters) == 0:
            return 0.0
            
        # Calculate current gradient norm
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type
        )
        
        # Adaptive threshold based on gradient history
        for p in parameters:
            if not hasattr(p, 'grad_norm_history'):
                p.grad_norm_history = []
            p.grad_norm_history.append(total_norm.item())
            
            # Keep only recent history
            if len(p.grad_norm_history) > 100:
                p.grad_norm_history.pop(0)
                
            # Adapt max_norm based on history
            if len(p.grad_norm_history) > 10:
                mean_norm = np.mean(p.grad_norm_history)
                std_norm = np.std(p.grad_norm_history)
                adaptive_max_norm = mean_norm + adaptive_rate * std_norm
                max_norm = min(max_norm, adaptive_max_norm)
                
        # Clip gradients
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.detach().mul_(clip_coef)
                
        return total_norm.item()


class EarlyStopping:
    """Early stopping with patience and delta threshold."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001,
                 mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop 