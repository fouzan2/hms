"""
Attention visualization and analysis for transformer-based EEG models.
Provides tools for understanding what the model is focusing on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score


@dataclass
class AttentionAnalysisResult:
    """Container for attention analysis results."""
    attention_weights: np.ndarray
    attention_rollout: Optional[np.ndarray]
    consistency_scores: Optional[Dict[str, float]]
    important_features: Optional[List[int]]
    metadata: Dict[str, any]


class AttentionVisualizer:
    """
    Visualize attention weights for transformer-based models.
    Creates interpretable visualizations of attention patterns.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        self.attention_weights = {}
        self._register_hooks()
        
    def _register_hooks(self):
        """Register hooks to capture attention weights."""
        def attention_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    # Multi-head attention returns (output, attention_weights)
                    self.attention_weights[name] = output[1].detach()
                return output
            return hook
            
        # Register hooks on all attention layers
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                module.register_forward_hook(attention_hook(name))
                
    def visualize(self,
                  input_tensor: torch.Tensor,
                  layer_names: Optional[List[str]] = None,
                  head_fusion: str = 'mean',
                  save_path: Optional[str] = None) -> AttentionAnalysisResult:
        """
        Visualize attention patterns.
        
        Args:
            input_tensor: Input EEG data
            layer_names: Specific layers to visualize (None for all)
            head_fusion: How to combine multi-head attention ('mean', 'max', 'all')
            save_path: Path to save visualization
            
        Returns:
            AttentionAnalysisResult with visualizations
        """
        # Forward pass to capture attention
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        # Select layers to visualize
        if layer_names is None:
            layer_names = list(self.attention_weights.keys())
            
        # Process attention weights
        processed_attention = {}
        for layer_name in layer_names:
            if layer_name in self.attention_weights:
                attn = self.attention_weights[layer_name]
                
                # Handle multi-head attention
                if head_fusion == 'mean':
                    attn = attn.mean(dim=1)  # Average across heads
                elif head_fusion == 'max':
                    attn = attn.max(dim=1)[0]  # Max across heads
                    
                processed_attention[layer_name] = attn.cpu().numpy()
                
        # Create visualizations
        self._create_attention_plots(processed_attention, save_path)
        
        # Compute attention rollout
        rollout = self._compute_attention_rollout(processed_attention)
        
        return AttentionAnalysisResult(
            attention_weights=processed_attention,
            attention_rollout=rollout,
            consistency_scores=None,
            important_features=None,
            metadata={'head_fusion': head_fusion}
        )
    
    def _create_attention_plots(self, attention_dict, save_path):
        """Create attention heatmap visualizations."""
        n_layers = len(attention_dict)
        fig, axes = plt.subplots(n_layers, 1, figsize=(12, 4 * n_layers))
        
        if n_layers == 1:
            axes = [axes]
            
        for idx, (layer_name, attn) in enumerate(attention_dict.items()):
            # Average over batch dimension
            attn_avg = attn.mean(axis=0)
            
            # Create heatmap
            sns.heatmap(
                attn_avg,
                ax=axes[idx],
                cmap='Blues',
                cbar_kws={'label': 'Attention Weight'},
                xticklabels=False,
                yticklabels=False
            )
            axes[idx].set_title(f'Attention Weights - {layer_name}')
            axes[idx].set_xlabel('Keys')
            axes[idx].set_ylabel('Queries')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _compute_attention_rollout(self, attention_dict):
        """Compute attention rollout across layers."""
        if not attention_dict:
            return None
            
        # Start with identity matrix
        rollout = None
        
        for layer_name, attn in attention_dict.items():
            if rollout is None:
                rollout = attn
            else:
                # Matrix multiplication to propagate attention
                rollout = np.matmul(attn, rollout)
                
        return rollout


class MultiHeadAttentionAnalyzer:
    """
    Analyze multi-head attention patterns to understand different focus areas.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def analyze_heads(self,
                      input_tensor: torch.Tensor,
                      layer_name: str,
                      target_class: Optional[int] = None) -> AttentionAnalysisResult:
        """
        Analyze individual attention heads.
        
        Args:
            input_tensor: Input EEG data
            layer_name: Name of attention layer to analyze
            target_class: Target class for analysis
            
        Returns:
            AttentionAnalysisResult with head-wise analysis
        """
        # Get attention weights for specific layer
        attention_weights = self._extract_attention_weights(
            input_tensor, layer_name
        )
        
        if attention_weights is None:
            raise ValueError(f"No attention weights found for layer {layer_name}")
            
        # Analyze each head
        head_analysis = self._analyze_individual_heads(attention_weights)
        
        # Compute head importance scores
        head_importance = self._compute_head_importance(
            input_tensor, attention_weights, target_class
        )
        
        # Find complementary heads
        complementary_heads = self._find_complementary_heads(attention_weights)
        
        return AttentionAnalysisResult(
            attention_weights=attention_weights.cpu().numpy(),
            attention_rollout=None,
            consistency_scores=head_importance,
            important_features=complementary_heads,
            metadata={'layer_name': layer_name, 'head_analysis': head_analysis}
        )
    
    def _extract_attention_weights(self, input_tensor, layer_name):
        """Extract attention weights from specific layer."""
        attention_weights = None
        
        def hook(module, input, output):
            nonlocal attention_weights
            if isinstance(output, tuple) and len(output) > 1:
                attention_weights = output[1].detach()
                
        # Register hook
        target_module = dict(self.model.named_modules())[layer_name]
        handle = target_module.register_forward_hook(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        handle.remove()
        return attention_weights
    
    def _analyze_individual_heads(self, attention_weights):
        """Analyze properties of individual attention heads."""
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        head_analysis = {}
        
        for head_idx in range(num_heads):
            head_attn = attention_weights[:, head_idx, :, :]
            
            # Compute entropy (focus vs. dispersed attention)
            head_entropy = []
            for b in range(batch_size):
                for s in range(seq_len):
                    head_entropy.append(
                        entropy(head_attn[b, s, :].cpu().numpy())
                    )
                    
            # Compute average attention distance
            positions = torch.arange(seq_len).float().to(attention_weights.device)
            avg_distance = 0
            for b in range(batch_size):
                for s in range(seq_len):
                    weights = head_attn[b, s, :]
                    expected_pos = (weights * positions).sum()
                    avg_distance += abs(expected_pos - s).item()
                    
            avg_distance /= (batch_size * seq_len)
            
            head_analysis[f'head_{head_idx}'] = {
                'avg_entropy': np.mean(head_entropy),
                'std_entropy': np.std(head_entropy),
                'avg_distance': avg_distance,
                'is_positional': avg_distance < seq_len * 0.1,
                'is_global': np.mean(head_entropy) > 0.8 * np.log(seq_len)
            }
            
        return head_analysis
    
    def _compute_head_importance(self, input_tensor, attention_weights, target_class):
        """Compute importance scores for each attention head."""
        num_heads = attention_weights.shape[1]
        head_importance = {}
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(input_tensor)
            if target_class is None:
                target_class = baseline_output.argmax(dim=1).item()
            baseline_prob = baseline_output[:, target_class].item()
            
        # Ablate each head and measure impact
        for head_idx in range(num_heads):
            # Create modified attention with head zeroed out
            modified_attn = attention_weights.clone()
            modified_attn[:, head_idx, :, :] = 0
            
            # Would need model modification to inject attention
            # This is a simplified importance score
            head_variance = attention_weights[:, head_idx, :, :].var().item()
            head_importance[f'head_{head_idx}'] = head_variance
            
        # Normalize importance scores
        total_importance = sum(head_importance.values())
        for key in head_importance:
            head_importance[key] /= total_importance
            
        return head_importance
    
    def _find_complementary_heads(self, attention_weights):
        """Find heads that focus on complementary patterns."""
        num_heads = attention_weights.shape[1]
        complementary_pairs = []
        
        # Compare attention patterns between heads
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                attn_i = attention_weights[:, i, :, :].flatten()
                attn_j = attention_weights[:, j, :, :].flatten()
                
                # Compute correlation
                correlation = torch.corrcoef(
                    torch.stack([attn_i, attn_j])
                )[0, 1].item()
                
                # Low correlation suggests complementary patterns
                if abs(correlation) < 0.3:
                    complementary_pairs.append((i, j))
                    
        return complementary_pairs


class AttentionRollout:
    """
    Implement attention rollout for deep attention networks.
    Shows how attention flows through the network.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def compute_rollout(self,
                        input_tensor: torch.Tensor,
                        method: str = 'mean',
                        normalize: bool = True) -> np.ndarray:
        """
        Compute attention rollout.
        
        Args:
            input_tensor: Input data
            method: Rollout method ('mean', 'max', 'flow')
            normalize: Whether to normalize attention weights
            
        Returns:
            Attention rollout matrix
        """
        # Extract all attention weights
        attention_layers = self._extract_all_attention_weights(input_tensor)
        
        if not attention_layers:
            raise ValueError("No attention layers found in model")
            
        # Initialize rollout
        rollout = None
        
        for layer_idx, attn_weights in enumerate(attention_layers):
            # Process multi-head attention
            if method == 'mean':
                attn = attn_weights.mean(dim=1)
            elif method == 'max':
                attn = attn_weights.max(dim=1)[0]
            elif method == 'flow':
                attn = self._compute_attention_flow(attn_weights)
            else:
                raise ValueError(f"Unknown rollout method: {method}")
                
            # Add residual connection
            eye = torch.eye(attn.shape[-1]).to(attn.device)
            attn = 0.5 * attn + 0.5 * eye
            
            # Normalize if requested
            if normalize:
                attn = attn / attn.sum(dim=-1, keepdim=True)
                
            # Compute rollout
            if rollout is None:
                rollout = attn
            else:
                rollout = torch.matmul(attn, rollout)
                
        return rollout.cpu().numpy()
    
    def _extract_all_attention_weights(self, input_tensor):
        """Extract attention weights from all layers."""
        attention_weights = []
        
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attention_weights.append(output[1].detach())
                
        # Register hooks on all attention layers
        handles = []
        for module in self.model.modules():
            if hasattr(module, 'attention') or 'attention' in str(type(module)).lower():
                handle = module.register_forward_hook(hook)
                handles.append(handle)
                
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        return attention_weights
    
    def _compute_attention_flow(self, attention_weights):
        """Compute attention flow for more accurate rollout."""
        # Implementation of attention flow
        # This considers the actual gradient flow through attention
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Compute head-wise flow
        flows = []
        for head in range(num_heads):
            head_attn = attention_weights[:, head, :, :]
            
            # Compute flow matrix based on attention gradients
            flow = head_attn
            for _ in range(3):  # Iterative refinement
                flow = 0.5 * flow + 0.5 * torch.matmul(flow, flow)
                flow = flow / flow.sum(dim=-1, keepdim=True)
                
            flows.append(flow)
            
        # Combine flows
        combined_flow = torch.stack(flows, dim=1).mean(dim=1)
        return combined_flow


class AttentionConsistencyChecker:
    """
    Check consistency of attention patterns across similar inputs.
    Helps identify stable vs. unstable attention mechanisms.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def check_consistency(self,
                          input_batch: torch.Tensor,
                          noise_levels: List[float] = [0.01, 0.05, 0.1],
                          n_samples: int = 10) -> Dict[str, float]:
        """
        Check attention consistency under input perturbations.
        
        Args:
            input_batch: Batch of input data
            noise_levels: Noise levels to test
            n_samples: Number of noisy samples per input
            
        Returns:
            Dictionary of consistency metrics
        """
        consistency_scores = {}
        
        # Get original attention
        original_attention = self._get_attention_weights(input_batch)
        
        for noise_level in noise_levels:
            similarities = []
            
            for _ in range(n_samples):
                # Add noise to input
                noise = torch.randn_like(input_batch) * noise_level
                noisy_input = input_batch + noise
                
                # Get attention for noisy input
                noisy_attention = self._get_attention_weights(noisy_input)
                
                # Compute similarity
                similarity = self._compute_attention_similarity(
                    original_attention, noisy_attention
                )
                similarities.append(similarity)
                
            consistency_scores[f'noise_{noise_level}'] = {
                'mean_similarity': np.mean(similarities),
                'std_similarity': np.std(similarities),
                'min_similarity': np.min(similarities)
            }
            
        # Check temporal consistency
        temporal_consistency = self._check_temporal_consistency(input_batch)
        consistency_scores['temporal'] = temporal_consistency
        
        return consistency_scores
    
    def _get_attention_weights(self, input_tensor):
        """Extract attention weights from model."""
        attention_weights = []
        
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attention_weights.append(output[1].detach())
                
        # Register hooks
        handles = []
        for module in self.model.modules():
            if 'attention' in str(type(module)).lower():
                handle = module.register_forward_hook(hook)
                handles.append(handle)
                
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        # Remove hooks
        for handle in handles:
            handle.remove()
            
        return attention_weights
    
    def _compute_attention_similarity(self, attn1_list, attn2_list):
        """Compute similarity between two sets of attention weights."""
        similarities = []
        
        for attn1, attn2 in zip(attn1_list, attn2_list):
            # Flatten attention matrices
            attn1_flat = attn1.flatten()
            attn2_flat = attn2.flatten()
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(
                attn1_flat.unsqueeze(0),
                attn2_flat.unsqueeze(0)
            ).item()
            
            similarities.append(similarity)
            
        return np.mean(similarities)
    
    def _check_temporal_consistency(self, input_batch):
        """Check if attention is consistent across time steps."""
        attention_weights = self._get_attention_weights(input_batch)
        
        temporal_scores = []
        for attn in attention_weights:
            # Check consistency across time dimension
            batch_size, num_heads, seq_len, _ = attn.shape
            
            # Compute temporal variance
            temporal_variance = []
            for b in range(batch_size):
                for h in range(num_heads):
                    # Variance across time steps
                    var = attn[b, h, :, :].var(dim=0).mean().item()
                    temporal_variance.append(var)
                    
            temporal_scores.append(np.mean(temporal_variance))
            
        return {
            'mean_temporal_variance': np.mean(temporal_scores),
            'std_temporal_variance': np.std(temporal_scores)
        }


class AttentionGuidedFeatureSelector:
    """
    Use attention weights to guide feature selection.
    Identifies most important input features based on attention patterns.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def select_features(self,
                        input_tensor: torch.Tensor,
                        n_features: int,
                        method: str = 'cumulative',
                        target_class: Optional[int] = None) -> List[int]:
        """
        Select important features based on attention.
        
        Args:
            input_tensor: Input data
            n_features: Number of features to select
            method: Selection method ('cumulative', 'gradient', 'mutual_info')
            target_class: Target class for feature selection
            
        Returns:
            List of selected feature indices
        """
        if method == 'cumulative':
            return self._cumulative_attention_selection(
                input_tensor, n_features, target_class
            )
        elif method == 'gradient':
            return self._gradient_attention_selection(
                input_tensor, n_features, target_class
            )
        elif method == 'mutual_info':
            return self._mutual_info_selection(
                input_tensor, n_features, target_class
            )
        else:
            raise ValueError(f"Unknown selection method: {method}")
            
    def _cumulative_attention_selection(self, input_tensor, n_features, target_class):
        """Select features based on cumulative attention weights."""
        # Get attention weights
        attention_weights = self._get_all_attention_weights(input_tensor)
        
        # Compute cumulative attention across layers
        cumulative_attention = None
        for attn in attention_weights:
            # Average across heads and batch
            attn_avg = attn.mean(dim=(0, 1))
            
            # Sum attention to each position
            position_importance = attn_avg.sum(dim=0)
            
            if cumulative_attention is None:
                cumulative_attention = position_importance
            else:
                cumulative_attention += position_importance
                
        # Select top-k features
        top_indices = torch.topk(cumulative_attention, n_features).indices
        return top_indices.cpu().tolist()
    
    def _gradient_attention_selection(self, input_tensor, n_features, target_class):
        """Select features based on attention gradients."""
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Get attention weights with gradients
        attention_weights = self._get_attention_with_gradients(
            input_tensor, target_class
        )
        
        # Compute feature importance based on attention gradients
        feature_importance = torch.zeros(input_tensor.shape[-1]).to(self.device)
        
        for attn, attn_grad in attention_weights:
            # Importance is attention * gradient magnitude
            importance = (attn * attn_grad.abs()).sum(dim=(0, 1, 2))
            feature_importance += importance
            
        # Select top features
        top_indices = torch.topk(feature_importance, n_features).indices
        return top_indices.cpu().tolist()
    
    def _mutual_info_selection(self, input_tensor, n_features, target_class):
        """Select features based on mutual information with attention."""
        # Get attention weights
        attention_weights = self._get_all_attention_weights(input_tensor)
        
        # Compute mutual information between input features and attention
        input_np = input_tensor.cpu().numpy().squeeze()
        feature_scores = []
        
        for feature_idx in range(input_tensor.shape[-1]):
            feature_values = input_np[:, feature_idx] if len(input_np.shape) > 1 else input_np[feature_idx]
            
            # Compute MI with each attention layer
            mi_scores = []
            for attn in attention_weights:
                attn_flat = attn.mean(dim=(0, 1)).sum(dim=0).cpu().numpy()
                
                # Discretize for MI computation
                feature_discrete = np.digitize(feature_values, bins=10)
                attn_discrete = np.digitize(attn_flat, bins=10)
                
                mi = mutual_info_score(feature_discrete, attn_discrete)
                mi_scores.append(mi)
                
            feature_scores.append(np.mean(mi_scores))
            
        # Select top features
        top_indices = np.argsort(feature_scores)[-n_features:].tolist()
        return top_indices
    
    def _get_all_attention_weights(self, input_tensor):
        """Get all attention weights from model."""
        attention_weights = []
        
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attention_weights.append(output[1].detach())
                
        handles = []
        for module in self.model.modules():
            if 'attention' in str(type(module)).lower():
                handle = module.register_forward_hook(hook)
                handles.append(handle)
                
        with torch.no_grad():
            _ = self.model(input_tensor)
            
        for handle in handles:
            handle.remove()
            
        return attention_weights
    
    def _get_attention_with_gradients(self, input_tensor, target_class):
        """Get attention weights and their gradients."""
        attention_data = []
        
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attn = output[1]
                attn.retain_grad()
                attention_data.append([attn, None])
                
        handles = []
        for module in self.model.modules():
            if 'attention' in str(type(module)).lower():
                handle = module.register_forward_hook(hook)
                handles.append(handle)
                
        # Forward and backward pass
        output = self.model(input_tensor)
        loss = output[:, target_class].sum()
        loss.backward()
        
        # Get gradients
        for i, (attn, _) in enumerate(attention_data):
            attention_data[i][1] = attn.grad
            
        for handle in handles:
            handle.remove()
            
        return attention_data 