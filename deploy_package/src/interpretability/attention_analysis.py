"""
Attention Analysis for EEG Harmful Brain Activity Classification.
Provides comprehensive analysis of attention mechanisms in neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
import warnings
from pathlib import Path
import json
from tqdm import tqdm
import cv2
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score


@dataclass
class AttentionWeights:
    """Container for attention weights and metadata."""
    weights: np.ndarray
    layer_name: str
    head_id: Optional[int] = None
    input_tokens: Optional[List[str]] = None
    spatial_dims: Optional[Tuple[int, int]] = None


@dataclass
class AttentionAnalysisResult:
    """Container for attention analysis results."""
    attention_weights: List[AttentionWeights]
    attention_maps: Optional[np.ndarray] = None
    rollout_attention: Optional[np.ndarray] = None
    gradient_attention: Optional[np.ndarray] = None
    entropy_scores: Optional[np.ndarray] = None
    consistency_scores: Optional[np.ndarray] = None


class AttentionVisualizer:
    """Visualize and analyze attention mechanisms in neural networks."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize attention visualizer.
        
        Args:
            model: Neural network model with attention mechanisms
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Find attention layers
        self.attention_layers = self._find_attention_layers()
        
        # Hook storage
        self.attention_weights = {}
        self.hooks = []
        
    def _find_attention_layers(self) -> List[str]:
        """Find attention layers in the model."""
        attention_layers = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.MultiheadAttention, nn.TransformerEncoderLayer)):
                attention_layers.append(name)
            # Check for custom attention layers
            elif hasattr(module, 'attention') or 'attention' in name.lower():
                attention_layers.append(name)
        
        return attention_layers
        
    def register_hooks(self):
        """Register forward hooks to capture attention weights."""
        def get_attention_hook(layer_name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    # MultiheadAttention returns (output, attention_weights)
                    attention_weights = output[1]
                    if attention_weights is not None:
                        self.attention_weights[layer_name] = attention_weights.detach().cpu().numpy()
                elif hasattr(module, 'attention_weights'):
                    # Custom attention modules
                    weights = module.attention_weights
                    if weights is not None:
                        self.attention_weights[layer_name] = weights.detach().cpu().numpy()
            return hook
        
        # Register hooks for each attention layer
        for layer_name in self.attention_layers:
            layer = dict(self.model.named_modules())[layer_name]
            hook = layer.register_forward_hook(get_attention_hook(layer_name))
            self.hooks.append(hook)
            
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def analyze_attention(self, x: torch.Tensor, 
                         target_class: Optional[int] = None) -> AttentionAnalysisResult:
        """
        Comprehensive attention analysis.
        
        Args:
            x: Input tensor
            target_class: Target class for gradient-based analysis
            
        Returns:
            AttentionAnalysisResult containing various attention analyses
        """
        x = x.to(self.device)
        self.model.eval()
        
        # Clear previous attention weights
        self.attention_weights.clear()
        
        # Register hooks
        self.register_hooks()
        
        try:
            # Forward pass to capture attention weights
            if target_class is not None:
                x.requires_grad_(True)
                
            with torch.set_grad_enabled(target_class is not None):
                output = self.model(x)
                if isinstance(output, dict):
                    logits = output.get('logits', output.get('output', output))
                else:
                    logits = output
                
                # Gradient-based attention if target class specified
                gradient_attention = None
                if target_class is not None:
                    class_score = logits[0, target_class]
                    self.model.zero_grad()
                    class_score.backward(retain_graph=True)
                    gradient_attention = x.grad.detach().cpu().numpy()
            
            # Process captured attention weights
            attention_weights_list = []
            for layer_name, weights in self.attention_weights.items():
                attention_weights_list.append(AttentionWeights(
                    weights=weights,
                    layer_name=layer_name
                ))
            
            # Calculate attention rollout
            rollout_attention = self._calculate_attention_rollout(attention_weights_list)
            
            # Calculate attention entropy
            entropy_scores = self._calculate_attention_entropy(attention_weights_list)
            
            # Calculate attention consistency
            consistency_scores = self._calculate_attention_consistency(attention_weights_list)
            
            return AttentionAnalysisResult(
                attention_weights=attention_weights_list,
                rollout_attention=rollout_attention,
                gradient_attention=gradient_attention,
                entropy_scores=entropy_scores,
                consistency_scores=consistency_scores
            )
            
        finally:
            # Always remove hooks
            self.remove_hooks()
            
    def _calculate_attention_rollout(self, attention_weights: List[AttentionWeights]) -> Optional[np.ndarray]:
        """Calculate attention rollout across layers."""
        if not attention_weights:
            return None
            
        # Start with identity matrix
        rollout = None
        
        for attention_weight in attention_weights:
            weights = attention_weight.weights
            
            # Average across heads if multi-head
            if len(weights.shape) == 4:  # [batch, heads, seq, seq]
                weights = np.mean(weights, axis=1)
            elif len(weights.shape) == 3:  # [heads, seq, seq]
                weights = np.mean(weights, axis=0)
                weights = weights[np.newaxis, :]  # Add batch dimension
            
            # Ensure square matrix for rollout
            if weights.shape[-2] == weights.shape[-1]:
                if rollout is None:
                    rollout = weights
                else:
                    # Matrix multiplication for rollout
                    rollout = np.matmul(rollout, weights)
        
        return rollout
        
    def _calculate_attention_entropy(self, attention_weights: List[AttentionWeights]) -> List[float]:
        """Calculate entropy of attention distributions."""
        entropy_scores = []
        
        for attention_weight in attention_weights:
            weights = attention_weight.weights
            
            # Flatten to 2D if needed
            if len(weights.shape) > 2:
                weights = weights.reshape(-1, weights.shape[-1])
            
            # Calculate entropy for each attention distribution
            entropies = []
            for i in range(weights.shape[0]):
                dist = weights[i]
                dist = dist + 1e-8  # Avoid log(0)
                entropy = -np.sum(dist * np.log(dist))
                entropies.append(entropy)
            
            entropy_scores.append(np.mean(entropies))
            
        return entropy_scores
        
    def _calculate_attention_consistency(self, attention_weights: List[AttentionWeights]) -> List[float]:
        """Calculate consistency of attention patterns across heads/layers."""
        consistency_scores = []
        
        for attention_weight in attention_weights:
            weights = attention_weight.weights
            
            if len(weights.shape) == 4:  # Multi-head attention
                # Calculate pairwise correlation between heads
                batch_size, num_heads, seq_len1, seq_len2 = weights.shape
                correlations = []
                
                for b in range(batch_size):
                    head_correlations = []
                    for i in range(num_heads):
                        for j in range(i + 1, num_heads):
                            head1 = weights[b, i].flatten()
                            head2 = weights[b, j].flatten()
                            corr = np.corrcoef(head1, head2)[0, 1]
                            if not np.isnan(corr):
                                head_correlations.append(corr)
                    
                    if head_correlations:
                        correlations.append(np.mean(head_correlations))
                
                consistency_scores.append(np.mean(correlations) if correlations else 0.0)
            else:
                # Single attention pattern, no consistency metric
                consistency_scores.append(1.0)
                
        return consistency_scores
        
    def visualize_attention_weights(self, attention_result: AttentionAnalysisResult,
                                  save_dir: Optional[str] = None) -> None:
        """
        Visualize attention weights.
        
        Args:
            attention_result: Attention analysis results
            save_dir: Directory to save visualizations
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, attention_weights in enumerate(attention_result.attention_weights):
            weights = attention_weights.weights
            layer_name = attention_weights.layer_name
            
            # Handle different attention weight shapes
            if len(weights.shape) == 4:  # [batch, heads, seq, seq]
                batch_size, num_heads, seq_len1, seq_len2 = weights.shape
                
                # Create subplot for each head
                fig, axes = plt.subplots(1, min(num_heads, 4), figsize=(4*min(num_heads, 4), 4))
                if num_heads == 1:
                    axes = [axes]
                
                for h in range(min(num_heads, 4)):
                    ax = axes[h] if num_heads > 1 else axes[0]
                    im = ax.imshow(weights[0, h], cmap='Blues', aspect='auto')
                    ax.set_title(f'Head {h+1}')
                    ax.set_xlabel('Key Position')
                    ax.set_ylabel('Query Position')
                    plt.colorbar(im, ax=ax)
                
                plt.suptitle(f'Attention Weights - {layer_name}')
                plt.tight_layout()
                
            elif len(weights.shape) == 3:  # [heads, seq, seq]
                num_heads, seq_len1, seq_len2 = weights.shape
                
                fig, axes = plt.subplots(1, min(num_heads, 4), figsize=(4*min(num_heads, 4), 4))
                if num_heads == 1:
                    axes = [axes]
                
                for h in range(min(num_heads, 4)):
                    ax = axes[h] if num_heads > 1 else axes[0]
                    im = ax.imshow(weights[h], cmap='Blues', aspect='auto')
                    ax.set_title(f'Head {h+1}')
                    ax.set_xlabel('Key Position')
                    ax.set_ylabel('Query Position')
                    plt.colorbar(im, ax=ax)
                
                plt.suptitle(f'Attention Weights - {layer_name}')
                plt.tight_layout()
                
            elif len(weights.shape) == 2:  # [seq, seq]
                plt.figure(figsize=(8, 6))
                plt.imshow(weights, cmap='Blues', aspect='auto')
                plt.title(f'Attention Weights - {layer_name}')
                plt.xlabel('Key Position')
                plt.ylabel('Query Position')
                plt.colorbar()
                
            if save_dir:
                plt.savefig(save_dir / f'attention_weights_{layer_name}_{i}.png', 
                           dpi=300, bbox_inches='tight')
            else:
                plt.show()
                
            plt.close()
            
    def visualize_attention_statistics(self, attention_result: AttentionAnalysisResult,
                                     save_dir: Optional[str] = None) -> None:
        """
        Visualize attention statistics (entropy, consistency).
        
        Args:
            attention_result: Attention analysis results
            save_dir: Directory to save visualizations
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Entropy scores
        if attention_result.entropy_scores:
            layer_names = [aw.layer_name for aw in attention_result.attention_weights]
            axes[0].bar(range(len(attention_result.entropy_scores)), attention_result.entropy_scores)
            axes[0].set_title('Attention Entropy by Layer')
            axes[0].set_xlabel('Layer')
            axes[0].set_ylabel('Entropy')
            axes[0].set_xticks(range(len(layer_names)))
            axes[0].set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45)
        
        # Consistency scores
        if attention_result.consistency_scores:
            layer_names = [aw.layer_name for aw in attention_result.attention_weights]
            axes[1].bar(range(len(attention_result.consistency_scores)), attention_result.consistency_scores)
            axes[1].set_title('Attention Consistency by Layer')
            axes[1].set_xlabel('Layer')
            axes[1].set_ylabel('Consistency')
            axes[1].set_xticks(range(len(layer_names)))
            axes[1].set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / 'attention_statistics.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()
        
    def export_attention_data(self, attention_result: AttentionAnalysisResult,
                            save_path: str) -> None:
        """
        Export attention analysis data to JSON.
        
        Args:
            attention_result: Attention analysis results
            save_path: Path to save the JSON file
        """
        # Prepare data for JSON export
        export_data = {
            'attention_weights': [],
            'entropy_scores': attention_result.entropy_scores,
            'consistency_scores': attention_result.consistency_scores
        }
        
        # Convert attention weights to serializable format
        for aw in attention_result.attention_weights:
            weight_data = {
                'layer_name': aw.layer_name,
                'shape': list(aw.weights.shape),
                'mean': float(np.mean(aw.weights)),
                'std': float(np.std(aw.weights)),
                'min': float(np.min(aw.weights)),
                'max': float(np.max(aw.weights))
            }
            
            if aw.head_id is not None:
                weight_data['head_id'] = aw.head_id
                
            export_data['attention_weights'].append(weight_data)
        
        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(export_data, f, indent=2)


class AttentionGuidedFeatureSelector:
    """Use attention patterns to guide feature selection."""
    
    def __init__(self, model: nn.Module, attention_visualizer: AttentionVisualizer):
        """
        Initialize attention-guided feature selector.
        
        Args:
            model: Neural network model
            attention_visualizer: Attention visualizer instance
        """
        self.model = model
        self.attention_visualizer = attention_visualizer
        
    def select_important_features(self, x: torch.Tensor, 
                                top_k: int = 10) -> Tuple[List[int], np.ndarray]:
        """
        Select most important features based on attention patterns.
        
        Args:
            x: Input tensor
            top_k: Number of top features to select
            
        Returns:
            Tuple of (feature_indices, importance_scores)
        """
        # Get attention analysis
        attention_result = self.attention_visualizer.analyze_attention(x)
        
        # Aggregate attention weights across layers
        total_attention = np.zeros(x.shape[-1])  # Assume last dimension is features
        
        for attention_weights in attention_result.attention_weights:
            weights = attention_weights.weights
            
            # Average across batch and heads if needed
            if len(weights.shape) > 2:
                weights = np.mean(weights, axis=tuple(range(len(weights.shape) - 2)))
            
            # Sum attention for each position/feature
            if weights.shape[0] == weights.shape[1]:  # Square attention matrix
                feature_attention = np.sum(weights, axis=0)
                if len(feature_attention) == len(total_attention):
                    total_attention += feature_attention
        
        # Get top-k features
        top_indices = np.argsort(total_attention)[-top_k:][::-1]
        importance_scores = total_attention[top_indices]
        
        return top_indices.tolist(), importance_scores


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