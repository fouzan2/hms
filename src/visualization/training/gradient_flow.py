"""Gradient flow visualization for training analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class GradientFlowVisualizer:
    """Visualizes gradient flow during model training."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize gradient flow visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir or Path("data/figures/gradients")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_gradient_flow(self, 
                         model: nn.Module,
                         title: str = "Gradient Flow",
                         save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot gradient flow through model layers.
        
        Args:
            model: PyTorch model
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Extract gradients
            ave_grads = []
            max_grads = []
            layers = []
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    layers.append(name)
                    ave_grads.append(param.grad.abs().mean().cpu().item())
                    max_grads.append(param.grad.abs().max().cpu().item())
            
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Create bar plot
            x_pos = np.arange(len(layers))
            bars1 = ax.bar(x_pos - 0.2, ave_grads, 0.4, 
                          label='Average Gradient', alpha=0.7)
            bars2 = ax.bar(x_pos + 0.2, max_grads, 0.4, 
                          label='Max Gradient', alpha=0.7)
            
            ax.set_xlabel('Layers')
            ax.set_ylabel('Gradient Magnitude')
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(layers, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Use log scale if gradients vary widely
            if max(max_grads) / min([g for g in ave_grads if g > 0]) > 100:
                ax.set_yscale('log')
            
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting gradient flow: {e}")
            raise
    
    def plot_gradient_histogram(self, 
                              model: nn.Module,
                              layer_name: Optional[str] = None,
                              title: str = "Gradient Distribution",
                              save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot histogram of gradient values.
        
        Args:
            model: PyTorch model
            layer_name: Specific layer to plot (None for all layers)
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            all_grads = []
            colors = plt.cm.Set3(np.linspace(0, 1, 10))
            
            for i, (name, param) in enumerate(model.named_parameters()):
                if param.requires_grad and param.grad is not None:
                    if layer_name is None or layer_name in name:
                        grads = param.grad.detach().cpu().numpy().flatten()
                        all_grads.extend(grads)
                        
                        # Plot individual layer if specified
                        if layer_name and layer_name in name:
                            ax.hist(grads, bins=50, alpha=0.7, 
                                   color=colors[i % len(colors)], 
                                   label=name, density=True)
            
            # Plot all gradients if no specific layer
            if layer_name is None:
                ax.hist(all_grads, bins=100, alpha=0.7, 
                       color='skyblue', density=True)
                ax.axvline(np.mean(all_grads), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(all_grads):.2e}')
                ax.axvline(np.std(all_grads), color='orange', linestyle='--', 
                          label=f'Std: {np.std(all_grads):.2e}')
            
            ax.set_xlabel('Gradient Value')
            ax.set_ylabel('Density')
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting gradient histogram: {e}")
            raise
    
    def plot_gradient_norms(self, 
                          gradient_history: List[Dict[str, float]],
                          title: str = "Gradient Norms Over Time",
                          save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot gradient norms over training iterations.
        
        Args:
            gradient_history: List of gradient norm dictionaries
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Extract data
            iterations = list(range(len(gradient_history)))
            total_norms = [entry.get('total_norm', 0) for entry in gradient_history]
            layer_norms = {}
            
            # Collect layer-specific norms
            for entry in gradient_history:
                for key, value in entry.items():
                    if key != 'total_norm' and 'norm' in key:
                        if key not in layer_norms:
                            layer_norms[key] = []
                        layer_norms[key].append(value)
            
            # Plot total gradient norm
            ax1.plot(iterations, total_norms, 'b-', linewidth=2, 
                    label='Total Gradient Norm')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Gradient Norm')
            ax1.set_title('Total Gradient Norm', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot layer-specific norms
            colors = plt.cm.Set3(np.linspace(0, 1, len(layer_norms)))
            for i, (layer, norms) in enumerate(layer_norms.items()):
                if len(norms) == len(iterations):
                    ax2.plot(iterations, norms, color=colors[i], 
                            label=layer, alpha=0.7)
            
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Layer Gradient Norm')
            ax2.set_title('Layer-wise Gradient Norms', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting gradient norms: {e}")
            raise
    
    def detect_gradient_issues(self, 
                             model: nn.Module,
                             threshold_vanishing: float = 1e-7,
                             threshold_exploding: float = 1.0) -> Dict[str, List[str]]:
        """
        Detect gradient vanishing/exploding issues.
        
        Args:
            model: PyTorch model
            threshold_vanishing: Threshold for vanishing gradients
            threshold_exploding: Threshold for exploding gradients
            
        Returns:
            Dictionary with problematic layers
        """
        try:
            issues = {
                'vanishing': [],
                'exploding': [],
                'dead': []
            }
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_abs = param.grad.abs()
                    max_grad = grad_abs.max().item()
                    mean_grad = grad_abs.mean().item()
                    
                    # Check for vanishing gradients
                    if mean_grad < threshold_vanishing:
                        issues['vanishing'].append(name)
                    
                    # Check for exploding gradients
                    if max_grad > threshold_exploding:
                        issues['exploding'].append(name)
                    
                    # Check for dead neurons (no gradients)
                    if max_grad == 0:
                        issues['dead'].append(name)
            
            return issues
            
        except Exception as e:
            logger.error(f"Error detecting gradient issues: {e}")
            raise
    
    def visualize_gradient_issues(self, 
                                model: nn.Module,
                                save_name: Optional[str] = None) -> plt.Figure:
        """
        Visualize gradient issues in the model.
        
        Args:
            model: PyTorch model
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            issues = self.detect_gradient_issues(model)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Gradient magnitudes by layer
            layers = []
            mean_grads = []
            max_grads = []
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    layers.append(name.split('.')[-1])  # Just the layer name
                    mean_grads.append(param.grad.abs().mean().cpu().item())
                    max_grads.append(param.grad.abs().max().cpu().item())
            
            x_pos = np.arange(len(layers))
            ax1.bar(x_pos, mean_grads, alpha=0.7, color='skyblue')
            ax1.set_xlabel('Layer')
            ax1.set_ylabel('Mean Gradient Magnitude')
            ax1.set_title('Mean Gradients by Layer')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(layers, rotation=45)
            ax1.set_yscale('log')
            
            # Plot 2: Max gradients
            ax2.bar(x_pos, max_grads, alpha=0.7, color='lightcoral')
            ax2.set_xlabel('Layer')
            ax2.set_ylabel('Max Gradient Magnitude')
            ax2.set_title('Max Gradients by Layer')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(layers, rotation=45)
            ax2.set_yscale('log')
            
            # Plot 3: Issues summary
            issue_types = ['Vanishing', 'Exploding', 'Dead']
            issue_counts = [len(issues['vanishing']), 
                           len(issues['exploding']), 
                           len(issues['dead'])]
            
            colors = ['orange', 'red', 'gray']
            bars = ax3.bar(issue_types, issue_counts, color=colors, alpha=0.7)
            ax3.set_ylabel('Number of Layers')
            ax3.set_title('Gradient Issues Summary')
            
            # Add value labels on bars
            for bar, count in zip(bars, issue_counts):
                if count > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(count), ha='center', va='bottom', fontweight='bold')
            
            # Plot 4: Gradient distribution
            all_grads = []
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    all_grads.extend(param.grad.detach().cpu().numpy().flatten())
            
            ax4.hist(all_grads, bins=50, alpha=0.7, color='green', density=True)
            ax4.set_xlabel('Gradient Value')
            ax4.set_ylabel('Density')
            ax4.set_title('Overall Gradient Distribution')
            ax4.axvline(0, color='red', linestyle='--', alpha=0.7)
            
            plt.suptitle('Gradient Analysis Report', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error visualizing gradient issues: {e}")
            raise 