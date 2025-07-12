"""Weight distribution visualization for training analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class WeightDistributionVisualizer:
    """Visualizes weight distributions during model training."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize weight distribution visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir or Path("data/figures/weights")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_weight_histograms(self, 
                             model: nn.Module,
                             layer_types: Optional[List[str]] = None,
                             title: str = "Weight Distributions",
                             save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot histograms of weight distributions for different layers.
        
        Args:
            model: PyTorch model
            layer_types: Types of layers to include (e.g., ['Linear', 'Conv2d'])
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Collect weights by layer type
            weight_data = {}
            
            for name, param in model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    layer_type = name.split('.')[0]
                    
                    # Filter by layer types if specified
                    if layer_types is None or any(lt in name for lt in layer_types):
                        if layer_type not in weight_data:
                            weight_data[layer_type] = []
                        weight_data[layer_type].extend(
                            param.detach().cpu().numpy().flatten()
                        )
            
            if not weight_data:
                logger.warning("No weight data found for visualization")
                return plt.figure()
            
            # Create subplots
            n_layers = len(weight_data)
            ncols = min(3, n_layers)
            nrows = (n_layers + ncols - 1) // ncols
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
            if n_layers == 1:
                axes = [axes]
            elif nrows == 1:
                axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
            else:
                axes = axes.flatten()
            
            colors = plt.cm.Set3(np.linspace(0, 1, n_layers))
            
            for i, (layer_name, weights) in enumerate(weight_data.items()):
                ax = axes[i]
                
                # Plot histogram
                ax.hist(weights, bins=50, alpha=0.7, color=colors[i], 
                       density=True, edgecolor='black', linewidth=0.5)
                
                # Add statistics
                mean_w = np.mean(weights)
                std_w = np.std(weights)
                ax.axvline(mean_w, color='red', linestyle='--', 
                          label=f'Mean: {mean_w:.3f}')
                ax.axvline(mean_w + std_w, color='orange', linestyle='--', 
                          label=f'+1σ: {mean_w + std_w:.3f}')
                ax.axvline(mean_w - std_w, color='orange', linestyle='--', 
                          label=f'-1σ: {mean_w - std_w:.3f}')
                
                ax.set_xlabel('Weight Value')
                ax.set_ylabel('Density')
                ax.set_title(f'{layer_name} Weights')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_layers, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting weight histograms: {e}")
            raise
    
    def plot_weight_evolution(self, 
                            weight_history: List[Dict[str, torch.Tensor]],
                            layer_name: str,
                            title: Optional[str] = None,
                            save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot evolution of weight statistics over training.
        
        Args:
            weight_history: List of weight dictionaries from different epochs
            layer_name: Name of layer to analyze
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            epochs = list(range(len(weight_history)))
            means = []
            stds = []
            mins = []
            maxs = []
            
            # Extract statistics
            for epoch_weights in weight_history:
                if layer_name in epoch_weights:
                    weights = epoch_weights[layer_name].detach().cpu().numpy().flatten()
                    means.append(np.mean(weights))
                    stds.append(np.std(weights))
                    mins.append(np.min(weights))
                    maxs.append(np.max(weights))
                else:
                    means.append(0)
                    stds.append(0)
                    mins.append(0)
                    maxs.append(0)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Mean and std
            ax1.plot(epochs, means, 'b-', label='Mean', linewidth=2)
            ax1.fill_between(epochs, 
                           np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           alpha=0.3, label='±1σ')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Weight Value')
            ax1.set_title('Mean ± Standard Deviation')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Min and max
            ax2.plot(epochs, mins, 'g-', label='Min', linewidth=2)
            ax2.plot(epochs, maxs, 'r-', label='Max', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Weight Value')
            ax2.set_title('Min and Max Values')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Standard deviation
            ax3.plot(epochs, stds, 'purple', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Standard Deviation')
            ax3.set_title('Weight Variance')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Range (max - min)
            ranges = np.array(maxs) - np.array(mins)
            ax4.plot(epochs, ranges, 'orange', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Range (Max - Min)')
            ax4.set_title('Weight Range')
            ax4.grid(True, alpha=0.3)
            
            if title is None:
                title = f'Weight Evolution: {layer_name}'
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting weight evolution: {e}")
            raise
    
    def plot_weight_matrix(self, 
                         weight_tensor: torch.Tensor,
                         title: str = "Weight Matrix",
                         save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot weight matrix as heatmap.
        
        Args:
            weight_tensor: 2D weight tensor
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            weights = weight_tensor.detach().cpu().numpy()
            
            # Handle different tensor shapes
            if len(weights.shape) > 2:
                # For conv layers, reshape or take a slice
                if len(weights.shape) == 4:  # Conv2d
                    weights = weights[0, 0]  # Take first filter, first channel
                else:
                    weights = weights.reshape(weights.shape[0], -1)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Heatmap
            im = ax1.imshow(weights, cmap='RdBu_r', aspect='auto')
            ax1.set_title('Weight Matrix Heatmap')
            ax1.set_xlabel('Input Dimension')
            ax1.set_ylabel('Output Dimension')
            plt.colorbar(im, ax=ax1)
            
            # Plot 2: Distribution
            ax2.hist(weights.flatten(), bins=50, alpha=0.7, 
                    color='skyblue', edgecolor='black', density=True)
            ax2.set_xlabel('Weight Value')
            ax2.set_ylabel('Density')
            ax2.set_title('Weight Distribution')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            mean_w = np.mean(weights)
            std_w = np.std(weights)
            ax2.axvline(mean_w, color='red', linestyle='--', 
                       label=f'Mean: {mean_w:.3f}')
            ax2.axvline(mean_w + std_w, color='orange', linestyle='--', 
                       label=f'+1σ: {mean_w + std_w:.3f}')
            ax2.axvline(mean_w - std_w, color='orange', linestyle='--', 
                       label=f'-1σ: {mean_w - std_w:.3f}')
            ax2.legend()
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting weight matrix: {e}")
            raise
    
    def analyze_weight_initialization(self, 
                                   model: nn.Module,
                                   save_name: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Analyze weight initialization across all layers.
        
        Args:
            model: PyTorch model
            save_name: Filename to save analysis plot
            
        Returns:
            Dictionary with weight statistics per layer
        """
        try:
            layer_stats = {}
            layer_names = []
            means = []
            stds = []
            
            for name, param in model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    weights = param.detach().cpu().numpy()
                    
                    stats = {
                        'mean': float(np.mean(weights)),
                        'std': float(np.std(weights)),
                        'min': float(np.min(weights)),
                        'max': float(np.max(weights)),
                        'shape': list(weights.shape),
                        'num_params': int(np.prod(weights.shape))
                    }
                    
                    layer_stats[name] = stats
                    layer_names.append(name.split('.')[-2] if '.' in name else name)
                    means.append(stats['mean'])
                    stds.append(stats['std'])
            
            # Create visualization
            if save_name:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                
                # Plot 1: Means by layer
                x_pos = np.arange(len(layer_names))
                ax1.bar(x_pos, means, alpha=0.7, color='skyblue')
                ax1.set_xlabel('Layer')
                ax1.set_ylabel('Mean Weight')
                ax1.set_title('Mean Weights by Layer')
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(layer_names, rotation=45, ha='right')
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Standard deviations by layer
                ax2.bar(x_pos, stds, alpha=0.7, color='lightcoral')
                ax2.set_xlabel('Layer')
                ax2.set_ylabel('Weight Std')
                ax2.set_title('Weight Standard Deviations by Layer')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(layer_names, rotation=45, ha='right')
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Mean vs Std scatter
                ax3.scatter(means, stds, alpha=0.7, s=100)
                ax3.set_xlabel('Mean Weight')
                ax3.set_ylabel('Weight Std')
                ax3.set_title('Mean vs Standard Deviation')
                ax3.grid(True, alpha=0.3)
                
                # Add layer labels
                for i, name in enumerate(layer_names):
                    ax3.annotate(name, (means[i], stds[i]), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.7)
                
                # Plot 4: Parameter count by layer
                param_counts = [layer_stats[name]['num_params'] 
                              for name in layer_stats.keys()]
                ax4.bar(x_pos, param_counts, alpha=0.7, color='lightgreen')
                ax4.set_xlabel('Layer')
                ax4.set_ylabel('Number of Parameters')
                ax4.set_title('Parameter Count by Layer')
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels(layer_names, rotation=45, ha='right')
                ax4.set_yscale('log')
                ax4.grid(True, alpha=0.3)
                
                plt.suptitle('Weight Initialization Analysis', 
                           fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            return layer_stats
            
        except Exception as e:
            logger.error(f"Error analyzing weight initialization: {e}")
            raise
    
    def detect_weight_issues(self, 
                           model: nn.Module,
                           dead_threshold: float = 1e-8,
                           saturated_threshold: float = 0.95) -> Dict[str, List[str]]:
        """
        Detect potential weight-related issues.
        
        Args:
            model: PyTorch model
            dead_threshold: Threshold for detecting dead weights
            saturated_threshold: Threshold for detecting saturated weights
            
        Returns:
            Dictionary with issue categories and affected layers
        """
        try:
            issues = {
                'dead_weights': [],
                'large_weights': [],
                'small_weights': [],
                'saturated_weights': []
            }
            
            for name, param in model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    weights = param.detach().cpu().numpy()
                    
                    # Check for dead weights (very small)
                    if np.mean(np.abs(weights)) < dead_threshold:
                        issues['dead_weights'].append(name)
                    
                    # Check for very large weights
                    if np.max(np.abs(weights)) > 10:
                        issues['large_weights'].append(name)
                    
                    # Check for very small weights
                    if np.max(np.abs(weights)) < 0.01:
                        issues['small_weights'].append(name)
                    
                    # Check for saturated weights (close to limits)
                    if (np.sum(np.abs(weights) > 0.9) / weights.size) > saturated_threshold:
                        issues['saturated_weights'].append(name)
            
            return issues
            
        except Exception as e:
            logger.error(f"Error detecting weight issues: {e}")
            raise 