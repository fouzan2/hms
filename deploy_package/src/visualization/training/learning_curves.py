"""
Learning Curve Visualizer for HMS EEG Classification System

This module provides visualization and analysis of learning curves including:
- Train/validation loss and accuracy curves
- Overfitting detection
- Early stopping point visualization
- Learning rate impact analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error
from scipy import signal
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class LearningCurveVisualizer:
    """Visualize and analyze learning curves for model training."""
    
    def __init__(self, log_dir: str = "logs/training"):
        """
        Initialize learning curve visualizer.
        
        Args:
            log_dir: Directory containing training logs
        """
        self.log_dir = Path(log_dir)
        self.metrics = None
        self.analysis_results = {}
        
        # Set visualization style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def load_metrics(self, metrics_file: Optional[str] = None) -> Dict:
        """
        Load training metrics from file.
        
        Args:
            metrics_file: Path to metrics JSON file
            
        Returns:
            Dictionary of training metrics
        """
        if metrics_file is None:
            metrics_file = self.log_dir / "training_metrics.json"
        
        with open(metrics_file, 'r') as f:
            self.metrics = json.load(f)
        
        # Convert to numpy arrays for easier processing
        for key in self.metrics:
            if isinstance(self.metrics[key], list):
                self.metrics[key] = np.array(self.metrics[key])
        
        logger.info(f"Loaded metrics from {metrics_file}")
        return self.metrics
    
    def detect_overfitting(self, 
                          patience: int = 20,
                          min_delta: float = 0.001) -> Dict:
        """
        Detect overfitting in training curves.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to consider as improvement
            
        Returns:
            Dictionary with overfitting analysis results
        """
        if self.metrics is None:
            raise ValueError("No metrics loaded. Call load_metrics() first.")
        
        val_loss = self.metrics['val_loss']
        train_loss = self.metrics['train_loss']
        
        # Remove NaN values
        valid_idx = ~np.isnan(val_loss)
        val_loss = val_loss[valid_idx]
        train_loss = train_loss[valid_idx]
        iterations = self.metrics['iteration'][valid_idx]
        
        # Find best validation loss
        best_val_idx = np.argmin(val_loss)
        best_val_loss = val_loss[best_val_idx]
        best_iteration = iterations[best_val_idx]
        
        # Check for overfitting after best point
        overfit_start = None
        for i in range(best_val_idx + 1, len(val_loss)):
            if val_loss[i] > best_val_loss + min_delta:
                # Check if this trend continues
                if i + patience < len(val_loss):
                    future_losses = val_loss[i:i+patience]
                    if np.all(future_losses > best_val_loss):
                        overfit_start = iterations[i]
                        break
        
        # Calculate overfitting metrics
        if overfit_start is not None:
            overfit_idx = np.where(iterations == overfit_start)[0][0]
            overfit_severity = (val_loss[-1] - best_val_loss) / best_val_loss
            train_val_gap = np.mean(val_loss[overfit_idx:] - train_loss[overfit_idx:])
        else:
            overfit_severity = 0
            train_val_gap = np.mean(val_loss - train_loss)
        
        self.analysis_results['overfitting'] = {
            'detected': overfit_start is not None,
            'start_iteration': overfit_start,
            'best_iteration': int(best_iteration),
            'best_val_loss': float(best_val_loss),
            'severity': float(overfit_severity),
            'train_val_gap': float(train_val_gap)
        }
        
        return self.analysis_results['overfitting']
    
    def analyze_convergence(self, window_size: int = 10) -> Dict:
        """
        Analyze convergence of training curves.
        
        Args:
            window_size: Window size for smoothing
            
        Returns:
            Dictionary with convergence analysis
        """
        if self.metrics is None:
            raise ValueError("No metrics loaded. Call load_metrics() first.")
        
        train_loss = self.metrics['train_loss']
        
        # Smooth the curve
        smoothed_loss = signal.savgol_filter(train_loss, window_size * 2 + 1, 3)
        
        # Calculate gradient
        gradient = np.gradient(smoothed_loss)
        
        # Find convergence point (where gradient is close to 0)
        convergence_threshold = np.std(gradient) * 0.1
        converged_idx = np.where(np.abs(gradient) < convergence_threshold)[0]
        
        if len(converged_idx) > 0:
            convergence_iteration = self.metrics['iteration'][converged_idx[0]]
            convergence_loss = float(train_loss[converged_idx[0]])
            is_converged = True
        else:
            convergence_iteration = None
            convergence_loss = None
            is_converged = False
        
        # Calculate convergence rate
        if len(train_loss) > 10:
            early_loss = np.mean(train_loss[:10])
            late_loss = np.mean(train_loss[-10:])
            convergence_rate = (early_loss - late_loss) / len(train_loss)
        else:
            convergence_rate = 0
        
        self.analysis_results['convergence'] = {
            'converged': is_converged,
            'convergence_iteration': convergence_iteration,
            'convergence_loss': convergence_loss,
            'convergence_rate': float(convergence_rate),
            'final_gradient': float(gradient[-1])
        }
        
        return self.analysis_results['convergence']
    
    def plot_learning_curves(self, 
                           save_path: Optional[str] = None,
                           show_analysis: bool = True) -> plt.Figure:
        """
        Create comprehensive learning curve visualization.
        
        Args:
            save_path: Path to save the figure
            show_analysis: Whether to show analysis annotations
            
        Returns:
            Matplotlib figure
        """
        if self.metrics is None:
            raise ValueError("No metrics loaded. Call load_metrics() first.")
        
        # Ensure all metrics are numpy arrays for consistent indexing
        for key, value in self.metrics.items():
            if isinstance(value, list):
                self.metrics[key] = np.array(value)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Learning Curve Analysis', fontsize=16)
        
        # Loss curves
        ax = axes[0, 0]
        ax.plot(self.metrics['iteration'], self.metrics['train_loss'], 
                label='Train Loss', alpha=0.8, linewidth=2)
        
        val_loss = self.metrics['val_loss']
        valid_idx = ~np.isnan(val_loss)
        ax.plot(self.metrics['iteration'][valid_idx], val_loss[valid_idx], 
                label='Validation Loss', alpha=0.8, linewidth=2)
        
        # Add overfitting analysis
        if show_analysis and 'overfitting' in self.analysis_results:
            overfit_info = self.analysis_results['overfitting']
            if overfit_info['detected']:
                ax.axvline(x=overfit_info['start_iteration'], 
                          color='red', linestyle='--', alpha=0.7,
                          label='Overfitting Start')
            ax.axvline(x=overfit_info['best_iteration'], 
                      color='green', linestyle='--', alpha=0.7,
                      label='Best Model')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax = axes[0, 1]
        train_acc = self.metrics['train_acc']
        val_acc = self.metrics['val_acc']
        
        if not np.all(np.isnan(train_acc)):
            ax.plot(self.metrics['iteration'], train_acc, 
                    label='Train Accuracy', alpha=0.8, linewidth=2)
        
        if not np.all(np.isnan(val_acc)):
            valid_idx = ~np.isnan(val_acc)
            ax.plot(self.metrics['iteration'][valid_idx], val_acc[valid_idx], 
                    label='Validation Accuracy', alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate schedule
        ax = axes[1, 0]
        lr = self.metrics['learning_rate']
        if not np.all(np.isnan(lr)):
            ax.plot(self.metrics['iteration'], lr, 
                    color='green', alpha=0.8, linewidth=2)
            ax.set_yscale('log')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        
        # Training statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create statistics text
        stats_text = "Training Statistics\n" + "="*30 + "\n"
        
        # Add basic stats
        stats_text += f"Total Iterations: {len(self.metrics['iteration'])}\n"
        stats_text += f"Total Epochs: {self.metrics['epoch'][-1]}\n"
        stats_text += f"Final Train Loss: {self.metrics['train_loss'][-1]:.4f}\n"
        
        if not np.all(np.isnan(val_loss)):
            final_val_loss = val_loss[~np.isnan(val_loss)][-1]
            stats_text += f"Final Val Loss: {final_val_loss:.4f}\n"
        
        # Add analysis results
        if 'overfitting' in self.analysis_results:
            stats_text += "\nOverfitting Analysis:\n"
            overfit_info = self.analysis_results['overfitting']
            stats_text += f"  Detected: {'Yes' if overfit_info['detected'] else 'No'}\n"
            stats_text += f"  Best Iteration: {overfit_info['best_iteration']}\n"
            stats_text += f"  Best Val Loss: {overfit_info['best_val_loss']:.4f}\n"
            if overfit_info['detected']:
                stats_text += f"  Severity: {overfit_info['severity']*100:.1f}%\n"
        
        if 'convergence' in self.analysis_results:
            stats_text += "\nConvergence Analysis:\n"
            conv_info = self.analysis_results['convergence']
            stats_text += f"  Converged: {'Yes' if conv_info['converged'] else 'No'}\n"
            if conv_info['converged']:
                stats_text += f"  At Iteration: {conv_info['convergence_iteration']}\n"
            stats_text += f"  Convergence Rate: {conv_info['convergence_rate']:.6f}\n"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved learning curve plot to {save_path}")
        
        return fig
    
    def create_interactive_plot(self) -> go.Figure:
        """
        Create interactive Plotly visualization of learning curves.
        
        Returns:
            Plotly figure object
        """
        if self.metrics is None:
            raise ValueError("No metrics loaded. Call load_metrics() first.")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss Curves', 'Accuracy Curves', 
                          'Learning Rate', 'Train vs Validation Gap'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Loss curves
        fig.add_trace(
            go.Scatter(x=self.metrics['iteration'], 
                      y=self.metrics['train_loss'],
                      name='Train Loss',
                      mode='lines',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        val_loss = self.metrics['val_loss']
        valid_idx = ~np.isnan(val_loss)
        fig.add_trace(
            go.Scatter(x=self.metrics['iteration'][valid_idx], 
                      y=val_loss[valid_idx],
                      name='Val Loss',
                      mode='lines',
                      line=dict(color='red', width=2)),
            row=1, col=1
        )
        
        # Add overfitting markers
        if 'overfitting' in self.analysis_results:
            overfit_info = self.analysis_results['overfitting']
            
            # Best model point
            fig.add_trace(
                go.Scatter(x=[overfit_info['best_iteration']], 
                          y=[overfit_info['best_val_loss']],
                          name='Best Model',
                          mode='markers',
                          marker=dict(color='green', size=10, symbol='star')),
                row=1, col=1
            )
            
            # Overfitting start
            if overfit_info['detected']:
                fig.add_vline(x=overfit_info['start_iteration'], 
                            line_dash="dash", line_color="red",
                            annotation_text="Overfitting Start",
                            row=1, col=1)
        
        # Accuracy curves
        train_acc = self.metrics['train_acc']
        val_acc = self.metrics['val_acc']
        
        if not np.all(np.isnan(train_acc)):
            fig.add_trace(
                go.Scatter(x=self.metrics['iteration'], 
                          y=train_acc,
                          name='Train Accuracy',
                          mode='lines',
                          line=dict(color='blue', width=2)),
                row=1, col=2
            )
        
        if not np.all(np.isnan(val_acc)):
            valid_idx = ~np.isnan(val_acc)
            fig.add_trace(
                go.Scatter(x=self.metrics['iteration'][valid_idx], 
                          y=val_acc[valid_idx],
                          name='Val Accuracy',
                          mode='lines',
                          line=dict(color='red', width=2)),
                row=1, col=2
            )
        
        # Learning rate
        lr = self.metrics['learning_rate']
        if not np.all(np.isnan(lr)):
            fig.add_trace(
                go.Scatter(x=self.metrics['iteration'], 
                          y=lr,
                          name='Learning Rate',
                          mode='lines',
                          line=dict(color='green', width=2)),
                row=2, col=1
            )
        
        # Train-validation gap
        train_loss = self.metrics['train_loss']
        val_loss = self.metrics['val_loss']
        valid_idx = ~np.isnan(val_loss)
        
        gap = val_loss[valid_idx] - train_loss[valid_idx]
        fig.add_trace(
            go.Scatter(x=self.metrics['iteration'][valid_idx], 
                      y=gap,
                      name='Loss Gap',
                      mode='lines',
                      fill='tozeroy',
                      line=dict(color='purple', width=2)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Learning Curve Analysis Dashboard',
            showlegend=True,
            height=800,
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Iteration", row=1, col=1)
        fig.update_xaxes(title_text="Iteration", row=1, col=2)
        fig.update_xaxes(title_text="Iteration", row=2, col=1)
        fig.update_xaxes(title_text="Iteration", row=2, col=2)
        
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
        fig.update_yaxes(title_text="Learning Rate", type="log", row=2, col=1)
        fig.update_yaxes(title_text="Gap", row=2, col=2)
        
        return fig
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive learning curve analysis report.
        
        Args:
            output_path: Path to save report
            
        Returns:
            Report text
        """
        if self.metrics is None:
            raise ValueError("No metrics loaded. Call load_metrics() first.")
        
        # Perform all analyses
        self.detect_overfitting()
        self.analyze_convergence()
        
        report = "HMS EEG Classification - Learning Curve Analysis Report\n"
        report += "="*60 + "\n\n"
        
        # Training overview
        report += "1. Training Overview\n"
        report += "-"*20 + "\n"
        report += f"Total Iterations: {len(self.metrics['iteration'])}\n"
        report += f"Total Epochs: {self.metrics['epoch'][-1]}\n"
        report += f"Final Train Loss: {self.metrics['train_loss'][-1]:.4f}\n"
        
        val_loss = self.metrics['val_loss']
        if not np.all(np.isnan(val_loss)):
            final_val_loss = val_loss[~np.isnan(val_loss)][-1]
            report += f"Final Validation Loss: {final_val_loss:.4f}\n"
        
        report += "\n"
        
        # Overfitting analysis
        report += "2. Overfitting Analysis\n"
        report += "-"*20 + "\n"
        overfit_info = self.analysis_results['overfitting']
        
        if overfit_info['detected']:
            report += "⚠️  Overfitting DETECTED\n"
            report += f"Started at iteration: {overfit_info['start_iteration']}\n"
            report += f"Severity: {overfit_info['severity']*100:.1f}% increase from best\n"
            report += f"Recommendation: Use model from iteration {overfit_info['best_iteration']}\n"
        else:
            report += "✅ No significant overfitting detected\n"
        
        report += f"Best validation loss: {overfit_info['best_val_loss']:.4f} "
        report += f"at iteration {overfit_info['best_iteration']}\n"
        report += f"Average train-val gap: {overfit_info['train_val_gap']:.4f}\n"
        report += "\n"
        
        # Convergence analysis
        report += "3. Convergence Analysis\n"
        report += "-"*20 + "\n"
        conv_info = self.analysis_results['convergence']
        
        if conv_info['converged']:
            report += "✅ Training has converged\n"
            report += f"Convergence at iteration: {conv_info['convergence_iteration']}\n"
            report += f"Convergence loss: {conv_info['convergence_loss']:.4f}\n"
        else:
            report += "⚠️  Training has NOT fully converged\n"
            report += "Consider training for more epochs\n"
        
        report += f"Convergence rate: {conv_info['convergence_rate']:.6f}\n"
        report += f"Final gradient: {conv_info['final_gradient']:.6f}\n"
        report += "\n"
        
        # Recommendations
        report += "4. Recommendations\n"
        report += "-"*20 + "\n"
        
        if overfit_info['detected']:
            report += "- Use early stopping or regularization to prevent overfitting\n"
            report += f"- Best model checkpoint is at iteration {overfit_info['best_iteration']}\n"
        
        if not conv_info['converged']:
            report += "- Consider training for more epochs to reach convergence\n"
            report += "- Monitor learning rate - may need adjustment\n"
        
        if overfit_info['train_val_gap'] > 0.1:
            report += "- Large train-validation gap suggests model may be too complex\n"
            report += "- Consider adding dropout or L2 regularization\n"
        
        # Save report
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Saved analysis report to {output_path}")
        
        return report
    
    def plot_learning_curve(self, 
                          train_sizes: np.ndarray,
                          train_scores: np.ndarray,
                          val_scores: np.ndarray,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot learning curve (alias for compatibility with tests).
        
        Args:
            train_sizes: Training set sizes
            train_scores: Training scores for each size
            val_scores: Validation scores for each size
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Create mock metrics from the provided data
        # Convert arrays to lists to avoid indexing issues
        train_sizes_list = train_sizes.tolist() if hasattr(train_sizes, 'tolist') else list(train_sizes)
        
        # Handle multi-dimensional score arrays
        if train_scores.ndim > 1:
            train_loss_vals = (1 - np.mean(train_scores, axis=1)).tolist()
            train_acc_vals = np.mean(train_scores, axis=1).tolist()
        else:
            train_loss_vals = (1 - train_scores).tolist()
            train_acc_vals = train_scores.tolist()
        
        if val_scores.ndim > 1:
            val_loss_vals = (1 - np.mean(val_scores, axis=1)).tolist()
            val_acc_vals = np.mean(val_scores, axis=1).tolist()
        else:
            val_loss_vals = (1 - val_scores).tolist()
            val_acc_vals = val_scores.tolist()
        
        self.metrics = {
            'iteration': train_sizes_list,
            'train_loss': train_loss_vals,
            'val_loss': val_loss_vals,
            'train_acc': train_acc_vals,
            'val_acc': val_acc_vals,
            'epoch': train_sizes_list,
            'learning_rate': [0.001] * len(train_sizes_list)
        }
        
        return self.plot_learning_curves(save_path=save_path, show_analysis=False)
    
    def plot_validation_curve(self, 
                            param_range: List[float],
                            train_scores: np.ndarray, 
                            val_scores: np.ndarray,
                            param_name: str = "Parameter",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot validation curve for parameter tuning.
        
        Args:
            param_range: Parameter values
            train_scores: Training scores for each parameter value
            val_scores: Validation scores for each parameter value
            param_name: Name of the parameter
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.plot(param_range, train_mean, 'o-', color='blue', 
               label='Training score')
        ax.fill_between(param_range, train_mean - train_std,
                       train_mean + train_std, alpha=0.1, color='blue')
        
        ax.plot(param_range, val_mean, 'o-', color='red',
               label='Cross-validation score')
        ax.fill_between(param_range, val_mean - val_std,
                       val_mean + val_std, alpha=0.1, color='red')
        
        ax.set_xlabel(param_name)
        ax.set_ylabel('Score')
        ax.set_title(f'Validation Curve for {param_name}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved validation curve plot to {save_path}")
        
        return fig 