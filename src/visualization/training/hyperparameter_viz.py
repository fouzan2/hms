"""Hyperparameter visualization components for training analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class HyperparameterVisualizer:
    """Visualizes hyperparameter tuning results and analysis."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize hyperparameter visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir or Path("data/figures/hyperparameters")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_parameter_importance(self, 
                                importance_data: Dict[str, float],
                                title: str = "Hyperparameter Importance",
                                save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot hyperparameter importance scores.
        
        Args:
            importance_data: Dict mapping parameter names to importance scores
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Sort parameters by importance
            sorted_params = sorted(importance_data.items(), 
                                 key=lambda x: x[1], reverse=True)
            params, scores = zip(*sorted_params)
            
            # Create bar plot
            bars = ax.barh(range(len(params)), scores, 
                          color=sns.color_palette("viridis", len(params)))
            ax.set_yticks(range(len(params)))
            ax.set_yticklabels(params)
            ax.set_xlabel('Importance Score')
            ax.set_title(title, fontsize=16, fontweight='bold')
            
            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, scores)):
                ax.text(score + 0.01, i, f'{score:.3f}', 
                       va='center', ha='left', fontweight='bold')
            
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting parameter importance: {e}")
            raise
    
    def plot_parameter_correlation(self, 
                                 param_data: pd.DataFrame,
                                 performance_col: str = 'accuracy',
                                 title: str = "Hyperparameter Correlations",
                                 save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation heatmap between hyperparameters and performance.
        
        Args:
            param_data: DataFrame with hyperparameters and performance metrics
            performance_col: Name of performance metric column
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Calculate correlation matrix
            corr_matrix = param_data.corr()
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                       center=0, square=True, ax=ax,
                       fmt='.3f', cbar_kws={'shrink': 0.8})
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting parameter correlation: {e}")
            raise
    
    def plot_optimization_history(self, 
                                trials_data: List[Dict[str, Any]],
                                target_metric: str = 'validation_accuracy',
                                title: str = "Optimization History",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot optimization history showing how performance improves over trials.
        
        Args:
            trials_data: List of trial results with metrics
            target_metric: Target metric to plot
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Extract data
            trial_nums = [trial.get('trial_id', i) for i, trial in enumerate(trials_data)]
            values = [trial.get(target_metric, trial.get('value', 0)) for trial in trials_data]
            
            # Plot 1: Trial values over time
            ax1.plot(trial_nums, values, 'b-', alpha=0.7, label='Trial Value')
            
            # Add running best
            best_values = []
            current_best = float('-inf')
            for value in values:
                current_best = max(current_best, value)
                best_values.append(current_best)
            
            ax1.plot(trial_nums, best_values, 'r-', linewidth=2, 
                    label='Best Value So Far')
            ax1.set_xlabel('Trial Number')
            ax1.set_ylabel(f'{target_metric}')
            ax1.set_title(f'{title} - Progress', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Distribution of trial values
            ax2.hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(np.mean(values), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(values):.3f}')
            ax2.axvline(np.max(values), color='green', linestyle='--', 
                       label=f'Best: {np.max(values):.3f}')
            ax2.set_xlabel(f'{target_metric}')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Trial Values', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting optimization history: {e}")
            raise
    
    def plot_parameter_space(self, 
                           param_data: pd.DataFrame,
                           x_param: str,
                           y_param: str,
                           performance_col: str = 'accuracy',
                           title: Optional[str] = None,
                           save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot 2D parameter space with performance as color.
        
        Args:
            param_data: DataFrame with hyperparameters and performance
            x_param: Name of parameter for x-axis
            y_param: Name of parameter for y-axis
            performance_col: Name of performance metric column
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            scatter = ax.scatter(param_data[x_param], param_data[y_param], 
                               c=param_data[performance_col], 
                               cmap='viridis', s=50, alpha=0.7)
            
            ax.set_xlabel(x_param)
            ax.set_ylabel(y_param)
            
            if title is None:
                title = f'{x_param} vs {y_param} Parameter Space'
            ax.set_title(title, fontsize=16, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(performance_col)
            
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting parameter space: {e}")
            raise
    
    def generate_optimization_report(self, 
                                   trials_data: List[Dict[str, Any]],
                                   best_params: Dict[str, Any],
                                   save_name: str = "hyperparameter_report") -> None:
        """
        Generate comprehensive hyperparameter optimization report.
        
        Args:
            trials_data: List of trial results
            best_params: Best hyperparameters found
            save_name: Base filename for report
        """
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 12))
            
            # Plot optimization history
            ax1 = plt.subplot(2, 2, 1)
            trial_nums = [trial['trial_number'] for trial in trials_data]
            values = [trial['value'] for trial in trials_data]
            
            plt.plot(trial_nums, values, 'b-', alpha=0.7)
            best_values = []
            current_best = float('-inf')
            for value in values:
                current_best = max(current_best, value)
                best_values.append(current_best)
            plt.plot(trial_nums, best_values, 'r-', linewidth=2)
            plt.xlabel('Trial Number')
            plt.ylabel('Objective Value')
            plt.title('Optimization Progress')
            plt.grid(True, alpha=0.3)
            
            # Plot value distribution
            ax2 = plt.subplot(2, 2, 2)
            plt.hist(values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(np.mean(values), color='red', linestyle='--')
            plt.axvline(np.max(values), color='green', linestyle='--')
            plt.xlabel('Objective Value')
            plt.ylabel('Frequency')
            plt.title('Value Distribution')
            plt.grid(True, alpha=0.3)
            
            # Best parameters table
            ax3 = plt.subplot(2, 1, 2)
            ax3.axis('tight')
            ax3.axis('off')
            
            # Create table data
            table_data = []
            for param, value in best_params.items():
                table_data.append([param, str(value)])
            
            table = ax3.table(cellText=table_data,
                            colLabels=['Parameter', 'Best Value'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            
            plt.suptitle('Hyperparameter Optimization Report', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plt.savefig(self.save_dir / f"{save_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Hyperparameter optimization report saved to {save_name}.png")
            
        except Exception as e:
            logger.error(f"Error generating optimization report: {e}")
            raise
    
    def plot_hyperparameter_importance(self, 
                                      trials_data: List[Dict[str, Any]],
                                      target_metric: str = 'validation_accuracy',
                                      title: str = "Hyperparameter Importance",
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot hyperparameter importance based on trial data.
        
        Args:
            trials_data: List of trial results with hyperparameters and metrics
            target_metric: Target metric to analyze importance for
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Convert trials data to DataFrame
            df = pd.DataFrame(trials_data)
            
            if target_metric not in df.columns:
                raise ValueError(f"Target metric '{target_metric}' not found in trials data")
            
            # Calculate correlation between each hyperparameter and target metric
            hyperparams = [col for col in df.columns if col not in ['trial_id', target_metric]]
            importance_scores = {}
            
            for param in hyperparams:
                try:
                    # Convert to numeric if possible
                    param_values = pd.to_numeric(df[param], errors='coerce')
                    target_values = pd.to_numeric(df[target_metric], errors='coerce')
                    
                    # Remove NaN values
                    valid_mask = ~(param_values.isna() | target_values.isna())
                    if valid_mask.sum() > 1:
                        correlation = param_values[valid_mask].corr(target_values[valid_mask])
                        importance_scores[param] = abs(correlation) if not np.isnan(correlation) else 0
                    else:
                        importance_scores[param] = 0
                except:
                    importance_scores[param] = 0
            
            # Sort by importance
            sorted_params = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            params, scores = zip(*sorted_params) if sorted_params else ([], [])
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if scores:
                # Create bar plot
                bars = ax.barh(range(len(params)), scores, 
                              color=sns.color_palette("viridis", len(params)))
                ax.set_yticks(range(len(params)))
                ax.set_yticklabels(params)
                ax.set_xlabel(f'Importance (|correlation| with {target_metric})')
                ax.set_title(title, fontsize=16, fontweight='bold')
                
                # Add value labels on bars
                for i, (bar, score) in enumerate(zip(bars, scores)):
                    ax.text(score + 0.01, i, f'{score:.3f}', 
                           va='center', ha='left', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No valid hyperparameter data', 
                       ha='center', va='center', transform=ax.transAxes)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting hyperparameter importance: {e}")
            raise 