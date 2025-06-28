"""Model comparison visualization for performance analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelComparisonVisualizer:
    """Visualizes comparisons between multiple models."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize model comparison visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir or Path("data/figures/performance")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_metrics_comparison(self, 
                              models_metrics: Dict[str, Dict[str, float]],
                              title: str = "Model Metrics Comparison",
                              save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of multiple metrics across models.
        
        Args:
            models_metrics: Dict mapping model names to metric dictionaries
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(models_metrics).T
            
            n_metrics = len(df.columns)
            n_models = len(df.index)
            
            # Determine subplot layout
            ncols = min(3, n_metrics)
            nrows = (n_metrics + ncols - 1) // ncols
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
            
            if n_metrics == 1:
                axes = [axes]
            elif nrows == 1:
                axes = axes if hasattr(axes, '__len__') else [axes]
            else:
                axes = axes.flatten()
            
            colors = sns.color_palette("Set2", n_models)
            
            # Plot each metric
            for i, metric in enumerate(df.columns):
                ax = axes[i]
                
                values = df[metric].values
                model_names = df.index.tolist()
                
                bars = ax.bar(range(n_models), values, color=colors, alpha=0.8)
                ax.set_xlabel('Model')
                ax.set_ylabel(metric.title())
                ax.set_title(f'{metric.title()} Comparison')
                ax.set_xticks(range(n_models))
                ax.set_xticklabels(model_names, rotation=45, ha='right')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for j, (bar, value) in enumerate(zip(bars, values)):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                
                # Highlight best performer
                best_idx = np.argmax(values) if metric in ['accuracy', 'precision', 'recall', 'f1', 'auc'] else np.argmin(values)
                bars[best_idx].set_color('gold')
                bars[best_idx].set_edgecolor('black')
                bars[best_idx].set_linewidth(2)
            
            # Hide unused subplots
            for i in range(n_metrics, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting metrics comparison: {e}")
            raise
    
    def plot_radar_comparison(self, 
                            models_metrics: Dict[str, Dict[str, float]],
                            title: str = "Model Performance Radar Chart",
                            save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot radar chart comparing models across multiple metrics.
        
        Args:
            models_metrics: Dict mapping model names to metric dictionaries
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            df = pd.DataFrame(models_metrics).T
            
            # Normalize metrics to 0-1 scale for fair comparison
            df_normalized = df.copy()
            for col in df.columns:
                if col.lower() in ['loss', 'error']:
                    # For loss metrics, invert: higher is worse, so we want 1-normalized_value
                    df_normalized[col] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                else:
                    # For performance metrics: higher is better
                    df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            
            # Set up radar chart
            metrics = list(df.columns)
            num_metrics = len(metrics)
            
            # Compute angles for each metric
            angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            colors = sns.color_palette("Set1", len(df))
            
            # Plot each model
            for i, (model_name, row) in enumerate(df_normalized.iterrows()):
                values = row.tolist()
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])
            
            # Customize the chart
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
            ax.grid(True)
            
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            plt.title(title, fontsize=16, fontweight='bold', pad=20)
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting radar comparison: {e}")
            raise
    
    def plot_model_ranking(self, 
                         models_metrics: Dict[str, Dict[str, float]],
                         weights: Optional[Dict[str, float]] = None,
                         title: str = "Model Ranking",
                         save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot model ranking based on weighted scores.
        
        Args:
            models_metrics: Dict mapping model names to metric dictionaries
            weights: Dict mapping metric names to weights (default: equal weights)
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            df = pd.DataFrame(models_metrics).T
            
            # Set default weights if not provided
            if weights is None:
                weights = {metric: 1.0 for metric in df.columns}
            
            # Normalize metrics and calculate weighted scores
            df_normalized = df.copy()
            for col in df.columns:
                if col.lower() in ['loss', 'error']:
                    # For loss metrics: lower is better
                    df_normalized[col] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                else:
                    # For performance metrics: higher is better
                    df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            
            # Calculate weighted scores
            weighted_scores = np.zeros(len(df))
            total_weight = sum(weights.values())
            
            for metric, weight in weights.items():
                if metric in df_normalized.columns:
                    weighted_scores += df_normalized[metric].values * (weight / total_weight)
            
            # Create ranking
            ranking_df = pd.DataFrame({
                'model': df.index,
                'score': weighted_scores
            }).sort_values('score', ascending=False)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot 1: Ranking bar chart
            colors = sns.color_palette("RdYlGn", len(ranking_df))
            bars = ax1.barh(range(len(ranking_df)), ranking_df['score'], color=colors)
            
            ax1.set_yticks(range(len(ranking_df)))
            ax1.set_yticklabels(ranking_df['model'])
            ax1.set_xlabel('Weighted Score')
            ax1.set_title('Model Ranking (Higher is Better)')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Add rank numbers and scores
            for i, (bar, score) in enumerate(zip(bars, ranking_df['score'])):
                ax1.text(score + 0.01, i, f'#{i+1} ({score:.3f})', 
                        va='center', ha='left', fontweight='bold')
            
            # Plot 2: Contribution of each metric
            model_names = ranking_df['model'].tolist()
            bottom = np.zeros(len(model_names))
            
            metric_colors = sns.color_palette("Set3", len(weights))
            
            for i, (metric, weight) in enumerate(weights.items()):
                if metric in df_normalized.columns:
                    contributions = []
                    for model in model_names:
                        contrib = df_normalized.loc[model, metric] * (weight / total_weight)
                        contributions.append(contrib)
                    
                    ax2.bar(range(len(model_names)), contributions, bottom=bottom,
                           label=f'{metric} (w={weight})', color=metric_colors[i], alpha=0.8)
                    bottom += contributions
            
            ax2.set_xticks(range(len(model_names)))
            ax2.set_xticklabels(model_names, rotation=45, ha='right')
            ax2.set_ylabel('Score Contribution')
            ax2.set_title('Score Breakdown by Metric')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting model ranking: {e}")
            raise
    
    def plot_performance_vs_complexity(self, 
                                     models_data: Dict[str, Dict[str, float]],
                                     performance_metric: str = 'accuracy',
                                     complexity_metric: str = 'parameters',
                                     title: str = "Performance vs Complexity",
                                     save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot model performance against complexity metrics.
        
        Args:
            models_data: Dict mapping model names to metrics including complexity
            performance_metric: Name of performance metric to use
            complexity_metric: Name of complexity metric to use
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            models = list(models_data.keys())
            performance = [models_data[model][performance_metric] for model in models]
            complexity = [models_data[model][complexity_metric] for model in models]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create scatter plot
            colors = sns.color_palette("viridis", len(models))
            scatter = ax.scatter(complexity, performance, c=colors, s=100, alpha=0.7)
            
            # Add model labels
            for i, model in enumerate(models):
                ax.annotate(model, (complexity[i], performance[i]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, alpha=0.8)
            
            ax.set_xlabel(f'{complexity_metric.title()}')
            ax.set_ylabel(f'{performance_metric.title()}')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Add efficiency frontier line (Pareto front)
            # Sort by complexity
            sorted_indices = np.argsort(complexity)
            sorted_complexity = np.array(complexity)[sorted_indices]
            sorted_performance = np.array(performance)[sorted_indices]
            
            # Find Pareto front
            pareto_front = []
            max_performance_so_far = -np.inf
            
            for i in range(len(sorted_complexity)):
                if sorted_performance[i] > max_performance_so_far:
                    pareto_front.append(i)
                    max_performance_so_far = sorted_performance[i]
            
            if len(pareto_front) > 1:
                pareto_complexity = sorted_complexity[pareto_front]
                pareto_performance = sorted_performance[pareto_front]
                ax.plot(pareto_complexity, pareto_performance, 'r--', 
                       linewidth=2, alpha=0.7, label='Efficiency Frontier')
                ax.legend()
            
            # Add quadrant lines
            mean_complexity = np.mean(complexity)
            mean_performance = np.mean(performance)
            ax.axhline(y=mean_performance, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(x=mean_complexity, color='gray', linestyle=':', alpha=0.5)
            
            # Add quadrant labels
            ax.text(0.05, 0.95, 'Low Complexity\nHigh Performance', 
                   transform=ax.transAxes, fontsize=10, 
                   bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.7))
            ax.text(0.95, 0.05, 'High Complexity\nLow Performance', 
                   transform=ax.transAxes, fontsize=10, ha='right',
                   bbox=dict(boxstyle="round", facecolor='lightcoral', alpha=0.7))
            
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting performance vs complexity: {e}")
            raise 