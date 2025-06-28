"""Feature importance visualization for model interpretability."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FeatureImportanceVisualizer:
    """Visualizes feature importance from various machine learning models."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize feature importance visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir or Path("data/figures/performance")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_feature_importance(self, 
                              importance_scores: np.ndarray,
                              feature_names: List[str],
                              title: str = "Feature Importance",
                              top_k: int = 20,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance scores as horizontal bar chart.
        
        Args:
            importance_scores: Importance scores for each feature
            feature_names: Names of features
            title: Plot title
            top_k: Maximum number of features to display (replaces max_features)
            save_path: Path to save plot (changed from save_name)
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Create DataFrame and sort by importance
            df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            })
            df = df.sort_values('importance', ascending=True)
            
            # Take top features
            if len(df) > top_k:
                df = df.tail(top_k)
            
            fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.4)))
            
            # Create horizontal bar plot
            colors = sns.color_palette("viridis", len(df))
            bars = ax.barh(range(len(df)), df['importance'], color=colors, alpha=0.8)
            
            # Customize plot
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df['feature'])
            ax.set_xlabel('Importance Score')
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (bar, importance) in enumerate(zip(bars, df['importance'])):
                ax.text(importance + 0.001, i, f'{importance:.3f}', 
                       va='center', ha='left', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
            raise
    
    def plot_permutation_importance(self, 
                                  feature_names: List[str],
                                  importance_mean: np.ndarray,
                                  importance_std: np.ndarray,
                                  title: str = "Permutation Feature Importance",
                                  max_features: int = 20,
                                  save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot permutation importance with error bars.
        
        Args:
            feature_names: Names of features
            importance_mean: Mean importance scores
            importance_std: Standard deviation of importance scores
            title: Plot title
            max_features: Maximum number of features to display
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Create DataFrame and sort by importance
            df = pd.DataFrame({
                'feature': feature_names,
                'importance_mean': importance_mean,
                'importance_std': importance_std
            })
            df = df.sort_values('importance_mean', ascending=True)
            
            # Take top features
            if len(df) > max_features:
                df = df.tail(max_features)
            
            fig, ax = plt.subplots(figsize=(12, max(8, len(df) * 0.4)))
            
            # Create horizontal bar plot with error bars
            y_pos = range(len(df))
            bars = ax.barh(y_pos, df['importance_mean'], 
                          xerr=df['importance_std'],
                          color='skyblue', alpha=0.8, capsize=5)
            
            # Customize plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(df['feature'])
            ax.set_xlabel('Importance Score')
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (mean_val, std_val) in enumerate(zip(df['importance_mean'], df['importance_std'])):
                ax.text(mean_val + std_val + 0.001, i, 
                       f'{mean_val:.3f}±{std_val:.3f}', 
                       va='center', ha='left', fontweight='bold', fontsize=9)
            
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting permutation importance: {e}")
            raise
    
    def plot_shap_summary(self, 
                         shap_values: np.ndarray,
                         feature_names: List[str],
                         feature_values: Optional[np.ndarray] = None,
                         title: str = "SHAP Feature Importance Summary",
                         max_features: int = 20,
                         save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot SHAP summary plot showing feature importance and impact.
        
        Args:
            shap_values: SHAP values matrix (samples x features)
            feature_names: Names of features
            feature_values: Feature values matrix (samples x features)
            title: Plot title
            max_features: Maximum number of features to display
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Calculate mean absolute SHAP values for importance ranking
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Create DataFrame and sort by importance
            df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_abs_shap
            })
            df = df.sort_values('importance', ascending=False)
            
            # Take top features
            if len(df) > max_features:
                df = df.head(max_features)
                top_indices = [feature_names.index(name) for name in df['feature']]
                shap_values = shap_values[:, top_indices]
                feature_names = df['feature'].tolist()
                if feature_values is not None:
                    feature_values = feature_values[:, top_indices]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(8, len(df) * 0.3)))
            
            # Plot 1: SHAP summary plot (bee swarm style)
            for i, feature in enumerate(reversed(feature_names)):
                feature_idx = len(feature_names) - 1 - i
                feature_shap = shap_values[:, feature_idx]
                
                if feature_values is not None:
                    # Color by feature value
                    feature_vals = feature_values[:, feature_idx]
                    scatter = ax1.scatter(feature_shap, [i] * len(feature_shap), 
                                        c=feature_vals, alpha=0.6, s=20, 
                                        cmap='coolwarm')
                else:
                    # Color by SHAP value
                    ax1.scatter(feature_shap, [i] * len(feature_shap), 
                              c=feature_shap, alpha=0.6, s=20, 
                              cmap='coolwarm')
            
            ax1.set_yticks(range(len(feature_names)))
            ax1.set_yticklabels(reversed(feature_names))
            ax1.set_xlabel('SHAP Value')
            ax1.set_title('SHAP Summary Plot')
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax1.grid(True, alpha=0.3)
            
            # Add colorbar
            if feature_values is not None:
                plt.colorbar(scatter, ax=ax1, label='Feature Value')
            
            # Plot 2: Feature importance (mean absolute SHAP)
            importance_sorted = df.sort_values('importance', ascending=True)
            colors = sns.color_palette("viridis", len(importance_sorted))
            bars = ax2.barh(range(len(importance_sorted)), 
                           importance_sorted['importance'], 
                           color=colors, alpha=0.8)
            
            ax2.set_yticks(range(len(importance_sorted)))
            ax2.set_yticklabels(importance_sorted['feature'])
            ax2.set_xlabel('Mean |SHAP Value|')
            ax2.set_title('Feature Importance')
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (bar, importance) in enumerate(zip(bars, importance_sorted['importance'])):
                ax2.text(importance + 0.001, i, f'{importance:.3f}', 
                        va='center', ha='left', fontweight='bold', fontsize=9)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting SHAP summary: {e}")
            raise
    
    def plot_feature_correlation_importance(self, 
                                          feature_names: List[str],
                                          importance_scores: np.ndarray,
                                          correlation_matrix: np.ndarray,
                                          title: str = "Feature Importance vs Correlation",
                                          save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance against feature correlations.
        
        Args:
            feature_names: Names of features
            importance_scores: Importance scores for each feature
            correlation_matrix: Feature correlation matrix
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Feature importance
            sorted_indices = np.argsort(importance_scores)[-20:]  # Top 20
            top_features = [feature_names[i] for i in sorted_indices]
            top_importance = importance_scores[sorted_indices]
            
            colors = sns.color_palette("viridis", len(top_features))
            bars = ax1.barh(range(len(top_features)), top_importance, 
                           color=colors, alpha=0.8)
            ax1.set_yticks(range(len(top_features)))
            ax1.set_yticklabels(top_features)
            ax1.set_xlabel('Importance Score')
            ax1.set_title('Top 20 Feature Importance')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Plot 2: Correlation heatmap (top features)
            top_corr_matrix = correlation_matrix[np.ix_(sorted_indices, sorted_indices)]
            im = ax2.imshow(top_corr_matrix, cmap='RdBu_r', aspect='auto', 
                           vmin=-1, vmax=1)
            ax2.set_xticks(range(len(top_features)))
            ax2.set_yticks(range(len(top_features)))
            ax2.set_xticklabels(top_features, rotation=45, ha='right')
            ax2.set_yticklabels(top_features)
            ax2.set_title('Correlation Matrix (Top Features)')
            plt.colorbar(im, ax=ax2, shrink=0.8)
            
            # Plot 3: Scatter plot - Importance vs Max Correlation
            max_correlations = []
            for i, importance in enumerate(importance_scores):
                # Find maximum absolute correlation with other features
                corr_row = np.abs(correlation_matrix[i])
                corr_row[i] = 0  # Exclude self-correlation
                max_correlations.append(np.max(corr_row))
            
            scatter = ax3.scatter(max_correlations, importance_scores, 
                                alpha=0.6, s=50, c=importance_scores, 
                                cmap='viridis')
            ax3.set_xlabel('Max Absolute Correlation with Other Features')
            ax3.set_ylabel('Feature Importance')
            ax3.set_title('Importance vs Maximum Correlation')
            ax3.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax3, shrink=0.8)
            
            # Plot 4: Distribution of correlations for high vs low importance features
            # Split features into high and low importance
            median_importance = np.median(importance_scores)
            high_importance_mask = importance_scores > median_importance
            
            high_imp_correlations = []
            low_imp_correlations = []
            
            for i in range(len(importance_scores)):
                corr_row = np.abs(correlation_matrix[i])
                corr_row[i] = 0  # Exclude self-correlation
                max_corr = np.max(corr_row)
                
                if high_importance_mask[i]:
                    high_imp_correlations.append(max_corr)
                else:
                    low_imp_correlations.append(max_corr)
            
            ax4.hist(low_imp_correlations, bins=20, alpha=0.7, 
                    label='Low Importance Features', density=True, color='lightblue')
            ax4.hist(high_imp_correlations, bins=20, alpha=0.7, 
                    label='High Importance Features', density=True, color='orange')
            ax4.set_xlabel('Maximum Absolute Correlation')
            ax4.set_ylabel('Density')
            ax4.set_title('Correlation Distribution by Importance Level')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting feature correlation importance: {e}")
            raise
    
    def compare_importance_methods(self, 
                                 feature_names: List[str],
                                 importance_dict: Dict[str, np.ndarray],
                                 title: str = "Feature Importance Methods Comparison",
                                 max_features: int = 15,
                                 save_name: Optional[str] = None) -> plt.Figure:
        """
        Compare feature importance from different methods.
        
        Args:
            feature_names: Names of features
            importance_dict: Dict mapping method names to importance scores
            title: Plot title
            max_features: Maximum number of features to display
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Create DataFrame with all importance methods
            df = pd.DataFrame({'feature': feature_names})
            
            for method_name, scores in importance_dict.items():
                df[method_name] = scores
            
            # Calculate mean importance across methods for ranking
            method_cols = [col for col in df.columns if col != 'feature']
            df['mean_importance'] = df[method_cols].mean(axis=1)
            
            # Sort by mean importance and take top features
            df = df.sort_values('mean_importance', ascending=False)
            if len(df) > max_features:
                df = df.head(max_features)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(8, len(df) * 0.3)))
            
            # Plot 1: Grouped bar chart
            x_pos = np.arange(len(df))
            width = 0.8 / len(method_cols)
            colors = sns.color_palette("Set2", len(method_cols))
            
            for i, method in enumerate(method_cols):
                offset = (i - len(method_cols) / 2 + 0.5) * width
                bars = ax1.bar(x_pos + offset, df[method], width, 
                              label=method, color=colors[i], alpha=0.8)
            
            ax1.set_xlabel('Features')
            ax1.set_ylabel('Importance Score')
            ax1.set_title('Feature Importance by Method')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(df['feature'], rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Plot 2: Correlation heatmap between methods
            method_corr = df[method_cols].corr()
            im = ax2.imshow(method_corr, cmap='RdBu_r', aspect='auto', 
                           vmin=-1, vmax=1)
            ax2.set_xticks(range(len(method_cols)))
            ax2.set_yticks(range(len(method_cols)))
            ax2.set_xticklabels(method_cols, rotation=45, ha='right')
            ax2.set_yticklabels(method_cols)
            ax2.set_title('Method Correlation')
            
            # Add correlation values as text
            for i in range(len(method_cols)):
                for j in range(len(method_cols)):
                    text = ax2.text(j, i, f'{method_corr.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", 
                                   fontweight='bold')
            
            plt.colorbar(im, ax=ax2, shrink=0.8)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error comparing importance methods: {e}")
            raise
    
    def plot_feature_stability(self, 
                             feature_names: List[str],
                             importance_history: List[np.ndarray],
                             title: str = "Feature Importance Stability",
                             max_features: int = 10,
                             save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot how feature importance changes over time/iterations.
        
        Args:
            feature_names: Names of features
            importance_history: List of importance arrays over time
            title: Plot title
            max_features: Maximum number of features to display
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Calculate mean importance across all iterations
            mean_importance = np.mean(importance_history, axis=0)
            
            # Select top features based on mean importance
            top_indices = np.argsort(mean_importance)[-max_features:]
            top_features = [feature_names[i] for i in top_indices]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Plot 1: Importance evolution over time
            iterations = range(len(importance_history))
            colors = sns.color_palette("tab10", max_features)
            
            for i, feature_idx in enumerate(top_indices):
                importance_series = [imp[feature_idx] for imp in importance_history]
                ax1.plot(iterations, importance_series, 
                        color=colors[i], linewidth=2, 
                        label=feature_names[feature_idx])
            
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Importance Score')
            ax1.set_title('Feature Importance Evolution')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Stability metrics (coefficient of variation)
            stability_scores = []
            feature_means = []
            
            for feature_idx in top_indices:
                importance_series = [imp[feature_idx] for imp in importance_history]
                mean_imp = np.mean(importance_series)
                std_imp = np.std(importance_series)
                cv = std_imp / mean_imp if mean_imp > 0 else 0  # Coefficient of variation
                
                stability_scores.append(cv)
                feature_means.append(mean_imp)
            
            bars = ax2.bar(range(len(top_features)), stability_scores, 
                          color=colors, alpha=0.8)
            ax2.set_xlabel('Feature')
            ax2.set_ylabel('Coefficient of Variation')
            ax2.set_title('Feature Importance Stability (Lower = More Stable)')
            ax2.set_xticks(range(len(top_features)))
            ax2.set_xticklabels(top_features, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, cv) in enumerate(zip(bars, stability_scores)):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{cv:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting feature stability: {e}")
            raise
    
    def plot_feature_importance_heatmap(self, 
                                      importance_matrix: np.ndarray,
                                      feature_names: List[str],
                                      model_names: List[str],
                                      title: str = "Feature Importance Heatmap",
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance heatmap for multiple models.
        
        Args:
            importance_matrix: Matrix of importance scores (models x features)
            feature_names: Names of features
            model_names: Names of models
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create heatmap
            sns.heatmap(importance_matrix, 
                       annot=True, 
                       fmt='.3f',
                       cmap='viridis',
                       xticklabels=feature_names,
                       yticklabels=model_names,
                       ax=ax)
            
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Features')
            ax.set_ylabel('Models')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting feature importance heatmap: {e}")
            raise 