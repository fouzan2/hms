"""Calibration visualization for model reliability analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CalibrationVisualizer:
    """Visualizes model calibration and reliability."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize calibration visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir or Path("data/figures/performance")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_calibration_curve(self, 
                             y_true: np.ndarray,
                             y_proba: np.ndarray,
                             n_bins: int = 10,
                             title: str = "Calibration Curve",
                             save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot calibration curve for binary classification.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            n_bins: Number of bins for calibration curve
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_proba, n_bins=n_bins
            )
            
            # Calculate Brier score
            brier_score = brier_score_loss(y_true, y_proba)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Calibration curve
            ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                    linewidth=2, label=f"Model (Brier: {brier_score:.3f})")
            ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            
            ax1.set_xlabel("Mean Predicted Probability")
            ax1.set_ylabel("Fraction of Positives")
            ax1.set_title("Calibration Plot (Reliability Diagram)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add calibration error text
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            ax1.text(0.05, 0.95, f'Calibration Error: {calibration_error:.3f}',
                    transform=ax1.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
            
            # Plot 2: Histogram of predicted probabilities
            ax2.hist(y_proba, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.set_xlabel("Predicted Probability")
            ax2.set_ylabel("Count")
            ax2.set_title("Distribution of Predicted Probabilities")
            ax2.grid(True, alpha=0.3)
            
            # Add vertical lines for bin edges
            bin_edges = np.linspace(0, 1, n_bins + 1)
            for edge in bin_edges:
                ax2.axvline(x=edge, color='red', linestyle='--', alpha=0.5)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting calibration curve: {e}")
            raise
    
    def compare_calibration(self, 
                          models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                          n_bins: int = 10,
                          title: str = "Calibration Comparison",
                          save_name: Optional[str] = None) -> plt.Figure:
        """
        Compare calibration curves for multiple models.
        
        Args:
            models_data: Dict mapping model names to (y_true, y_proba) tuples
            n_bins: Number of bins for calibration curves
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(models_data)))
            calibration_metrics = {}
            
            # Plot calibration curves
            for i, (model_name, (y_true, y_proba)) in enumerate(models_data.items()):
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, y_proba, n_bins=n_bins
                )
                brier_score = brier_score_loss(y_true, y_proba)
                calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
                
                calibration_metrics[model_name] = {
                    'brier_score': brier_score,
                    'calibration_error': calibration_error
                }
                
                ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                        color=colors[i], linewidth=2, 
                        label=f"{model_name} (Brier: {brier_score:.3f})")
            
            # Perfect calibration line
            ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            ax1.set_xlabel("Mean Predicted Probability")
            ax1.set_ylabel("Fraction of Positives")
            ax1.set_title("Calibration Curves Comparison")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Metrics comparison
            model_names = list(calibration_metrics.keys())
            brier_scores = [calibration_metrics[name]['brier_score'] for name in model_names]
            cal_errors = [calibration_metrics[name]['calibration_error'] for name in model_names]
            
            x_pos = np.arange(len(model_names))
            width = 0.35
            
            bars1 = ax2.bar(x_pos - width/2, brier_scores, width, 
                           label='Brier Score', alpha=0.8, color='lightcoral')
            bars2 = ax2.bar(x_pos + width/2, cal_errors, width, 
                           label='Calibration Error', alpha=0.8, color='skyblue')
            
            ax2.set_xlabel('Model')
            ax2.set_ylabel('Score')
            ax2.set_title('Calibration Metrics Comparison')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(model_names, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars1, brier_scores):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            for bar, value in zip(bars2, cal_errors):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error comparing calibration: {e}")
            raise 