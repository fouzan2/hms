"""ROC curve visualization for model performance analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ROCCurveVisualizer:
    """Visualizes ROC curves for binary and multi-class classification."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize ROC curve visualizer.
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir or Path("data/figures/performance")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_binary_roc(self, 
                       y_true: np.ndarray,
                       y_scores: np.ndarray,
                       title: str = "ROC Curve",
                       save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve for binary classification.
        
        Args:
            y_true: True binary labels
            y_scores: Predicted scores/probabilities
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot ROC curve
            ax.plot(fpr, tpr, color='darkorange', lw=3, 
                   label=f'ROC curve (AUC = {roc_auc:.3f})')
            
            # Plot diagonal line (random classifier)
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                   label='Random classifier (AUC = 0.5)')
            
            # Find optimal threshold (Youden's J statistic)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            optimal_fpr = fpr[optimal_idx]
            optimal_tpr = tpr[optimal_idx]
            
            # Mark optimal point
            ax.plot(optimal_fpr, optimal_tpr, 'ro', markersize=10,
                   label=f'Optimal threshold = {optimal_threshold:.3f}')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate (1 - Specificity)')
            ax.set_ylabel('True Positive Rate (Sensitivity)')
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            # Add text box with statistics
            stats_text = f"""
            AUC: {roc_auc:.3f}
            Optimal Threshold: {optimal_threshold:.3f}
            Sensitivity: {optimal_tpr:.3f}
            Specificity: {1-optimal_fpr:.3f}
            """
            
            ax.text(0.02, 0.98, stats_text.strip(), 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle="round", 
                   facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting binary ROC curve: {e}")
            raise
    
    def plot_multiclass_roc(self, 
                           y_true: np.ndarray,
                           y_scores: np.ndarray,
                           class_names: Optional[List[str]] = None,
                           title: str = "Multi-class ROC Curves",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            y_true: True class labels
            y_scores: Predicted scores/probabilities for each class
            class_names: Names of classes
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            n_classes = y_scores.shape[1] if y_scores.ndim > 1 else len(np.unique(y_true))
            
            if class_names is None:
                class_names = [f'Class {i}' for i in range(n_classes)]
            
            # Binarize labels for multi-class ROC
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Colors for different classes
            colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
            
            # Plot 1: Individual class ROC curves
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                if y_true_bin.ndim > 1:
                    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
                else:
                    # Handle binary case
                    fpr[i], tpr[i], _ = roc_curve(y_true == i, 
                                                y_scores[:, i] if y_scores.ndim > 1 else y_scores)
                roc_auc[i] = auc(fpr[i], tpr[i])
                
                ax1.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                        label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
            
            # Plot diagonal line
            ax1.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('ROC Curves by Class')
            ax1.legend(loc="lower right")
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Micro and Macro averaged ROC
            if y_true_bin.ndim > 1 and y_true_bin.shape[1] > 1:
                # Compute micro-average ROC curve
                fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), 
                                                   y_scores.ravel())
                roc_auc_micro = auc(fpr_micro, tpr_micro)
                
                ax2.plot(fpr_micro, tpr_micro, 
                        label=f'Micro-average (AUC = {roc_auc_micro:.3f})',
                        color='deeppink', linestyle=':', linewidth=4)
                
                # Compute macro-average ROC curve
                all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                mean_tpr = np.zeros_like(all_fpr)
                
                for i in range(n_classes):
                    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                
                mean_tpr /= n_classes
                roc_auc_macro = auc(all_fpr, mean_tpr)
                
                ax2.plot(all_fpr, mean_tpr,
                        label=f'Macro-average (AUC = {roc_auc_macro:.3f})',
                        color='navy', linestyle=':', linewidth=4)
            
            # Plot individual curves in second plot too
            for i in range(n_classes):
                ax2.plot(fpr[i], tpr[i], color=colors[i], lw=1, alpha=0.3)
            
            ax2.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('Averaged ROC Curves')
            ax2.legend(loc="lower right")
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting multiclass ROC curves: {e}")
            raise
    
    def plot_roc_comparison(self, 
                          models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                          title: str = "ROC Curves Comparison",
                          save_name: Optional[str] = None) -> plt.Figure:
        """
        Compare ROC curves for multiple models.
        
        Args:
            models_data: Dict mapping model names to (y_true, y_scores) tuples
            title: Plot title
            save_name: Filename to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(models_data)))
            auc_scores = {}
            
            # Plot 1: ROC curves
            for i, (model_name, (y_true, y_scores)) in enumerate(models_data.items()):
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                auc_scores[model_name] = roc_auc
                
                ax1.plot(fpr, tpr, color=colors[i], lw=3,
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
            
            # Plot diagonal line
            ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                    label='Random classifier')
            
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('ROC Curves Comparison')
            ax1.legend(loc="lower right")
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: AUC scores comparison
            model_names = list(auc_scores.keys())
            auc_values = list(auc_scores.values())
            
            bars = ax2.bar(range(len(model_names)), auc_values, 
                          color=colors, alpha=0.7)
            ax2.set_xlabel('Model')
            ax2.set_ylabel('AUC Score')
            ax2.set_title('AUC Scores Comparison')
            ax2.set_xticks(range(len(model_names)))
            ax2.set_xticklabels(model_names, rotation=45)
            ax2.set_ylim([0, 1])
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, auc_val in zip(bars, auc_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{auc_val:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting ROC comparison: {e}")
            raise
    
    def plot_precision_recall_curves(self, 
                                    y_true: np.ndarray,
                                    y_scores: np.ndarray,
                                    class_names: Optional[List[str]] = None,
                                    title: str = "Precision-Recall Curves",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot precision-recall curves for multi-class classification.
        
        Args:
            y_true: True class labels
            y_scores: Predicted scores/probabilities for each class
            class_names: Names of classes
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            matplotlib Figure object
        """
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            
            n_classes = y_scores.shape[1] if y_scores.ndim > 1 else len(np.unique(y_true))
            
            if class_names is None:
                class_names = [f'Class {i}' for i in range(n_classes)]
            
            # Binarize labels for multi-class PR curves
            y_true_bin = label_binarize(y_true, classes=range(n_classes))
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Colors for different classes
            colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
            
            for i in range(n_classes):
                if y_true_bin.ndim > 1:
                    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
                    avg_precision = average_precision_score(y_true_bin[:, i], y_scores[:, i])
                else:
                    # Handle binary case
                    precision, recall, _ = precision_recall_curve(y_true == i, 
                                                                y_scores[:, i] if y_scores.ndim > 1 else y_scores)
                    avg_precision = average_precision_score(y_true == i, 
                                                          y_scores[:, i] if y_scores.ndim > 1 else y_scores)
                
                ax.plot(recall, precision, color=colors[i], lw=2,
                       label=f'{class_names[i]} (AP = {avg_precision:.3f})')
            
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(title)
            ax.legend(loc="lower left")
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting precision-recall curves: {e}")
            raise
    
    def analyze_threshold_performance(self, 
                                    y_true: np.ndarray,
                                    y_scores: np.ndarray,
                                    save_name: Optional[str] = None) -> Dict[str, float]:
        """
        Analyze performance metrics across different thresholds.
        
        Args:
            y_true: True binary labels
            y_scores: Predicted scores/probabilities
            save_name: Filename to save plot
            
        Returns:
            Dictionary with optimal threshold information
        """
        try:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            
            # Calculate metrics for each threshold
            metrics = {
                'threshold': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'specificity': [],
                'sensitivity': [],
                'youden_j': []
            }
            
            for threshold in thresholds:
                y_pred = (y_scores >= threshold).astype(int)
                
                # Handle edge cases
                if len(np.unique(y_pred)) == 1:
                    precision = 0 if np.unique(y_pred)[0] == 0 else precision_score(y_true, y_pred, zero_division=0)
                    recall = 0 if np.unique(y_pred)[0] == 0 else recall_score(y_true, y_pred, zero_division=0)
                    f1 = 0
                else:
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    recall = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                metrics['threshold'].append(threshold)
                metrics['precision'].append(precision)
                metrics['recall'].append(recall)
                metrics['f1'].append(f1)
                metrics['specificity'].append(specificity)
                metrics['sensitivity'].append(recall)
                metrics['youden_j'].append(recall + specificity - 1)
            
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(metrics)
            
            # Find optimal thresholds
            optimal_f1_idx = df['f1'].idxmax()
            optimal_youden_idx = df['youden_j'].idxmax()
            
            optimal_thresholds = {
                'f1_optimal': {
                    'threshold': df.loc[optimal_f1_idx, 'threshold'],
                    'f1': df.loc[optimal_f1_idx, 'f1'],
                    'precision': df.loc[optimal_f1_idx, 'precision'],
                    'recall': df.loc[optimal_f1_idx, 'recall']
                },
                'youden_optimal': {
                    'threshold': df.loc[optimal_youden_idx, 'threshold'],
                    'youden_j': df.loc[optimal_youden_idx, 'youden_j'],
                    'sensitivity': df.loc[optimal_youden_idx, 'sensitivity'],
                    'specificity': df.loc[optimal_youden_idx, 'specificity']
                }
            }
            
            # Create visualization
            if save_name:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # Plot 1: Precision, Recall, F1 vs Threshold
                ax1.plot(df['threshold'], df['precision'], 'b-', label='Precision', linewidth=2)
                ax1.plot(df['threshold'], df['recall'], 'r-', label='Recall', linewidth=2)
                ax1.plot(df['threshold'], df['f1'], 'g-', label='F1-Score', linewidth=2)
                
                # Mark optimal F1 point
                ax1.axvline(x=optimal_thresholds['f1_optimal']['threshold'], 
                           color='green', linestyle='--', alpha=0.7,
                           label=f"Optimal F1 = {optimal_thresholds['f1_optimal']['threshold']:.3f}")
                
                ax1.set_xlabel('Threshold')
                ax1.set_ylabel('Score')
                ax1.set_title('Precision, Recall, F1 vs Threshold')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Sensitivity, Specificity vs Threshold
                ax2.plot(df['threshold'], df['sensitivity'], 'r-', label='Sensitivity (TPR)', linewidth=2)
                ax2.plot(df['threshold'], df['specificity'], 'b-', label='Specificity (TNR)', linewidth=2)
                ax2.plot(df['threshold'], df['youden_j'], 'purple', label="Youden's J", linewidth=2)
                
                # Mark optimal Youden point
                ax2.axvline(x=optimal_thresholds['youden_optimal']['threshold'], 
                           color='purple', linestyle='--', alpha=0.7,
                           label=f"Optimal Youden = {optimal_thresholds['youden_optimal']['threshold']:.3f}")
                
                ax2.set_xlabel('Threshold')
                ax2.set_ylabel('Score')
                ax2.set_title('Sensitivity, Specificity vs Threshold')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: ROC curve with optimal points
                ax3.plot(fpr, tpr, 'b-', linewidth=3, label=f'ROC (AUC = {auc(fpr, tpr):.3f})')
                ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                
                # Mark optimal points
                for i, threshold in enumerate(thresholds):
                    if abs(threshold - optimal_thresholds['youden_optimal']['threshold']) < 0.001:
                        ax3.plot(fpr[i], tpr[i], 'ro', markersize=10, 
                               label=f"Youden Optimal")
                        break
                
                ax3.set_xlabel('False Positive Rate')
                ax3.set_ylabel('True Positive Rate')
                ax3.set_title('ROC Curve with Optimal Points')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # Plot 4: Threshold distribution
                ax4.hist(df['threshold'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                ax4.axvline(x=optimal_thresholds['f1_optimal']['threshold'], 
                           color='green', linestyle='--', linewidth=2, label='F1 Optimal')
                ax4.axvline(x=optimal_thresholds['youden_optimal']['threshold'], 
                           color='purple', linestyle='--', linewidth=2, label='Youden Optimal')
                
                ax4.set_xlabel('Threshold')
                ax4.set_ylabel('Frequency')
                ax4.set_title('Threshold Distribution')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                plt.suptitle('Threshold Performance Analysis', fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                plt.savefig(self.save_dir / f"{save_name}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            return optimal_thresholds
            
        except Exception as e:
            logger.error(f"Error in threshold performance analysis: {e}")
            raise 