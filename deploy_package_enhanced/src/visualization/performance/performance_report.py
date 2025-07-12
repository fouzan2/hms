"""Performance report generation for comprehensive model evaluation."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceReportGenerator:
    """Generates comprehensive performance reports for model evaluation."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize performance report generator.
        
        Args:
            save_dir: Directory to save reports
        """
        self.save_dir = save_dir or Path("data/figures/reports")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def generate_classification_report(self, 
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     y_proba: Optional[np.ndarray] = None,
                                     class_names: Optional[List[str]] = None,
                                     model_name: str = "Model",
                                     save_name: str = "classification_report") -> None:
        """
        Generate comprehensive classification performance report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for binary classification)
            class_names: Names of classes
            model_name: Name of the model
            save_name: Base filename for report
        """
        try:
            from sklearn.metrics import (
                classification_report, confusion_matrix, 
                roc_auc_score, precision_recall_fscore_support
            )
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 16))
            
            # Basic metrics
            if y_proba is not None and len(np.unique(y_true)) == 2:
                auc_score = roc_auc_score(y_true, y_proba)
            else:
                auc_score = None
            
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None
            )
            
            # Plot 1: Confusion Matrix
            ax1 = plt.subplot(3, 3, 1)
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix', fontweight='bold')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Plot 2: Class-wise Performance
            ax2 = plt.subplot(3, 3, 2)
            if class_names is None:
                class_names = [f'Class {i}' for i in range(len(precision))]
            
            x_pos = np.arange(len(class_names))
            width = 0.25
            
            plt.bar(x_pos - width, precision, width, label='Precision', alpha=0.8)
            plt.bar(x_pos, recall, width, label='Recall', alpha=0.8)
            plt.bar(x_pos + width, f1, width, label='F1-Score', alpha=0.8)
            
            plt.xlabel('Classes')
            plt.ylabel('Score')
            plt.title('Class-wise Performance Metrics')
            plt.xticks(x_pos, class_names, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            
            # Plot 3: ROC Curve (if binary classification)
            if y_proba is not None and len(np.unique(y_true)) == 2:
                from sklearn.metrics import roc_curve, auc
                
                ax3 = plt.subplot(3, 3, 3)
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                        label=f'ROC curve (AUC = {roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Plot 4: Support (sample count) by class
            ax4 = plt.subplot(3, 3, 4)
            bars = plt.bar(range(len(class_names)), support, alpha=0.8, color='lightgreen')
            plt.xlabel('Classes')
            plt.ylabel('Number of Samples')
            plt.title('Support by Class')
            plt.xticks(range(len(class_names)), class_names, rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, supp in zip(bars, support):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        str(supp), ha='center', va='bottom', fontweight='bold')
            
            # Plot 5: Normalized Confusion Matrix
            ax5 = plt.subplot(3, 3, 5)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Normalized Confusion Matrix', fontweight='bold')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Plot 6: Precision-Recall Curve (if binary)
            if y_proba is not None and len(np.unique(y_true)) == 2:
                from sklearn.metrics import precision_recall_curve, average_precision_score
                
                ax6 = plt.subplot(3, 3, 6)
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
                avg_precision = average_precision_score(y_true, y_proba)
                
                plt.plot(recall_curve, precision_curve, color='darkgreen', lw=2,
                        label=f'PR curve (AP = {avg_precision:.3f})')
                
                # Baseline
                baseline = np.sum(y_true) / len(y_true)
                plt.axhline(y=baseline, color='navy', linestyle='--', 
                           label=f'Baseline (AP = {baseline:.3f})')
                
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Summary Statistics Table
            ax7 = plt.subplot(3, 1, 3)
            ax7.axis('tight')
            ax7.axis('off')
            
            # Calculate overall metrics
            accuracy = np.mean(y_true == y_pred)
            macro_precision = np.mean(precision)
            macro_recall = np.mean(recall)
            macro_f1 = np.mean(f1)
            
            # Weighted averages
            weights = support / np.sum(support)
            weighted_precision = np.sum(precision * weights)
            weighted_recall = np.sum(recall * weights)
            weighted_f1 = np.sum(f1 * weights)
            
            # Create summary table
            summary_data = [
                ['Accuracy', f'{accuracy:.4f}'],
                ['Macro Avg Precision', f'{macro_precision:.4f}'],
                ['Macro Avg Recall', f'{macro_recall:.4f}'],
                ['Macro Avg F1-Score', f'{macro_f1:.4f}'],
                ['Weighted Avg Precision', f'{weighted_precision:.4f}'],
                ['Weighted Avg Recall', f'{weighted_recall:.4f}'],
                ['Weighted Avg F1-Score', f'{weighted_f1:.4f}'],
                ['Total Samples', str(len(y_true))]
            ]
            
            if auc_score is not None:
                summary_data.insert(-1, ['AUC Score', f'{auc_score:.4f}'])
            
            table = ax7.table(cellText=summary_data,
                            colLabels=['Metric', 'Value'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            
            plt.suptitle(f'{model_name} - Classification Performance Report\n'
                        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plt.savefig(self.save_dir / f"{save_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Classification report saved to {save_name}.png")
            
        except Exception as e:
            logger.error(f"Error generating classification report: {e}")
            raise
    
    def generate_model_comparison_report(self, 
                                       models_results: Dict[str, Dict[str, Any]],
                                       save_name: str = "model_comparison_report") -> None:
        """
        Generate comprehensive model comparison report.
        
        Args:
            models_results: Dict mapping model names to result dictionaries
            save_name: Base filename for report
        """
        try:
            fig = plt.figure(figsize=(20, 16))
            
            # Extract metrics for comparison
            model_names = list(models_results.keys())
            metrics_data = {}
            
            for model_name, results in models_results.items():
                metrics = results.get('metrics', {})
                for metric_name, value in metrics.items():
                    if metric_name not in metrics_data:
                        metrics_data[metric_name] = {}
                    metrics_data[metric_name][model_name] = value
            
            # Plot 1: Accuracy Comparison
            ax1 = plt.subplot(3, 3, 1)
            if 'accuracy' in metrics_data:
                accuracies = [metrics_data['accuracy'].get(model, 0) for model in model_names]
                bars = plt.bar(range(len(model_names)), accuracies, alpha=0.8, 
                              color=sns.color_palette("viridis", len(model_names)))
                plt.xlabel('Model')
                plt.ylabel('Accuracy')
                plt.title('Accuracy Comparison')
                plt.xticks(range(len(model_names)), model_names, rotation=45)
                plt.grid(True, alpha=0.3, axis='y')
                
                # Highlight best model
                best_idx = np.argmax(accuracies)
                bars[best_idx].set_color('gold')
                bars[best_idx].set_edgecolor('black')
                bars[best_idx].set_linewidth(2)
                
                # Add value labels
                for bar, acc in zip(bars, accuracies):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                            f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 2: F1-Score Comparison
            ax2 = plt.subplot(3, 3, 2)
            if 'f1_score' in metrics_data or 'f1' in metrics_data:
                metric_key = 'f1_score' if 'f1_score' in metrics_data else 'f1'
                f1_scores = [metrics_data[metric_key].get(model, 0) for model in model_names]
                bars = plt.bar(range(len(model_names)), f1_scores, alpha=0.8,
                              color=sns.color_palette("plasma", len(model_names)))
                plt.xlabel('Model')
                plt.ylabel('F1-Score')
                plt.title('F1-Score Comparison')
                plt.xticks(range(len(model_names)), model_names, rotation=45)
                plt.grid(True, alpha=0.3, axis='y')
                
                # Highlight best model
                best_idx = np.argmax(f1_scores)
                bars[best_idx].set_color('gold')
                bars[best_idx].set_edgecolor('black')
                bars[best_idx].set_linewidth(2)
                
                # Add value labels
                for bar, f1 in zip(bars, f1_scores):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                            f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 3: Training Time Comparison
            ax3 = plt.subplot(3, 3, 3)
            training_times = []
            for model_name in model_names:
                train_time = models_results[model_name].get('training_time', 0)
                training_times.append(train_time)
            
            if any(t > 0 for t in training_times):
                bars = plt.bar(range(len(model_names)), training_times, alpha=0.8,
                              color=sns.color_palette("coolwarm", len(model_names)))
                plt.xlabel('Model')
                plt.ylabel('Training Time (seconds)')
                plt.title('Training Time Comparison')
                plt.xticks(range(len(model_names)), model_names, rotation=45)
                plt.grid(True, alpha=0.3, axis='y')
                
                # Highlight fastest model
                if training_times:
                    fastest_idx = np.argmin([t for t in training_times if t > 0])
                    bars[fastest_idx].set_color('lightgreen')
                    bars[fastest_idx].set_edgecolor('black')
                    bars[fastest_idx].set_linewidth(2)
            
            # Plot 4: Multi-metric Radar Chart
            ax4 = plt.subplot(3, 3, 4, projection='polar')
            
            # Select key metrics for radar chart
            radar_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            available_metrics = [m for m in radar_metrics if m in metrics_data]
            
            if available_metrics:
                angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
                angles += angles[:1]
                
                colors = sns.color_palette("Set1", len(model_names))
                
                for i, model_name in enumerate(model_names):
                    values = []
                    for metric in available_metrics:
                        values.append(metrics_data[metric].get(model_name, 0))
                    values += values[:1]
                    
                    ax4.plot(angles, values, 'o-', linewidth=2, 
                            label=model_name, color=colors[i])
                    ax4.fill(angles, values, alpha=0.25, color=colors[i])
                
                ax4.set_xticks(angles[:-1])
                ax4.set_xticklabels(available_metrics)
                ax4.set_ylim(0, 1)
                ax4.set_title('Multi-metric Performance', fontweight='bold', pad=20)
                ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            # Plot 5: Confusion Matrix Comparison (for first two models)
            if len(model_names) >= 2:
                for idx, model_name in enumerate(model_names[:2]):
                    ax = plt.subplot(3, 3, 5 + idx)
                    
                    cm = models_results[model_name].get('confusion_matrix')
                    if cm is not None:
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                        plt.title(f'{model_name} - Confusion Matrix')
                        plt.ylabel('True Label')
                        plt.xlabel('Predicted Label')
            
            # Summary Table
            ax7 = plt.subplot(3, 1, 3)
            ax7.axis('tight')
            ax7.axis('off')
            
            # Create comprehensive comparison table
            table_data = []
            headers = ['Model'] + list(metrics_data.keys())
            
            for model_name in model_names:
                row = [model_name]
                for metric_name in metrics_data.keys():
                    value = metrics_data[metric_name].get(model_name, 0)
                    row.append(f'{value:.4f}')
                table_data.append(row)
            
            table = ax7.table(cellText=table_data,
                            colLabels=headers,
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2.0)
            
            # Highlight best performers in each metric
            for col_idx, metric_name in enumerate(metrics_data.keys()):
                values = [metrics_data[metric_name].get(model, 0) for model in model_names]
                if metric_name.lower() in ['loss', 'error']:
                    best_row_idx = np.argmin(values) + 1  # +1 for header
                else:
                    best_row_idx = np.argmax(values) + 1  # +1 for header
                
                table[(best_row_idx, col_idx + 1)].set_facecolor('lightgreen')
                table[(best_row_idx, col_idx + 1)].set_text_props(weight='bold')
            
            plt.suptitle(f'Model Comparison Report\n'
                        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plt.savefig(self.save_dir / f"{save_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Model comparison report saved to {save_name}.png")
            
        except Exception as e:
            logger.error(f"Error generating model comparison report: {e}")
            raise 