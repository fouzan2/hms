"""
Confusion Matrix Visualizer for HMS EEG Classification System

This module provides comprehensive confusion matrix visualization including:
- Multi-class confusion matrices with percentages
- Per-class performance metrics
- Interactive heatmaps
- Clinical significance highlighting
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfusionMatrixVisualizer:
    """Create and analyze confusion matrices for EEG classification."""
    
    # EEG activity classes
    CLASSES = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
    
    # Clinical significance matrix (how critical misclassifications are)
    CLINICAL_WEIGHTS = {
        ('Seizure', 'Other'): 5.0,  # Missing seizure is very critical
        ('Seizure', 'LPD'): 3.0,
        ('Seizure', 'GPD'): 3.0,
        ('LPD', 'Other'): 3.0,
        ('GPD', 'Other'): 3.0,
        ('LRDA', 'Other'): 2.0,
        ('GRDA', 'Other'): 2.0,
    }
    
    def __init__(self, 
                 class_names: Optional[List[str]] = None,
                 clinical_mode: bool = True):
        """
        Initialize confusion matrix visualizer.
        
        Args:
            class_names: Custom class names
            clinical_mode: Enable clinical significance highlighting
        """
        self.class_names = class_names or self.CLASSES
        self.clinical_mode = clinical_mode
        self.cm = None
        self.y_true = None
        self.y_pred = None
        
    def compute_matrix(self, 
                      y_true: Union[List, np.ndarray],
                      y_pred: Union[List, np.ndarray],
                      normalize: str = 'true') -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization mode ('true', 'pred', 'all', None)
            
        Returns:
            Confusion matrix
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        
        # Compute confusion matrix
        self.cm = confusion_matrix(y_true, y_pred)
        
        # Normalize if requested
        if normalize == 'true':
            self.cm = self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]
        elif normalize == 'pred':
            self.cm = self.cm.astype('float') / self.cm.sum(axis=0)
        elif normalize == 'all':
            self.cm = self.cm.astype('float') / self.cm.sum()
        
        return self.cm
    
    def plot_matrix(self, 
                   normalize: str = 'true',
                   title: Optional[str] = None,
                   cmap: str = 'Blues',
                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create static confusion matrix visualization.
        
        Args:
            normalize: Normalization mode
            title: Plot title
            cmap: Colormap
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if self.cm is None:
            raise ValueError("No confusion matrix computed. Call compute_matrix() first.")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create annotation matrix
        if normalize:
            annot = np.array([[f'{self.cm[i,j]:.2%}\n({int(self.cm[i,j]*self.cm.sum()):d})' 
                              for j in range(len(self.class_names))] 
                             for i in range(len(self.class_names))])
        else:
            annot = self.cm.astype(int)
        
        # Create heatmap
        sns.heatmap(self.cm, 
                   annot=annot if not normalize else False,
                   fmt='' if normalize else 'd',
                   cmap=cmap,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8},
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax)
        
        # Add custom annotations for normalized view
        if normalize:
            for i in range(len(self.class_names)):
                for j in range(len(self.class_names)):
                    text = ax.text(j + 0.5, i + 0.5, annot[i, j],
                                 ha="center", va="center",
                                 color="white" if self.cm[i, j] > 0.5 else "black",
                                 fontsize=9)
        
        # Highlight clinical significance
        if self.clinical_mode:
            self._add_clinical_highlights(ax)
        
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        if title is None:
            title = f'Confusion Matrix ({normalize} normalized)' if normalize else 'Confusion Matrix'
        ax.set_title(title, fontsize=14, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        return fig
    
    def _add_clinical_highlights(self, ax):
        """Add clinical significance highlights to confusion matrix."""
        for (true_class, pred_class), weight in self.CLINICAL_WEIGHTS.items():
            if true_class in self.class_names and pred_class in self.class_names:
                i = self.class_names.index(true_class)
                j = self.class_names.index(pred_class)
                
                # Add border around critical cells
                rect = plt.Rectangle((j, i), 1, 1, 
                                   fill=False, 
                                   edgecolor='red', 
                                   linewidth=3*weight/5,
                                   alpha=0.7)
                ax.add_patch(rect)
    
    def create_interactive_matrix(self, 
                                normalize: str = 'true',
                                title: Optional[str] = None) -> go.Figure:
        """
        Create interactive Plotly confusion matrix.
        
        Args:
            normalize: Normalization mode
            title: Plot title
            
        Returns:
            Plotly figure
        """
        if self.cm is None:
            raise ValueError("No confusion matrix computed. Call compute_matrix() first.")
        
        # Prepare data
        if normalize:
            z = self.cm
            text = [[f'{self.cm[i,j]:.2%}<br>Count: {int(self.cm[i,j]*self.cm.sum()):d}' 
                    for j in range(len(self.class_names))] 
                   for i in range(len(self.class_names))]
        else:
            z = self.cm
            text = [[str(val) for val in row] for row in self.cm]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=self.class_names,
            y=self.class_names,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorscale='Blues',
            showscale=True,
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Value: %{z}<br>%{text}<extra></extra>"
        ))
        
        # Add clinical significance annotations
        if self.clinical_mode:
            shapes = []
            annotations = []
            
            for (true_class, pred_class), weight in self.CLINICAL_WEIGHTS.items():
                if true_class in self.class_names and pred_class in self.class_names:
                    i = self.class_names.index(true_class)
                    j = self.class_names.index(pred_class)
                    
                    shapes.append(dict(
                        type='rect',
                        x0=j-0.5, x1=j+0.5,
                        y0=i-0.5, y1=i+0.5,
                        line=dict(color='red', width=3*weight/5),
                        fillcolor='rgba(255,0,0,0)'
                    ))
                    
                    if weight >= 4:
                        annotations.append(dict(
                            x=j, y=i,
                            text='⚠️',
                            showarrow=False,
                            font=dict(size=20, color='red'),
                            xshift=20, yshift=-20
                        ))
            
            fig.update_layout(shapes=shapes, annotations=annotations)
        
        # Update layout
        if title is None:
            title = f'Interactive Confusion Matrix ({normalize} normalized)' if normalize else 'Interactive Confusion Matrix'
        
        fig.update_layout(
            title=title,
            xaxis=dict(title='Predicted Label', side='bottom'),
            yaxis=dict(title='True Label', autorange='reversed'),
            width=800,
            height=700,
            template='plotly_white'
        )
        
        return fig
    
    def get_class_metrics(self) -> pd.DataFrame:
        """
        Calculate per-class performance metrics.
        
        Returns:
            DataFrame with metrics for each class
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("No predictions available. Call compute_matrix() first.")
        
        # Get classification report
        report = classification_report(self.y_true, self.y_pred, 
                                     target_names=self.class_names,
                                     output_dict=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(report).transpose()
        
        # Add additional metrics
        for i, class_name in enumerate(self.class_names):
            # True positives, false positives, false negatives
            tp = self.cm[i, i]
            fp = self.cm[:, i].sum() - tp
            fn = self.cm[i, :].sum() - tp
            tn = self.cm.sum() - tp - fp - fn
            
            # Additional metrics
            df.loc[class_name, 'sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            df.loc[class_name, 'specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            df.loc[class_name, 'ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            df.loc[class_name, 'npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # Clinical significance score
            if self.clinical_mode:
                clinical_score = 0
                for j in range(len(self.class_names)):
                    if i != j:
                        weight = self.CLINICAL_WEIGHTS.get(
                            (class_name, self.class_names[j]), 1.0
                        )
                        clinical_score += self.cm[i, j] * weight
                df.loc[class_name, 'clinical_impact'] = clinical_score
        
        return df
    
    def plot_class_performance(self, 
                             metrics: List[str] = ['precision', 'recall', 'f1-score'],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot per-class performance metrics.
        
        Args:
            metrics: Metrics to plot
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        df = self.get_class_metrics()
        
        # Filter to requested metrics and classes only
        class_df = df.loc[self.class_names, metrics]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create grouped bar chart
        x = np.arange(len(self.class_names))
        width = 0.8 / len(metrics)
        
        for i, metric in enumerate(metrics):
            offset = (i - len(metrics)/2) * width + width/2
            bars = ax.bar(x + offset, class_df[metric], width, 
                         label=metric.capitalize(), alpha=0.8)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Brain Activity Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved class performance plot to {save_path}")
        
        return fig
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive confusion matrix analysis report.
        
        Args:
            output_path: Path to save report
            
        Returns:
            Report text
        """
        if self.cm is None:
            raise ValueError("No confusion matrix computed. Call compute_matrix() first.")
        
        report = "HMS EEG Classification - Confusion Matrix Analysis\n"
        report += "="*50 + "\n\n"
        
        # Overall accuracy
        accuracy = np.trace(self.cm) / self.cm.sum()
        report += f"Overall Accuracy: {accuracy:.2%}\n\n"
        
        # Per-class metrics
        report += "Per-Class Performance:\n"
        report += "-"*30 + "\n"
        
        df = self.get_class_metrics()
        for class_name in self.class_names:
            report += f"\n{class_name}:\n"
            report += f"  Precision: {df.loc[class_name, 'precision']:.2%}\n"
            report += f"  Recall: {df.loc[class_name, 'recall']:.2%}\n"
            report += f"  F1-Score: {df.loc[class_name, 'f1-score']:.2%}\n"
            report += f"  Support: {int(df.loc[class_name, 'support'])}\n"
            
            if self.clinical_mode and 'clinical_impact' in df.columns:
                report += f"  Clinical Impact: {df.loc[class_name, 'clinical_impact']:.2f}\n"
        
        # Most confused pairs
        report += "\n\nMost Confused Pairs:\n"
        report += "-"*30 + "\n"
        
        # Get off-diagonal elements
        confusion_pairs = []
        for i in range(len(self.class_names)):
            for j in range(len(self.class_names)):
                if i != j and self.cm[i, j] > 0:
                    confusion_pairs.append((
                        self.class_names[i],
                        self.class_names[j],
                        self.cm[i, j],
                        self.CLINICAL_WEIGHTS.get((self.class_names[i], self.class_names[j]), 1.0)
                    ))
        
        # Sort by frequency
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        
        for true_class, pred_class, count, clinical_weight in confusion_pairs[:10]:
            report += f"{true_class} → {pred_class}: {count}"
            if self.clinical_mode and clinical_weight > 1:
                report += f" (Clinical Weight: {clinical_weight})"
            report += "\n"
        
        # Clinical recommendations
        if self.clinical_mode:
            report += "\n\nClinical Recommendations:\n"
            report += "-"*30 + "\n"
            
            # Check for critical misclassifications
            seizure_recall = df.loc['Seizure', 'recall'] if 'Seizure' in self.class_names else 1.0
            if seizure_recall < 0.9:
                report += "⚠️  Seizure detection recall is below 90%. "
                report += "Consider adjusting decision threshold.\n"
            
            # Check for high false positive rate
            if 'Seizure' in self.class_names:
                seizure_precision = df.loc['Seizure', 'precision']
                if seizure_precision < 0.7:
                    report += "⚠️  High false positive rate for seizure detection. "
                    report += "May lead to unnecessary interventions.\n"
        
        # Save report
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Saved confusion matrix report to {output_path}")
        
        return report
    
    def plot_confusion_matrix(self, 
                            cm: np.ndarray,
                            class_names: List[str],
                            normalize: str = 'true',
                            title: Optional[str] = None,
                            cmap: str = 'Blues',
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Create confusion matrix visualization (alias for compatibility with tests).
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            normalize: Normalization mode
            title: Plot title
            cmap: Colormap
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Set the confusion matrix and class names
        self.cm = cm
        self.class_names = class_names
        
        # Call the existing plot_matrix method
        return self.plot_matrix(
            normalize=normalize,
            title=title,
            cmap=cmap,
            save_path=save_path
        )
        
    def plot_normalized_confusion_matrix(self,
                                       cm: np.ndarray,
                                       class_names: List[str],
                                       title: Optional[str] = None,
                                       cmap: str = 'Blues',
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Create normalized confusion matrix visualization.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            title: Plot title
            cmap: Colormap
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        return self.plot_confusion_matrix(
            cm=cm,
            class_names=class_names,
            normalize='true',
            title=title or 'Normalized Confusion Matrix',
            cmap=cmap,
            save_path=save_path
        ) 