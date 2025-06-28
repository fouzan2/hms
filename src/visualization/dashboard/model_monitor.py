"""Model monitoring dashboard for performance tracking."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelMonitor:
    """Monitors model performance and health."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir or Path("data/figures/dashboard")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def create_model_dashboard(self, 
                             model_metrics: Dict,
                             title: str = "Model Performance Monitor",
                             save_name: Optional[str] = None) -> plt.Figure:
        """Create model monitoring dashboard."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Performance metrics over time
            if 'performance_history' in model_metrics:
                history = model_metrics['performance_history']
                metrics = ['accuracy', 'precision', 'recall', 'f1_score']
                
                for metric in metrics:
                    if metric in history:
                        ax1.plot(history[metric], label=metric.title())
                
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Score')
                ax1.set_title('Performance Metrics Over Time')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Current metrics
            if 'current_metrics' in model_metrics:
                current = model_metrics['current_metrics']
                metric_names = list(current.keys())
                metric_values = list(current.values())
                bars = ax2.bar(metric_names, metric_values, color='skyblue')
                ax2.set_ylabel('Score')
                ax2.set_title('Current Performance Metrics')
                ax2.set_ylim(0, 1)
                ax2.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, metric_values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Model health indicators
            if 'health_indicators' in model_metrics:
                health = model_metrics['health_indicators']
                indicators = list(health.keys())
                values = list(health.values())
                colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in values]
                ax3.bar(indicators, values, color=colors)
                ax3.set_ylabel('Health Score')
                ax3.set_title('Model Health Indicators')
                ax3.set_ylim(0, 1)
                ax3.grid(True, alpha=0.3)
            
            # System information
            ax4.axis('off')
            info_text = "Model Information:\n\n"
            info_text += f"Model Type: {model_metrics.get('model_type', 'N/A')}\n"
            info_text += f"Version: {model_metrics.get('version', 'N/A')}\n"
            info_text += f"Last Updated: {model_metrics.get('last_updated', 'N/A')}\n"
            info_text += f"Predictions Made: {model_metrics.get('prediction_count', 'N/A')}\n"
            info_text += f"Uptime: {model_metrics.get('uptime', 'N/A')}\n"
            
            ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating model dashboard: {e}")
            raise 