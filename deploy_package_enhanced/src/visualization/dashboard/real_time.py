"""Real-time monitoring dashboard for EEG analysis."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class RealTimeMonitor:
    """Real-time monitoring dashboard for EEG data."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir or Path("data/figures/dashboard")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def create_realtime_dashboard(self, 
                                current_data: Dict,
                                title: str = "Real-time EEG Monitor",
                                save_name: Optional[str] = None) -> plt.Figure:
        """Create real-time monitoring dashboard."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Current EEG signal
            if 'eeg_signal' in current_data:
                eeg_data = current_data['eeg_signal']
                time_axis = np.arange(len(eeg_data)) / 500
                ax1.plot(time_axis, eeg_data, 'b-', linewidth=1)
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Amplitude (μV)')
                ax1.set_title('Current EEG Signal')
                ax1.grid(True, alpha=0.3)
            
            # Current predictions
            if 'predictions' in current_data:
                predictions = current_data['predictions']
                classes = list(predictions.keys())
                probs = list(predictions.values())
                bars = ax2.bar(classes, probs, color=['red' if p > 0.5 else 'green' for p in probs])
                ax2.set_ylabel('Probability')
                ax2.set_title('Current Predictions')
                ax2.set_ylim(0, 1)
                ax2.grid(True, alpha=0.3)
                
                # Add threshold line
                ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7)
            
            # Signal quality metrics
            if 'quality_metrics' in current_data:
                metrics = current_data['quality_metrics']
                metric_names = list(metrics.keys())
                metric_values = list(metrics.values())
                ax3.bar(metric_names, metric_values, color='lightblue')
                ax3.set_ylabel('Quality Score')
                ax3.set_title('Signal Quality Metrics')
                ax3.set_ylim(0, 1)
                ax3.grid(True, alpha=0.3)
            
            # System status
            ax4.axis('off')
            status_text = "System Status:\n\n"
            status_text += f"Recording: {'Active' if current_data.get('recording', False) else 'Inactive'}\n"
            status_text += f"Sample Rate: {current_data.get('sample_rate', 500)} Hz\n"
            status_text += f"Channels: {current_data.get('num_channels', 'N/A')}\n"
            status_text += f"Buffer Size: {current_data.get('buffer_size', 'N/A')}\n"
            
            status_color = 'lightgreen' if current_data.get('recording', False) else 'lightcoral'
            ax4.text(0.1, 0.9, status_text, transform=ax4.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=status_color, alpha=0.8))
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating real-time dashboard: {e}")
            raise 