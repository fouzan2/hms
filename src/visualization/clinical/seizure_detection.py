"""Seizure detection visualization for clinical analysis."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SeizureDetectionVisualizer:
    """Visualizes seizure detection results."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir or Path("data/figures/clinical")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_seizure_detection(self, 
                             eeg_data: np.ndarray,
                             detection_results: Dict,
                             title: str = "Seizure Detection",
                             save_name: Optional[str] = None) -> plt.Figure:
        """Plot EEG with seizure detection overlay."""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            time_axis = np.arange(len(eeg_data)) / 500  # Assuming 500 Hz
            
            # Plot EEG signal
            ax1.plot(time_axis, eeg_data, 'b-', linewidth=0.5)
            ax1.set_ylabel('EEG Amplitude (μV)')
            ax1.set_title('EEG Signal')
            ax1.grid(True, alpha=0.3)
            
            # Highlight seizure periods
            seizure_periods = detection_results.get('seizure_periods', [])
            for start, end in seizure_periods:
                ax1.axvspan(start, end, alpha=0.3, color='red', label='Seizure')
            
            # Plot detection confidence
            confidence = detection_results.get('confidence', np.zeros_like(time_axis))
            ax2.plot(time_axis, confidence, 'r-', linewidth=1)
            ax2.axhline(y=0.5, color='orange', linestyle='--', label='Threshold')
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Seizure Confidence')
            ax2.set_title('Seizure Detection Confidence')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting seizure detection: {e}")
            raise 