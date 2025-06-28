"""Temporal analysis visualization for EEG data."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TemporalAnalysisVisualizer:
    """Visualizes temporal patterns in EEG data."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir or Path("data/figures/clinical")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_temporal_patterns(self, 
                             temporal_data: Dict,
                             title: str = "Temporal Analysis",
                             save_name: Optional[str] = None) -> plt.Figure:
        """Plot temporal patterns and trends."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Power over time
            if 'power_timeline' in temporal_data:
                power_data = temporal_data['power_timeline']
                time_axis = np.arange(len(power_data))
                ax1.plot(time_axis, power_data)
                ax1.set_xlabel('Time Windows')
                ax1.set_ylabel('Power')
                ax1.set_title('Power Over Time')
                ax1.grid(True, alpha=0.3)
            
            # Frequency bands
            if 'frequency_bands' in temporal_data:
                bands = temporal_data['frequency_bands']
                band_names = list(bands.keys())
                band_powers = list(bands.values())
                ax2.bar(band_names, band_powers)
                ax2.set_ylabel('Power')
                ax2.set_title('Frequency Band Power')
                ax2.grid(True, alpha=0.3)
            
            # Spectogram (if available)
            if 'spectrogram' in temporal_data:
                spec_data = temporal_data['spectrogram']
                im = ax3.imshow(spec_data, aspect='auto', origin='lower')
                ax3.set_xlabel('Time')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Spectrogram')
                plt.colorbar(im, ax=ax3)
            
            # Statistics
            ax4.axis('off')
            stats_text = "Temporal Analysis Summary:\n\n"
            if 'mean_power' in temporal_data:
                stats_text += f"Mean Power: {temporal_data['mean_power']:.3f}\n"
            if 'power_variance' in temporal_data:
                stats_text += f"Power Variance: {temporal_data['power_variance']:.3f}\n"
            
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                    fontsize=12, verticalalignment='top')
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting temporal analysis: {e}")
            raise 