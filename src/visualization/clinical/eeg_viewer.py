"""EEG signal viewer for clinical analysis."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EEGSignalViewer:
    """Visualizes EEG signals for clinical analysis."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir or Path("data/figures/clinical")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_eeg_signal(self, 
                       eeg_data: np.ndarray,
                       sample_rate: int = 500,
                       channel_names: Optional[List[str]] = None,
                       title: str = "EEG Signal",
                       save_name: Optional[str] = None) -> plt.Figure:
        """Plot EEG signal channels."""
        try:
            n_channels = eeg_data.shape[0] if eeg_data.ndim > 1 else 1
            if eeg_data.ndim == 1:
                eeg_data = eeg_data.reshape(1, -1)
                
            fig, axes = plt.subplots(n_channels, 1, figsize=(15, 2*n_channels), sharex=True)
            if n_channels == 1:
                axes = [axes]
                
            time_axis = np.arange(eeg_data.shape[1]) / sample_rate
            
            for i, ax in enumerate(axes):
                ax.plot(time_axis, eeg_data[i], linewidth=0.5)
                channel_name = channel_names[i] if channel_names else f"Channel {i+1}"
                ax.set_ylabel(f"{channel_name}\n(μV)")
                ax.grid(True, alpha=0.3)
                
            axes[-1].set_xlabel("Time (seconds)")
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting EEG signal: {e}")
            raise
    
    def plot_eeg_signals(self, 
                        eeg_data: np.ndarray,
                        channel_names: List[str],
                        sampling_rate: int,
                        duration: float = 5.0,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot EEG signals (plural - alias for compatibility with tests).
        
        Args:
            eeg_data: EEG data array (channels x timepoints)
            channel_names: List of channel names
            sampling_rate: Sampling rate in Hz
            duration: Duration to plot in seconds
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Calculate how many samples to plot
        samples_to_plot = int(duration * sampling_rate)
        if samples_to_plot > eeg_data.shape[1]:
            samples_to_plot = eeg_data.shape[1]
        
        # Truncate data to requested duration
        plot_data = eeg_data[:, :samples_to_plot]
        
        # Use the existing plot_eeg_signal method
        fig = self.plot_eeg_signal(
            eeg_data=plot_data,
            sample_rate=sampling_rate,
            channel_names=channel_names,
            title=f"EEG Signals ({duration}s)",
            save_name=None  # Don't save from internal call
        )
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved EEG signals plot to {save_path}")
        
        return fig
    
    def plot_topographic_map(self, 
                           topo_data: np.ndarray,
                           channel_names: List[str],
                           title: str = "EEG Topographic Map",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot topographic map of EEG data.
        
        Args:
            topo_data: Topographic data (one value per channel)
            channel_names: List of channel names
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create a simple scatter plot representation
            # This is a simplified version - real topographic maps need electrode positions
            n_channels = len(channel_names)
            
            # Create circular arrangement for channels
            angles = np.linspace(0, 2*np.pi, n_channels, endpoint=False)
            x = np.cos(angles)
            y = np.sin(angles)
            
            # Create scatter plot with color representing values
            scatter = ax.scatter(x, y, c=topo_data, s=200, cmap='RdBu_r', 
                               alpha=0.8, edgecolors='black')
            
            # Add channel labels
            for i, (xi, yi, name) in enumerate(zip(x, y, channel_names)):
                ax.annotate(name, (xi, yi), xytext=(0, 0), 
                           textcoords='offset points', ha='center', va='center',
                           fontsize=8, fontweight='bold')
            
            # Style the plot
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Amplitude (μV)', rotation=270, labelpad=15)
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved topographic map to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.warning(f"Topographic map plotting failed: {e}")
            # Return a simple figure if topographic plotting fails
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Topographic map not available\n(requires electrode positions)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            ax.axis('off')
            return fig 