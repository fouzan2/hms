"""System metrics monitoring for dashboard."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SystemMetricsMonitor:
    """Monitors system performance and resource usage."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir or Path("data/figures/dashboard")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def create_system_dashboard(self, 
                              system_metrics: Dict,
                              title: str = "System Metrics Monitor",
                              save_name: Optional[str] = None) -> plt.Figure:
        """Create system metrics monitoring dashboard."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # CPU and Memory usage over time
            if 'resource_history' in system_metrics:
                history = system_metrics['resource_history']
                time_points = range(len(history.get('cpu_usage', [])))
                
                if 'cpu_usage' in history:
                    ax1.plot(time_points, history['cpu_usage'], 'b-', label='CPU %', linewidth=2)
                if 'memory_usage' in history:
                    ax1.plot(time_points, history['memory_usage'], 'r-', label='Memory %', linewidth=2)
                
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Usage (%)')
                ax1.set_title('Resource Usage Over Time')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(0, 100)
            
            # Current resource usage
            if 'current_usage' in system_metrics:
                usage = system_metrics['current_usage']
                resources = list(usage.keys())
                values = list(usage.values())
                colors = ['red' if v > 80 else 'orange' if v > 60 else 'green' for v in values]
                bars = ax2.bar(resources, values, color=colors)
                ax2.set_ylabel('Usage (%)')
                ax2.set_title('Current Resource Usage')
                ax2.set_ylim(0, 100)
                ax2.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Processing statistics
            if 'processing_stats' in system_metrics:
                stats = system_metrics['processing_stats']
                stat_names = list(stats.keys())
                stat_values = list(stats.values())
                ax3.bar(stat_names, stat_values, color='lightgreen')
                ax3.set_ylabel('Count/Rate')
                ax3.set_title('Processing Statistics')
                ax3.grid(True, alpha=0.3)
            
            # System status
            ax4.axis('off')
            status_text = "System Status:\n\n"
            
            if 'system_info' in system_metrics:
                info = system_metrics['system_info']
                status_text += f"Status: {info.get('status', 'Unknown')}\n"
                status_text += f"Uptime: {info.get('uptime', 'N/A')}\n"
                status_text += f"Total Memory: {info.get('total_memory', 'N/A')} GB\n"
                status_text += f"CPU Cores: {info.get('cpu_cores', 'N/A')}\n"
                status_text += f"GPU Available: {info.get('gpu_available', 'N/A')}\n"
                status_text += f"Active Processes: {info.get('active_processes', 'N/A')}\n"
            
            # Color based on system health
            health_score = system_metrics.get('health_score', 0.5)
            status_color = 'lightgreen' if health_score > 0.8 else 'yellow' if health_score > 0.6 else 'lightcoral'
            
            ax4.text(0.1, 0.9, status_text, transform=ax4.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=status_color, alpha=0.8))
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_name:
                plt.savefig(self.save_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating system dashboard: {e}")
            raise 