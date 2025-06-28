"""Clinical alert visualization for medical monitoring."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ClinicalAlertVisualizer:
    """Visualizes clinical alerts and monitoring data."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir or Path("data/figures/clinical")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_alert_timeline(self, 
                           alerts: List[Dict],
                           title: str = "Clinical Alerts Timeline",
                           save_path: Optional[str] = None) -> plt.Figure:
        """Plot timeline of clinical alerts."""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            if not alerts:
                ax.text(0.5, 0.5, 'No alerts to display', 
                       ha='center', va='center', transform=ax.transAxes)
                return fig
                
            # Extract alert data
            times = [alert.get('timestamp', i) for i, alert in enumerate(alerts)]
            severity_colors = {'low': 'green', 'medium': 'orange', 'high': 'red'}
            colors = [severity_colors.get(alert.get('severity', 'low'), 'blue') for alert in alerts]
            
            # Plot alerts
            ax.scatter(times, range(len(alerts)), c=colors, s=100, alpha=0.7)
            
            # Add alert text
            for i, alert in enumerate(alerts):
                ax.text(times[i], i, alert.get('message', 'Alert'), 
                       ha='left', va='center', fontsize=8)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Alert Index')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting alert timeline: {e}")
            raise
    
    def create_alert_dashboard(self, 
                             alerts: List[Dict],
                             title: str = "Clinical Alert Dashboard",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive alert dashboard.
        
        Args:
            alerts: List of alert dictionaries
            title: Dashboard title
            save_path: Path to save the dashboard
            
        Returns:
            Matplotlib figure
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            if not alerts:
                # Show empty dashboard
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, 'No alerts to display', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('No Data')
                return fig
            
            # Extract alert data
            severity_counts = {'low': 0, 'medium': 0, 'high': 0}
            type_counts = {}
            timestamps = []
            
            for alert in alerts:
                # Count by severity
                severity = alert.get('severity', 'low')
                if severity in severity_counts:
                    severity_counts[severity] += 1
                
                # Count by type
                alert_type = alert.get('type', 'Unknown')
                type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
                
                # Collect timestamps
                if 'timestamp' in alert:
                    timestamps.append(alert['timestamp'])
            
            # Plot 1: Severity distribution
            severities = list(severity_counts.keys())
            counts = list(severity_counts.values())
            colors = ['green', 'orange', 'red']
            
            bars = ax1.bar(severities, counts, color=colors, alpha=0.7)
            ax1.set_title('Alert Severity Distribution')
            ax1.set_ylabel('Count')
            
            # Add count labels
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(count), ha='center', va='bottom', fontweight='bold')
            
            # Plot 2: Alert type distribution
            if type_counts:
                types = list(type_counts.keys())
                type_vals = list(type_counts.values())
                
                ax2.pie(type_vals, labels=types, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Alert Type Distribution')
            else:
                ax2.text(0.5, 0.5, 'No type data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Alert Types')
            
            # Plot 3: Timeline of alerts
            if timestamps:
                # Convert timestamps to hours for plotting
                try:
                    if isinstance(timestamps[0], str):
                        # Convert string timestamps to datetime
                        time_points = [datetime.fromisoformat(ts.replace('Z', '+00:00')) if 'Z' in ts 
                                     else datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') 
                                     for ts in timestamps]
                    else:
                        time_points = timestamps
                    
                    # Plot as scatter
                    severity_vals = [alert.get('severity', 'low') for alert in alerts]
                    severity_colors = {'low': 'green', 'medium': 'orange', 'high': 'red'}
                    colors = [severity_colors.get(sev, 'blue') for sev in severity_vals]
                    
                    ax3.scatter(time_points, range(len(alerts)), c=colors, alpha=0.7)
                    ax3.set_xlabel('Time')
                    ax3.set_ylabel('Alert Index')
                    ax3.set_title('Alert Timeline')
                    
                except Exception as e:
                    ax3.text(0.5, 0.5, f'Timeline error: {str(e)[:50]}...', 
                           ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title('Alert Timeline (Error)')
            else:
                ax3.text(0.5, 0.5, 'No timestamp data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Alert Timeline')
            
            # Plot 4: Alert summary statistics
            ax4.axis('off')
            total_alerts = len(alerts)
            high_priority = severity_counts.get('high', 0)
            recent_alerts = len([a for a in alerts[-10:]]) if len(alerts) > 10 else total_alerts
            
            summary_text = f"""
            ALERT SUMMARY
            
            Total Alerts: {total_alerts}
            High Priority: {high_priority}
            Recent (last 10): {recent_alerts}
            
            Alert Types:
            """
            
            for alert_type, count in type_counts.items():
                summary_text += f"• {alert_type}: {count}\n"
            
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Alert dashboard saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating alert dashboard: {e}")
            raise 