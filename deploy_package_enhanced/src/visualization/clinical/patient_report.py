"""Patient report generation for clinical visualization."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PatientReportGenerator:
    """Generates comprehensive patient reports for clinical use."""
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize patient report generator.
        
        Args:
            save_dir: Directory to save reports
        """
        self.save_dir = save_dir or Path("data/figures/clinical")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def generate_patient_report(self, 
                              patient_data: Dict[str, Any],
                              analysis_results: Dict[str, Any],
                              eeg_data: np.ndarray,
                              channel_names: List[str],
                              save_path: str = "patient_report") -> None:
        """
        Generate comprehensive patient report.
        
        Args:
            patient_data: Patient information and metrics
            analysis_results: Analysis results from the model
            eeg_data: EEG signal data
            channel_names: List of channel names
            save_path: Path to save report (changed from save_name)
        """
        try:
            fig = plt.figure(figsize=(20, 16))
            
            # Patient Information Table
            ax1 = plt.subplot(3, 2, 1)
            ax1.axis('tight')
            ax1.axis('off')
            
            patient_info = [
                ['Patient ID', patient_data.get('patient_id', 'N/A')],
                ['Age', str(patient_data.get('age', 'N/A'))],
                ['Gender', patient_data.get('gender', 'N/A')],
                ['Condition', patient_data.get('condition', patient_data.get('diagnosis', 'N/A'))],
                ['Recording Date', patient_data.get('recording_date', 'N/A')],
                ['Duration (min)', str(patient_data.get('duration_minutes', 'N/A'))],
            ]
            
            table = ax1.table(cellText=patient_info,
                            colLabels=['Parameter', 'Value'],
                            cellLoc='left',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            ax1.set_title('Patient Information', fontsize=14, fontweight='bold')
            
            # EEG Signal Plot
            ax2 = plt.subplot(3, 2, 2)
            time_axis = np.arange(len(eeg_data[0])) / 500  # Assuming 500 Hz
            sample_length = min(5000, eeg_data.shape[1])  # First 10 seconds
            ax2.plot(time_axis[:sample_length], eeg_data[0][:sample_length])  # Plot first channel
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Amplitude (μV)')
            ax2.set_title('EEG Signal Sample (First 10 seconds)')
            ax2.grid(True, alpha=0.3)
            
            # Predictions
            if analysis_results and 'predictions' in analysis_results:
                ax3 = plt.subplot(3, 2, 3)
                predictions = analysis_results['predictions']
                confidences = analysis_results.get('confidences', [1.0] * len(predictions))
                
                pred_names = predictions[:5] if len(predictions) > 5 else predictions  # Top 5
                pred_values = confidences[:5] if len(confidences) > 5 else confidences
                
                colors = sns.color_palette("viridis", len(pred_names))
                bars = ax3.bar(range(len(pred_names)), pred_values, color=colors)
                ax3.set_xlabel('Condition')
                ax3.set_ylabel('Confidence')
                ax3.set_title('Model Predictions')
                ax3.set_xticks(range(len(pred_names)))
                ax3.set_xticklabels(pred_names, rotation=45)
                ax3.set_ylim(0, 1)
                
                # Add value labels
                for bar, value in zip(bars, pred_values):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Report Summary
            ax4 = plt.subplot(3, 1, 3)
            ax4.axis('off')
            
            summary_text = f"""
            CLINICAL EEG ANALYSIS REPORT
            
            Patient: {patient_data.get('patient_id', 'N/A')}
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            FINDINGS:
            • EEG recording duration: {patient_data.get('duration_minutes', 'N/A')} minutes
            • Signal quality: {analysis_results.get('quality_score', 'Good')}
            • Total seizures detected: {analysis_results.get('total_seizures', 0)}
            """
            
            if analysis_results and 'seizure_duration' in analysis_results:
                summary_text += f"• Seizure duration: {analysis_results['seizure_duration']:.1f} seconds\n"
            
            if analysis_results and 'predictions' in analysis_results:
                predictions = analysis_results['predictions']
                confidences = analysis_results.get('confidences', [])
                if predictions and confidences:
                    max_idx = confidences.index(max(confidences))
                    max_pred = predictions[max_idx]
                    max_conf = confidences[max_idx]
                    summary_text += f"• Primary classification: {max_pred} ({max_conf:.1%} confidence)\n"
            
            summary_text += """
            RECOMMENDATIONS:
            • Clinical correlation recommended
            • Follow-up as clinically indicated
            • Report reviewed by qualified neurologist
            
            Note: This automated analysis is for clinical decision support only.
            Final interpretation should always be made by qualified medical personnel.
            """
            
            ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
            
            plt.suptitle('Clinical EEG Analysis Report', fontsize=18, fontweight='bold')
            plt.tight_layout()
            
            # Handle save_path - could be full path or just filename
            save_path_str = str(save_path)  # Convert to string to handle PosixPath objects
            
            # Check if user wants HTML format
            if save_path_str.endswith('.html'):
                # Generate HTML report instead of image
                self._generate_html_report(patient_data, analysis_results, save_path_str)
            else:
                # Save as image using matplotlib
                # Matplotlib supported formats
                supported_formats = ['.png', '.pdf', '.eps', '.svg', '.ps', '.jpg', '.jpeg', '.tiff', '.tif']
                
                # Check if save_path has a supported extension
                has_supported_ext = any(save_path_str.lower().endswith(ext) for ext in supported_formats)
                
                if has_supported_ext:
                    save_file = save_path_str
                else:
                    # Default to PNG if no supported extension
                    save_file = f"{save_path_str}.png"
                
                # Extract directory and filename
                save_path_obj = Path(save_file)
                if save_path_obj.parent != Path('.'):
                    # Full path provided
                    plt.savefig(save_file, dpi=300, bbox_inches='tight')
                else:
                    # Just filename provided, use save_dir
                    plt.savefig(self.save_dir / save_file, dpi=300, bbox_inches='tight')
            
            plt.close()
            
            logger.info(f"Patient report saved to {save_path_str}")
            
        except Exception as e:
            logger.error(f"Error generating patient report: {e}")
            raise
    
    def _generate_html_report(self, patient_data: Dict[str, Any], analysis_results: Dict[str, Any], save_path: str):
        """Generate HTML version of the patient report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Clinical EEG Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; }}
                .section {{ margin: 20px 0; }}
                .info-table {{ width: 100%; border-collapse: collapse; }}
                .info-table th, .info-table td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
                .info-table th {{ background-color: #f2f2f2; }}
                .predictions {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                .summary {{ background-color: #e6f3ff; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Clinical EEG Analysis Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Patient Information</h2>
                <table class="info-table">
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Patient ID</td><td>{patient_data.get('patient_id', 'N/A')}</td></tr>
                    <tr><td>Age</td><td>{patient_data.get('age', 'N/A')}</td></tr>
                    <tr><td>Gender</td><td>{patient_data.get('gender', 'N/A')}</td></tr>
                    <tr><td>Condition</td><td>{patient_data.get('condition', patient_data.get('diagnosis', 'N/A'))}</td></tr>
                    <tr><td>Recording Date</td><td>{patient_data.get('recording_date', 'N/A')}</td></tr>
                    <tr><td>Duration (min)</td><td>{patient_data.get('duration_minutes', 'N/A')}</td></tr>
                </table>
            </div>
        """
        
        # Add predictions section if available
        if analysis_results and 'predictions' in analysis_results:
            predictions = analysis_results['predictions']
            confidences = analysis_results.get('confidences', [1.0] * len(predictions))
            
            html_content += """
            <div class="section">
                <h2>Model Predictions</h2>
                <div class="predictions">
            """
            
            for pred, conf in zip(predictions[:5], confidences[:5]):  # Top 5
                html_content += f"<p><strong>{pred}:</strong> {conf:.1%} confidence ({conf})</p>"
            
            html_content += "</div></div>"
        
        # Add summary section
        html_content += f"""
            <div class="section">
                <h2>Clinical Summary</h2>
                <div class="summary">
                    <p><strong>Patient:</strong> {patient_data.get('patient_id', 'N/A')}</p>
                    <p><strong>EEG recording duration:</strong> {patient_data.get('duration_minutes', 'N/A')} minutes</p>
                    <p><strong>Signal quality:</strong> {analysis_results.get('quality_score', 'Good')}</p>
                    <p><strong>Total seizures detected:</strong> {analysis_results.get('total_seizures', 0)}</p>
        """
        
        if analysis_results and 'seizure_duration' in analysis_results:
            html_content += f"<p><strong>Seizure duration:</strong> {analysis_results['seizure_duration']:.1f} seconds</p>"
        
        if analysis_results and 'predictions' in analysis_results:
            predictions = analysis_results['predictions']
            confidences = analysis_results.get('confidences', [])
            if predictions and confidences:
                max_idx = confidences.index(max(confidences))
                max_pred = predictions[max_idx]
                max_conf = confidences[max_idx]
                html_content += f"<p><strong>Primary classification:</strong> {max_pred} ({max_conf:.1%} confidence)</p>"
        
        html_content += """
                    <h3>Recommendations:</h3>
                    <ul>
                        <li>Clinical correlation recommended</li>
                        <li>Follow-up as clinically indicated</li>
                        <li>Report reviewed by qualified neurologist</li>
                    </ul>
                    <p><em>Note: This automated analysis is for clinical decision support only.
                    Final interpretation should always be made by qualified medical personnel.</em></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save HTML file
        with open(save_path, 'w') as f:
            f.write(html_content) 