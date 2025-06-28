#!/usr/bin/env python3
"""
Generate Sample Patient Report for HMS EEG Classification System
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from visualization.clinical.patient_report import PatientReportGenerator

def generate_sample_report():
    """Generate a sample patient report."""
    print("🏥 Generating Sample Patient Report...")
    
    # Create patient report generator
    generator = PatientReportGenerator(save_dir=Path("data/figures/clinical"))
    
    # Mock patient data
    patient_data = {
        'patient_id': 'HMS_P001',
        'age': 34,
        'gender': 'Female',
        'recording_date': '2024-12-27',
        'diagnosis': 'Epilepsy monitoring',
        'duration_minutes': 120,
        'medications': ['Levetiracetam', 'Phenytoin']
    }
    
    # Mock analysis results
    analysis_results = {
        'predictions': ['Seizure', 'LPD', 'Other', 'GPD', 'GRDA'],
        'confidences': [0.95, 0.78, 0.65, 0.43, 0.21],
        'timestamps': ['10:15:30', '10:45:12', '11:20:45', '11:35:20', '12:05:15'],
        'total_seizures': 2,
        'seizure_duration': 87.3,
        'quality_score': 'Excellent (0.92)'
    }
    
    # Mock EEG data (19 channels, 5000 time points = 10 seconds at 500Hz)
    np.random.seed(42)  # For reproducible demo data
    eeg_data = np.random.randn(19, 5000) * 50  # 50 μV amplitude
    
    # Add some realistic EEG patterns
    # Alpha waves (8-12 Hz) in occipital channels
    time = np.linspace(0, 10, 5000)
    alpha_wave = 30 * np.sin(2 * np.pi * 10 * time)
    eeg_data[8:10] += alpha_wave  # O1, O2 channels
    
    # Beta waves (13-30 Hz) in frontal channels
    beta_wave = 15 * np.sin(2 * np.pi * 20 * time)
    eeg_data[0:2] += beta_wave  # Fp1, Fp2 channels
    
    # Simulate seizure-like activity in temporal channels
    seizure_start = 2000  # 4 seconds in
    seizure_duration = 1500  # 3 seconds
    seizure_freq = 5  # 5 Hz spike-wave
    seizure_wave = 100 * np.sin(2 * np.pi * seizure_freq * time[seizure_start:seizure_start+seizure_duration])
    eeg_data[10:14, seizure_start:seizure_start+seizure_duration] += seizure_wave  # Temporal channels
    
    # Standard 10-20 electrode system channel names
    channel_names = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
    ]
    
    # Generate HTML report
    print("📄 Generating HTML report...")
    html_path = "data/figures/clinical/sample_patient_report.html"
    generator.generate_patient_report(
        patient_data, analysis_results, eeg_data, 
        channel_names, save_path=html_path
    )
    
    # Generate PNG report 
    print("🖼️ Generating PNG report...")
    png_path = "data/figures/clinical/sample_patient_report.png"
    generator.generate_patient_report(
        patient_data, analysis_results, eeg_data, 
        channel_names, save_path=png_path
    )
    
    print(f"\n✅ Sample patient reports generated successfully!")
    print(f"📁 HTML Report: {Path(html_path).absolute()}")
    print(f"📁 PNG Report: {Path(png_path).absolute()}")
    print(f"\n🌐 To view the HTML report, open it in your web browser:")
    print(f"   file://{Path(html_path).absolute()}")

if __name__ == "__main__":
    generate_sample_report() 