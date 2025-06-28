#!/usr/bin/env python3
"""
Local test data preparation - creates mock data compatible with your pipeline
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

def create_test_data():
    """Create minimal test data that matches HMS format"""
    
    # Create directories
    os.makedirs('data/raw/train_eegs', exist_ok=True)
    os.makedirs('data/raw/train_spectrograms', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Create mock train.csv
    test_data = {
        'eeg_id': [f'100000{i:04d}' for i in range(100)],
        'expert_consensus': np.random.choice(['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other'], 100)
    }
    
    train_df = pd.DataFrame(test_data)
    train_df.to_csv('data/raw/train.csv', index=False)
    
    print(f"✅ Created mock train.csv with {len(train_df)} samples")
    
    # Create mock EEG files (parquet format as in your code)
    for eeg_id in train_df['eeg_id'].head(10):  # Just 10 files for testing
        # Mock EEG data: 20 channels x 10000 samples (50 seconds * 200 Hz)
        mock_eeg = np.random.randn(20, 10000).astype(np.float32)
        
        # Create DataFrame matching expected format
        eeg_df = pd.DataFrame(mock_eeg.T)  # Transpose to match expected format
        eeg_df.columns = [f'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
                         'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'EKG']
        
        # Save as parquet
        eeg_df.to_parquet(f'data/raw/train_eegs/{eeg_id}.parquet')
    
    print(f"✅ Created 10 mock EEG files")
    
    # Create mock spectrograms (just empty for now)
    for eeg_id in train_df['eeg_id'].head(5):
        mock_spec = np.random.rand(400, 300, 3).astype(np.uint8)  # Mock spectrogram
        np.save(f'data/raw/train_spectrograms/{eeg_id}.npy', mock_spec)
    
    print(f"✅ Created 5 mock spectrogram files")
    print(f"📁 Test data structure:")
    os.system('find data -type f | head -20')

if __name__ == "__main__":
    create_test_data()