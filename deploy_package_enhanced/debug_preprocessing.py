#!/usr/bin/env python3
"""
Debug script to identify the preprocessing issue.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Set up paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.preprocessing.eeg_preprocessor import EEGPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_preprocessing():
    """Debug the preprocessing step by step."""
    
    # Load config
    config_path = "config/novita_enhanced_config.yaml"
    
    # Load a sample EEG file
    eeg_file = Path("data/raw/train_eegs/1000913311.parquet")
    eeg_data = pd.read_parquet(eeg_file)
    
    print(f"EEG data shape: {eeg_data.shape}")
    print(f"EEG columns: {eeg_data.columns.tolist()}")
    
    # Convert to numpy array
    eeg_array = eeg_data.values.T  # Transpose to get channels as first dimension
    print(f"EEG array shape after transpose: {eeg_array.shape}")
    
    # Get channel names from config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    channel_names = config['eeg']['channels']
    print(f"Channel names: {channel_names}")
    print(f"Number of channels: {len(channel_names)}")
    
    # Check if shapes match
    print(f"EEG array shape[0]: {eeg_array.shape[0]}")
    print(f"Channel names length: {len(channel_names)}")
    print(f"Shapes match: {eeg_array.shape[0] == len(channel_names)}")
    
    # Initialize preprocessor
    print("\nInitializing EEGPreprocessor...")
    eeg_preprocessor = EEGPreprocessor(config_path)
    
    # Test preprocessing
    print("\nTesting preprocessing...")
    try:
        preprocessed_eeg, info = eeg_preprocessor.preprocess_eeg(eeg_array, channel_names)
        print(f"✅ Preprocessing successful!")
        print(f"Preprocessed shape: {preprocessed_eeg.shape}")
        print(f"Processing info: {info}")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_preprocessing() 