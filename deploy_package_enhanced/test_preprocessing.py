#!/usr/bin/env python3
"""
Test script for preprocessing to verify GPU compatibility and fix serialization issues.
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Set up paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.preprocessing.eeg_preprocessor import EEGPreprocessor
from src.preprocessing.spectrogram_generator import SpectrogramGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_preprocessing():
    """Test preprocessing with a small batch to verify it works."""
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # Load config
    config_path = "config/novita_enhanced_config.yaml"
    
    try:
        # Initialize preprocessors
        logger.info("Initializing EEG preprocessor...")
        eeg_preprocessor = EEGPreprocessor(config_path)
        
        logger.info("Initializing spectrogram generator...")
        spectrogram_generator = SpectrogramGenerator(config_path)
        
        # Create dummy EEG data (20 channels, 10000 samples)
        logger.info("Creating dummy EEG data...")
        dummy_eeg = np.random.randn(20, 10000) * 0.1  # Small amplitude noise
        
        # Get channel names from config
        channel_names = [
            "Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
            "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG"
        ]
        
        # Test single EEG preprocessing
        logger.info("Testing single EEG preprocessing...")
        try:
            preprocessed_eeg, info = eeg_preprocessor.preprocess_eeg(dummy_eeg, channel_names)
            logger.info(f"✅ Single EEG preprocessing successful. Output shape: {preprocessed_eeg.shape}")
            logger.info(f"Preprocessing info: {info}")
        except Exception as e:
            logger.error(f"❌ Single EEG preprocessing failed: {e}")
            return False
        
        # Test spectrogram generation
        logger.info("Testing spectrogram generation...")
        try:
            spectrogram = spectrogram_generator.generate_multichannel_spectrogram(preprocessed_eeg)
            logger.info(f"✅ Spectrogram generation successful. Output shape: {spectrogram.shape}")
        except Exception as e:
            logger.error(f"❌ Spectrogram generation failed: {e}")
            return False
        
        # Test batch processing (small batch)
        logger.info("Testing batch processing...")
        try:
            eeg_batch = [dummy_eeg, dummy_eeg * 0.5, dummy_eeg * 2.0]  # 3 samples
            
            # Process sequentially (avoiding multiprocessing)
            processed_batch = []
            for eeg_data in eeg_batch:
                preprocessed, _ = eeg_preprocessor.preprocess_eeg(eeg_data, channel_names)
                processed_batch.append(preprocessed)
            
            # Generate spectrograms
            spectrograms = spectrogram_generator.generate_batch(processed_batch)
            
            logger.info(f"✅ Batch processing successful. Processed {len(processed_batch)} samples")
            logger.info(f"Spectrograms shape: {[spec.shape for spec in spectrograms]}")
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            return False
        
        logger.info("🎉 All preprocessing tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_preprocessing()
    if success:
        print("✅ Preprocessing test completed successfully!")
        sys.exit(0)
    else:
        print("❌ Preprocessing test failed!")
        sys.exit(1) 