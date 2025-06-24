#!/usr/bin/env python3
"""
Simplified data preparation script for HMS Brain Activity Classification.
This version does basic data validation and organization without complex preprocessing.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import yaml
from tqdm import tqdm
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleDataPreparer:
    """Simple data preparation for HMS dataset."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data preparer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.data_dir = Path(self.config['dataset']['raw_data_path'])
        self.processed_dir = Path(self.config['dataset']['processed_data_path'])
        
        # Create output directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_dataset(self) -> Dict:
        """Validate the dataset structure and files."""
        logger.info("Validating dataset structure...")
        
        validation_results = {
            'train_csv_exists': False,
            'test_csv_exists': False,
            'train_eegs_count': 0,
            'train_spectrograms_count': 0,
            'test_eegs_count': 0,
            'test_spectrograms_count': 0,
            'missing_files': []
        }
        
        # Check CSV files
        train_csv_path = self.data_dir / 'train.csv'
        test_csv_path = self.data_dir / 'test.csv'
        
        if train_csv_path.exists():
            validation_results['train_csv_exists'] = True
            train_df = pd.read_csv(train_csv_path)
            logger.info(f"Found train.csv with {len(train_df)} entries")
            
            # Check for train EEG files
            train_eeg_dir = self.data_dir / 'train_eegs'
            if train_eeg_dir.exists():
                eeg_files = list(train_eeg_dir.glob('*.parquet'))
                validation_results['train_eegs_count'] = len(eeg_files)
                logger.info(f"Found {len(eeg_files)} train EEG files")
                
                # Check for missing files
                expected_eeg_ids = set(train_df['eeg_id'].unique())
                found_eeg_ids = {int(f.stem) for f in eeg_files}
                missing_eeg_ids = expected_eeg_ids - found_eeg_ids
                
                if missing_eeg_ids:
                    validation_results['missing_files'].extend([f"train_eegs/{eid}.parquet" for eid in missing_eeg_ids])
                    logger.warning(f"Missing {len(missing_eeg_ids)} train EEG files")
            
            # Check for train spectrogram files
            train_spec_dir = self.data_dir / 'train_spectrograms'
            if train_spec_dir.exists():
                spec_files = list(train_spec_dir.glob('*.parquet'))
                validation_results['train_spectrograms_count'] = len(spec_files)
                logger.info(f"Found {len(spec_files)} train spectrogram files")
        
        if test_csv_path.exists():
            validation_results['test_csv_exists'] = True
            test_df = pd.read_csv(test_csv_path)
            logger.info(f"Found test.csv with {len(test_df)} entries")
            
            # Check for test files
            test_eeg_dir = self.data_dir / 'test_eegs'
            test_spec_dir = self.data_dir / 'test_spectrograms'
            
            if test_eeg_dir.exists():
                validation_results['test_eegs_count'] = len(list(test_eeg_dir.glob('*.parquet')))
            if test_spec_dir.exists():
                validation_results['test_spectrograms_count'] = len(list(test_spec_dir.glob('*.parquet')))
        
        # Save validation results
        with open(self.processed_dir / 'dataset_validation.yaml', 'w') as f:
            yaml.dump(validation_results, f)
        
        return validation_results
    
    def analyze_data_sample(self, n_samples: int = 5):
        """Analyze a sample of the data to understand structure."""
        logger.info(f"Analyzing {n_samples} sample files...")
        
        train_csv_path = self.data_dir / 'train.csv'
        train_df = pd.read_csv(train_csv_path)
        
        # Sample some files
        sample_df = train_df.head(n_samples)
        
        analysis_results = []
        
        for _, row in sample_df.iterrows():
            eeg_id = row['eeg_id']
            eeg_file = self.data_dir / 'train_eegs' / f'{eeg_id}.parquet'
            
            if eeg_file.exists():
                try:
                    # Read EEG data
                    eeg_df = pd.read_parquet(eeg_file)
                    
                    result = {
                        'eeg_id': eeg_id,
                        'label': row['expert_consensus'],
                        'shape': eeg_df.shape,
                        'columns': eeg_df.columns.tolist(),
                        'duration_seconds': eeg_df.shape[0] / 200,  # Assuming 200Hz sampling
                        'has_nulls': eeg_df.isnull().any().any(),
                        'data_ranges': {
                            col: {'min': float(eeg_df[col].min()), 
                                  'max': float(eeg_df[col].max()),
                                  'mean': float(eeg_df[col].mean())}
                            for col in eeg_df.columns[:5]  # First 5 channels
                        }
                    }
                    
                    analysis_results.append(result)
                    logger.info(f"Analyzed EEG {eeg_id}: shape={eeg_df.shape}, label={row['expert_consensus']}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing {eeg_file}: {str(e)}")
        
        # Save analysis results
        analysis_df = pd.DataFrame(analysis_results)
        analysis_df.to_csv(self.processed_dir / 'data_sample_analysis.csv', index=False)
        
        # Also save detailed results
        with open(self.processed_dir / 'data_sample_details.yaml', 'w') as f:
            yaml.dump(analysis_results, f)
        
        return analysis_results
    
    def create_train_val_split(self, val_split: float = 0.2, random_state: int = 42):
        """Create train/validation split maintaining class balance."""
        logger.info(f"Creating train/validation split with {val_split:.0%} validation")
        
        # Load train.csv
        train_csv_path = self.data_dir / 'train.csv'
        train_df = pd.read_csv(train_csv_path)
        
        # Get unique EEG IDs
        unique_eeg_ids = train_df['eeg_id'].unique()
        logger.info(f"Total unique EEG files: {len(unique_eeg_ids)}")
        
        # Create a mapping of eeg_id to label
        eeg_labels = train_df.groupby('eeg_id')['expert_consensus'].first()
        
        # Stratified split
        from sklearn.model_selection import train_test_split
        
        train_ids, val_ids = train_test_split(
            unique_eeg_ids,
            test_size=val_split,
            stratify=eeg_labels[unique_eeg_ids],
            random_state=random_state
        )
        
        # Save splits
        np.save(self.processed_dir / 'train_ids.npy', train_ids)
        np.save(self.processed_dir / 'val_ids.npy', val_ids)
        
        # Create split DataFrames
        train_split_df = train_df[train_df['eeg_id'].isin(train_ids)]
        val_split_df = train_df[train_df['eeg_id'].isin(val_ids)]
        
        train_split_df.to_csv(self.processed_dir / 'train_split.csv', index=False)
        val_split_df.to_csv(self.processed_dir / 'val_split.csv', index=False)
        
        logger.info(f"Train set: {len(train_ids)} files, {len(train_split_df)} total entries")
        logger.info(f"Validation set: {len(val_ids)} files, {len(val_split_df)} total entries")
        
        # Check class distribution
        train_labels = train_split_df['expert_consensus'].value_counts()
        val_labels = val_split_df['expert_consensus'].value_counts()
        
        logger.info(f"Train label distribution:\n{train_labels}")
        logger.info(f"Validation label distribution:\n{val_labels}")
        
        # Save split statistics
        split_stats = {
            'train_files': int(len(train_ids)),
            'val_files': int(len(val_ids)),
            'train_entries': int(len(train_split_df)),
            'val_entries': int(len(val_split_df)),
            'train_label_dist': train_labels.to_dict(),
            'val_label_dist': val_labels.to_dict()
        }
        
        with open(self.processed_dir / 'split_statistics.yaml', 'w') as f:
            yaml.dump(split_stats, f)


def main():
    """Main function to run simple data preparation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple HMS EEG data preparation")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--n-samples', type=int, default=5,
                       help='Number of samples to analyze')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Initialize preparer
    preparer = SimpleDataPreparer(config_path=args.config)
    
    # Validate dataset
    validation_results = preparer.validate_dataset()
    logger.info(f"Dataset validation complete: {validation_results}")
    
    # Analyze data samples
    if validation_results['train_csv_exists'] and validation_results['train_eegs_count'] > 0:
        preparer.analyze_data_sample(n_samples=args.n_samples)
    
    # Create train/val split
    if validation_results['train_csv_exists']:
        preparer.create_train_val_split(val_split=args.val_split)
    
    logger.info("Simple data preparation completed successfully!")


if __name__ == "__main__":
    main() 