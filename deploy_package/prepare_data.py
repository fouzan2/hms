#!/usr/bin/env python3
"""
Data preparation script for HMS Brain Activity Classification.
Uses comprehensive preprocessing pipeline to prepare EEG data for training.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import yaml
from tqdm import tqdm
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from preprocessing import (
    MultiFormatEEGReader,
    SignalQualityAssessor,
    EEGFilter,
    EEGFeatureExtractor,
    SpectrogramGenerator,
    read_eeg_file,
    assess_signal_quality,
    create_eeg_filter
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreparer:
    """Comprehensive data preparation pipeline."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data preparer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.data_dir = Path(self.config['dataset']['raw_data_path'])
        self.processed_dir = Path(self.config['dataset']['processed_data_path'])
        self.sampling_rate = self.config['dataset']['eeg_sampling_rate']
        
        # Initialize components
        self.eeg_reader = MultiFormatEEGReader(sampling_rate=self.sampling_rate)
        self.quality_assessor = SignalQualityAssessor(
            sampling_rate=self.sampling_rate,
            config=self.config.get('signal_quality', {})
        )
        self.eeg_filter = create_eeg_filter(
            sampling_rate=self.sampling_rate,
            config=self.config.get('preprocessing', {})
        )
        self.feature_extractor = EEGFeatureExtractor(
            sampling_rate=self.sampling_rate,
            config=self.config.get('features', {})
        )
        self.spectrogram_generator = SpectrogramGenerator(config_path=config_path)
        
        # Create output directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        (self.processed_dir / 'eeg').mkdir(exist_ok=True)
        (self.processed_dir / 'spectrograms').mkdir(exist_ok=True)
        (self.processed_dir / 'features').mkdir(exist_ok=True)
        (self.processed_dir / 'quality_reports').mkdir(exist_ok=True)
        
    def load_metadata(self) -> pd.DataFrame:
        """Load and prepare metadata."""
        logger.info("Loading metadata...")
        
        # Load train.csv
        metadata_path = self.data_dir / 'train.csv'
        metadata = pd.read_csv(metadata_path)
        
        # Rename columns to match expected names
        metadata = metadata.rename(columns={
            'eeg_id': 'file_id',
            'expert_consensus': 'label'
        })
        
        logger.info(f"Loaded {len(metadata)} samples")
        logger.info(f"Classes distribution:\n{metadata['label'].value_counts()}")
        
        return metadata
        
    def process_eeg_file(self, 
                        file_path: Path,
                        metadata_row: pd.Series) -> Dict:
        """Process a single EEG file through the complete pipeline."""
        try:
            # 1. Read EEG file
            eeg_data, file_metadata = self.eeg_reader.read(file_path)
            
            # 2. Assess signal quality
            quality_metrics = self.quality_assessor.assess_quality(
                eeg_data,
                channel_names=file_metadata.get('channel_names')
            )
            
            # Log quality info
            logger.info(f"Signal quality score: {quality_metrics.overall_quality_score:.2f}")
            
            # 3. Filter and clean signal
            # Get bad channel indices
            bad_channel_indices = []
            if quality_metrics.bad_channels:
                channel_names = file_metadata.get('channel_names', [])
                for bad_ch in quality_metrics.bad_channels:
                    if bad_ch in channel_names:
                        bad_channel_indices.append(channel_names.index(bad_ch))
            
            # Apply comprehensive filtering
            filtered_data = self.eeg_filter.apply_filters(
                eeg_data,
                bad_channels=bad_channel_indices
            )
            
            # 4. Segment signal into windows
            window_length = self.config['preprocessing']['window_length']
            overlap = self.config['preprocessing'].get('overlap', 0.5)
            
            segments = self.eeg_filter.segment_signal(
                filtered_data,
                segment_length=window_length,
                overlap=overlap
            )
            
            # 5. Extract features from each segment
            all_features = []
            all_spectrograms = []
            
            for segment in segments:
                # Extract features
                feature_set = self.feature_extractor.extract_features(
                    segment,
                    channel_names=file_metadata.get('channel_names')
                )
                all_features.append(feature_set.feature_vector)
                
                # Generate spectrogram
                spectrogram = self.spectrogram_generator.generate_multichannel_spectrogram(segment)
                all_spectrograms.append(spectrogram)
            
            # 6. Prepare output
            result = {
                'file_id': metadata_row['file_id'],
                'label': metadata_row['label'],
                'filtered_eeg': filtered_data,
                'segments': segments,
                'features': np.array(all_features),
                'spectrograms': np.array(all_spectrograms),
                'quality_metrics': quality_metrics,
                'metadata': file_metadata,
                'n_segments': len(segments)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return None
            
    def save_processed_data(self, processed_data: Dict, output_prefix: str):
        """Save processed data to disk."""
        file_id = processed_data['file_id']
        
        # Save filtered EEG
        eeg_path = self.processed_dir / 'eeg' / f'{file_id}_filtered.npy'
        np.save(eeg_path, processed_data['filtered_eeg'])
        
        # Save features
        features_path = self.processed_dir / 'features' / f'{file_id}_features.npy'
        np.save(features_path, processed_data['features'])
        
        # Save spectrograms
        spec_path = self.processed_dir / 'spectrograms' / f'{file_id}_spectrograms.npy'
        np.save(spec_path, processed_data['spectrograms'])
        
        # Save quality report
        quality_report = {
            'file_id': file_id,
            'overall_quality_score': processed_data['quality_metrics'].overall_quality_score,
            'snr': processed_data['quality_metrics'].snr,
            'artifact_ratio': processed_data['quality_metrics'].artifact_ratio,
            'bad_channels': processed_data['quality_metrics'].bad_channels,
            'n_segments': processed_data['n_segments']
        }
        
        quality_path = self.processed_dir / 'quality_reports' / f'{file_id}_quality.joblib'
        joblib.dump(quality_report, quality_path)
        
    def process_dataset(self, 
                       max_samples: Optional[int] = None,
                       n_jobs: int = 1):
        """Process the entire dataset."""
        # Load metadata
        metadata = self.load_metadata()
        
        if max_samples:
            metadata = metadata.head(max_samples)
            logger.info(f"Processing only {max_samples} samples")
        
        # Process each file
        successful = 0
        failed = 0
        
        # Create summary DataFrame
        summary_data = []
        
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing EEG files"):
            # Construct file path
            file_path = self.data_dir / 'train_eegs' / f"{row['file_id']}.parquet"
            
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                failed += 1
                continue
                
            # Process file
            result = self.process_eeg_file(file_path, row)
            
            if result:
                # Save processed data
                self.save_processed_data(result, f"{row['file_id']}")
                
                # Add to summary
                summary_data.append({
                    'file_id': row['file_id'],
                    'label': row['label'],
                    'quality_score': result['quality_metrics'].overall_quality_score,
                    'n_segments': result['n_segments'],
                    'n_bad_channels': len(result['quality_metrics'].bad_channels),
                    'artifact_ratio': result['quality_metrics'].artifact_ratio
                })
                
                successful += 1
            else:
                failed += 1
                
        # Save summary
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.processed_dir / 'processing_summary.csv', index=False)
        else:
            logger.warning("No data to save in processing summary")
        
        logger.info(f"Processing completed: {successful} successful, {failed} failed")
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            logger.info(f"Summary statistics:")
            logger.info(f"Average quality score: {summary_df['quality_score'].mean():.3f}")
            logger.info(f"Average segments per file: {summary_df['n_segments'].mean():.1f}")
        else:
            logger.warning("No files were processed successfully")
        
        # Save preprocessing statistics
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            stats = {
                'total_processed': successful,
                'total_failed': failed,
                'quality_stats': summary_df['quality_score'].describe().to_dict(),
                'segment_stats': summary_df['n_segments'].describe().to_dict(),
                'label_distribution': summary_df['label'].value_counts().to_dict()
            }
        else:
            stats = {
                'total_processed': successful,
                'total_failed': failed,
                'quality_stats': {},
                'segment_stats': {},
                'label_distribution': {}
            }
        
        with open(self.processed_dir / 'preprocessing_stats.yaml', 'w') as f:
            yaml.dump(stats, f)
            
    def create_train_val_split(self, val_split: float = 0.2, random_state: int = 42):
        """Create train/validation split maintaining class balance when possible."""
        logger.info(f"Creating train/validation split with {val_split:.0%} validation")
        
        # Check if processing summary exists
        summary_path = self.processed_dir / 'processing_summary.csv'
        if not summary_path.exists() or summary_path.stat().st_size == 0:
            logger.error("No processing summary found. Please process some files successfully first.")
            return
        
        # Load processing summary
        summary_df = pd.read_csv(summary_path)
        
        if len(summary_df) == 0:
            logger.error("Processing summary is empty. No files were processed successfully.")
            return
            
        # Check if we have enough samples for splitting
        if len(summary_df) < 5:  # Need at least 5 samples for meaningful split
            logger.warning(f"Only {len(summary_df)} samples available. Need at least 5 for train/val split.")
            logger.info("All samples will be used for training.")
            # Save all as training
            np.save(self.processed_dir / 'train_ids.npy', summary_df['file_id'].values)
            np.save(self.processed_dir / 'val_ids.npy', np.array([]))
            return
        
        # Check class distribution for stratification
        class_counts = summary_df['label'].value_counts()
        min_class_count = class_counts.min()
        
        logger.info(f"Class distribution: {class_counts.to_dict()}")
        logger.info(f"Minimum class count: {min_class_count}")
        
        # Stratified split
        from sklearn.model_selection import train_test_split
        
        # Try stratified split if all classes have at least 2 samples
        if min_class_count >= 2:
            try:
                logger.info("Using stratified train/validation split")
                train_ids, val_ids = train_test_split(
                    summary_df['file_id'].values,
                    test_size=val_split,
                    stratify=summary_df['label'].values,
                    random_state=random_state
                )
            except ValueError as e:
                logger.warning(f"Stratified split failed: {e}")
                logger.info("Falling back to random split")
                train_ids, val_ids = train_test_split(
                    summary_df['file_id'].values,
                    test_size=val_split,
                    random_state=random_state
                )
        else:
            logger.warning(f"Some classes have only {min_class_count} sample(s). Using random split instead of stratified.")
            train_ids, val_ids = train_test_split(
                summary_df['file_id'].values,
                test_size=val_split,
                random_state=random_state
            )
        
        # Save splits
        np.save(self.processed_dir / 'train_ids.npy', train_ids)
        np.save(self.processed_dir / 'val_ids.npy', val_ids)
        
        logger.info(f"Train set: {len(train_ids)} samples")
        logger.info(f"Validation set: {len(val_ids)} samples")
        
        # Check class distribution
        train_df = summary_df[summary_df['file_id'].isin(train_ids)]
        val_df = summary_df[summary_df['file_id'].isin(val_ids)]
        
        train_labels = train_df['label'].value_counts()
        val_labels = val_df['label'].value_counts()
        
        logger.info(f"Train label distribution:\n{train_labels}")
        logger.info(f"Validation label distribution:\n{val_labels}")


def main():
    """Main function to run data preparation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare HMS EEG data for training")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel jobs')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Initialize preparer
    preparer = DataPreparer(config_path=args.config)
    
    # Process dataset
    preparer.process_dataset(
        max_samples=args.max_samples,
        n_jobs=args.n_jobs
    )
    
    # Create train/val split only if processing was successful
    summary_path = preparer.processed_dir / 'processing_summary.csv'
    if summary_path.exists() and summary_path.stat().st_size > 0:
        preparer.create_train_val_split(val_split=args.val_split)
    else:
        logger.warning("Skipping train/val split as no files were processed successfully")
    
    logger.info("Data preparation completed!")


if __name__ == "__main__":
    main() 