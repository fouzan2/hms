"""
Dataset classes for HMS brain activity classification.
Handles both raw EEG signals and spectrograms with patient-independent splits.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import h5py
import logging
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from collections import defaultdict
import random
import yaml

from ..preprocessing.eeg_preprocessor import EEGPreprocessor
from ..preprocessing.spectrogram_generator import SpectrogramGenerator

logger = logging.getLogger(__name__)


class HMSDataset(Dataset):
    """Main dataset class for HMS brain activity classification."""
    
    def __init__(self, 
                 data_path: str,
                 metadata_df: pd.DataFrame,
                 config: Dict,
                 mode: str = 'train',
                 transform_eeg: Optional[callable] = None,
                 transform_spectrogram: Optional[callable] = None,
                 preprocess: bool = True,
                 cache_preprocessed: bool = True):
        """
        Initialize HMS dataset.
        
        Args:
            data_path: Path to data directory
            metadata_df: DataFrame with file paths and labels
            config: Configuration dictionary
            mode: 'train', 'val', or 'test'
            transform_eeg: Transformations for EEG data
            transform_spectrogram: Transformations for spectrogram data
            preprocess: Whether to apply preprocessing
            cache_preprocessed: Whether to cache preprocessed data
        """
        self.data_path = Path(data_path)
        self.metadata_df = metadata_df
        self.config = config
        self.mode = mode
        self.transform_eeg = transform_eeg
        self.transform_spectrogram = transform_spectrogram
        self.preprocess = preprocess
        self.cache_preprocessed = cache_preprocessed
        
        # Initialize preprocessors
        if self.preprocess:
            self.eeg_preprocessor = EEGPreprocessor(config)
            self.spectrogram_generator = SpectrogramGenerator(config)
            
        # Cache for preprocessed data
        self.cache = {} if cache_preprocessed else None
        
        # Class mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(config['classes'])}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Load channel names
        self.channel_names = config['eeg']['channels']
        
        logger.info(f"Initialized {mode} dataset with {len(self.metadata_df)} samples")
        
    def __len__(self) -> int:
        return len(self.metadata_df)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        # Check cache first
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]
            
        # Get metadata
        row = self.metadata_df.iloc[idx]
        
        # Load EEG data
        eeg_data = self._load_eeg_data(row)
        
        # Load or generate spectrogram
        spectrogram_data = self._load_or_generate_spectrogram(row, eeg_data)
        
        # Preprocess if enabled
        if self.preprocess:
            eeg_data = self.eeg_preprocessor.preprocess_eeg(
                eeg_data, 
                self.channel_names,
                apply_ica=(self.mode == 'train')  # Only apply ICA during training
            )
            
        # Apply transforms
        if self.transform_eeg is not None:
            eeg_data = self.transform_eeg(eeg_data)
            
        if self.transform_spectrogram is not None:
            spectrogram_data = self.transform_spectrogram(spectrogram_data)
            
        # Convert to tensors
        eeg_tensor = torch.FloatTensor(eeg_data)
        spectrogram_tensor = torch.FloatTensor(spectrogram_data)
        
        # Get label
        label = self.class_to_idx[row['label']]
        label_tensor = torch.LongTensor([label])
        
        # Additional labels for multi-task learning
        is_seizure = 1 if row['label'] == 'Seizure' else 0
        has_artifact = row.get('has_artifact', 0)
        
        sample = {
            'eeg': eeg_tensor,
            'spectrogram': spectrogram_tensor,
            'label': label_tensor.squeeze(),
            'seizure_label': torch.FloatTensor([is_seizure]).squeeze(),
            'artifact_label': torch.FloatTensor([has_artifact]).squeeze(),
            'patient_id': row['patient_id'],
            'recording_id': row['recording_id'],
            'idx': idx
        }
        
        # Cache if enabled
        if self.cache is not None:
            self.cache[idx] = sample
            
        return sample
        
    def _load_eeg_data(self, row: pd.Series) -> np.ndarray:
        """Load EEG data from file."""
        eeg_path = self.data_path / row['eeg_path']
        
        if eeg_path.suffix == '.npy':
            data = np.load(eeg_path)
        elif eeg_path.suffix == '.h5':
            with h5py.File(eeg_path, 'r') as f:
                data = f['eeg'][:]
        else:
            raise ValueError(f"Unsupported file format: {eeg_path.suffix}")
            
        # Ensure correct shape (channels, time)
        if data.shape[0] > data.shape[1]:
            data = data.T
            
        return data
        
    def _load_or_generate_spectrogram(self, row: pd.Series, 
                                     eeg_data: np.ndarray) -> np.ndarray:
        """Load existing spectrogram or generate from EEG data."""
        if 'spectrogram_path' in row and pd.notna(row['spectrogram_path']):
            # Load existing spectrogram
            spec_path = self.data_path / row['spectrogram_path']
            
            if spec_path.suffix == '.npy':
                spectrogram = np.load(spec_path)
            elif spec_path.suffix == '.h5':
                with h5py.File(spec_path, 'r') as f:
                    spectrogram = f['spectrogram'][:]
            else:
                raise ValueError(f"Unsupported file format: {spec_path.suffix}")
                
        else:
            # Generate spectrogram from EEG
            spectrogram = self.spectrogram_generator.generate_multichannel_spectrogram(
                eeg_data, 
                method='stft'
            )
            
            # Create 3-channel representation for CNN
            spectrogram = self.spectrogram_generator.create_3d_spectrogram_representation(
                spectrogram,
                output_shape=(224, 224, 3)
            )
            
        return spectrogram
        
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalance."""
        class_counts = self.metadata_df['label'].value_counts()
        total_samples = len(self.metadata_df)
        
        weights = []
        for cls in self.config['classes']:
            count = class_counts.get(cls, 1)  # Avoid division by zero
            weight = total_samples / (len(self.config['classes']) * count)
            weights.append(weight)
            
        return torch.FloatTensor(weights)


class PatientGroupedSampler(Sampler):
    """Sampler that ensures all recordings from same patient are in same batch."""
    
    def __init__(self, metadata_df: pd.DataFrame, batch_size: int, 
                 shuffle: bool = True):
        self.metadata_df = metadata_df
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by patient
        self.patient_groups = defaultdict(list)
        for idx, row in metadata_df.iterrows():
            self.patient_groups[row['patient_id']].append(idx)
            
        self.patient_ids = list(self.patient_groups.keys())
        
    def __iter__(self):
        # Shuffle patients if required
        if self.shuffle:
            random.shuffle(self.patient_ids)
            
        # Create batches ensuring patient grouping
        batch = []
        for patient_id in self.patient_ids:
            patient_indices = self.patient_groups[patient_id]
            
            if self.shuffle:
                random.shuffle(patient_indices)
                
            for idx in patient_indices:
                batch.append(idx)
                
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    
        # Yield remaining samples
        if batch:
            yield batch
            
    def __len__(self):
        return (len(self.metadata_df) + self.batch_size - 1) // self.batch_size


class DataAugmentation:
    """Data augmentation for EEG signals and spectrograms."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.time_aug_config = config['training']['augmentation']['time_domain']
        self.freq_aug_config = config['training']['augmentation']['frequency_domain']
        
    def augment_eeg(self, eeg: np.ndarray) -> np.ndarray:
        """Apply time-domain augmentations to EEG signal."""
        augmented = eeg.copy()
        
        # Time shift
        if np.random.random() < 0.5:
            shift = int(self.time_aug_config['time_shift'] * eeg.shape[1])
            shift = np.random.randint(-shift, shift)
            augmented = np.roll(augmented, shift, axis=1)
            
        # Amplitude scaling
        if np.random.random() < 0.5:
            scale = 1 + np.random.uniform(
                -self.time_aug_config['scaling'],
                self.time_aug_config['scaling']
            )
            augmented = augmented * scale
            
        # Add jitter
        if np.random.random() < 0.5:
            jitter = np.random.normal(
                0, 
                self.time_aug_config['jitter'], 
                augmented.shape
            )
            augmented = augmented + jitter
            
        # Channel dropout
        if np.random.random() < 0.3:
            n_channels = augmented.shape[0]
            n_drop = np.random.randint(1, max(2, n_channels // 4))
            drop_channels = np.random.choice(n_channels, n_drop, replace=False)
            augmented[drop_channels] = 0
            
        return augmented
        
    def augment_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """Apply frequency-domain augmentations to spectrogram."""
        augmented = spectrogram.copy()
        
        # Frequency masking
        if np.random.random() < 0.5:
            freq_mask_size = int(
                self.freq_aug_config['freq_mask'] * augmented.shape[1]
            )
            if freq_mask_size > 0:
                start = np.random.randint(0, augmented.shape[1] - freq_mask_size)
                augmented[:, start:start + freq_mask_size, :] = 0
                
        # Time masking
        if np.random.random() < 0.5:
            time_mask_size = int(
                self.freq_aug_config['time_mask'] * augmented.shape[2]
            )
            if time_mask_size > 0:
                start = np.random.randint(0, augmented.shape[2] - time_mask_size)
                augmented[:, :, start:start + time_mask_size] = 0
                
        return augmented


def create_data_splits(metadata_path: str, config: Dict, 
                      n_folds: int = 5) -> Dict[str, pd.DataFrame]:
    """Create patient-independent train/val/test splits."""
    metadata_df = pd.read_csv(metadata_path)
    
    # Ensure patient_id exists
    if 'patient_id' not in metadata_df.columns:
        # Extract from recording ID or create dummy
        metadata_df['patient_id'] = metadata_df['recording_id'].str.split('_').str[0]
        
    # Create stratified group k-fold
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Get splits
    X = metadata_df.index
    y = metadata_df['label']
    groups = metadata_df['patient_id']
    
    splits = list(sgkf.split(X, y, groups))
    
    # Use first fold for train/val, last fold for test
    train_idx, val_idx = splits[0]
    test_idx = splits[-1][1]
    
    # Create split dataframes
    train_df = metadata_df.iloc[train_idx]
    val_df = metadata_df.iloc[val_idx]
    test_df = metadata_df.iloc[test_idx]
    
    # Log split statistics
    logger.info(f"Train: {len(train_df)} samples, {train_df['patient_id'].nunique()} patients")
    logger.info(f"Val: {len(val_df)} samples, {val_df['patient_id'].nunique()} patients")
    logger.info(f"Test: {len(test_df)} samples, {test_df['patient_id'].nunique()} patients")
    
    # Verify no patient overlap
    train_patients = set(train_df['patient_id'])
    val_patients = set(val_df['patient_id'])
    test_patients = set(test_df['patient_id'])
    
    assert len(train_patients & val_patients) == 0, "Patient overlap between train and val"
    assert len(train_patients & test_patients) == 0, "Patient overlap between train and test"
    assert len(val_patients & test_patients) == 0, "Patient overlap between val and test"
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


def create_dataloaders(config: Dict, 
                      batch_size: Optional[int] = None) -> Dict[str, DataLoader]:
    """Create dataloaders for train, validation, and test sets."""
    # Load metadata and create splits
    metadata_path = Path(config['dataset']['raw_data_path']) / 'train.csv'
    splits = create_data_splits(metadata_path, config)
    
    # Initialize augmentation
    augmentation = DataAugmentation(config)
    
    # Create datasets
    datasets = {}
    for split_name, split_df in splits.items():
        # Set transforms based on split
        if split_name == 'train':
            transform_eeg = augmentation.augment_eeg
            transform_spec = augmentation.augment_spectrogram
        else:
            transform_eeg = None
            transform_spec = None
            
        datasets[split_name] = HMSDataset(
            data_path=config['dataset']['raw_data_path'],
            metadata_df=split_df,
            config=config,
            mode=split_name,
            transform_eeg=transform_eeg,
            transform_spectrogram=transform_spec,
            preprocess=True,
            cache_preprocessed=(split_name != 'train')  # Cache val/test sets
        )
        
    # Create dataloaders
    if batch_size is None:
        batch_size = config['training']['batch_size']
        
    dataloaders = {}
    
    # Training dataloader with patient grouping
    train_sampler = PatientGroupedSampler(
        splits['train'], 
        batch_size, 
        shuffle=True
    )
    dataloaders['train'] = DataLoader(
        datasets['train'],
        batch_sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Val/test dataloaders
    for split in ['val', 'test']:
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
    return dataloaders 