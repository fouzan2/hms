"""
Data loader utilities for HMS brain activity classification.
Enterprise-grade data loading with balanced sampling and patient-independent splits.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from .dataset import HMSDataset, create_data_splits, create_dataloaders

logger = logging.getLogger(__name__)


def create_balanced_loaders(dataset: HMSDataset,
                          val_split: float = 0.15,
                          batch_size: int = 32,
                          num_workers: int = 4,
                          pin_memory: bool = True,
                          shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create balanced train and validation data loaders.
    
    Args:
        dataset: HMSDataset instance
        val_split: Fraction of data to use for validation
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Calculate split indices
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Create indices
    indices = list(range(total_size))
    
    if shuffle:
        np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    
    # Calculate class weights for balanced sampling
    train_labels = [dataset[idx]['label'].item() for idx in train_indices]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_labels]
    
    # Create weighted sampler for training
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_subset),
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    logger.info(f"Created balanced loaders: Train={len(train_subset)}, Val={len(val_subset)}")
    
    return train_loader, val_loader


def create_patient_balanced_loaders(metadata_df: pd.DataFrame,
                                  config: Dict,
                                  val_split: float = 0.15,
                                  batch_size: int = 32,
                                  num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Create patient-balanced train and validation data loaders.
    Ensures no patient overlap between train and validation sets.
    
    Args:
        metadata_df: DataFrame with patient information
        config: Configuration dictionary
        val_split: Fraction of patients to use for validation
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get unique patients
    unique_patients = metadata_df['patient_id'].unique()
    np.random.shuffle(unique_patients)
    
    # Split patients
    val_patient_count = int(len(unique_patients) * val_split)
    val_patients = unique_patients[:val_patient_count]
    train_patients = unique_patients[val_patient_count:]
    
    # Split data
    train_df = metadata_df[metadata_df['patient_id'].isin(train_patients)]
    val_df = metadata_df[metadata_df['patient_id'].isin(val_patients)]
    
    # Create datasets
    train_dataset = HMSDataset(
        data_path=config['dataset']['processed_data_path'],
        metadata_df=train_df,
        config=config,
        mode='train'
    )
    
    val_dataset = HMSDataset(
        data_path=config['dataset']['processed_data_path'],
        metadata_df=val_df,
        config=config,
        mode='val'
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f"Created patient-balanced loaders: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    return train_loader, val_loader


def create_cross_validation_loaders(metadata_df: pd.DataFrame,
                                  config: Dict,
                                  n_folds: int = 5,
                                  batch_size: int = 32,
                                  num_workers: int = 4) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Create cross-validation data loaders.
    
    Args:
        metadata_df: DataFrame with patient information
        config: Configuration dictionary
        n_folds: Number of cross-validation folds
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        
    Returns:
        List of (train_loader, val_loader) tuples for each fold
    """
    from sklearn.model_selection import StratifiedGroupKFold
    
    # Create cross-validation splits
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    loaders = []
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(
        metadata_df, 
        metadata_df['label'], 
        groups=metadata_df['patient_id']
    )):
        train_df = metadata_df.iloc[train_idx]
        val_df = metadata_df.iloc[val_idx]
        
        # Create datasets
        train_dataset = HMSDataset(
            data_path=config['dataset']['processed_data_path'],
            metadata_df=train_df,
            config=config,
            mode='train'
        )
        
        val_dataset = HMSDataset(
            data_path=config['dataset']['processed_data_path'],
            metadata_df=val_df,
            config=config,
            mode='val'
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        loaders.append((train_loader, val_loader))
        
        logger.info(f"Created fold {fold + 1}: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    return loaders


# Export the main classes and functions
__all__ = [
    'HMSDataset',
    'create_balanced_loaders',
    'create_patient_balanced_loaders',
    'create_cross_validation_loaders',
    'create_data_splits',
    'create_dataloaders'
] 