"""
Cross-validation strategies for EEG classification model evaluation.
Includes patient-wise, time-based, and stratified approaches.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Dict, List, Tuple, Optional, Union, Iterator
from sklearn.model_selection import (
    StratifiedKFold, KFold, TimeSeriesSplit,
    StratifiedGroupKFold, GroupKFold
)
from dataclasses import dataclass
import logging
from tqdm import tqdm
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CVFold:
    """Container for cross-validation fold data."""
    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


class CrossValidationStrategy:
    """
    Base class for cross-validation strategies.
    Handles data splitting for EEG classification tasks.
    """
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        
    def split(self, dataset: Dataset, labels: np.ndarray, 
              groups: Optional[np.ndarray] = None) -> Iterator[CVFold]:
        """
        Generate cross-validation splits.
        
        Args:
            dataset: PyTorch dataset
            labels: Target labels
            groups: Group labels (e.g., patient IDs)
            
        Yields:
            CVFold objects containing train/val indices
        """
        raise NotImplementedError("Subclasses must implement split method")
        
    def get_fold_statistics(self, dataset: Dataset, fold: CVFold) -> Dict:
        """Get statistics for a fold."""
        train_labels = [dataset[i]['label'] for i in fold.train_indices]
        val_labels = [dataset[i]['label'] for i in fold.val_indices]
        
        train_dist = np.bincount(train_labels)
        val_dist = np.bincount(val_labels)
        
        stats = {
            'n_train': len(fold.train_indices),
            'n_val': len(fold.val_indices),
            'train_class_distribution': train_dist.tolist(),
            'val_class_distribution': val_dist.tolist()
        }
        
        if fold.test_indices is not None:
            test_labels = [dataset[i]['label'] for i in fold.test_indices]
            test_dist = np.bincount(test_labels)
            stats['n_test'] = len(fold.test_indices)
            stats['test_class_distribution'] = test_dist.tolist()
            
        return stats


class StandardKFold(CrossValidationStrategy):
    """Standard k-fold cross-validation."""
    
    def split(self, dataset: Dataset, labels: np.ndarray,
              groups: Optional[np.ndarray] = None) -> Iterator[CVFold]:
        """Generate standard k-fold splits."""
        kf = KFold(n_splits=self.n_splits, shuffle=True, 
                   random_state=self.random_state)
        
        indices = np.arange(len(dataset))
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            yield CVFold(
                fold_idx=fold_idx,
                train_indices=train_idx,
                val_indices=val_idx,
                metadata={'strategy': 'standard_kfold'}
            )


class StratifiedKFoldCV(CrossValidationStrategy):
    """Stratified k-fold ensuring balanced class distribution."""
    
    def split(self, dataset: Dataset, labels: np.ndarray,
              groups: Optional[np.ndarray] = None) -> Iterator[CVFold]:
        """Generate stratified k-fold splits."""
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                             random_state=self.random_state)
        
        indices = np.arange(len(dataset))
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
            yield CVFold(
                fold_idx=fold_idx,
                train_indices=train_idx,
                val_indices=val_idx,
                metadata={'strategy': 'stratified_kfold'}
            )


class PatientWiseCV(CrossValidationStrategy):
    """
    Patient-wise cross-validation.
    Ensures no patient data appears in both train and validation sets.
    Critical for medical applications to avoid data leakage.
    """
    
    def split(self, dataset: Dataset, labels: np.ndarray,
              groups: Optional[np.ndarray] = None) -> Iterator[CVFold]:
        """Generate patient-wise splits."""
        if groups is None:
            raise ValueError("Patient IDs (groups) required for patient-wise CV")
            
        # Use GroupKFold for patient-wise splitting
        gkf = GroupKFold(n_splits=self.n_splits)
        indices = np.arange(len(dataset))
        
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(indices, labels, groups)):
            # Verify no patient overlap
            train_patients = set(groups[train_idx])
            val_patients = set(groups[val_idx])
            
            if train_patients & val_patients:
                logger.warning(f"Patient overlap detected in fold {fold_idx}")
                
            yield CVFold(
                fold_idx=fold_idx,
                train_indices=train_idx,
                val_indices=val_idx,
                metadata={
                    'strategy': 'patient_wise',
                    'n_train_patients': len(train_patients),
                    'n_val_patients': len(val_patients)
                }
            )


class StratifiedPatientCV(CrossValidationStrategy):
    """
    Stratified patient-wise cross-validation.
    Maintains class balance while ensuring patient separation.
    """
    
    def split(self, dataset: Dataset, labels: np.ndarray,
              groups: Optional[np.ndarray] = None) -> Iterator[CVFold]:
        """Generate stratified patient-wise splits."""
        if groups is None:
            raise ValueError("Patient IDs (groups) required for stratified patient CV")
            
        # Compute patient-level labels for stratification
        patient_labels = self._compute_patient_labels(labels, groups)
        
        # Use StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True,
                                   random_state=self.random_state)
        
        indices = np.arange(len(dataset))
        
        for fold_idx, (train_idx, val_idx) in enumerate(
            sgkf.split(indices, labels, groups)
        ):
            yield CVFold(
                fold_idx=fold_idx,
                train_indices=train_idx,
                val_indices=val_idx,
                metadata={
                    'strategy': 'stratified_patient',
                    'class_balance_maintained': True
                }
            )
            
    def _compute_patient_labels(self, labels: np.ndarray, 
                               groups: np.ndarray) -> Dict[str, int]:
        """Compute predominant label for each patient."""
        patient_labels = {}
        
        for patient_id in np.unique(groups):
            patient_mask = groups == patient_id
            patient_label_counts = np.bincount(labels[patient_mask])
            patient_labels[patient_id] = np.argmax(patient_label_counts)
            
        return patient_labels


class TimeBasedCV(CrossValidationStrategy):
    """
    Time-based cross-validation for temporal data.
    Ensures temporal order is preserved (train on past, validate on future).
    """
    
    def __init__(self, n_splits: int = 5, gap: int = 0):
        super().__init__(n_splits)
        self.gap = gap  # Gap between train and validation sets
        
    def split(self, dataset: Dataset, labels: np.ndarray,
              timestamps: np.ndarray) -> Iterator[CVFold]:
        """Generate time-based splits."""
        # Sort by timestamp
        time_order = np.argsort(timestamps)
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits, gap=self.gap)
        
        for fold_idx, (train_idx, val_idx) in enumerate(
            tscv.split(time_order)
        ):
            # Map back to original indices
            train_indices = time_order[train_idx]
            val_indices = time_order[val_idx]
            
            yield CVFold(
                fold_idx=fold_idx,
                train_indices=train_indices,
                val_indices=val_indices,
                metadata={
                    'strategy': 'time_based',
                    'train_time_range': (
                        timestamps[train_indices].min(),
                        timestamps[train_indices].max()
                    ),
                    'val_time_range': (
                        timestamps[val_indices].min(),
                        timestamps[val_indices].max()
                    )
                }
            )


class BlockedTimeSeriesCV(CrossValidationStrategy):
    """
    Blocked time series cross-validation.
    Creates contiguous time blocks for train/val splits.
    """
    
    def __init__(self, n_splits: int = 5, block_size: str = '1D'):
        super().__init__(n_splits)
        self.block_size = block_size  # e.g., '1D' for daily blocks
        
    def split(self, dataset: Dataset, labels: np.ndarray,
              timestamps: pd.DatetimeIndex) -> Iterator[CVFold]:
        """Generate blocked time series splits."""
        # Create time blocks
        blocks = pd.Grouper(freq=self.block_size)
        time_df = pd.DataFrame({
            'idx': np.arange(len(dataset)),
            'timestamp': timestamps,
            'label': labels
        })
        
        # Group by time blocks
        time_df['block'] = time_df.groupby(blocks).ngroup()
        unique_blocks = time_df['block'].unique()
        
        # Create folds from blocks
        block_folds = np.array_split(unique_blocks, self.n_splits)
        
        for fold_idx in range(self.n_splits):
            val_blocks = block_folds[fold_idx]
            train_blocks = np.concatenate(
                [block_folds[i] for i in range(self.n_splits) if i != fold_idx]
            )
            
            train_mask = time_df['block'].isin(train_blocks)
            val_mask = time_df['block'].isin(val_blocks)
            
            yield CVFold(
                fold_idx=fold_idx,
                train_indices=time_df[train_mask]['idx'].values,
                val_indices=time_df[val_mask]['idx'].values,
                metadata={
                    'strategy': 'blocked_time_series',
                    'block_size': self.block_size,
                    'n_train_blocks': len(train_blocks),
                    'n_val_blocks': len(val_blocks)
                }
            )


class NestedCV:
    """
    Nested cross-validation for hyperparameter tuning and evaluation.
    Provides unbiased performance estimates.
    """
    
    def __init__(self, 
                 outer_cv: CrossValidationStrategy,
                 inner_cv: CrossValidationStrategy):
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        
    def split(self, dataset: Dataset, labels: np.ndarray,
              groups: Optional[np.ndarray] = None) -> Iterator[Tuple[CVFold, List[CVFold]]]:
        """
        Generate nested CV splits.
        
        Yields:
            Tuples of (outer_fold, inner_folds)
        """
        for outer_fold in self.outer_cv.split(dataset, labels, groups):
            # Create subset for inner CV
            inner_dataset = Subset(dataset, outer_fold.train_indices)
            inner_labels = labels[outer_fold.train_indices]
            inner_groups = groups[outer_fold.train_indices] if groups is not None else None
            
            # Generate inner folds
            inner_folds = list(self.inner_cv.split(
                inner_dataset, inner_labels, inner_groups
            ))
            
            yield outer_fold, inner_folds


class MonteCarloCV(CrossValidationStrategy):
    """
    Monte Carlo cross-validation with random train/val splits.
    Useful for stability analysis.
    """
    
    def __init__(self, n_splits: int = 10, val_size: float = 0.2,
                 random_state: int = 42):
        super().__init__(n_splits, random_state)
        self.val_size = val_size
        
    def split(self, dataset: Dataset, labels: np.ndarray,
              groups: Optional[np.ndarray] = None) -> Iterator[CVFold]:
        """Generate Monte Carlo CV splits."""
        n_samples = len(dataset)
        n_val = int(n_samples * self.val_size)
        
        rng = np.random.RandomState(self.random_state)
        
        for fold_idx in range(self.n_splits):
            # Random permutation
            indices = rng.permutation(n_samples)
            
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]
            
            yield CVFold(
                fold_idx=fold_idx,
                train_indices=train_indices,
                val_indices=val_indices,
                metadata={
                    'strategy': 'monte_carlo',
                    'val_size': self.val_size,
                    'random_seed': self.random_state + fold_idx
                }
            )


class LeaveOnePatientOut(CrossValidationStrategy):
    """
    Leave-one-patient-out cross-validation.
    Each patient serves as validation set once.
    """
    
    def split(self, dataset: Dataset, labels: np.ndarray,
              groups: Optional[np.ndarray] = None) -> Iterator[CVFold]:
        """Generate leave-one-patient-out splits."""
        if groups is None:
            raise ValueError("Patient IDs (groups) required for LOPO CV")
            
        unique_patients = np.unique(groups)
        self.n_splits = len(unique_patients)
        
        for fold_idx, val_patient in enumerate(unique_patients):
            val_mask = groups == val_patient
            train_mask = ~val_mask
            
            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]
            
            yield CVFold(
                fold_idx=fold_idx,
                train_indices=train_indices,
                val_indices=val_indices,
                metadata={
                    'strategy': 'leave_one_patient_out',
                    'val_patient': val_patient,
                    'n_val_samples': len(val_indices)
                }
            )


class CrossValidationManager:
    """
    Manages cross-validation experiments.
    Handles fold creation, data loading, and result tracking.
    """
    
    def __init__(self, 
                 cv_strategy: CrossValidationStrategy,
                 dataset: Dataset,
                 batch_size: int = 32,
                 num_workers: int = 4):
        self.cv_strategy = cv_strategy
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Extract metadata
        self.labels = np.array([dataset[i]['label'] for i in range(len(dataset))])
        self.groups = None
        self.timestamps = None
        
        # Try to extract groups and timestamps
        try:
            self.groups = np.array([dataset[i]['patient_id'] for i in range(len(dataset))])
        except:
            pass
            
        try:
            self.timestamps = np.array([dataset[i]['timestamp'] for i in range(len(dataset))])
        except:
            pass
            
    def create_fold_loaders(self, fold: CVFold) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """Create data loaders for a fold."""
        # Create subsets
        train_subset = Subset(self.dataset, fold.train_indices)
        val_subset = Subset(self.dataset, fold.val_indices)
        
        # Create loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = None
        if fold.test_indices is not None:
            test_subset = Subset(self.dataset, fold.test_indices)
            test_loader = DataLoader(
                test_subset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
            
        return train_loader, val_loader, test_loader
    
    def run_cv_experiment(self, 
                          train_fn,
                          eval_fn,
                          model_init_fn,
                          save_dir: Path) -> Dict:
        """
        Run complete cross-validation experiment.
        
        Args:
            train_fn: Function to train model (model, train_loader, val_loader) -> trained_model
            eval_fn: Function to evaluate model (model, loader) -> metrics
            model_init_fn: Function to initialize model () -> model
            save_dir: Directory to save results
            
        Returns:
            Dictionary with CV results
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        # Generate folds based on strategy type
        if isinstance(self.cv_strategy, TimeBasedCV) and self.timestamps is not None:
            folds = self.cv_strategy.split(self.dataset, self.labels, self.timestamps)
        else:
            folds = self.cv_strategy.split(self.dataset, self.labels, self.groups)
            
        for fold in folds:
            logger.info(f"Processing fold {fold.fold_idx + 1}/{self.cv_strategy.n_splits}")
            
            # Create data loaders
            train_loader, val_loader, test_loader = self.create_fold_loaders(fold)
            
            # Initialize model
            model = model_init_fn()
            
            # Train model
            logger.info(f"Training fold {fold.fold_idx}")
            trained_model = train_fn(model, train_loader, val_loader)
            
            # Evaluate on validation set
            logger.info(f"Evaluating fold {fold.fold_idx}")
            val_metrics = eval_fn(trained_model, val_loader)
            
            # Evaluate on test set if available
            test_metrics = None
            if test_loader is not None:
                test_metrics = eval_fn(trained_model, test_loader)
                
            # Save fold results
            fold_results = {
                'fold_idx': fold.fold_idx,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'fold_metadata': fold.metadata,
                'fold_statistics': self.cv_strategy.get_fold_statistics(self.dataset, fold)
            }
            
            all_results.append(fold_results)
            
            # Save intermediate results
            with open(save_dir / f'fold_{fold.fold_idx}_results.json', 'w') as f:
                json.dump(fold_results, f, indent=2)
                
            # Save model checkpoint
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'fold_idx': fold.fold_idx,
                'val_metrics': val_metrics
            }, save_dir / f'fold_{fold.fold_idx}_model.pt')
            
        # Aggregate results
        aggregated_results = self._aggregate_cv_results(all_results)
        
        # Save final results
        with open(save_dir / 'cv_results.json', 'w') as f:
            json.dump({
                'strategy': self.cv_strategy.__class__.__name__,
                'n_splits': self.cv_strategy.n_splits,
                'fold_results': all_results,
                'aggregated_results': aggregated_results
            }, f, indent=2)
            
        return aggregated_results
    
    def _aggregate_cv_results(self, fold_results: List[Dict]) -> Dict:
        """Aggregate results across folds."""
        # Extract metric values
        metric_values = {}
        
        for fold in fold_results:
            for metric_name, metric_value in fold['val_metrics'].items():
                if metric_name not in metric_values:
                    metric_values[metric_name] = []
                    
                # Handle different metric types
                if isinstance(metric_value, (int, float)):
                    metric_values[metric_name].append(metric_value)
                elif hasattr(metric_value, 'value'):
                    metric_values[metric_name].append(metric_value.value)
                    
        # Compute statistics
        aggregated = {}
        for metric_name, values in metric_values.items():
            if values and all(isinstance(v, (int, float)) for v in values):
                aggregated[f'{metric_name}_mean'] = np.mean(values)
                aggregated[f'{metric_name}_std'] = np.std(values)
                aggregated[f'{metric_name}_min'] = np.min(values)
                aggregated[f'{metric_name}_max'] = np.max(values)
                aggregated[f'{metric_name}_values'] = values
                
        return aggregated 