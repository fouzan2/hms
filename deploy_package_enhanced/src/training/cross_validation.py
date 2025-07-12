"""
Cross-validation framework for EEG classification.
Implements patient-independent splits and temporal validation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, TimeSeriesSplit,
    StratifiedGroupKFold
)
from typing import Dict, List, Tuple, Optional, Iterator, Union
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from collections import defaultdict, Counter
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class PatientIndependentCV:
    """Patient-independent cross-validation to prevent data leakage."""
    
    def __init__(self, n_splits: int = 5, stratify: bool = True, 
                 random_state: int = 42):
        self.n_splits = n_splits
        self.stratify = stratify
        self.random_state = random_state
        
    def split(self, X: np.ndarray, y: np.ndarray, 
              groups: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate patient-independent train/val splits.
        
        Args:
            X: Features array
            y: Labels array
            groups: Patient IDs array
            
        Yields:
            Train and validation indices
        """
        if self.stratify:
            # Use StratifiedGroupKFold to maintain class distribution
            cv = StratifiedGroupKFold(n_splits=self.n_splits, 
                                     shuffle=True,
                                     random_state=self.random_state)
            for train_idx, val_idx in cv.split(X, y, groups):
                yield train_idx, val_idx
        else:
            # Simple GroupKFold
            cv = GroupKFold(n_splits=self.n_splits)
            for train_idx, val_idx in cv.split(X, y, groups):
                yield train_idx, val_idx
                
    def get_patient_stats(self, y: np.ndarray, groups: np.ndarray) -> Dict:
        """Get statistics about patient distribution."""
        patient_labels = defaultdict(list)
        for label, patient in zip(y, groups):
            patient_labels[patient].append(label)
            
        stats = {
            'n_patients': len(patient_labels),
            'labels_per_patient': {
                patient: Counter(labels) 
                for patient, labels in patient_labels.items()
            },
            'patients_per_class': defaultdict(set)
        }
        
        for patient, labels in patient_labels.items():
            for label in set(labels):
                stats['patients_per_class'][label].add(patient)
                
        return stats


class TimeSeriesCV:
    """Time-series aware cross-validation for temporal data."""
    
    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None,
                 gap: int = 0):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap  # Gap between train and test to avoid leakage
        
    def split(self, X: np.ndarray, timestamps: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time-based train/val splits.
        
        Args:
            X: Features array
            timestamps: Timestamps array
            
        Yields:
            Train and validation indices
        """
        # Sort by timestamp
        sorted_indices = np.argsort(timestamps)
        n_samples = len(X)
        
        if self.test_size is None:
            self.test_size = n_samples // (self.n_splits + 1)
            
        tscv = TimeSeriesSplit(n_splits=self.n_splits, 
                              test_size=self.test_size,
                              gap=self.gap)
        
        for train_idx, val_idx in tscv.split(sorted_indices):
            yield sorted_indices[train_idx], sorted_indices[val_idx]


class NestedCV:
    """Nested cross-validation for unbiased performance estimation."""
    
    def __init__(self, outer_cv, inner_cv, 
                 param_grid: Dict, scoring: str = 'balanced_accuracy'):
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.param_grid = param_grid
        self.scoring = scoring
        
    def fit_predict(self, estimator, X: np.ndarray, y: np.ndarray,
                   groups: Optional[np.ndarray] = None) -> Dict:
        """
        Perform nested cross-validation.
        
        Returns:
            Dictionary with scores and best parameters for each fold
        """
        outer_scores = []
        best_params_per_fold = []
        
        for fold, (train_idx, test_idx) in enumerate(self.outer_cv.split(X, y, groups)):
            logger.info(f"Outer fold {fold + 1}/{self.outer_cv.n_splits}")
            
            # Outer split
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx] if groups is not None else None
            
            # Inner CV for hyperparameter tuning
            best_score = -np.inf
            best_params = None
            
            for params in self._generate_param_combinations():
                inner_scores = []
                
                for inner_train_idx, inner_val_idx in self.inner_cv.split(
                    X_train, y_train, groups_train
                ):
                    # Set parameters
                    estimator.set_params(**params)
                    
                    # Train on inner training set
                    estimator.fit(X_train[inner_train_idx], y_train[inner_train_idx])
                    
                    # Evaluate on inner validation set
                    score = estimator.score(X_train[inner_val_idx], y_train[inner_val_idx])
                    inner_scores.append(score)
                    
                mean_inner_score = np.mean(inner_scores)
                if mean_inner_score > best_score:
                    best_score = mean_inner_score
                    best_params = params
                    
            # Train on full training set with best params
            estimator.set_params(**best_params)
            estimator.fit(X_train, y_train)
            
            # Evaluate on test set
            test_score = estimator.score(X_test, y_test)
            outer_scores.append(test_score)
            best_params_per_fold.append(best_params)
            
            logger.info(f"Fold {fold + 1} score: {test_score:.4f}")
            
        return {
            'scores': outer_scores,
            'mean_score': np.mean(outer_scores),
            'std_score': np.std(outer_scores),
            'best_params_per_fold': best_params_per_fold
        }
        
    def _generate_param_combinations(self) -> List[Dict]:
        """Generate all parameter combinations from grid."""
        from itertools import product
        
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        
        for combination in product(*values):
            yield dict(zip(keys, combination))


class StratifiedBatchSampler(torch.utils.data.Sampler):
    """Stratified batch sampler to ensure balanced classes in each batch."""
    
    def __init__(self, labels: np.ndarray, batch_size: int, 
                 drop_last: bool = False):
        self.labels = labels
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
            
        # Calculate samples per class in each batch
        self.n_classes = len(self.class_indices)
        self.samples_per_class = self.batch_size // self.n_classes
        
    def __iter__(self):
        # Shuffle indices within each class
        for label in self.class_indices:
            np.random.shuffle(self.class_indices[label])
            
        # Create batches
        batch = []
        class_counters = {label: 0 for label in self.class_indices}
        
        while True:
            # Sample from each class
            for label in self.class_indices:
                for _ in range(self.samples_per_class):
                    if class_counters[label] < len(self.class_indices[label]):
                        idx = self.class_indices[label][class_counters[label]]
                        batch.append(idx)
                        class_counters[label] += 1
                        
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
                
            # Check if we've used all samples
            if all(class_counters[label] >= len(self.class_indices[label]) 
                   for label in self.class_indices):
                if not self.drop_last and len(batch) > 0:
                    yield batch
                break
                
    def __len__(self):
        total_samples = sum(len(indices) for indices in self.class_indices.values())
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size


class ValidationMonitor:
    """Monitor validation performance across folds."""
    
    def __init__(self, metrics: List[str], save_dir: Path):
        self.metrics = metrics
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.fold_results = defaultdict(lambda: defaultdict(list))
        self.best_scores = defaultdict(lambda: -np.inf)
        self.best_epochs = defaultdict(int)
        
    def update(self, fold: int, epoch: int, train_metrics: Dict, 
               val_metrics: Dict):
        """Update metrics for current fold and epoch."""
        for metric in self.metrics:
            if metric in train_metrics:
                self.fold_results[fold][f'train_{metric}'].append(train_metrics[metric])
            if metric in val_metrics:
                self.fold_results[fold][f'val_{metric}'].append(val_metrics[metric])
                
                # Track best validation score
                if val_metrics[metric] > self.best_scores[fold]:
                    self.best_scores[fold] = val_metrics[metric]
                    self.best_epochs[fold] = epoch
                    
    def save_results(self):
        """Save validation results to disk."""
        results = {
            'fold_results': dict(self.fold_results),
            'best_scores': dict(self.best_scores),
            'best_epochs': dict(self.best_epochs),
            'summary': self.get_summary()
        }
        
        with open(self.save_dir / 'cv_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
    def get_summary(self) -> Dict:
        """Get summary statistics across folds."""
        all_scores = list(self.best_scores.values())
        
        return {
            'mean_score': np.mean(all_scores),
            'std_score': np.std(all_scores),
            'min_score': np.min(all_scores),
            'max_score': np.max(all_scores),
            'scores_per_fold': all_scores
        }
        
    def plot_learning_curves(self):
        """Plot learning curves for each fold."""
        import matplotlib.pyplot as plt
        
        n_folds = len(self.fold_results)
        fig, axes = plt.subplots(n_folds, len(self.metrics), 
                                figsize=(5 * len(self.metrics), 4 * n_folds))
        
        if n_folds == 1:
            axes = axes.reshape(1, -1)
        if len(self.metrics) == 1:
            axes = axes.reshape(-1, 1)
            
        for fold_idx, fold in enumerate(sorted(self.fold_results.keys())):
            for metric_idx, metric in enumerate(self.metrics):
                ax = axes[fold_idx, metric_idx]
                
                train_key = f'train_{metric}'
                val_key = f'val_{metric}'
                
                if train_key in self.fold_results[fold]:
                    ax.plot(self.fold_results[fold][train_key], 
                           label='Train', color='blue')
                if val_key in self.fold_results[fold]:
                    ax.plot(self.fold_results[fold][val_key], 
                           label='Val', color='red')
                    
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.set_title(f'Fold {fold + 1} - {metric}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
        plt.tight_layout()
        plt.savefig(self.save_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


class CrossValidationPipeline:
    """Complete cross-validation pipeline for EEG classification."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.cv_config = config.get('cross_validation', {})
        
        # Setup cross-validation strategy
        self.cv_strategy = self._setup_cv_strategy()
        
        # Setup monitoring
        self.monitor = ValidationMonitor(
            metrics=config['evaluation']['metrics'],
            save_dir=Path(config['paths']['logs_dir']) / 'cv_results'
        )
        
    def _setup_cv_strategy(self):
        """Setup appropriate CV strategy based on configuration."""
        cv_type = self.cv_config.get('type', 'patient_independent')
        n_splits = self.cv_config.get('n_splits', 5)
        
        if cv_type == 'patient_independent':
            return PatientIndependentCV(
                n_splits=n_splits,
                stratify=self.cv_config.get('stratify', True),
                random_state=self.cv_config.get('random_state', 42)
            )
        elif cv_type == 'time_series':
            return TimeSeriesCV(
                n_splits=n_splits,
                test_size=self.cv_config.get('test_size'),
                gap=self.cv_config.get('gap', 0)
            )
        else:
            raise ValueError(f"Unknown CV type: {cv_type}")
            
    def run(self, dataset: Dataset, model_fn, train_fn, eval_fn) -> Dict:
        """
        Run complete cross-validation pipeline.
        
        Args:
            dataset: PyTorch dataset
            model_fn: Function to create model
            train_fn: Training function
            eval_fn: Evaluation function
            
        Returns:
            Cross-validation results
        """
        # Extract labels and groups
        labels = np.array([dataset[i]['label'] for i in range(len(dataset))])
        groups = np.array([dataset[i]['patient_id'] for i in range(len(dataset))])
        
        # Run cross-validation
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(
            self.cv_strategy.split(np.arange(len(dataset)), labels, groups)
        ):
            logger.info(f"\nFold {fold + 1}/{self.cv_strategy.n_splits}")
            logger.info(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
            
            # Create data loaders
            train_loader = self._create_balanced_loader(dataset, train_idx, True)
            val_loader = self._create_balanced_loader(dataset, val_idx, False)
            
            # Create model
            model = model_fn()
            
            # Train model
            fold_metrics = train_fn(
                model, train_loader, val_loader,
                fold=fold, monitor=self.monitor
            )
            
            # Final evaluation
            final_metrics = eval_fn(model, val_loader)
            fold_metrics['final'] = final_metrics
            
            fold_results.append(fold_metrics)
            
        # Save results
        self.monitor.save_results()
        self.monitor.plot_learning_curves()
        
        return {
            'fold_results': fold_results,
            'summary': self.monitor.get_summary()
        }
        
    def _create_balanced_loader(self, dataset: Dataset, indices: np.ndarray,
                               is_training: bool) -> DataLoader:
        """Create balanced data loader for training."""
        if is_training and self.cv_config.get('stratified_batches', True):
            # Get labels for indices
            labels = np.array([dataset[i]['label'] for i in indices])
            
            # Create stratified batch sampler
            sampler = StratifiedBatchSampler(
                labels, 
                batch_size=self.config['training']['batch_size'],
                drop_last=True
            )
            
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=self.config['hardware']['num_workers'],
                pin_memory=self.config['hardware']['pin_memory']
            )
        else:
            # Regular sampler
            sampler = SubsetRandomSampler(indices)
            
            return DataLoader(
                dataset,
                sampler=sampler,
                batch_size=self.config['inference']['batch_size'],
                num_workers=self.config['hardware']['num_workers'],
                pin_memory=self.config['hardware']['pin_memory']
            ) 