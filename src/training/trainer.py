"""
Comprehensive trainer for HMS harmful brain activity classification.
Integrates all training strategies and optimization techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
import logging
import mlflow
import wandb
import optuna
from tqdm import tqdm
import json
import yaml
import time
from collections import defaultdict

from ..models.ensemble_model import HMSEnsembleModel, MetaLearner
from ..models.resnet1d_gru import ResNet1D_GRU_Trainer
from ..models.efficientnet_spectrogram import EfficientNetTrainer
from ..evaluation.evaluator import ModelEvaluator, ClinicalMetrics
from ..utils.dataset import create_dataloaders
from .cross_validation import CrossValidationPipeline, ValidationMonitor
from .augmentation import AugmentationPipeline, TestTimeAugmentation
from .class_balancing import BalancedDataLoader, HardExampleMining
from .hyperparameter_optimization import HyperparameterOptimizationPipeline
from ..models.optimization import (
    EarlyStopping, GradientClipping, CosineAnnealingWarmRestarts,
    SAM, LookAheadOptimizer
)

logger = logging.getLogger(__name__)


class HMSTrainer:
    """Comprehensive trainer for HMS brain activity classification."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['hardware']['device'])
        
        # Initialize components
        self.augmentation = AugmentationPipeline(config)
        self.data_loader_factory = BalancedDataLoader(config)
        self.cv_pipeline = CrossValidationPipeline(config)
        self.hyperopt_pipeline = HyperparameterOptimizationPipeline(config)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = -np.inf
        self.training_history = defaultdict(list)
        
        # Setup experiment tracking
        self._setup_experiment_tracking()
        
    def _setup_experiment_tracking(self):
        """Setup MLflow and WandB tracking."""
        # MLflow setup
        if self.config['logging']['mlflow']['enabled']:
            mlflow.set_tracking_uri(self.config['logging']['mlflow']['tracking_uri'])
            mlflow.set_experiment(self.config['logging']['experiment_name'])
            
        # WandB setup
        if self.config['logging']['wandb']['enabled']:
            wandb.init(
                project=self.config['logging']['wandb']['project'],
                entity=self.config['logging']['wandb']['entity'],
                config=self.config,
                name=self.config['logging']['experiment_name']
            )
            
    def train_model(self, model: nn.Module, train_dataset: Dataset,
                   val_dataset: Dataset, model_name: str = 'model') -> Dict:
        """
        Train a single model with all strategies.
        
        Args:
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            model_name: Name for logging
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Starting training for {model_name}")
        
        # Move model to device
        model = model.to(self.device)
        
        # Create data loaders
        train_loader = self.data_loader_factory.create_loader(train_dataset, is_training=True)
        val_loader = self.data_loader_factory.create_loader(val_dataset, is_training=False)
        
        # Get class counts for loss function
        labels = [train_dataset[i]['label'] for i in range(len(train_dataset))]
        class_counts = [labels.count(i) for i in range(self.config['classes'].__len__())]
        
        # Setup loss function
        criterion = self.data_loader_factory.get_loss_function(class_counts)
        
        # Setup optimizer
        optimizer = self._setup_optimizer(model, model_name)
        
        # Setup scheduler
        scheduler = self._setup_scheduler(optimizer, len(train_loader))
        
        # Setup early stopping
        early_stopping = EarlyStopping(
            patience=self.config['training']['early_stopping']['patience'],
            min_delta=self.config['training']['early_stopping']['min_delta'],
            mode=self.config['training']['early_stopping']['mode']
        )
        
        # Hard example mining
        hard_mining = HardExampleMining(
            mining_type='class_aware',
            percentile=0.3
        )
        
        # Training loop
        for epoch in range(self.config['models'][model_name]['training']['epochs']):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self._train_epoch(
                model, train_loader, criterion, optimizer, 
                scheduler, hard_mining
            )
            
            # Validate
            val_metrics = self._validate(model, val_loader, criterion)
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Check early stopping
            if early_stopping(val_metrics['loss']):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
                
            # Save checkpoint if best
            if val_metrics['balanced_accuracy'] > self.best_metric:
                self.best_metric = val_metrics['balanced_accuracy']
                self._save_checkpoint(model, optimizer, epoch, val_metrics, model_name)
                
        # Final evaluation
        final_metrics = self._final_evaluation(model, val_loader)
        
        return {
            'best_metric': self.best_metric,
            'final_metrics': final_metrics,
            'training_history': dict(self.training_history)
        }
        
    def train_with_cross_validation(self, model_class, dataset: Dataset,
                                   model_name: str = 'model') -> Dict:
        """Train model with cross-validation."""
        
        def model_fn():
            return model_class(self.config)
            
        def train_fn(model, train_loader, val_loader, fold, monitor):
            # Training function for CV
            return self._cv_train(model, train_loader, val_loader, fold, monitor, model_name)
            
        def eval_fn(model, val_loader):
            # Evaluation function for CV
            criterion = nn.CrossEntropyLoss()
            return self._validate(model, val_loader, criterion)
            
        # Run cross-validation
        cv_results = self.cv_pipeline.run(
            dataset, model_fn, train_fn, eval_fn
        )
        
        return cv_results
        
    def optimize_hyperparameters(self, model_class, dataset: Dataset,
                                model_name: str = 'model') -> Dict:
        """Optimize hyperparameters for a model."""
        
        def train_func(model, dataset, config, trial=None):
            # Split dataset
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            # Train model
            results = self.train_model(
                model, train_dataset, val_dataset, model_name
            )
            
            return results['final_metrics']
            
        def eval_func(model, dataset, config):
            # Create loader
            loader = DataLoader(
                dataset, 
                batch_size=config['inference']['batch_size'],
                shuffle=False
            )
            criterion = nn.CrossEntropyLoss()
            return self._validate(model, loader, criterion)
            
        # Run hyperparameter optimization
        results = self.hyperopt_pipeline.optimize_hyperparameters(
            model_name, train_func, eval_func, dataset
        )
        
        return results
        
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader,
                    criterion: nn.Module, optimizer, scheduler,
                    hard_mining: HardExampleMining) -> Dict[str, float]:
        """Train one epoch."""
        model.train()
        
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Apply augmentation
            batch = self.augmentation.augment_batch(batch, training=True)
            
            # Move to device
            if 'eeg' in batch:
                inputs = batch['eeg'].to(self.device)
            else:
                inputs = batch['spectrogram'].to(self.device)
                
            # Handle mixup/cutmix
            if 'mix_lambda' in batch:
                targets_a = batch['label_a'].to(self.device)
                targets_b = batch['label_b'].to(self.device)
                lam = batch['mix_lambda']
            else:
                targets = batch['label'].to(self.device)
                targets_a = targets_b = targets
                lam = 1.0
                
            # Forward pass
            outputs = model(inputs)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
                
            # Calculate loss
            if lam < 1.0:
                loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
            else:
                loss_unreduced = F.cross_entropy(logits, targets, reduction='none')
                
                # Hard example mining
                hard_mask = hard_mining.mine_hard_examples(loss_unreduced, targets)
                loss = loss_unreduced[hard_mask].mean()
                
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = GradientClipping.adaptive_clip_grad_norm_(
                model.parameters(),
                max_norm=self.config['training']['optimization']['gradient_clip']
            )
            
            # Optimizer step
            if isinstance(optimizer, SAM):
                optimizer.first_step(zero_grad=True)
                
                # Second forward-backward pass
                outputs_2 = model(inputs)
                logits_2 = outputs_2['logits'] if isinstance(outputs_2, dict) else outputs_2
                
                if lam < 1.0:
                    loss_2 = lam * criterion(logits_2, targets_a) + (1 - lam) * criterion(logits_2, targets_b)
                else:
                    loss_2 = criterion(logits_2, targets)
                    
                loss_2.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()
                
            # Update scheduler
            if scheduler is not None:
                scheduler.step()
                
            # Update metrics
            epoch_loss += loss.item() * inputs.size(0)
            
            if lam >= 1.0:
                preds = torch.argmax(logits, dim=1)
                epoch_correct += (preds == targets).sum().item()
                epoch_total += targets.size(0)
            else:
                # For mixup, use weighted accuracy
                preds = torch.argmax(logits, dim=1)
                correct_a = (preds == targets_a).float()
                correct_b = (preds == targets_b).float()
                epoch_correct += (lam * correct_a + (1 - lam) * correct_b).sum().item()
                epoch_total += targets_a.size(0)
                
            # Update progress bar
            if epoch_total > 0:
                pbar.set_postfix({
                    'loss': epoch_loss / epoch_total,
                    'acc': epoch_correct / epoch_total,
                    'lr': optimizer.param_groups[0]['lr']
                })
                
            self.global_step += 1
            
        return {
            'loss': epoch_loss / epoch_total,
            'accuracy': epoch_correct / epoch_total
        }
        
    def _validate(self, model: nn.Module, val_loader: DataLoader,
                 criterion: nn.Module) -> Dict[str, float]:
        """Validate model."""
        model.eval()
        
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                # Move to device
                if 'eeg' in batch:
                    inputs = batch['eeg'].to(self.device)
                else:
                    inputs = batch['spectrogram'].to(self.device)
                targets = batch['label'].to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                    
                # Calculate loss
                loss = criterion(logits, targets)
                
                # Update metrics
                val_loss += loss.item() * inputs.size(0)
                
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        # Calculate metrics
        from sklearn.metrics import balanced_accuracy_score, f1_score
        
        metrics = {
            'loss': val_loss / val_total,
            'accuracy': val_correct / val_total,
            'balanced_accuracy': balanced_accuracy_score(all_targets, all_preds),
            'f1_macro': f1_score(all_targets, all_preds, average='macro'),
            'predictions': np.array(all_preds),
            'targets': np.array(all_targets),
            'probabilities': np.array(all_probs)
        }
        
        return metrics
        
    def _final_evaluation(self, model: nn.Module, 
                         test_loader: DataLoader) -> Dict[str, float]:
        """Final evaluation with test-time augmentation."""
        if self.config['inference']['use_tta']:
            tta = TestTimeAugmentation(
                n_augmentations=self.config['inference']['tta_transforms']
            )
            
            # TTA predictions
            all_probs = []
            
            for batch in tqdm(test_loader, desc='TTA Evaluation'):
                if 'eeg' in batch:
                    inputs = batch['eeg'].to(self.device)
                else:
                    inputs = batch['spectrogram'].to(self.device)
                    
                # Get TTA predictions
                probs = tta(inputs, model)
                all_probs.extend(probs.cpu().numpy())
                
            # Convert to predictions
            all_probs = np.array(all_probs)
            all_preds = np.argmax(all_probs, axis=1)
            
            # Get true labels
            all_targets = []
            for batch in test_loader:
                all_targets.extend(batch['label'].numpy())
            all_targets = np.array(all_targets)
            
        else:
            # Standard evaluation
            criterion = nn.CrossEntropyLoss()
            metrics = self._validate(model, test_loader, criterion)
            all_preds = metrics['predictions']
            all_targets = metrics['targets']
            all_probs = metrics['probabilities']
            
        # Calculate comprehensive metrics
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, f1_score,
            precision_recall_fscore_support, roc_auc_score,
            confusion_matrix, classification_report
        )
        
        final_metrics = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'balanced_accuracy': balanced_accuracy_score(all_targets, all_preds),
            'f1_macro': f1_score(all_targets, all_preds, average='macro'),
            'f1_weighted': f1_score(all_targets, all_preds, average='weighted'),
            'confusion_matrix': confusion_matrix(all_targets, all_preds),
            'classification_report': classification_report(all_targets, all_preds),
            'probabilities': all_probs
        }
        
        # Calculate AUC if binary or one-vs-rest
        try:
            if len(np.unique(all_targets)) == 2:
                final_metrics['auc_roc'] = roc_auc_score(all_targets, all_probs[:, 1])
            else:
                final_metrics['auc_roc'] = roc_auc_score(
                    all_targets, all_probs, multi_class='ovr'
                )
        except:
            pass
            
        return final_metrics
        
    def _setup_optimizer(self, model: nn.Module, model_name: str):
        """Setup optimizer based on configuration."""
        opt_config = self.config['training']['optimization']
        model_config = self.config['models'][model_name]['training']
        
        # Base optimizer
        if opt_config['optimizer'] == 'adamw':
            base_optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=model_config['learning_rate'],
                weight_decay=model_config['weight_decay']
            )
        elif opt_config['optimizer'] == 'sgd':
            base_optimizer = torch.optim.SGD(
                model.parameters(),
                lr=model_config['learning_rate'],
                momentum=0.9,
                weight_decay=model_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['optimizer']}")
            
        # Wrap with advanced optimizers
        if opt_config['optimizer'] == 'sam':
            optimizer = SAM(base_optimizer, rho=0.05)
        elif opt_config['optimizer'] == 'lookahead':
            optimizer = LookAheadOptimizer(base_optimizer, k=5, alpha=0.5)
        else:
            optimizer = base_optimizer
            
        return optimizer
        
    def _setup_scheduler(self, optimizer, steps_per_epoch: int):
        """Setup learning rate scheduler."""
        sched_config = self.config['training']['optimization']
        
        if sched_config['scheduler'] == 'cosine_warmup':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10 * steps_per_epoch,
                T_mult=2,
                warmup_steps=sched_config['warmup_epochs'] * steps_per_epoch
            )
        elif sched_config['scheduler'] == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=optimizer.param_groups[0]['lr'],
                epochs=self.config['models']['resnet1d_gru']['training']['epochs'],
                steps_per_epoch=steps_per_epoch
            )
        else:
            scheduler = None
            
        return scheduler
        
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log metrics to experiment tracking."""
        # Store in history
        for key, value in train_metrics.items():
            self.training_history[f'train_{key}'].append(value)
        for key, value in val_metrics.items():
            if not isinstance(value, (list, np.ndarray)):
                self.training_history[f'val_{key}'].append(value)
                
        # MLflow logging
        if self.config['logging']['mlflow']['enabled']:
            mlflow.log_metrics({
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items() 
                   if not isinstance(v, (list, np.ndarray))}
            }, step=epoch)
            
        # WandB logging
        if self.config['logging']['wandb']['enabled']:
            wandb.log({
                **{f'train/{k}': v for k, v in train_metrics.items()},
                **{f'val/{k}': v for k, v in val_metrics.items()
                   if not isinstance(v, (list, np.ndarray))},
                'epoch': epoch
            })
            
    def _save_checkpoint(self, model: nn.Module, optimizer, epoch: int,
                        metrics: Dict, model_name: str):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config['paths']['models_dir']) / model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f'best_model_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # MLflow artifact
        if self.config['logging']['mlflow']['enabled']:
            mlflow.log_artifact(str(checkpoint_path))
            
    def _cv_train(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, fold: int, monitor: ValidationMonitor,
                 model_name: str) -> Dict:
        """Training function for cross-validation."""
        # Get loss function
        labels = []
        for batch in train_loader:
            labels.extend(batch['label'].numpy())
        class_counts = [labels.count(i) for i in range(len(self.config['classes']))]
        
        criterion = self.data_loader_factory.get_loss_function(class_counts)
        
        # Setup optimizer and scheduler
        optimizer = self._setup_optimizer(model, model_name)
        scheduler = self._setup_scheduler(optimizer, len(train_loader))
        
        # Training loop
        for epoch in range(self.config['models'][model_name]['training']['epochs']):
            # Train
            train_metrics = self._train_epoch(
                model, train_loader, criterion, optimizer, scheduler,
                HardExampleMining()
            )
            
            # Validate
            val_metrics = self._validate(model, val_loader, criterion)
            
            # Update monitor
            monitor.update(fold, epoch, train_metrics, val_metrics)
            
        return {
            'final_train_metrics': train_metrics,
            'final_val_metrics': val_metrics
        }


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna."""
    
    def __init__(self, config: Dict, n_trials: int = 50):
        self.config = config
        self.n_trials = n_trials
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization."""
        # Suggest hyperparameters
        suggested_config = self._suggest_hyperparameters(trial)
        
        # Update config
        config = self._update_config(self.config.copy(), suggested_config)
        
        # Create data loaders with suggested batch size
        dataloaders = create_dataloaders(
            config,
            batch_size=suggested_config['batch_size']
        )
        
        # Train model
        trainer = HMSTrainer(config)
        
        try:
            metrics = trainer.train_model(
                dataloaders['train'],
                dataloaders['val']
            )
            
            # Return validation loss as objective
            return metrics['final_metrics']['loss']
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float('inf')
            
    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """Suggest hyperparameters for trial."""
        return {
            # Training parameters
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'learning_rate_resnet': trial.suggest_loguniform('lr_resnet', 1e-5, 1e-2),
            'learning_rate_efficientnet': trial.suggest_loguniform('lr_efficientnet', 1e-6, 1e-3),
            
            # Model parameters
            'resnet_dropout': trial.suggest_uniform('resnet_dropout', 0.1, 0.5),
            'efficientnet_dropout': trial.suggest_uniform('efficientnet_dropout', 0.1, 0.5),
            'gru_hidden_size': trial.suggest_categorical('gru_hidden', [128, 256, 512]),
            
            # Ensemble parameters
            'ensemble_method': trial.suggest_categorical('ensemble', ['stacking', 'attention']),
            
            # Regularization
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-2),
        }
        
    def _update_config(self, config: Dict, suggestions: Dict) -> Dict:
        """Update configuration with suggested values."""
        config['training']['batch_size'] = suggestions['batch_size']
        
        config['models']['resnet1d_gru']['training']['learning_rate'] = suggestions['learning_rate_resnet']
        config['models']['resnet1d_gru']['resnet']['dropout'] = suggestions['resnet_dropout']
        config['models']['resnet1d_gru']['gru']['hidden_size'] = suggestions['gru_hidden_size']
        config['models']['resnet1d_gru']['training']['weight_decay'] = suggestions['weight_decay']
        
        config['models']['efficientnet']['training']['learning_rate'] = suggestions['learning_rate_efficientnet']
        config['models']['efficientnet']['dropout'] = suggestions['efficientnet_dropout']
        config['models']['efficientnet']['training']['weight_decay'] = suggestions['weight_decay']
        
        config['models']['ensemble']['method'] = suggestions['ensemble_method']
        
        return config
        
    def optimize(self) -> Dict:
        """Run hyperparameter optimization."""
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.config['training']['optimization']['timeout']
        )
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {best_value}")
        logger.info(f"Best parameters: {best_params}")
        
        # Save optimization results
        results = {
            'best_params': best_params,
            'best_value': best_value,
            'study_stats': {
                'n_trials': len(study.trials),
                'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            }
        }
        
        with open('optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        return results 