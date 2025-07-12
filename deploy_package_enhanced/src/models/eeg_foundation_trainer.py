"""
Fine-tuning Framework and Transfer Learning for EEG Foundation Model
Provides comprehensive training, fine-tuning, and evaluation capabilities
for the EEG Foundation Model on downstream tasks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import wandb
from collections import defaultdict

from .eeg_foundation_model import EEGFoundationModel, EEGFoundationConfig, EEGFoundationPreTrainer

logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning EEG Foundation Model."""
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 50
    batch_size: int = 16
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Fine-tuning strategy
    freeze_backbone: bool = False
    freeze_epochs: int = 10  # Epochs to keep backbone frozen
    unfreeze_layers: int = 3  # Number of top layers to unfreeze first
    gradual_unfreezing: bool = True
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = 'cosine'  # 'cosine', 'linear', 'exponential'
    scheduler_patience: int = 5
    min_lr: float = 1e-6
    
    # Regularization
    dropout_rate: float = 0.1
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    use_mixup: bool = False
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Model checkpointing
    save_best_model: bool = True
    save_last_model: bool = True
    output_dir: str = 'checkpoints/eeg_foundation_finetune'
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_probability: float = 0.3


class EEGDataset(Dataset):
    """Dataset wrapper for EEG data with foundation model compatibility."""
    
    def __init__(self, 
                 eeg_data: List[np.ndarray],
                 labels: List[int],
                 channel_names: Optional[List[str]] = None,
                 transform: Optional[Callable] = None):
        """
        Initialize EEG dataset.
        
        Args:
            eeg_data: List of EEG arrays (n_channels, seq_length)
            labels: List of labels
            channel_names: Optional channel names
            transform: Optional data transformation function
        """
        self.eeg_data = eeg_data
        self.labels = labels
        self.channel_names = channel_names
        self.transform = transform
        
        assert len(eeg_data) == len(labels), "Data and labels must have same length"
        
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        label = self.labels[idx]
        
        # Apply transformations if provided
        if self.transform:
            eeg = self.transform(eeg)
            
        # Convert to tensor
        if not isinstance(eeg, torch.Tensor):
            eeg = torch.tensor(eeg, dtype=torch.float32)
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
            
        return eeg, label


class EEGFoundationTrainer:
    """Comprehensive trainer for EEG Foundation Model with pre-training and fine-tuning."""
    
    def __init__(self,
                 model: EEGFoundationModel,
                 config: FineTuningConfig,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model: EEG Foundation Model
            config: Fine-tuning configuration
            device: Training device
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Initialize pre-trainer for self-supervised learning
        self.pre_trainer = EEGFoundationPreTrainer(model, device)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_optimizer_and_scheduler(self, train_loader: DataLoader):
        """Setup optimizer and learning rate scheduler."""
        # Optimizer
        optimizer_params = [
            {'params': [p for n, p in self.model.named_parameters() 
                       if 'classification_head' in n], 'lr': self.config.learning_rate},
            {'params': [p for n, p in self.model.named_parameters() 
                       if 'classification_head' not in n], 'lr': self.config.learning_rate * 0.1}
        ]
        
        self.optimizer = optim.AdamW(
            optimizer_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        if self.config.use_scheduler:
            total_steps = len(train_loader) * self.config.epochs
            
            if self.config.scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=total_steps,
                    eta_min=self.config.min_lr
                )
            elif self.config.scheduler_type == 'linear':
                self.scheduler = optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=self.config.min_lr / self.config.learning_rate,
                    total_iters=total_steps
                )
            elif self.config.scheduler_type == 'exponential':
                gamma = (self.config.min_lr / self.config.learning_rate) ** (1 / total_steps)
                self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
            else:
                self.scheduler = None
        else:
            self.scheduler = None
            
    def _apply_mixup(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
        """Apply mixup data augmentation."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def _mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Compute mixup loss."""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def _compute_metrics(self, predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics."""
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _log_metrics(self, metrics: Dict[str, float], phase: str, epoch: int):
        """Log metrics to console and wandb if available."""
        logger.info(f"{phase} Epoch {epoch}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
            
        # Log to wandb if available
        try:
            wandb.log({f"{phase}_{k}": v for k, v in metrics.items()}, step=epoch)
        except:
            pass  # wandb not initialized
            
    def pretrain(self,
                train_loader: DataLoader,
                val_loader: Optional[DataLoader] = None,
                epochs: int = 100,
                save_steps: int = 1000):
        """
        Pre-train the foundation model with self-supervised learning.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of pre-training epochs
            save_steps: Save model every N steps
        """
        logger.info(f"Starting pre-training for {epochs} epochs")
        
        # Setup optimizer for pre-training
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.model.train()
        global_step = 0
        
        for epoch in range(epochs):
            epoch_losses = defaultdict(list)
            
            pbar = tqdm(train_loader, desc=f"Pre-training Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (eeg_data, _) in enumerate(pbar):
                eeg_data = eeg_data.to(self.device)
                
                # Pre-training step
                total_loss, losses = self.pre_trainer.pretrain_step(eeg_data)
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    
                self.optimizer.step()
                
                # Update metrics
                for key, value in losses.items():
                    epoch_losses[key].append(value)
                    
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{losses['total_loss']:.4f}",
                    'masked': f"{losses.get('masked_eeg_loss', 0):.4f}",
                    'contrastive': f"{losses.get('contrastive_loss', 0):.4f}"
                })
                
                global_step += 1
                
                # Save checkpoint
                if global_step % save_steps == 0:
                    checkpoint_path = self.output_dir / f"pretrain_checkpoint_{global_step}.pt"
                    self.save_checkpoint(checkpoint_path, epoch, global_step)
                    
            # Log epoch metrics
            avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
            self._log_metrics(avg_losses, "Pretrain", epoch + 1)
            
            # Validation (if provided)
            if val_loader is not None:
                val_losses = self.validate_pretraining(val_loader)
                self._log_metrics(val_losses, "Pretrain_Val", epoch + 1)
                
        logger.info("Pre-training completed")
        
        # Save final pre-trained model
        final_path = self.output_dir / "pretrained_model"
        self.model.save_pretrained(final_path)
        logger.info(f"Pre-trained model saved to {final_path}")
        
    def validate_pretraining(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate pre-training performance."""
        self.model.eval()
        val_losses = defaultdict(list)
        
        with torch.no_grad():
            for eeg_data, _ in val_loader:
                eeg_data = eeg_data.to(self.device)
                
                # Pre-training step (without gradients)
                _, losses = self.pre_trainer.pretrain_step(eeg_data)
                
                for key, value in losses.items():
                    val_losses[key].append(value)
                    
        self.model.train()
        return {k: np.mean(v) for k, v in val_losses.items()}
    
    def finetune(self,
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_classes: int,
                class_names: Optional[List[str]] = None):
        """
        Fine-tune the foundation model on a downstream task.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_classes: Number of classes for the task
            class_names: Optional class names for logging
        """
        logger.info(f"Starting fine-tuning for {self.config.epochs} epochs")
        
        # Add classification head
        self.model.add_classification_head(num_classes, self.config.dropout_rate)
        
        # Setup optimizer and scheduler
        self._setup_optimizer_and_scheduler(train_loader)
        
        # Loss function with label smoothing
        if self.config.label_smoothing > 0:
            criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
            
        # Gradual unfreezing strategy
        if self.config.freeze_backbone:
            self.model.freeze_backbone()
            logger.info("Backbone frozen for initial training")
            
        best_val_metric = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Gradual unfreezing
            if (self.config.gradual_unfreezing and 
                epoch >= self.config.freeze_epochs and 
                epoch % (self.config.freeze_epochs // self.config.unfreeze_layers) == 0):
                self.model.unfreeze_backbone()
                logger.info(f"Unfroze backbone at epoch {epoch}")
                
            # Training phase
            train_metrics = self.train_epoch(train_loader, criterion)
            self._log_metrics(train_metrics, "Train", epoch + 1)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader, criterion)
            self._log_metrics(val_metrics, "Val", epoch + 1)
            
            # Store metrics
            for key, value in train_metrics.items():
                self.train_metrics[key].append(value)
            for key, value in val_metrics.items():
                self.val_metrics[key].append(value)
                
            # Check for best model
            current_metric = val_metrics['accuracy']  # Use accuracy as primary metric
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                patience_counter = 0
                
                if self.config.save_best_model:
                    best_path = self.output_dir / "best_model"
                    self.model.save_pretrained(best_path)
                    logger.info(f"New best model saved with accuracy: {current_metric:.4f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if (self.config.early_stopping and 
                patience_counter >= self.config.patience):
                logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                break
                
            # Save checkpoint
            if (epoch + 1) % (self.config.save_steps // len(train_loader)) == 0:
                checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path, epoch, self.global_step)
                
        # Save final model
        if self.config.save_last_model:
            final_path = self.output_dir / "final_model"
            self.model.save_pretrained(final_path)
            logger.info(f"Final model saved to {final_path}")
            
        # Generate training plots
        self.plot_training_curves()
        
        logger.info("Fine-tuning completed")
        return best_val_metric
    
    def train_epoch(self, train_loader: DataLoader, criterion) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch + 1}")
        
        for batch_idx, (eeg_data, labels) in enumerate(pbar):
            eeg_data = eeg_data.to(self.device)
            labels = labels.to(self.device)
            
            # Apply mixup if enabled
            if self.config.use_mixup and np.random.random() < self.config.augmentation_probability:
                eeg_data, labels_a, labels_b, lam = self._apply_mixup(
                    eeg_data, labels, self.config.mixup_alpha
                )
                
                # Forward pass
                logits = self.model.classify(eeg_data)
                loss = self._mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                
                # For metrics, use original labels
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:
                # Regular forward pass
                logits = self.model.classify(eeg_data)
                loss = criterion(logits, labels)
                
                # Collect predictions for metrics
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
                
            # Update metrics
            epoch_losses.append(loss.item())
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
        # Compute epoch metrics
        avg_loss = np.mean(epoch_losses)
        metrics = self._compute_metrics(
            np.array(all_predictions), 
            np.array(all_labels)
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate_epoch(self, val_loader: DataLoader, criterion) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = []
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for eeg_data, labels in val_loader:
                eeg_data = eeg_data.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model.classify(eeg_data)
                loss = criterion(logits, labels)
                
                # Collect predictions
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                epoch_losses.append(loss.item())
                
        # Compute metrics
        avg_loss = np.mean(epoch_losses)
        metrics = self._compute_metrics(
            np.array(all_predictions), 
            np.array(all_labels)
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Comprehensive evaluation on test set."""
        logger.info("Running comprehensive evaluation...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for eeg_data, labels in tqdm(test_loader, desc="Evaluating"):
                eeg_data = eeg_data.to(self.device)
                
                # Get predictions
                logits = self.model.classify(eeg_data)
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        probabilities = np.array(all_probabilities)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, labels)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Create comprehensive results
        results = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'predictions': predictions,
            'labels': labels,
            'probabilities': probabilities
        }
        
        # Save results
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'metrics': metrics,
                'confusion_matrix': cm.tolist(),
                'predictions': predictions.tolist(),
                'labels': labels.tolist()
            }
            json.dump(json_results, f, indent=2)
            
        logger.info(f"Evaluation results saved to {results_path}")
        
        return results
    
    def plot_training_curves(self):
        """Plot training and validation curves."""
        if not self.train_metrics or not self.val_metrics:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        epochs = range(1, len(self.train_metrics['loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.train_metrics['loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.val_metrics['loss'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.train_metrics['accuracy'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.val_metrics['accuracy'], 'r-', label='Validation', linewidth=2)
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score curves
        axes[1, 0].plot(epochs, self.train_metrics['f1'], 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, self.val_metrics['f1'], 'r-', label='Validation', linewidth=2)
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Precision-Recall curves
        axes[1, 1].plot(epochs, self.train_metrics['precision'], 'b-', label='Train Precision', linewidth=2)
        axes[1, 1].plot(epochs, self.val_metrics['precision'], 'r-', label='Val Precision', linewidth=2)
        axes[1, 1].plot(epochs, self.train_metrics['recall'], 'b--', label='Train Recall', linewidth=2)
        axes[1, 1].plot(epochs, self.val_metrics['recall'], 'r--', label='Val Recall', linewidth=2)
        axes[1, 1].set_title('Precision & Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training curves saved to {plot_path}")
    
    def save_checkpoint(self, path: Path, epoch: int, global_step: int):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics),
            'best_metric': self.best_metric
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        self.train_metrics = defaultdict(list, checkpoint['train_metrics'])
        self.val_metrics = defaultdict(list, checkpoint['val_metrics'])
        
        logger.info(f"Checkpoint loaded from {path}")


class TransferLearningPipeline:
    """Complete transfer learning pipeline for EEG Foundation Model."""
    
    def __init__(self, 
                 foundation_model_path: Optional[Path] = None,
                 config: Optional[FineTuningConfig] = None):
        """
        Initialize transfer learning pipeline.
        
        Args:
            foundation_model_path: Path to pre-trained foundation model
            config: Fine-tuning configuration
        """
        self.foundation_model_path = foundation_model_path
        self.config = config or FineTuningConfig()
        
    def run_pipeline(self,
                   train_data: List[np.ndarray],
                   train_labels: List[int],
                   val_data: List[np.ndarray],
                   val_labels: List[int],
                   test_data: List[np.ndarray],
                   test_labels: List[int],
                   num_classes: int,
                   class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run complete transfer learning pipeline.
        
        Args:
            train_data: Training EEG data
            train_labels: Training labels
            val_data: Validation EEG data
            val_labels: Validation labels
            test_data: Test EEG data
            test_labels: Test labels
            num_classes: Number of classes
            class_names: Optional class names
            
        Returns:
            Dictionary with results and trained model
        """
        logger.info("Starting transfer learning pipeline...")
        
        # Load or create foundation model
        if self.foundation_model_path and self.foundation_model_path.exists():
            logger.info(f"Loading pre-trained model from {self.foundation_model_path}")
            model = EEGFoundationModel.from_pretrained(self.foundation_model_path)
        else:
            logger.info("Creating new foundation model")
            config = EEGFoundationConfig()
            model = EEGFoundationModel(config)
            
        # Create datasets
        train_dataset = EEGDataset(train_data, train_labels)
        val_dataset = EEGDataset(val_data, val_labels)
        test_dataset = EEGDataset(test_data, test_labels)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=4
        )
        
        # Initialize trainer
        trainer = EEGFoundationTrainer(model, self.config)
        
        # Fine-tune model
        best_val_metric = trainer.finetune(
            train_loader, 
            val_loader, 
            num_classes,
            class_names
        )
        
        # Evaluate on test set
        test_results = trainer.evaluate(test_loader)
        
        # Prepare final results
        results = {
            'best_val_metric': best_val_metric,
            'test_results': test_results,
            'model': model,
            'trainer': trainer,
            'config': self.config
        }
        
        logger.info("Transfer learning pipeline completed successfully")
        return results 