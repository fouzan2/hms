#!/usr/bin/env python3
"""
HMS EEG Classification - Novita AI Optimized Training Script
Full dataset training on H100 GPU for >90% accuracy
"""

import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import wandb
import mlflow
import mlflow.pytorch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/novita_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class NovitaTrainingPipeline:
    """Optimized training pipeline for Novita AI H100"""
    
    def __init__(self, config_path: str = "config/novita_production_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_environment()
        self.setup_logging()
        
    def load_config(self) -> Dict:
        """Load configuration file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
        
    def setup_environment(self):
        """Setup optimized environment for H100"""
        logger.info("Setting up Novita AI H100 environment...")
        
        # CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Memory optimizations
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
        
        # Set optimal number of threads
        torch.set_num_threads(16)
        
        # Enable mixed precision
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        logger.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            
    def setup_logging(self):
        """Setup experiment tracking"""
        logger.info("Setting up experiment tracking...")
        
        # Initialize Weights & Biases
        try:
            wandb.init(
                project=self.config['logging']['wandb_project'],
                name=f"{self.config['logging']['run_name_prefix']}-{int(time.time())}",
                config=self.config,
                mode="online" if os.getenv('WANDB_API_KEY') else "disabled"
            )
            logger.info("Weights & Biases initialized")
        except Exception as e:
            logger.warning(f"Could not initialize W&B: {e}")
            
        # Initialize MLflow
        try:
            mlflow.set_tracking_uri(self.config['logging']['mlflow_tracking_uri'])
            mlflow.set_experiment(self.config['logging']['experiment_name'])
            mlflow.start_run()
            logger.info("MLflow initialized")
        except Exception as e:
            logger.warning(f"Could not initialize MLflow: {e}")
            
    def download_data(self):
        """Download HMS dataset from Kaggle"""
        logger.info("Downloading HMS dataset...")
        
        try:
            # Check if data already exists
            data_dir = Path(self.config['dataset']['raw_data_path'])
            if (data_dir / "train.csv").exists():
                logger.info("Data already exists, skipping download")
                return True
                
            # Download using Kaggle API
            import kaggle
            kaggle.api.competition_download_files(
                self.config['dataset']['kaggle_competition'],
                path=str(data_dir),
                unzip=True
            )
            
            logger.info("Dataset downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Data download failed: {e}")
            return False
            
    def preprocess_data(self):
        """Preprocess EEG data with optimizations"""
        logger.info("Starting optimized data preprocessing...")
        
        # Import preprocessing modules
        sys.path.append(str(Path(__file__).parent / "src"))
        from preprocessing.advanced_preprocessing import AdvancedEEGPreprocessor
        from preprocessing.spectrogram_generator import SpectrogramGenerator
        
        try:
            # Initialize preprocessors
            eeg_preprocessor = AdvancedEEGPreprocessor(self.config)
            spec_generator = SpectrogramGenerator(self.config)
            
            # Load raw data
            train_df = pd.read_csv(f"{self.config['dataset']['raw_data_path']}/train.csv")
            logger.info(f"Loaded {len(train_df)} training samples")
            
            # Use full dataset for production
            if self.config['dataset']['max_samples'] is None:
                logger.info("Using full dataset for training")
            else:
                train_df = train_df.head(self.config['dataset']['max_samples'])
                logger.info(f"Using {len(train_df)} samples for training")
                
            # Parallel preprocessing
            processed_data = eeg_preprocessor.process_parallel(
                train_df, 
                n_jobs=16,  # Utilize all CPU cores
                batch_size=1000
            )
            
            # Generate spectrograms
            spectrograms = spec_generator.generate_parallel(
                processed_data,
                n_jobs=8,  # Balance CPU/GPU usage
                batch_size=500
            )
            
            # Save processed data
            processed_dir = Path(self.config['dataset']['processed_data_path'])
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            np.savez_compressed(
                processed_dir / "eeg_processed.npz",
                **processed_data
            )
            
            np.savez_compressed(
                processed_dir / "spectrograms.npz",
                **spectrograms
            )
            
            logger.info("Data preprocessing completed")
            return True
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return False
            
    def create_data_loaders(self):
        """Create optimized data loaders"""
        logger.info("Creating optimized data loaders...")
        
        sys.path.append(str(Path(__file__).parent / "src"))
        from utils.data_loader import HMSDataset, create_balanced_loaders
        
        try:
            # Load processed data
            processed_dir = Path(self.config['dataset']['processed_data_path'])
            
            eeg_data = np.load(processed_dir / "eeg_processed.npz")
            spec_data = np.load(processed_dir / "spectrograms.npz")
            
            # Create datasets
            dataset = HMSDataset(
                eeg_data=eeg_data,
                spectrogram_data=spec_data,
                config=self.config,
                mode='train'
            )
            
            # Create balanced loaders with optimizations
            train_loader, val_loader = create_balanced_loaders(
                dataset,
                val_split=self.config['dataset']['val_split'],
                batch_size=self.config['models']['resnet1d_gru']['training']['batch_size'],
                num_workers=self.config['training']['num_workers'],
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4
            )
            
            logger.info(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}")
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"Data loader creation failed: {e}")
            return None, None
            
    def create_models(self):
        """Create optimized models"""
        logger.info("Creating optimized models...")
        
        sys.path.append(str(Path(__file__).parent / "src"))
        from models.resnet1d_gru import ResNet1D_GRU_Advanced
        from models.efficientnet_classifier import EfficientNetClassifier
        from models.ensemble import AdvancedEnsemble
        
        models = {}
        
        try:
            # ResNet1D-GRU for raw EEG
            if self.config['models']['resnet1d_gru']['enabled']:
                resnet_gru = ResNet1D_GRU_Advanced(
                    num_channels=self.config['preprocessing']['num_channels'],
                    num_classes=self.config['dataset']['num_classes'],
                    config=self.config['models']['resnet1d_gru']
                ).to(self.device)
                
                # Compile model for speed (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    resnet_gru = torch.compile(resnet_gru)
                    
                models['resnet1d_gru'] = resnet_gru
                logger.info("ResNet1D-GRU model created")
                
            # EfficientNet for spectrograms
            if self.config['models']['efficientnet']['enabled']:
                efficientnet = EfficientNetClassifier(
                    model_name=self.config['models']['efficientnet']['model_name'],
                    num_classes=self.config['dataset']['num_classes'],
                    config=self.config['models']['efficientnet']
                ).to(self.device)
                
                if hasattr(torch, 'compile'):
                    efficientnet = torch.compile(efficientnet)
                    
                models['efficientnet'] = efficientnet
                logger.info("EfficientNet model created")
                
            # Ensemble model
            if self.config['models']['ensemble']['enabled'] and len(models) > 1:
                ensemble = AdvancedEnsemble(
                    models=list(models.values()),
                    config=self.config['models']['ensemble']
                ).to(self.device)
                
                models['ensemble'] = ensemble
                logger.info("Ensemble model created")
                
            return models
            
        except Exception as e:
            logger.error(f"Model creation failed: {e}")
            return {}
            
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, model_name: str) -> Dict:
        """Train individual model with optimizations"""
        logger.info(f"Training {model_name}...")
        
        # Get model-specific config
        model_config = self.config['models'][model_name]['training']
        
        # Optimizer with advanced settings
        optimizer = optim.AdamW(
            model.parameters(),
            lr=model_config['learning_rate'],
            weight_decay=model_config['weight_decay'],
            betas=(0.9, 0.95),  # Better for large models
            eps=1e-6
        )
        
        # Advanced scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=model_config['epochs'] // 4,
            T_mult=2,
            eta_min=model_config['learning_rate'] * 0.01
        )
        
        # Loss function with label smoothing
        criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.get('advanced_training', {}).get('label_smoothing', 0.1)
        )
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler(enabled=self.config['training']['mixed_precision']['enabled'])
        
        # Training metrics
        best_accuracy = 0.0
        patience_counter = 0
        training_history = []
        
        for epoch in range(model_config['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{model_config['epochs']}")
            
            for batch_idx, (data, targets) in enumerate(progress_bar):
                data, targets = data.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Mixed precision forward pass
                with torch.cuda.amp.autocast(enabled=self.config['training']['mixed_precision']['enabled']):
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('advanced_training', {}).get('gradient_clipping'):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config['advanced_training']['gradient_clipping']
                    )
                
                scaler.step(optimizer)
                scaler.update()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
                
                # Log to wandb
                if hasattr(wandb, 'log'):
                    wandb.log({
                        f'{model_name}_batch_loss': loss.item(),
                        f'{model_name}_batch_acc': 100.*train_correct/train_total,
                        'epoch': epoch
                    })
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                    
                    with torch.cuda.amp.autocast(enabled=self.config['training']['mixed_precision']['enabled']):
                        outputs = model(data)
                        loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
            
            # Calculate metrics
            train_accuracy = 100. * train_correct / train_total
            val_accuracy = 100. * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Update scheduler
            scheduler.step()
            
            # Log metrics
            epoch_metrics = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            training_history.append(epoch_metrics)
            
            logger.info(
                f"Epoch {epoch+1}: Train Acc: {train_accuracy:.2f}%, "
                f"Val Acc: {val_accuracy:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Log to experiment tracking
            if hasattr(wandb, 'log'):
                wandb.log({f'{model_name}_{k}': v for k, v in epoch_metrics.items()})
                
            if hasattr(mlflow, 'log_metric'):
                for k, v in epoch_metrics.items():
                    mlflow.log_metric(f'{model_name}_{k}', v, step=epoch)
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0
                
                # Save checkpoint
                checkpoint_path = f"models/checkpoints/{model_name}_best.pth"
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_accuracy': best_accuracy,
                    'config': self.config
                }, checkpoint_path)
                
                logger.info(f"New best model saved with accuracy: {best_accuracy:.2f}%")
                
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= model_config['early_stopping_patience']:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        best_checkpoint = torch.load(f"models/checkpoints/{model_name}_best.pth")
        model.load_state_dict(best_checkpoint['model_state_dict'])
        
        # Save final model
        final_path = f"models/final/{model_name}_final.pth"
        Path(final_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), final_path)
        
        return {
            'best_accuracy': best_accuracy,
            'training_history': training_history,
            'final_model_path': final_path
        }
        
    def export_to_onnx(self, models: Dict[str, nn.Module]):
        """Export trained models to ONNX format"""
        logger.info("Exporting models to ONNX...")
        
        onnx_dir = Path("models/onnx")
        onnx_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in models.items():
            if model_name == 'ensemble':
                continue  # Skip ensemble for ONNX export
                
            try:
                model.eval()
                
                # Create dummy input
                if model_name == 'resnet1d_gru':
                    dummy_input = torch.randn(1, self.config['preprocessing']['num_channels'], 10000).to(self.device)
                elif model_name == 'efficientnet':
                    dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                
                # Export to ONNX
                onnx_path = onnx_dir / f"{model_name}.onnx"
                
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                
                logger.info(f"Exported {model_name} to ONNX: {onnx_path}")
                
            except Exception as e:
                logger.error(f"ONNX export failed for {model_name}: {e}")
                
    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("🚀 Starting HMS Novita AI Training Pipeline")
        logger.info("="*60)
        
        start_time = time.time()
        results = {}
        
        try:
            # Step 1: Download data
            if not self.download_data():
                raise Exception("Data download failed")
            results['data_download'] = 'success'
            
            # Step 2: Preprocess data
            if not self.preprocess_data():
                raise Exception("Data preprocessing failed")
            results['preprocessing'] = 'success'
            
            # Step 3: Create data loaders
            train_loader, val_loader = self.create_data_loaders()
            if train_loader is None:
                raise Exception("Data loader creation failed")
            results['data_loaders'] = 'success'
            
            # Step 4: Create models
            models = self.create_models()
            if not models:
                raise Exception("Model creation failed")
            results['model_creation'] = 'success'
            
            # Step 5: Train models
            training_results = {}
            for model_name, model in models.items():
                if model_name == 'ensemble':
                    continue  # Train ensemble separately
                    
                model_results = self.train_model(model, train_loader, val_loader, model_name)
                training_results[model_name] = model_results
                
                logger.info(f"✅ {model_name} training completed - Best Accuracy: {model_results['best_accuracy']:.2f}%")
            
            results['training'] = training_results
            
            # Step 6: Train ensemble if enabled
            if 'ensemble' in models:
                ensemble_results = self.train_ensemble(models, train_loader, val_loader)
                training_results['ensemble'] = ensemble_results
                logger.info(f"✅ Ensemble training completed - Best Accuracy: {ensemble_results['best_accuracy']:.2f}%")
            
            # Step 7: Export to ONNX
            self.export_to_onnx(models)
            results['onnx_export'] = 'success'
            
            # Calculate total time and cost
            total_time = (time.time() - start_time) / 3600
            estimated_cost = total_time * 3.35  # H100 cost per hour
            
            # Final results
            best_accuracy = max([r['best_accuracy'] for r in training_results.values()])
            
            final_results = {
                'success': True,
                'best_accuracy': best_accuracy,
                'total_time_hours': total_time,
                'estimated_cost_usd': estimated_cost,
                'training_results': training_results,
                'pipeline_results': results
            }
            
            # Save results
            with open('training_results.json', 'w') as f:
                json.dump(final_results, f, indent=2)
            
            logger.info("="*60)
            logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
            logger.info(f"🎯 Best Accuracy: {best_accuracy:.2f}%")
            logger.info(f"⏱️  Total Time: {total_time:.1f} hours")
            logger.info(f"💰 Estimated Cost: ${estimated_cost:.2f}")
            logger.info(f"🎯 Target Achieved: {'✅ YES' if best_accuracy >= 90.0 else '❌ NO'}")
            logger.info("="*60)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {'success': False, 'error': str(e)}
            
        finally:
            # Cleanup
            if hasattr(wandb, 'finish'):
                wandb.finish()
            if hasattr(mlflow, 'end_run'):
                mlflow.end_run()

def main():
    parser = argparse.ArgumentParser(description='HMS Novita AI Training Pipeline')
    parser.add_argument('--config', default='config/novita_production_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', help='Resume from checkpoint')
    parser.add_argument('--dry-run', action='store_true', help='Dry run without training')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = NovitaTrainingPipeline(args.config)
    
    if args.dry_run:
        logger.info("Dry run mode - checking configuration and setup")
        return
        
    # Run pipeline
    results = pipeline.run_complete_pipeline()
    
    # Exit with appropriate code
    sys.exit(0 if results.get('success', False) else 1)

if __name__ == '__main__':
    main() 