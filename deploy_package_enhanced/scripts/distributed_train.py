#!/usr/bin/env python3
"""
Distributed Training Script for HMS EEG Classification System

This script implements distributed training using:
- Data parallelism with DDP
- Gradient compression
- Fault tolerance
- Multi-GPU support
"""

import argparse
import logging
import yaml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.deployment.distributed_training import (
    DistributedConfig,
    create_distributed_trainer,
    DistributedTrainingManager
)
from src.deployment.memory_optimization import (
    MemoryConfig,
    MemoryOptimizer,
    MixedPrecisionOptimizer
)
from src.utils import HMSDataModule
from src.preprocessing import EEGPreprocessor, SpectrogramGenerator
from src.models import ResNet1D_GRU, EfficientNetSpectrogram, HMSEnsembleModel
from src.training import HMSTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """Setup distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    # Set device
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def train_distributed(rank: int, world_size: int, config: Dict[str, Any], args):
    """Training function for each distributed process."""
    logger.info(f"Running distributed training on rank {rank}")
    
    # Setup distributed
    setup_distributed(rank, world_size, args.backend)
    
    # Create distributed config
    dist_config = DistributedConfig(
        backend=args.backend,
        world_size=world_size,
        enable_gradient_compression=args.gradient_compression,
        compression_ratio=args.compression_ratio,
        enable_fault_tolerance=True,
        checkpoint_interval=args.checkpoint_interval,
        use_zero_optimizer=args.zero_optimizer,
        gradient_accumulation_steps=args.gradient_accumulation,
        sync_batch_norm=True
    )
    
    # Memory optimization config
    memory_config = MemoryConfig(
        enable_gradient_checkpointing=args.gradient_checkpointing,
        enable_mixed_precision=True,
        mixed_precision_dtype="fp16" if args.fp16 else "fp32",
        prefetch_factor=2,
        num_workers=args.num_workers
    )
    
    # Initialize preprocessors
    eeg_preprocessor = EEGPreprocessor(config)
    spectrogram_generator = SpectrogramGenerator(config)
    
    # Create data module
    data_module = HMSDataModule(
        config=config,
        eeg_preprocessor=eeg_preprocessor,
        spectrogram_generator=spectrogram_generator
    )
    data_module.setup()
    
    # Select model to train
    if args.model == 'resnet':
        model = ResNet1D_GRU(
            num_channels=config['preprocessing']['num_channels'],
            num_classes=config['dataset']['num_classes'],
            **config['models']['resnet1d_gru']
        )
    elif args.model == 'efficientnet':
        model = EfficientNetSpectrogram(
            num_classes=config['dataset']['num_classes'],
            **config['models']['efficientnet']
        )
    else:
        # Load base models for ensemble
        base_models = {}
        # ... load pre-trained base models ...
        model = HMSEnsembleModel(config)
    
    # Create distributed trainer
    trainer_components = create_distributed_trainer(model, dist_config)
    
    # Extract components
    ddp_model = trainer_components['model']
    optimizer = trainer_components['optimizer']
    fault_tolerant = trainer_components['fault_tolerant']
    
    # Apply memory optimizations
    memory_optimizer = MemoryOptimizer(memory_config)
    ddp_model = memory_optimizer.optimize_model_memory(ddp_model)
    
    # Mixed precision setup
    mixed_precision = MixedPrecisionOptimizer(memory_config)
    scaler = mixed_precision.scaler if args.fp16 else None
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        checkpoint_info = fault_tolerant.load_checkpoint(ddp_model, optimizer)
        start_epoch = checkpoint_info['epoch'] + 1
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    num_epochs = config['models'][args.model]['training']['epochs']
    
    for epoch in range(start_epoch, num_epochs):
        # Set epoch for distributed sampler
        data_module.train_sampler.set_epoch(epoch)
        
        # Training epoch
        ddp_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(data_module.train_dataloader()):
            # Move data to device
            if args.model == 'ensemble':
                eeg_data = batch['eeg'].cuda(rank)
                spec_data = batch['spectrogram'].cuda(rank)
                labels = batch['label'].cuda(rank)
                inputs = (eeg_data, spec_data)
            else:
                inputs = batch['data'].cuda(rank)
                labels = batch['label'].cuda(rank)
            
            # Forward pass with mixed precision
            with mixed_precision.get_autocast_context():
                outputs = ddp_model(inputs)
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                    
                loss = torch.nn.functional.cross_entropy(logits, labels)
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % dist_config.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                
                if (batch_idx + 1) % dist_config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            # Log progress
            if rank == 0 and batch_idx % 10 == 0:
                logger.info(f"Epoch [{epoch}/{num_epochs}] "
                          f"Batch [{batch_idx}/{len(data_module.train_dataloader())}] "
                          f"Loss: {loss.item():.4f}")
        
        # Calculate epoch metrics
        train_accuracy = train_correct / train_total
        avg_train_loss = train_loss / len(data_module.train_dataloader())
        
        # Validation
        if rank == 0:
            ddp_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in data_module.val_dataloader():
                    # Similar to training loop...
                    pass
            
            val_accuracy = val_correct / val_total if val_total > 0 else 0
            avg_val_loss = val_loss / len(data_module.val_dataloader()) if len(data_module.val_dataloader()) > 0 else 0
            
            logger.info(f"Epoch {epoch} - "
                      f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Save checkpoint
        if rank == 0 and (epoch + 1) % args.checkpoint_interval == 0:
            metrics = {
                'train_loss': avg_train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': avg_val_loss if rank == 0 else 0,
                'val_accuracy': val_accuracy if rank == 0 else 0
            }
            fault_tolerant.save_checkpoint(epoch, ddp_model, optimizer, metrics)
    
    # Cleanup
    cleanup_distributed()


def main():
    """Main distributed training entry point."""
    parser = argparse.ArgumentParser(description='Distributed Training for HMS EEG Classification')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--world-size', type=int, default=-1,
                        help='Number of distributed processes')
    parser.add_argument('--backend', type=str, default='nccl',
                        choices=['nccl', 'gloo'], help='Distributed backend')
    parser.add_argument('--model', type=str, default='resnet',
                        choices=['resnet', 'efficientnet', 'ensemble'],
                        help='Model to train')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--gradient-compression', action='store_true',
                        help='Enable gradient compression')
    parser.add_argument('--compression-ratio', type=float, default=0.1,
                        help='Gradient compression ratio')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing')
    parser.add_argument('--zero-optimizer', action='store_true',
                        help='Use ZeRO optimizer')
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                        help='Checkpoint save interval (epochs)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine world size
    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info(f"Auto-detected {args.world_size} GPUs")
    
    if args.world_size < 1:
        logger.error("No GPUs available for distributed training")
        return
    
    # Launch distributed training
    if args.world_size > 1:
        mp.spawn(
            train_distributed,
            args=(args.world_size, config, args),
            nprocs=args.world_size,
            join=True
        )
    else:
        # Single GPU training
        train_distributed(0, 1, config, args)
    
    logger.info("Distributed training complete!")


if __name__ == "__main__":
    main() 