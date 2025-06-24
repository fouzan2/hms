#!/usr/bin/env python3
"""Main training script for HMS Brain Activity Classification System."""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
import warnings
warnings.filterwarnings('ignore')

from src.utils import download_dataset, HMSDataModule
from src.preprocessing import EEGPreprocessor, SpectrogramGenerator
from src.models import ResNet1D_GRU, EfficientNetSpectrogram, HMSEnsembleModel
from src.training import HMSTrainer
from src.evaluation import HMSEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: dict):
    """Download and prepare data for training."""
    logger.info("Downloading dataset...")
    download_dataset(
        output_dir=config['dataset']['raw_data_path'],
        dataset_name=config['dataset']['name'],
        batch_size=config['dataset']['download_batch_size']
    )
    
    logger.info("Data preparation complete!")


def preprocess_data(config: dict):
    """Preprocess EEG data and generate spectrograms."""
    logger.info("Preprocessing EEG data...")
    
    # Initialize preprocessors
    eeg_preprocessor = EEGPreprocessor(config)
    spectrogram_generator = SpectrogramGenerator(config)
    
    # Process data
    processed_data_path = Path(config['dataset']['processed_data_path'])
    processed_data_path.mkdir(parents=True, exist_ok=True)
    
    # Note: Actual preprocessing would be done within the dataset/dataloader
    logger.info("Preprocessing configuration initialized")
    
    return eeg_preprocessor, spectrogram_generator


def train_individual_models(config: dict, data_module: HMSDataModule):
    """Train individual models (ResNet1D-GRU and EfficientNet)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    models = {}
    
    # Train ResNet1D-GRU
    if config['models']['resnet1d_gru']['enabled']:
        logger.info("Training ResNet1D-GRU model...")
        resnet_model = ResNet1D_GRU(
            num_channels=config['preprocessing']['num_channels'],
            num_classes=config['dataset']['num_classes'],
            **config['models']['resnet1d_gru']
        ).to(device)
        
        resnet_trainer = HMSTrainer(
            model=resnet_model,
            config=config,
            model_name='resnet1d_gru'
        )
        
        resnet_trainer.train(data_module)
        models['resnet1d_gru'] = resnet_model
    
    # Train EfficientNet
    if config['models']['efficientnet']['enabled']:
        logger.info("Training EfficientNet-Spectrogram model...")
        efficientnet_model = EfficientNetSpectrogram(
            num_classes=config['dataset']['num_classes'],
            **config['models']['efficientnet']
        ).to(device)
        
        efficientnet_trainer = HMSTrainer(
            model=efficientnet_model,
            config=config,
            model_name='efficientnet'
        )
        
        efficientnet_trainer.train(data_module)
        models['efficientnet'] = efficientnet_model
    
    return models


def train_ensemble(config: dict, base_models: dict, data_module: HMSDataModule):
    """Train ensemble model."""
    if not config['models']['ensemble']['enabled']:
        return None
    
    logger.info("Training ensemble model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ensemble_model = HMSEnsembleModel(
        base_models=base_models,
        num_classes=config['dataset']['num_classes'],
        config=config['models']['ensemble']
    ).to(device)
    
    ensemble_trainer = HMSTrainer(
        model=ensemble_model,
        config=config,
        model_name='ensemble'
    )
    
    ensemble_trainer.train(data_module)
    
    return ensemble_model


def evaluate_models(config: dict, models: dict, data_module: HMSDataModule):
    """Evaluate all trained models."""
    logger.info("Evaluating models...")
    
    evaluator = HMSEvaluator(config)
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        metrics = evaluator.evaluate(
            model=model,
            dataloader=data_module.val_dataloader(),
            model_name=model_name
        )
        results[model_name] = metrics
        
        # Log results
        logger.info(f"{model_name} Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    # Generate evaluation report
    evaluator.generate_report(results)
    
    return results


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train HMS Brain Activity Classification Models')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--download-only', action='store_true',
                        help='Only download the dataset')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip dataset download')
    parser.add_argument('--model', type=str, choices=['resnet', 'efficientnet', 'ensemble', 'all'],
                        default='all', help='Which model to train')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Only evaluate existing models')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create necessary directories
    Path('logs').mkdir(exist_ok=True)
    Path(config['dataset']['processed_data_path']).mkdir(parents=True, exist_ok=True)
    Path(config['training']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    # Download data if needed
    if not args.skip_download:
        prepare_data(config)
        
    if args.download_only:
        logger.info("Download complete. Exiting.")
        return
    
    # Initialize preprocessors
    eeg_preprocessor, spectrogram_generator = preprocess_data(config)
    
    # Create data module
    data_module = HMSDataModule(
        config=config,
        eeg_preprocessor=eeg_preprocessor,
        spectrogram_generator=spectrogram_generator
    )
    data_module.setup()
    
    if args.evaluate_only:
        # Load existing models and evaluate
        logger.info("Loading existing models for evaluation...")
        # Implementation for loading models from checkpoints
        pass
    else:
        # Train models
        if args.model == 'all':
            # Train individual models
            base_models = train_individual_models(config, data_module)
            
            # Train ensemble
            ensemble_model = train_ensemble(config, base_models, data_module)
            
            # Combine all models for evaluation
            all_models = base_models.copy()
            if ensemble_model:
                all_models['ensemble'] = ensemble_model
        else:
            # Train specific model
            all_models = {}
            if args.model == 'resnet':
                config['models']['efficientnet']['enabled'] = False
                config['models']['ensemble']['enabled'] = False
            elif args.model == 'efficientnet':
                config['models']['resnet1d_gru']['enabled'] = False
                config['models']['ensemble']['enabled'] = False
            elif args.model == 'ensemble':
                base_models = train_individual_models(config, data_module)
                ensemble_model = train_ensemble(config, base_models, data_module)
                all_models = {'ensemble': ensemble_model}
            else:
                base_models = train_individual_models(config, data_module)
                all_models = base_models
        
        # Evaluate models
        if all_models:
            evaluate_models(config, all_models, data_module)
    
    logger.info("Training pipeline complete!")


if __name__ == "__main__":
    main() 