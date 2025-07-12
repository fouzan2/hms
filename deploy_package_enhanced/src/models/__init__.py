"""
Models package for HMS harmful brain activity classification.
Provides deep learning architectures for EEG analysis.
"""

from .resnet1d_gru import (
    ResNet1D_GRU,
    ResNet1D_GRU_Trainer,
    BasicBlock1D,
    AttentionGRU,
    MultiHeadTemporalAttention,
    SEBlock1D,
    MultiScaleBlock1D,
    FocalLoss,
    TemporalConsistencyLoss,
    CombinedLoss
)

from .efficientnet_spectrogram import (
    EfficientNetSpectrogram,
    EfficientNetTrainer,
    SpectrogramAttention,
    FrequencyAwarePooling,
    SpectrogramAugmentation,
    FrequencyMasking,
    TimeMasking,
    ChannelSEBlock,
    DepthwiseSeparableConv2d,
    ProgressiveResizing
)

from .ensemble_model import (
    HMSEnsembleModel,
    StackingEnsemble,
    AttentionFusion,
    MetaLearner,
    BayesianModelAveraging,
    EnsembleDiversity,
    AdaptiveEnsembleSelection
)

from .eeg_foundation_model import (
    EEGFoundationModel,
    EEGFoundationConfig,
    EEGFoundationPreTrainer,
    MultiScaleTemporalEncoder,
    ChannelAttention,
    EEGTransformerBlock
)

from .eeg_foundation_trainer import (
    EEGFoundationTrainer,
    FineTuningConfig,
    EEGDataset,
    TransferLearningPipeline
)


def create_model(model_type: str, config: dict, **kwargs):
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('resnet1d_gru', 'efficientnet', 'ensemble', 'eeg_foundation')
        config: Configuration dictionary
        **kwargs: Additional model parameters
        
    Returns:
        Instantiated model
    """
    if model_type == 'resnet1d_gru':
        return ResNet1D_GRU(config, **kwargs)
    elif model_type == 'efficientnet':
        return EfficientNetSpectrogram(config, **kwargs)
    elif model_type == 'ensemble':
        # For ensemble, we need base models
        base_models = kwargs.get('base_models', {})
        return HMSEnsembleModel(base_models, config, **kwargs)
    elif model_type == 'eeg_foundation':
        # For foundation model, config should be EEGFoundationConfig
        if isinstance(config, dict):
            foundation_config = EEGFoundationConfig(**config)
        else:
            foundation_config = config
        return EEGFoundationModel(foundation_config, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_class(model_type: str):
    """
    Get model class by type.
    
    Args:
        model_type: Type of model
        
    Returns:
        Model class
    """
    model_classes = {
        'resnet1d_gru': ResNet1D_GRU,
        'efficientnet': EfficientNetSpectrogram,
        'ensemble': HMSEnsembleModel,
        'eeg_foundation': EEGFoundationModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model_classes[model_type]


def create_foundation_trainer(model: EEGFoundationModel, config: dict = None, **kwargs):
    """
    Create EEG Foundation Model trainer.
    
    Args:
        model: EEG Foundation Model instance
        config: Training configuration dictionary
        **kwargs: Additional trainer parameters
        
    Returns:
        EEGFoundationTrainer instance
    """
    if config is None:
        trainer_config = FineTuningConfig(**kwargs)
    elif isinstance(config, dict):
        trainer_config = FineTuningConfig(**config)
    else:
        trainer_config = config
        
    return EEGFoundationTrainer(model, trainer_config, **kwargs)


def create_transfer_learning_pipeline(foundation_model_path: str = None, config: dict = None, **kwargs):
    """
    Create transfer learning pipeline for EEG Foundation Model.
    
    Args:
        foundation_model_path: Path to pre-trained foundation model
        config: Fine-tuning configuration dictionary
        **kwargs: Additional pipeline parameters
        
    Returns:
        TransferLearningPipeline instance
    """
    if config is None:
        pipeline_config = FineTuningConfig(**kwargs)
    elif isinstance(config, dict):
        pipeline_config = FineTuningConfig(**config)
    else:
        pipeline_config = config
        
    return TransferLearningPipeline(foundation_model_path, pipeline_config)


__all__ = [
    # Factory functions
    'create_model',
    'get_model_class',
    'create_foundation_trainer',
    'create_transfer_learning_pipeline',
    
    # ResNet1D-GRU components
    'ResNet1D_GRU',
    'ResNet1D_GRU_Trainer',
    'BasicBlock1D',
    'AttentionGRU',
    'MultiHeadTemporalAttention',
    'SEBlock1D',
    'MultiScaleBlock1D',
    'FocalLoss',
    'TemporalConsistencyLoss',
    'CombinedLoss',
    
    # EfficientNet components
    'EfficientNetSpectrogram',
    'EfficientNetTrainer',
    'SpectrogramAttention',
    'FrequencyAwarePooling',
    'SpectrogramAugmentation',
    'FrequencyMasking',
    'TimeMasking',
    'ChannelSEBlock',
    'DepthwiseSeparableConv2d',
    'ProgressiveResizing',
    
    # Ensemble components
    'HMSEnsembleModel',
    'StackingEnsemble',
    'AttentionFusion',
    'MetaLearner',
    'BayesianModelAveraging',
    'EnsembleDiversity',
    'AdaptiveEnsembleSelection',
    
    # EEG Foundation Model components
    'EEGFoundationModel',
    'EEGFoundationConfig',
    'EEGFoundationPreTrainer',
    'MultiScaleTemporalEncoder',
    'ChannelAttention',
    'EEGTransformerBlock',
    'EEGFoundationTrainer',
    'FineTuningConfig',
    'EEGDataset',
    'TransferLearningPipeline'
] 