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


def create_model(model_type: str, config: dict, **kwargs):
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('resnet1d_gru', 'efficientnet', 'ensemble')
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
        'ensemble': HMSEnsembleModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model_classes[model_type]


__all__ = [
    # Factory functions
    'create_model',
    'get_model_class',
    
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
    'AdaptiveEnsembleSelection'
] 