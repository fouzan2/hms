"""
Training package for HMS harmful brain activity classification.
Provides comprehensive training strategies and optimization tools.
"""

from .cross_validation import (
    PatientIndependentCV,
    TimeSeriesCV,
    NestedCV,
    StratifiedBatchSampler,
    ValidationMonitor,
    CrossValidationPipeline
)

from .augmentation import (
    TimeDomainAugmentation,
    FrequencyDomainAugmentation,
    TimeShift,
    AmplitudeScale,
    GaussianNoise,
    ChannelDropout,
    TimeWarp,
    RandomBandStop,
    FrequencyMasking,
    TimeMasking,
    SpecAugment,
    SpectrogramMixup,
    CutMix,
    AugmentationPipeline,
    TestTimeAugmentation
)

from .class_balancing import (
    ClassWeightCalculator,
    BalancedBatchSampler,
    WeightedSampler,
    SMOTEBalancer,
    FocalLoss,
    ClassBalancedLoss,
    LDAMLoss,
    MixupLoss,
    BalancedDataLoader,
    HardExampleMining
)

from .hyperparameter_optimization import (
    BayesianOptimization,
    PopulationBasedTrainer,
    MultiObjectiveOptimization,
    NeuralArchitectureSearch,
    HyperparameterOptimizationPipeline,
    AutoMLPipeline
)

__all__ = [
    # Cross-validation
    'PatientIndependentCV',
    'TimeSeriesCV',
    'NestedCV',
    'StratifiedBatchSampler',
    'ValidationMonitor',
    'CrossValidationPipeline',
    
    # Augmentation
    'TimeDomainAugmentation',
    'FrequencyDomainAugmentation',
    'TimeShift',
    'AmplitudeScale',
    'GaussianNoise',
    'ChannelDropout',
    'TimeWarp',
    'RandomBandStop',
    'FrequencyMasking',
    'TimeMasking',
    'SpecAugment',
    'SpectrogramMixup',
    'CutMix',
    'AugmentationPipeline',
    'TestTimeAugmentation',
    
    # Class balancing
    'ClassWeightCalculator',
    'BalancedBatchSampler',
    'WeightedSampler',
    'SMOTEBalancer',
    'FocalLoss',
    'ClassBalancedLoss',
    'LDAMLoss',
    'MixupLoss',
    'BalancedDataLoader',
    'HardExampleMining',
    
    # Hyperparameter optimization
    'BayesianOptimization',
    'PopulationBasedTrainer',
    'MultiObjectiveOptimization',
    'NeuralArchitectureSearch',
    'HyperparameterOptimizationPipeline',
    'AutoMLPipeline'
] 