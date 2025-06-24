# Model Interpretability Module for EEG Harmful Brain Activity Classification

from .gradient_attribution import (
    IntegratedGradients,
    GradCAM,
    GuidedBackpropagation,
    SaliencyMaps,
    LayerwiseRelevancePropagation
)

from .attention_analysis import (
    AttentionVisualizer,
    MultiHeadAttentionAnalyzer,
    AttentionRollout,
    AttentionConsistencyChecker,
    AttentionGuidedFeatureSelector
)

from .uncertainty import (
    MonteCarloDropout,
    EnsembleUncertainty,
    BayesianNeuralNetwork,
    TemperatureScaling,
    UncertaintyVisualizer,
    UncertaintyBasedActiveLearning
)

from .feature_importance import (
    PermutationImportance,
    PartialDependencePlotter,
    AccumulatedLocalEffects,
    ClinicalFeatureValidator,
    FeatureInteractionAnalyzer,
    FeatureImportanceReporter,
    FeatureStabilityAnalyzer
)

from .explainers import (
    SHAPExplainer,
    LIMEExplainer,
    ModelInterpreter,
    ClinicalExplanationGenerator
)

__all__ = [
    # Gradient Attribution
    'IntegratedGradients',
    'GradCAM',
    'GuidedBackpropagation',
    'SaliencyMaps',
    'LayerwiseRelevancePropagation',
    
    # Attention Analysis
    'AttentionVisualizer',
    'MultiHeadAttentionAnalyzer',
    'AttentionRollout',
    'AttentionConsistencyChecker',
    'AttentionGuidedFeatureSelector',
    
    # Uncertainty
    'MonteCarloDropout',
    'EnsembleUncertainty',
    'BayesianNeuralNetwork',
    'TemperatureScaling',
    'UncertaintyVisualizer',
    'UncertaintyBasedActiveLearning',
    
    # Feature Importance
    'PermutationImportance',
    'PartialDependencePlotter',
    'AccumulatedLocalEffects',
    'ClinicalFeatureValidator',
    'FeatureInteractionAnalyzer',
    'FeatureImportanceReporter',
    'FeatureStabilityAnalyzer',
    
    # Explainers
    'SHAPExplainer',
    'LIMEExplainer',
    'ModelInterpreter',
    'ClinicalExplanationGenerator'
] 