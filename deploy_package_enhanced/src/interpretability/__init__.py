"""
Interpretability package for HMS Brain Activity Classification.
Provides explainable AI, counterfactual reasoning, and gradient-based explanations.
"""

from .explainable_ai import (
    ExplainableAI,
    CounterfactualGenerator,
    SHAPExplainer,
    AttentionVisualizer,
    ClinicalInterpreter,
    ExplanationConfig
)

from .gradient_explanations import (
    GradientExplanationFramework,
    GradCAM,
    IntegratedGradients,
    GuidedBackpropagation,
    SmoothGrad,
    LayerActivationAnalysis
)


def create_explainable_ai(model, config=None, device='auto'):
    """
    Create explainable AI framework for a given model.
    
    Args:
        model: Trained model to explain
        config: ExplanationConfig instance or dict
        device: Computing device
        
    Returns:
        ExplainableAI instance
    """
    if device == 'auto':
        device = 'cuda' if hasattr(model, 'device') else 'cpu'
    
    if config is None:
        config = ExplanationConfig()
    elif isinstance(config, dict):
        config = ExplanationConfig(**config)
    
    return ExplainableAI(model, config, device)


def create_gradient_explainer(model, target_layers=None, device='auto'):
    """
    Create gradient-based explanation framework.
    
    Args:
        model: Trained model to explain
        target_layers: List of layer names for Grad-CAM
        device: Computing device
        
    Returns:
        GradientExplanationFramework instance
    """
    if device == 'auto':
        device = 'cuda' if hasattr(model, 'device') else 'cpu'
    
    return GradientExplanationFramework(model, target_layers, device)


__all__ = [
    # Main frameworks
    'ExplainableAI',
    'GradientExplanationFramework',
    
    # Explainable AI components
    'CounterfactualGenerator',
    'SHAPExplainer',
    'AttentionVisualizer',
    'ClinicalInterpreter',
    'ExplanationConfig',
    
    # Gradient explanation components
    'GradCAM',
    'IntegratedGradients',
    'GuidedBackpropagation',
    'SmoothGrad',
    'LayerActivationAnalysis',
    
    # Factory functions
    'create_explainable_ai',
    'create_gradient_explainer'
] 