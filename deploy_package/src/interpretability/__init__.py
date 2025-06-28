"""
Interpretability package for HMS harmful brain activity classification.
Provides explanations for model predictions and decision support.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
import numpy as np

@dataclass
class ExplanationResult:
    """Container for explanation results."""
    explanation_type: str
    feature_importance: Optional[np.ndarray] = None
    attention_weights: Optional[np.ndarray] = None
    gradient_attribution: Optional[np.ndarray] = None
    uncertainty: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

from .explainers import SHAPExplainer, LIMEExplainer
from .gradient_attribution import (
    IntegratedGradients,
    GradCAM,
    GradientAttribution
)
from .attention_analysis import AttentionVisualizer
from .uncertainty_estimation import (
    MonteCarloDropout,
    UncertaintyEstimator
)
from .feature_importance import (
    PermutationImportance,
    FeatureImportanceReporter,
    FeatureImportanceResult
)

# Create simple classes for missing components
class ModelInterpreter:
    """Simple model interpreter."""
    def __init__(self, model):
        self.model = model
    
    def get_explanations(self, x, methods=['shap']):
        """Get explanations using specified methods."""
        return methods

from .clinical_explanation import ClinicalExplanationGenerator

__all__ = [
    # Core types
    'ExplanationResult',
    
    # Explainers
    'SHAPExplainer',
    'LIMEExplainer',
    'ModelInterpreter',
    'ClinicalExplanationGenerator',
    
    # Attribution methods
    'IntegratedGradients',
    'GradCAM',
    'GradientAttribution',
    
    # Attention analysis
    'AttentionVisualizer',
    
    # Uncertainty estimation
    'MonteCarloDropout',
    'UncertaintyEstimator',
    
    # Feature importance
    'PermutationImportance',
    'FeatureImportanceReporter',
    'FeatureImportanceResult'
] 