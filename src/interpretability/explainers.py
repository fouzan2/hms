"""
SHAP and LIME explainers for EEG model interpretability.
Provides model-agnostic explanations for individual predictions.
"""

import torch
import torch.nn as nn
import numpy as np
import shap
import lime
import lime.lime_tabular
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm


@dataclass
class ExplanationResult:
    """Container for explanation results."""
    instance_idx: int
    prediction: np.ndarray
    true_label: Optional[int]
    explanation_values: np.ndarray
    feature_names: List[str]
    method: str
    metadata: Dict[str, any]


class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) for model interpretability.
    Provides unified framework for feature attribution.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda', background_data: Optional[torch.Tensor] = None):
        self.model = model
        self.device = device
        self.model.eval()
        self.background_data = background_data
        self.explainer = None
        
    def setup_explainer(self, 
                        X_train: torch.Tensor,
                        explainer_type: str = 'deep',
                        n_background: int = 100):
        """
        Set up SHAP explainer based on model type.
        
        Args:
            X_train: Training data for background
            explainer_type: Type of explainer ('deep', 'gradient', 'kernel')
            n_background: Number of background samples
        """
        # Select background samples
        if self.background_data is None:
            indices = torch.randperm(len(X_train))[:n_background]
            self.background_data = X_train[indices].to(self.device)
            
        # Create prediction function wrapper
        def model_predict(x):
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                output = self.model(x)
                if output.shape[-1] > 1:  # Classification
                    output = torch.softmax(output, dim=-1)
            return output.cpu().numpy()
            
        # Initialize appropriate explainer
        if explainer_type == 'deep':
            self.explainer = shap.DeepExplainer(self.model, self.background_data)
        elif explainer_type == 'gradient':
            self.explainer = shap.GradientExplainer(self.model, self.background_data)
        elif explainer_type == 'kernel':
            # For kernel SHAP, we need to work with numpy arrays
            background_np = self.background_data.cpu().numpy()
            self.explainer = shap.KernelExplainer(model_predict, background_np)
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
            
        self.explainer_type = explainer_type
        
    def explain_instance(self,
                         instance: torch.Tensor,
                         target_class: Optional[int] = None,
                         feature_names: Optional[List[str]] = None) -> ExplanationResult:
        """
        Explain a single instance prediction.
        
        Args:
            instance: Single input instance
            target_class: Class to explain (None for predicted class)
            feature_names: Names of features
            
        Returns:
            ExplanationResult with SHAP values
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call setup_explainer first.")
            
        # Get prediction
        with torch.no_grad():
            prediction = self.model(instance.unsqueeze(0).to(self.device))
            if prediction.shape[-1] > 1:
                prediction = torch.softmax(prediction, dim=-1)
            prediction = prediction.cpu().numpy()
            
        if target_class is None and prediction.shape[-1] > 1:
            target_class = prediction.argmax()
            
        # Compute SHAP values
        if self.explainer_type in ['deep', 'gradient']:
            shap_values = self.explainer.shap_values(
                instance.unsqueeze(0).to(self.device)
            )
        else:  # kernel
            shap_values = self.explainer.shap_values(
                instance.unsqueeze(0).cpu().numpy()
            )
            
        # Handle multi-class case
        if isinstance(shap_values, list):
            if target_class is not None:
                shap_values = shap_values[target_class]
            else:
                # Use values for predicted class
                shap_values = shap_values[prediction.argmax()]
                
        # Ensure proper shape
        if len(shap_values.shape) > 2:
            shap_values = shap_values.reshape(shap_values.shape[0], -1)
            
        # Generate feature names if not provided
        if feature_names is None:
            n_features = shap_values.shape[-1]
            feature_names = [f"Feature_{i}" for i in range(n_features)]
            
        return ExplanationResult(
            instance_idx=0,
            prediction=prediction.squeeze(),
            true_label=None,
            explanation_values=shap_values.squeeze(),
            feature_names=feature_names,
            method=f'SHAP_{self.explainer_type}',
            metadata={
                'target_class': target_class,
                'base_value': self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else None
            }
        )
        
    def explain_batch(self,
                      X_batch: torch.Tensor,
                      y_batch: Optional[torch.Tensor] = None,
                      target_classes: Optional[List[int]] = None,
                      feature_names: Optional[List[str]] = None) -> List[ExplanationResult]:
        """
        Explain multiple instances.
        
        Args:
            X_batch: Batch of inputs
            y_batch: True labels (optional)
            target_classes: Classes to explain for each instance
            feature_names: Feature names
            
        Returns:
            List of ExplanationResults
        """
        explanations = []
        
        for idx in tqdm(range(len(X_batch)), desc="Computing SHAP explanations"):
            instance = X_batch[idx]
            true_label = y_batch[idx].item() if y_batch is not None else None
            target_class = target_classes[idx] if target_classes else None
            
            result = self.explain_instance(instance, target_class, feature_names)
            result.instance_idx = idx
            result.true_label = true_label
            
            explanations.append(result)
            
        return explanations
        
    def plot_explanation(self,
                         explanation: ExplanationResult,
                         plot_type: str = 'waterfall',
                         max_features: int = 20,
                         save_path: Optional[str] = None):
        """
        Visualize SHAP explanation.
        
        Args:
            explanation: ExplanationResult to visualize
            plot_type: Type of plot ('waterfall', 'bar', 'beeswarm')
            max_features: Maximum features to show
            save_path: Path to save plot
        """
        # Select top features by absolute SHAP value
        shap_values = explanation.explanation_values
        feature_names = explanation.feature_names
        
        if len(shap_values) > max_features:
            top_indices = np.argsort(np.abs(shap_values))[-max_features:]
            shap_values = shap_values[top_indices]
            feature_names = [feature_names[i] for i in top_indices]
            
        # Create appropriate plot
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'waterfall':
            # Create waterfall plot manually
            cumsum = np.cumsum(shap_values)
            base_value = explanation.metadata.get('base_value', 0)
            
            y_pos = np.arange(len(shap_values))
            colors = ['red' if v < 0 else 'blue' for v in shap_values]
            
            plt.barh(y_pos, shap_values, left=cumsum - shap_values + base_value,
                    color=colors, alpha=0.7)
            plt.yticks(y_pos, feature_names)
            plt.xlabel('SHAP Value')
            plt.title('SHAP Waterfall Plot')
            plt.axvline(x=base_value, color='black', linestyle='--', alpha=0.5)
            
        elif plot_type == 'bar':
            # Bar plot
            y_pos = np.arange(len(shap_values))
            colors = ['red' if v < 0 else 'blue' for v in shap_values]
            
            plt.barh(y_pos, shap_values, color=colors, alpha=0.7)
            plt.yticks(y_pos, feature_names)
            plt.xlabel('SHAP Value')
            plt.title('SHAP Feature Importance')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def compute_global_importance(self,
                                  X_sample: torch.Tensor,
                                  feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute global feature importance using SHAP values.
        
        Args:
            X_sample: Sample of data to compute importance
            feature_names: Feature names
            
        Returns:
            Dictionary of feature importances
        """
        # Get SHAP values for sample
        if self.explainer_type in ['deep', 'gradient']:
            shap_values = self.explainer.shap_values(X_sample.to(self.device))
        else:
            shap_values = self.explainer.shap_values(X_sample.cpu().numpy())
            
        # Handle multi-class
        if isinstance(shap_values, list):
            # Average across classes
            shap_values = np.mean(shap_values, axis=0)
            
        # Compute mean absolute SHAP values
        if len(shap_values.shape) > 2:
            shap_values = shap_values.reshape(shap_values.shape[0], -1)
            
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Create feature names if needed
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(mean_abs_shap))]
            
        # Create importance dictionary
        importance_dict = {
            feature: importance 
            for feature, importance in zip(feature_names, mean_abs_shap)
        }
        
        return importance_dict


class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) for EEG models.
    Provides local linear approximations of model behavior.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        self.explainer = None
        
    def setup_explainer(self,
                        training_data: torch.Tensor,
                        feature_names: Optional[List[str]] = None,
                        categorical_features: Optional[List[int]] = None,
                        discretize_continuous: bool = True):
        """
        Set up LIME explainer.
        
        Args:
            training_data: Training data for statistics
            feature_names: Names of features
            categorical_features: Indices of categorical features
            discretize_continuous: Whether to discretize continuous features
        """
        # Flatten data if needed
        if len(training_data.shape) > 2:
            training_data_flat = training_data.reshape(training_data.shape[0], -1)
        else:
            training_data_flat = training_data
            
        # Create feature names if not provided
        if feature_names is None:
            n_features = training_data_flat.shape[1]
            feature_names = [f"Feature_{i}" for i in range(n_features)]
            
        # Create prediction function wrapper
        def predict_fn(X):
            if isinstance(X, np.ndarray):
                X_tensor = torch.tensor(X, dtype=torch.float32)
                
                # Reshape back if needed
                if len(training_data.shape) > 2:
                    original_shape = training_data.shape[1:]
                    X_tensor = X_tensor.reshape(-1, *original_shape)
                    
                X_tensor = X_tensor.to(self.device)
                
            with torch.no_grad():
                output = self.model(X_tensor)
                if output.shape[-1] > 1:  # Classification
                    output = torch.softmax(output, dim=-1)
                    
            return output.cpu().numpy()
            
        # Initialize LIME explainer
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data_flat.cpu().numpy(),
            feature_names=feature_names,
            categorical_features=categorical_features,
            mode='classification' if self._is_classification() else 'regression',
            discretize_continuous=discretize_continuous,
            random_state=42
        )
        
        self.predict_fn = predict_fn
        self.original_shape = training_data.shape[1:] if len(training_data.shape) > 2 else None
        
    def explain_instance(self,
                         instance: torch.Tensor,
                         target_class: Optional[int] = None,
                         num_features: int = 10,
                         num_samples: int = 5000) -> ExplanationResult:
        """
        Explain a single instance using LIME.
        
        Args:
            instance: Instance to explain
            target_class: Class to explain (None for all)
            num_features: Number of features in explanation
            num_samples: Number of samples for local approximation
            
        Returns:
            ExplanationResult with LIME explanation
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call setup_explainer first.")
            
        # Flatten instance if needed
        if len(instance.shape) > 1:
            instance_flat = instance.flatten()
        else:
            instance_flat = instance
            
        # Get prediction for instance
        with torch.no_grad():
            pred = self.model(instance.unsqueeze(0).to(self.device))
            if pred.shape[-1] > 1:
                pred = torch.softmax(pred, dim=-1)
            prediction = pred.cpu().numpy().squeeze()
            
        # Generate explanation
        explanation = self.explainer.explain_instance(
            instance_flat.cpu().numpy(),
            self.predict_fn,
            labels=[target_class] if target_class is not None else None,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Extract explanation values
        if target_class is None and self._is_classification():
            target_class = prediction.argmax()
            
        exp_list = explanation.as_list(label=target_class if self._is_classification() else 0)
        
        # Parse explanation into arrays
        feature_names = []
        explanation_values = []
        
        for feature_exp, value in exp_list:
            feature_names.append(feature_exp)
            explanation_values.append(value)
            
        # Pad to full feature size
        full_explanation = np.zeros(instance_flat.shape[0])
        for i, (feature_exp, value) in enumerate(exp_list):
            # Extract feature index from name
            if 'Feature_' in feature_exp:
                try:
                    idx = int(feature_exp.split('_')[1].split()[0])
                    full_explanation[idx] = value
                except:
                    pass
                    
        return ExplanationResult(
            instance_idx=0,
            prediction=prediction,
            true_label=None,
            explanation_values=full_explanation,
            feature_names=[f"Feature_{i}" for i in range(len(full_explanation))],
            method='LIME',
            metadata={
                'target_class': target_class,
                'num_features': num_features,
                'num_samples': num_samples,
                'local_explanation': exp_list
            }
        )
        
    def _is_classification(self):
        """Check if model is for classification based on output."""
        # Simple heuristic: if output has more than 1 dimension, it's classification
        dummy_input = torch.randn(1, 10).to(self.device)  # Dummy input
        try:
            with torch.no_grad():
                output = self.model(dummy_input)
            return output.shape[-1] > 1
        except:
            return True  # Default to classification
            
    def plot_explanation(self,
                         explanation: ExplanationResult,
                         save_path: Optional[str] = None):
        """
        Plot LIME explanation.
        
        Args:
            explanation: ExplanationResult to plot
            save_path: Path to save plot
        """
        # Get local explanation from metadata
        local_exp = explanation.metadata.get('local_explanation', [])
        
        if not local_exp:
            print("No local explanation found in metadata")
            return
            
        # Extract features and values
        features = []
        values = []
        for feature_exp, value in local_exp:
            features.append(feature_exp)
            values.append(value)
            
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = np.arange(len(features))
        colors = ['red' if v < 0 else 'green' for v in values]
        
        ax.barh(y_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('LIME Weight')
        ax.set_title('LIME Local Explanation')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class ModelInterpreter:
    """
    Unified interface for model interpretation using multiple methods.
    Combines SHAP, LIME, and other interpretation techniques.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Initialize explainers
        self.shap_explainer = SHAPExplainer(model, device)
        self.lime_explainer = LIMEExplainer(model, device)
        
        # Store explanations for aggregation
        self.explanations = []
        
    def setup(self,
              training_data: torch.Tensor,
              feature_names: Optional[List[str]] = None,
              shap_type: str = 'deep',
              n_background: int = 100):
        """
        Set up all explainers.
        
        Args:
            training_data: Training data for explainer initialization
            feature_names: Feature names
            shap_type: Type of SHAP explainer
            n_background: Number of background samples for SHAP
        """
        # Setup SHAP
        self.shap_explainer.setup_explainer(
            training_data, shap_type, n_background
        )
        
        # Setup LIME
        self.lime_explainer.setup_explainer(
            training_data, feature_names
        )
        
        self.feature_names = feature_names
        
    def explain_instance(self,
                         instance: torch.Tensor,
                         methods: List[str] = ['shap', 'lime'],
                         target_class: Optional[int] = None) -> Dict[str, ExplanationResult]:
        """
        Explain instance using multiple methods.
        
        Args:
            instance: Instance to explain
            methods: List of methods to use
            target_class: Target class for explanation
            
        Returns:
            Dictionary of explanations by method
        """
        explanations = {}
        
        if 'shap' in methods:
            explanations['shap'] = self.shap_explainer.explain_instance(
                instance, target_class, self.feature_names
            )
            
        if 'lime' in methods:
            explanations['lime'] = self.lime_explainer.explain_instance(
                instance, target_class
            )
            
        return explanations
        
    def compare_explanations(self,
                             explanations: Dict[str, ExplanationResult],
                             save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Compare explanations from different methods.
        
        Args:
            explanations: Dictionary of explanations
            save_path: Path to save comparison plot
            
        Returns:
            Comparison metrics
        """
        if len(explanations) < 2:
            return {}
            
        # Extract explanation values
        method_values = {}
        for method, exp in explanations.items():
            method_values[method] = exp.explanation_values
            
        # Compute correlation between methods
        correlations = {}
        methods = list(method_values.keys())
        
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1, method2 = methods[i], methods[j]
                corr = np.corrcoef(
                    method_values[method1], 
                    method_values[method2]
                )[0, 1]
                correlations[f"{method1}_vs_{method2}"] = corr
                
        # Plot comparison if requested
        if save_path:
            self._plot_explanation_comparison(explanations, save_path)
            
        # Compute agreement metrics
        agreement_metrics = self._compute_agreement_metrics(method_values)
        
        return {
            'correlations': correlations,
            'agreement_metrics': agreement_metrics
        }
        
    def _plot_explanation_comparison(self, explanations, save_path):
        """Plot comparison of explanations from different methods."""
        n_methods = len(explanations)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6))
        
        if n_methods == 1:
            axes = [axes]
            
        for idx, (method, exp) in enumerate(explanations.items()):
            ax = axes[idx]
            
            # Select top features
            values = exp.explanation_values
            n_show = min(20, len(values))
            top_indices = np.argsort(np.abs(values))[-n_show:]
            
            top_values = values[top_indices]
            top_names = [f"Feature_{i}" for i in top_indices]
            
            # Plot
            y_pos = np.arange(n_show)
            colors = ['red' if v < 0 else 'blue' for v in top_values]
            
            ax.barh(y_pos, top_values, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_names)
            ax.set_xlabel(f'{method.upper()} Value')
            ax.set_title(f'{method.upper()} Explanation')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _compute_agreement_metrics(self, method_values):
        """Compute agreement metrics between methods."""
        metrics = {}
        
        # Rank agreement
        method_ranks = {}
        for method, values in method_values.items():
            ranks = len(values) - np.argsort(np.argsort(np.abs(values)))
            method_ranks[method] = ranks
            
        # Compute Spearman correlation between ranks
        methods = list(method_ranks.keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1, method2 = methods[i], methods[j]
                from scipy.stats import spearmanr
                corr, _ = spearmanr(method_ranks[method1], method_ranks[method2])
                metrics[f"rank_correlation_{method1}_vs_{method2}"] = corr
                
        # Top-k agreement
        k = 10
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1, method2 = methods[i], methods[j]
                
                top_k_1 = set(np.argsort(np.abs(method_values[method1]))[-k:])
                top_k_2 = set(np.argsort(np.abs(method_values[method2]))[-k:])
                
                overlap = len(top_k_1 & top_k_2) / k
                metrics[f"top_{k}_overlap_{method1}_vs_{method2}"] = overlap
                
        return metrics


class ClinicalExplanationGenerator:
    """
    Generate clinical-friendly explanations for EEG predictions.
    Translates technical explanations into clinical language.
    """
    
    def __init__(self, 
                 class_names: Dict[int, str],
                 feature_descriptions: Optional[Dict[str, str]] = None):
        self.class_names = class_names
        self.feature_descriptions = feature_descriptions or {}
        
        # Clinical importance thresholds
        self.importance_thresholds = {
            'very_high': 0.8,
            'high': 0.6,
            'moderate': 0.4,
            'low': 0.2
        }
        
    def generate_clinical_explanation(self,
                                      explanation: ExplanationResult,
                                      patient_info: Optional[Dict] = None) -> str:
        """
        Generate clinical explanation from technical explanation.
        
        Args:
            explanation: Technical explanation result
            patient_info: Optional patient information
            
        Returns:
            Clinical explanation text
        """
        # Start with prediction summary
        prediction = explanation.prediction
        if len(prediction.shape) > 0 and prediction.shape[-1] > 1:
            predicted_class = prediction.argmax()
            confidence = prediction[predicted_class]
            class_name = self.class_names.get(predicted_class, f"Class {predicted_class}")
        else:
            predicted_class = 0
            confidence = float(prediction)
            class_name = "Positive"
            
        explanation_text = f"## Clinical Analysis Report\n\n"
        
        # Add patient info if available
        if patient_info:
            explanation_text += f"**Patient ID:** {patient_info.get('id', 'Unknown')}\n"
            explanation_text += f"**Age:** {patient_info.get('age', 'Unknown')}\n"
            explanation_text += f"**Recording Date:** {patient_info.get('date', 'Unknown')}\n\n"
            
        # Prediction summary
        explanation_text += f"### Prediction Summary\n"
        explanation_text += f"**Predicted Condition:** {class_name}\n"
        explanation_text += f"**Confidence Level:** {confidence:.1%}\n"
        explanation_text += f"**Clinical Confidence:** {self._interpret_confidence(confidence)}\n\n"
        
        # Key findings
        explanation_text += f"### Key EEG Findings\n"
        
        # Get top contributing features
        feature_importance = explanation.explanation_values
        top_indices = np.argsort(np.abs(feature_importance))[-5:][::-1]
        
        for idx in top_indices:
            feature_name = explanation.feature_names[idx]
            importance = feature_importance[idx]
            
            # Translate feature to clinical terms
            clinical_feature = self._translate_feature(feature_name, idx)
            importance_level = self._categorize_importance(np.abs(importance))
            direction = "increased" if importance > 0 else "decreased"
            
            explanation_text += f"- **{clinical_feature}**: {importance_level} importance ({direction} activity)\n"
            
        # Clinical interpretation
        explanation_text += f"\n### Clinical Interpretation\n"
        explanation_text += self._generate_clinical_interpretation(
            predicted_class, top_indices, feature_importance
        )
        
        # Recommendations
        explanation_text += f"\n### Recommendations\n"
        explanation_text += self._generate_recommendations(confidence, predicted_class)
        
        return explanation_text
        
    def _interpret_confidence(self, confidence: float) -> str:
        """Interpret confidence level in clinical terms."""
        if confidence > 0.9:
            return "Very High - Strong evidence"
        elif confidence > 0.7:
            return "High - Clear evidence"
        elif confidence > 0.5:
            return "Moderate - Some evidence"
        else:
            return "Low - Uncertain, consider additional testing"
            
    def _translate_feature(self, feature_name: str, feature_idx: int) -> str:
        """Translate technical feature name to clinical description."""
        if feature_name in self.feature_descriptions:
            return self.feature_descriptions[feature_name]
            
        # Default translations based on common EEG features
        if 'delta' in feature_name.lower():
            return "Delta wave activity (0.5-4 Hz)"
        elif 'theta' in feature_name.lower():
            return "Theta wave activity (4-8 Hz)"
        elif 'alpha' in feature_name.lower():
            return "Alpha wave activity (8-13 Hz)"
        elif 'beta' in feature_name.lower():
            return "Beta wave activity (13-30 Hz)"
        elif 'gamma' in feature_name.lower():
            return "Gamma wave activity (30-50 Hz)"
        elif 'spike' in feature_name.lower():
            return "Spike activity"
        elif 'sharp' in feature_name.lower():
            return "Sharp wave activity"
        else:
            return f"EEG Pattern {feature_idx + 1}"
            
    def _categorize_importance(self, importance: float) -> str:
        """Categorize importance level."""
        for level, threshold in self.importance_thresholds.items():
            if importance >= threshold:
                return level.replace('_', ' ').title()
        return "minimal"
        
    def _generate_clinical_interpretation(self, 
                                          predicted_class: int,
                                          top_features: List[int],
                                          importances: np.ndarray) -> str:
        """Generate clinical interpretation based on features."""
        class_name = self.class_names.get(predicted_class, f"Class {predicted_class}")
        
        interpretation = f"The EEG analysis indicates patterns consistent with **{class_name}**. "
        
        # Add class-specific interpretations
        if predicted_class == 0:  # Assuming 0 is normal
            interpretation += "The EEG shows predominantly normal background activity with no significant abnormalities detected. "
        elif predicted_class == 1:  # Assuming 1 is seizure
            interpretation += "The EEG demonstrates epileptiform activity suggestive of seizure activity. "
            interpretation += "This includes sharp waves, spikes, or spike-wave complexes that are characteristic of seizure disorders. "
        elif predicted_class == 2:  # Assuming 2 is other abnormality
            interpretation += "The EEG shows abnormal patterns that may indicate underlying neurological dysfunction. "
            
        # Add dominant frequency information
        interpretation += "\n\nThe most significant EEG patterns contributing to this classification are present in the "
        
        # Identify dominant frequency bands
        dominant_bands = []
        for idx in top_features[:3]:
            feature_name = self._translate_feature(f"feature_{idx}", idx)
            if any(band in feature_name.lower() for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']):
                dominant_bands.append(feature_name)
                
        if dominant_bands:
            interpretation += f"{', '.join(dominant_bands)} frequency ranges."
        else:
            interpretation += "identified EEG patterns."
            
        return interpretation
        
    def _generate_recommendations(self, confidence: float, predicted_class: int) -> str:
        """Generate clinical recommendations based on prediction."""
        recommendations = []
        
        # Confidence-based recommendations
        if confidence < 0.7:
            recommendations.append("- Consider repeat EEG recording for confirmation due to moderate confidence level")
            recommendations.append("- Correlate with clinical symptoms and patient history")
            
        # Class-specific recommendations
        if predicted_class == 1:  # Seizure
            recommendations.append("- Immediate neurological consultation recommended")
            recommendations.append("- Consider continuous EEG monitoring")
            recommendations.append("- Evaluate for anti-epileptic medication if clinically appropriate")
        elif predicted_class == 2:  # Other abnormality
            recommendations.append("- Further neurological evaluation recommended")
            recommendations.append("- Consider additional neuroimaging studies")
            recommendations.append("- Follow-up EEG in 3-6 months")
        else:  # Normal
            recommendations.append("- No immediate intervention required based on EEG findings")
            recommendations.append("- Continue routine clinical monitoring")
            
        # Always add
        recommendations.append("- Clinical correlation is essential for final diagnosis")
        recommendations.append("- This analysis should be reviewed by a qualified neurologist")
        
        return '\n'.join(recommendations)
        
    def generate_summary_report(self,
                                explanations: List[ExplanationResult],
                                save_path: Optional[str] = None) -> str:
        """
        Generate summary report for multiple explanations.
        
        Args:
            explanations: List of explanation results
            save_path: Path to save report
            
        Returns:
            Summary report text
        """
        report = "# EEG Analysis Summary Report\n\n"
        report += f"**Total Recordings Analyzed:** {len(explanations)}\n"
        report += f"**Analysis Date:** {np.datetime64('today')}\n\n"
        
        # Class distribution
        report += "## Classification Summary\n"
        class_counts = {}
        for exp in explanations:
            if len(exp.prediction.shape) > 0 and exp.prediction.shape[-1] > 1:
                predicted_class = exp.prediction.argmax()
            else:
                predicted_class = 0
            class_name = self.class_names.get(predicted_class, f"Class {predicted_class}")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        for class_name, count in class_counts.items():
            percentage = count / len(explanations) * 100
            report += f"- **{class_name}:** {count} ({percentage:.1f}%)\n"
            
        # Average confidence
        confidences = []
        for exp in explanations:
            if len(exp.prediction.shape) > 0 and exp.prediction.shape[-1] > 1:
                conf = exp.prediction.max()
            else:
                conf = float(exp.prediction)
            confidences.append(conf)
            
        avg_confidence = np.mean(confidences)
        report += f"\n**Average Confidence:** {avg_confidence:.1%}\n"
        
        # Feature importance summary
        report += "\n## Common EEG Patterns\n"
        feature_importance_sum = np.zeros_like(explanations[0].explanation_values)
        
        for exp in explanations:
            feature_importance_sum += np.abs(exp.explanation_values)
            
        feature_importance_avg = feature_importance_sum / len(explanations)
        top_features = np.argsort(feature_importance_avg)[-5:][::-1]
        
        for idx in top_features:
            feature_name = self._translate_feature(f"feature_{idx}", idx)
            avg_importance = feature_importance_avg[idx]
            report += f"- **{feature_name}:** Average importance {avg_importance:.3f}\n"
            
        # Recommendations summary
        report += "\n## General Recommendations\n"
        report += "- All positive findings should be clinically correlated\n"
        report += "- Consider patient history and symptoms in interpretation\n"
        report += "- Follow-up testing may be indicated for uncertain results\n"
        report += "- This automated analysis supplements but does not replace expert review\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
                
        return report 