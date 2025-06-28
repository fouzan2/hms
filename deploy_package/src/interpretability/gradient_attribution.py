"""
Gradient-based attribution methods for EEG model interpretability.
Implements Integrated Gradients, GradCAM, Guided Backpropagation, and more.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import warnings
from tqdm import tqdm
import cv2


@dataclass
class AttributionResult:
    """Container for attribution results."""
    attributions: np.ndarray
    input_data: np.ndarray
    predictions: np.ndarray
    target_class: int
    method_name: str
    metadata: Dict[str, any]


class IntegratedGradients:
    """
    Integrated Gradients for precise feature attribution.
    Computes the integral of gradients along the path from baseline to input.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def attribute(self, 
                  input_tensor: torch.Tensor,
                  target_class: Optional[int] = None,
                  baseline: Optional[torch.Tensor] = None,
                  steps: int = 50,
                  return_convergence_delta: bool = False) -> AttributionResult:
        """
        Compute Integrated Gradients attribution.
        
        Args:
            input_tensor: Input EEG data (batch_size, channels, time)
            target_class: Target class for attribution (None for predicted class)
            baseline: Baseline input (None for zero baseline)
            steps: Number of integration steps
            return_convergence_delta: Whether to return convergence information
            
        Returns:
            AttributionResult with computed attributions
        """
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
            
        # Get predictions
        with torch.no_grad():
            predictions = self.model(input_tensor).detach()
            if target_class is None:
                target_class = predictions.argmax(dim=1).item()
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps + 1).to(self.device)
        
        # Compute gradients at each interpolation step
        integrated_gradients = torch.zeros_like(input_tensor)
        
        for i in tqdm(range(steps), desc="Computing integrated gradients"):
            alpha = alphas[i]
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad = True
            
            output = self.model(interpolated)
            
            # Select target class output
            if len(output.shape) > 1:
                output = output[:, target_class]
            
            # Compute gradients
            self.model.zero_grad()
            output.backward(retain_graph=True)
            
            # Accumulate gradients
            integrated_gradients += interpolated.grad / steps
            
        # Final attribution
        attributions = (input_tensor - baseline) * integrated_gradients
        
        # Convergence check
        metadata = {}
        if return_convergence_delta:
            delta = self._compute_convergence_delta(
                attributions, input_tensor, baseline, target_class
            )
            metadata['convergence_delta'] = delta
            
        return AttributionResult(
            attributions=attributions.cpu().numpy(),
            input_data=input_tensor.cpu().numpy(),
            predictions=predictions.cpu().numpy(),
            target_class=target_class,
            method_name='integrated_gradients',
            metadata=metadata
        )
    
    def _compute_convergence_delta(self, attributions, input_tensor, baseline, target_class):
        """Check convergence of integrated gradients."""
        with torch.no_grad():
            baseline_output = self.model(baseline)[:, target_class]
            input_output = self.model(input_tensor)[:, target_class]
            attribution_sum = attributions.sum(dim=(1, 2))
            expected_diff = input_output - baseline_output
            delta = (attribution_sum - expected_diff).abs().mean().item()
        return delta


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN models.
    Highlights important regions in spectrograms.
    """
    
    def __init__(self, model: nn.Module, target_layer: str, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.target_layer = target_layer
        self.model.eval()
        
        # Register hooks
        self.activations = None
        self.gradients = None
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""
        target_module = dict(self.model.named_modules())[self.target_layer]
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        target_module.register_forward_hook(forward_hook)
        target_module.register_full_backward_hook(backward_hook)
        
    def attribute(self,
                  input_tensor: torch.Tensor,
                  target_class: Optional[int] = None,
                  eigen_smooth: bool = True) -> AttributionResult:
        """
        Compute GradCAM attribution.
        
        Args:
            input_tensor: Input spectrogram (batch_size, channels, freq, time)
            target_class: Target class for attribution
            eigen_smooth: Apply eigenvalue-based smoothing
            
        Returns:
            AttributionResult with GradCAM heatmap
        """
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[:, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute GradCAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize and resize
        cam = self._normalize_cam(cam)
        cam_resized = self._resize_cam(cam, input_tensor.shape[-2:])
        
        # Apply eigen smoothing if requested
        if eigen_smooth:
            cam_resized = self._eigen_smooth(cam_resized)
            
        return AttributionResult(
            attributions=cam_resized.cpu().numpy(),
            input_data=input_tensor.cpu().numpy(),
            predictions=output.detach().cpu().numpy(),
            target_class=target_class,
            method_name='gradcam',
            metadata={'target_layer': self.target_layer}
        )
    
    def _normalize_cam(self, cam):
        """Normalize CAM values to [0, 1]."""
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam
    
    def _resize_cam(self, cam, target_size):
        """Resize CAM to match input size."""
        cam_np = cam.cpu().numpy().squeeze()
        cam_resized = cv2.resize(cam_np, target_size[::-1])
        return torch.from_numpy(cam_resized).unsqueeze(0).unsqueeze(0)
    
    def _eigen_smooth(self, cam):
        """Apply eigenvalue-based smoothing to reduce noise."""
        # Implementation of eigen smoothing for cleaner visualizations
        cam_np = cam.cpu().numpy().squeeze()
        U, S, Vt = np.linalg.svd(cam_np, full_matrices=False)
        # Keep top components
        k = min(10, len(S))
        S[k:] = 0
        cam_smoothed = U @ np.diag(S) @ Vt
        return torch.from_numpy(cam_smoothed).unsqueeze(0).unsqueeze(0)


class GuidedBackpropagation:
    """
    Guided Backpropagation for refined attribution maps.
    Only propagates positive gradients through ReLU layers.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        """Register hooks to modify backward pass through ReLUs."""
        def relu_hook(module, grad_input, grad_output):
            return (torch.clamp(grad_input[0], min=0.0),)
            
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                hook = module.register_backward_hook(relu_hook)
                self.hooks.append(hook)
                
    def attribute(self,
                  input_tensor: torch.Tensor,
                  target_class: Optional[int] = None) -> AttributionResult:
        """
        Compute guided backpropagation attribution.
        
        Args:
            input_tensor: Input data
            target_class: Target class for attribution
            
        Returns:
            AttributionResult with guided gradients
        """
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        output[:, target_class].backward(retain_graph=True)
        
        # Get guided gradients
        guided_gradients = input_tensor.grad.detach()
        
        # Clean up hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        return AttributionResult(
            attributions=guided_gradients.cpu().numpy(),
            input_data=input_tensor.cpu().numpy(),
            predictions=output.detach().cpu().numpy(),
            target_class=target_class,
            method_name='guided_backpropagation',
            metadata={}
        )


class SaliencyMaps:
    """
    Simple gradient-based saliency maps.
    Shows which input features most affect the output.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
        
    def attribute(self,
                  input_tensor: torch.Tensor,
                  target_class: Optional[int] = None,
                  abs_value: bool = True,
                  smooth_samples: int = 0,
                  smooth_std: float = 0.1) -> AttributionResult:
        """
        Compute saliency map attribution.
        
        Args:
            input_tensor: Input data
            target_class: Target class
            abs_value: Whether to take absolute value of gradients
            smooth_samples: Number of samples for SmoothGrad
            smooth_std: Standard deviation for noise in SmoothGrad
            
        Returns:
            AttributionResult with saliency maps
        """
        if smooth_samples > 0:
            return self._smooth_grad(
                input_tensor, target_class, abs_value, smooth_samples, smooth_std
            )
            
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        output[:, target_class].backward(retain_graph=True)
        
        # Get gradients
        saliency = input_tensor.grad.detach()
        
        if abs_value:
            saliency = saliency.abs()
            
        return AttributionResult(
            attributions=saliency.cpu().numpy(),
            input_data=input_tensor.cpu().numpy(),
            predictions=output.detach().cpu().numpy(),
            target_class=target_class,
            method_name='saliency_map',
            metadata={'abs_value': abs_value}
        )
    
    def _smooth_grad(self, input_tensor, target_class, abs_value, samples, std):
        """Implement SmoothGrad for noise-robust saliency maps."""
        accumulated_gradients = torch.zeros_like(input_tensor)
        
        for _ in range(samples):
            noise = torch.randn_like(input_tensor) * std
            noisy_input = input_tensor + noise
            
            result = self.attribute(noisy_input, target_class, abs_value)
            accumulated_gradients += torch.from_numpy(result.attributions).to(self.device)
            
        averaged_gradients = accumulated_gradients / samples
        
        # Get final predictions with original input
        with torch.no_grad():
            output = self.model(input_tensor)
            
        return AttributionResult(
            attributions=averaged_gradients.cpu().numpy(),
            input_data=input_tensor.cpu().numpy(),
            predictions=output.cpu().numpy(),
            target_class=target_class,
            method_name='smoothgrad',
            metadata={'samples': samples, 'std': std, 'abs_value': abs_value}
        )


class LayerwiseRelevancePropagation:
    """
    Layer-wise Relevance Propagation for deep network interpretation.
    Decomposes the prediction in terms of input relevances.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda', epsilon: float = 1e-9):
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.model.eval()
        
    def attribute(self,
                  input_tensor: torch.Tensor,
                  target_class: Optional[int] = None,
                  rule: str = 'epsilon') -> AttributionResult:
        """
        Compute LRP attribution.
        
        Args:
            input_tensor: Input data
            target_class: Target class
            rule: LRP rule ('epsilon', 'gamma', 'alpha-beta')
            
        Returns:
            AttributionResult with relevance scores
        """
        # Forward pass to get activations
        activations = self._forward_and_store_activations(input_tensor)
        
        # Get predictions
        output = activations[-1]
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Initialize relevance scores
        R = torch.zeros_like(output)
        R[:, target_class] = output[:, target_class]
        
        # Backward relevance propagation
        for i in reversed(range(len(activations) - 1)):
            R = self._propagate_relevance(
                activations[i], activations[i + 1], R, rule
            )
            
        return AttributionResult(
            attributions=R.cpu().numpy(),
            input_data=input_tensor.cpu().numpy(),
            predictions=output.detach().cpu().numpy(),
            target_class=target_class,
            method_name='lrp',
            metadata={'rule': rule, 'epsilon': self.epsilon}
        )
    
    def _forward_and_store_activations(self, input_tensor):
        """Forward pass storing all layer activations."""
        activations = [input_tensor]
        x = input_tensor
        
        for layer in self.model.children():
            x = layer(x)
            activations.append(x)
            
        return activations
    
    def _propagate_relevance(self, a_lower, a_higher, R_higher, rule):
        """Propagate relevance from higher to lower layer."""
        if rule == 'epsilon':
            return self._epsilon_rule(a_lower, a_higher, R_higher)
        elif rule == 'gamma':
            return self._gamma_rule(a_lower, a_higher, R_higher)
        elif rule == 'alpha-beta':
            return self._alpha_beta_rule(a_lower, a_higher, R_higher)
        else:
            raise ValueError(f"Unknown LRP rule: {rule}")
            
    def _epsilon_rule(self, a_lower, a_higher, R_higher):
        """Epsilon-LRP rule for stable relevance propagation."""
        # This is a simplified implementation
        # In practice, would need layer-specific handling
        z = a_lower + self.epsilon
        s = R_higher / z
        c = s.sum()
        R_lower = a_lower * s / c
        return R_lower
    
    def _gamma_rule(self, a_lower, a_higher, R_higher, gamma=0.25):
        """Gamma-LRP rule favoring positive contributions."""
        # Implementation of gamma rule
        z_plus = F.relu(a_lower) + self.epsilon
        s = R_higher / z_plus
        R_lower = a_lower * s
        return R_lower
    
    def _alpha_beta_rule(self, a_lower, a_higher, R_higher, alpha=2.0, beta=1.0):
        """Alpha-Beta LRP rule with separate positive/negative relevance."""
        # Split into positive and negative parts
        a_pos = F.relu(a_lower)
        a_neg = a_lower - a_pos
        
        # Propagate positive and negative relevance separately
        R_pos = alpha * a_pos * (R_higher / (a_pos.sum() + self.epsilon))
        R_neg = beta * a_neg * (R_higher / (a_neg.sum() + self.epsilon))
        
        return R_pos + R_neg 


class GradientAttribution:
    """
    Unified interface for gradient-based attribution methods.
    Combines multiple attribution techniques in a single class.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize gradient attribution analyzer.
        
        Args:
            model: Neural network model
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Initialize attribution methods
        self.integrated_gradients = IntegratedGradients(model, device)
        self.saliency_maps = SaliencyMaps(model, device)
        self.guided_backprop = GuidedBackpropagation(model, device)
        
        # GradCAM requires target layer specification
        self.gradcam = None
        
    def set_gradcam_layer(self, target_layer: str):
        """Set target layer for GradCAM analysis."""
        self.gradcam = GradCAM(self.model, target_layer, self.device)
        
    def get_attribution(self, 
                       x: torch.Tensor,
                       method: str = 'integrated_gradients',
                       target_class: Optional[int] = None,
                       **kwargs) -> AttributionResult:
        """
        Get attribution using specified method.
        
        Args:
            x: Input tensor
            method: Attribution method name
            target_class: Target class for attribution
            **kwargs: Method-specific arguments
            
        Returns:
            AttributionResult containing attributions
        """
        x = x.to(self.device)
        
        if method == 'integrated_gradients':
            return self.integrated_gradients.attribute(x, target_class, **kwargs)
        elif method == 'saliency':
            return self.saliency_maps.attribute(x, target_class, **kwargs)
        elif method == 'guided_backprop':
            return self.guided_backprop.attribute(x, target_class, **kwargs)
        elif method == 'gradcam':
            if self.gradcam is None:
                raise ValueError("GradCAM target layer not set. Call set_gradcam_layer() first.")
            return self.gradcam.attribute(x, target_class, **kwargs)
        else:
            raise ValueError(f"Unknown attribution method: {method}")
            
    def compare_methods(self, 
                       x: torch.Tensor,
                       target_class: Optional[int] = None,
                       methods: Optional[List[str]] = None) -> Dict[str, AttributionResult]:
        """
        Compare multiple attribution methods.
        
        Args:
            x: Input tensor
            target_class: Target class for attribution
            methods: List of methods to compare
            
        Returns:
            Dictionary mapping method names to AttributionResults
        """
        if methods is None:
            methods = ['integrated_gradients', 'saliency', 'guided_backprop']
            
        results = {}
        for method in methods:
            try:
                results[method] = self.get_attribution(x, method, target_class)
            except Exception as e:
                print(f"Warning: Failed to compute {method}: {e}")
                
        return results 