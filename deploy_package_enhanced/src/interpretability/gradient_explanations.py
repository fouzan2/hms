"""
Gradient-based Explanation Methods for HMS EEG Classification
Implements Grad-CAM, Integrated Gradients, Guided Backpropagation,
and other gradient-based interpretability methods for EEG models.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
import cv2

logger = logging.getLogger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping for EEG models."""
    
    def __init__(self, model: nn.Module, target_layers: List[str], device: str = 'cpu'):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained model
            target_layers: Names of layers to extract gradients from
            device: Computing device
        """
        self.model = model.to(device)
        self.device = device
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook
        
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(forward_hook(name)))
                self.hooks.append(module.register_full_backward_hook(backward_hook(name)))
    
    def generate_cam(self, 
                    x: torch.Tensor, 
                    class_idx: Optional[int] = None,
                    layer_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Generate Class Activation Maps.
        
        Args:
            x: Input tensor (1, n_channels, seq_length)
            class_idx: Target class index (if None, uses predicted class)
            layer_name: Specific layer to analyze (if None, analyzes all target layers)
            
        Returns:
            Dictionary containing CAMs for each layer
        """
        x = x.to(self.device)
        x.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(x)
        if isinstance(outputs, dict):
            logits = outputs.get('logits', list(outputs.values())[0])
        else:
            logits = outputs
        
        # Get target class
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target_score = logits[0, class_idx]
        target_score.backward(retain_graph=True)
        
        # Generate CAMs
        cams = {}
        layers_to_process = [layer_name] if layer_name else self.target_layers
        
        for layer in layers_to_process:
            if layer in self.gradients and layer in self.activations:
                gradients = self.gradients[layer]
                activations = self.activations[layer]
                
                # Global average pooling of gradients
                weights = torch.mean(gradients, dim=(0, 2), keepdim=True)
                
                # Weighted combination of activation maps
                cam = torch.sum(weights * activations, dim=1, keepdim=True)
                
                # Apply ReLU
                cam = F.relu(cam)
                
                # Normalize
                cam = cam - cam.min()
                if cam.max() > 0:
                    cam = cam / cam.max()
                
                cams[layer] = cam
        
        return cams
    
    def __del__(self):
        """Remove hooks when object is destroyed."""
        for hook in self.hooks:
            hook.remove()


class IntegratedGradients:
    """Integrated Gradients for EEG classification models."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize Integrated Gradients.
        
        Args:
            model: Trained model
            device: Computing device
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def generate_attributions(self,
                            x: torch.Tensor,
                            baseline: Optional[torch.Tensor] = None,
                            target_class: Optional[int] = None,
                            steps: int = 50) -> Dict[str, torch.Tensor]:
        """
        Generate integrated gradients attributions.
        
        Args:
            x: Input tensor (1, n_channels, seq_length)
            baseline: Baseline tensor (if None, uses zero baseline)
            target_class: Target class index
            steps: Number of integration steps
            
        Returns:
            Dictionary containing attributions and metadata
        """
        x = x.to(self.device)
        
        # Create baseline
        if baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = baseline.to(self.device)
        
        # Get target class
        if target_class is None:
            with torch.no_grad():
                outputs = self.model(x)
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', list(outputs.values())[0])
                else:
                    logits = outputs
                target_class = torch.argmax(logits, dim=1).item()
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps, device=self.device).view(-1, 1, 1)
        interpolated_inputs = baseline + alphas * (x - baseline)
        
        # Compute gradients for each interpolated input
        gradients = []
        for i in range(steps):
            input_tensor = interpolated_inputs[i:i+1]
            input_tensor.requires_grad_(True)
            
            outputs = self.model(input_tensor)
            if isinstance(outputs, dict):
                logits = outputs.get('logits', list(outputs.values())[0])
            else:
                logits = outputs
            
            target_score = logits[0, target_class]
            
            grad = torch.autograd.grad(target_score, input_tensor, create_graph=False)[0]
            gradients.append(grad)
        
        # Stack gradients
        gradients = torch.stack(gradients, dim=0)
        
        # Integrate gradients (trapezoidal rule)
        integrated_gradients = torch.mean(gradients, dim=0) * (x - baseline)
        
        # Compute attribution statistics
        channel_attributions = torch.mean(torch.abs(integrated_gradients), dim=2)
        temporal_attributions = torch.mean(torch.abs(integrated_gradients), dim=1)
        
        return {
            'integrated_gradients': integrated_gradients,
            'channel_attributions': channel_attributions,
            'temporal_attributions': temporal_attributions,
            'total_attribution': torch.sum(torch.abs(integrated_gradients)),
            'baseline': baseline,
            'target_class': target_class,
            'steps': steps
        }


class GuidedBackpropagation:
    """Guided Backpropagation for visualization of relevant features."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize Guided Backpropagation.
        
        Args:
            model: Trained model
            device: Computing device
        """
        self.model = model.to(device)
        self.device = device
        self.original_relu_backward = {}
        
        # Replace ReLU backward pass
        self._replace_relu_backward()
    
    def _replace_relu_backward(self):
        """Replace ReLU backward function with guided version."""
        def guided_relu_backward(module, grad_input, grad_output):
            return (F.relu(grad_input[0]),) if grad_input[0] is not None else grad_input
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                # Store original backward function
                self.original_relu_backward[name] = module.register_full_backward_hook(guided_relu_backward)
    
    def generate_guided_gradients(self,
                                x: torch.Tensor,
                                target_class: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Generate guided gradients.
        
        Args:
            x: Input tensor (1, n_channels, seq_length)
            target_class: Target class index
            
        Returns:
            Dictionary containing guided gradients
        """
        x = x.to(self.device)
        x.requires_grad_(True)
        
        # Forward pass
        outputs = self.model(x)
        if isinstance(outputs, dict):
            logits = outputs.get('logits', list(outputs.values())[0])
        else:
            logits = outputs
        
        # Get target class
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        target_score = logits[0, target_class]
        target_score.backward()
        
        # Get guided gradients
        guided_gradients = x.grad.clone()
        
        # Compute statistics
        channel_gradients = torch.mean(torch.abs(guided_gradients), dim=2)
        temporal_gradients = torch.mean(torch.abs(guided_gradients), dim=1)
        
        return {
            'guided_gradients': guided_gradients,
            'channel_gradients': channel_gradients,
            'temporal_gradients': temporal_gradients,
            'total_gradient': torch.sum(torch.abs(guided_gradients)),
            'target_class': target_class
        }
    
    def __del__(self):
        """Restore original ReLU functions."""
        for hook in self.original_relu_backward.values():
            hook.remove()


class SmoothGrad:
    """SmoothGrad for reducing noise in gradient-based attributions."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize SmoothGrad.
        
        Args:
            model: Trained model
            device: Computing device
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def generate_smooth_gradients(self,
                                x: torch.Tensor,
                                target_class: Optional[int] = None,
                                noise_level: float = 0.1,
                                n_samples: int = 50) -> Dict[str, torch.Tensor]:
        """
        Generate smooth gradients by averaging over noisy samples.
        
        Args:
            x: Input tensor (1, n_channels, seq_length)
            target_class: Target class index
            noise_level: Standard deviation of noise
            n_samples: Number of noisy samples
            
        Returns:
            Dictionary containing smooth gradients
        """
        x = x.to(self.device)
        
        # Get target class
        if target_class is None:
            with torch.no_grad():
                outputs = self.model(x)
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', list(outputs.values())[0])
                else:
                    logits = outputs
                target_class = torch.argmax(logits, dim=1).item()
        
        # Generate noisy samples and compute gradients
        gradients = []
        for _ in range(n_samples):
            # Add noise
            noise = torch.randn_like(x) * noise_level
            noisy_x = x + noise
            noisy_x.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(noisy_x)
            if isinstance(outputs, dict):
                logits = outputs.get('logits', list(outputs.values())[0])
            else:
                logits = outputs
            
            # Backward pass
            target_score = logits[0, target_class]
            grad = torch.autograd.grad(target_score, noisy_x, create_graph=False)[0]
            gradients.append(grad)
        
        # Average gradients
        smooth_gradients = torch.mean(torch.stack(gradients), dim=0)
        
        # Compute statistics
        channel_gradients = torch.mean(torch.abs(smooth_gradients), dim=2)
        temporal_gradients = torch.mean(torch.abs(smooth_gradients), dim=1)
        
        return {
            'smooth_gradients': smooth_gradients,
            'channel_gradients': channel_gradients,
            'temporal_gradients': temporal_gradients,
            'total_gradient': torch.sum(torch.abs(smooth_gradients)),
            'target_class': target_class,
            'noise_level': noise_level,
            'n_samples': n_samples
        }


class LayerActivationAnalysis:
    """Analyze layer activations to understand model behavior."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        Initialize layer activation analysis.
        
        Args:
            model: Trained model
            device: Computing device
        """
        self.model = model.to(device)
        self.device = device
        self.activations = {}
        self.hooks = []
        
    def register_activation_hooks(self, layer_names: List[str]):
        """Register hooks to capture activations."""
        def get_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        for name, module in self.model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(get_activation(name))
                self.hooks.append(hook)
    
    def analyze_activations(self, 
                          x: torch.Tensor,
                          layer_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze activations for given input.
        
        Args:
            x: Input tensor
            layer_names: Specific layers to analyze
            
        Returns:
            Dictionary containing activation analysis
        """
        x = x.to(self.device)
        
        # Register hooks if needed
        if layer_names and not self.hooks:
            self.register_activation_hooks(layer_names)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(x)
        
        # Analyze activations
        analysis = {}
        for layer_name, activation in self.activations.items():
            layer_analysis = self._analyze_layer_activation(activation)
            analysis[layer_name] = layer_analysis
        
        return analysis
    
    def _analyze_layer_activation(self, activation: torch.Tensor) -> Dict[str, Any]:
        """Analyze single layer activation."""
        activation = activation.cpu().numpy()
        
        return {
            'shape': activation.shape,
            'mean': np.mean(activation),
            'std': np.std(activation),
            'min': np.min(activation),
            'max': np.max(activation),
            'sparsity': np.mean(activation == 0),
            'entropy': self._compute_activation_entropy(activation),
            'dead_neurons': np.sum(np.max(activation.reshape(activation.shape[0], -1), axis=1) == 0)
        }
    
    def _compute_activation_entropy(self, activation: np.ndarray) -> float:
        """Compute entropy of activation distribution."""
        flat_activation = activation.flatten()
        hist, _ = np.histogram(flat_activation, bins=50, density=True)
        hist = hist + 1e-8  # Avoid log(0)
        entropy = -np.sum(hist * np.log(hist))
        return entropy
    
    def __del__(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()


class GradientExplanationFramework:
    """
    Comprehensive framework for gradient-based explanations.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 target_layers: Optional[List[str]] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize gradient explanation framework.
        
        Args:
            model: Trained model
            target_layers: Target layers for Grad-CAM
            device: Computing device
        """
        self.model = model.to(device)
        self.device = device
        
        # Initialize explanation methods
        self.grad_cam = GradCAM(model, target_layers or [], device) if target_layers else None
        self.integrated_gradients = IntegratedGradients(model, device)
        self.guided_backprop = GuidedBackpropagation(model, device)
        self.smooth_grad = SmoothGrad(model, device)
        self.activation_analyzer = LayerActivationAnalysis(model, device)
    
    def explain(self,
               x: torch.Tensor,
               methods: List[str] = None,
               target_class: Optional[int] = None,
               **kwargs) -> Dict[str, Any]:
        """
        Generate comprehensive gradient-based explanations.
        
        Args:
            x: Input tensor (1, n_channels, seq_length)
            methods: List of explanation methods to use
            target_class: Target class index
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Dictionary containing all explanations
        """
        if methods is None:
            methods = ['integrated_gradients', 'guided_backprop', 'smooth_grad']
            if self.grad_cam:
                methods.append('grad_cam')
        
        x = x.to(self.device)
        explanations = {}
        
        # Integrated Gradients
        if 'integrated_gradients' in methods:
            try:
                ig_results = self.integrated_gradients.generate_attributions(
                    x, target_class=target_class, **kwargs.get('ig_params', {})
                )
                explanations['integrated_gradients'] = ig_results
            except Exception as e:
                logger.warning(f"Integrated Gradients failed: {e}")
                explanations['integrated_gradients'] = {'error': str(e)}
        
        # Guided Backpropagation
        if 'guided_backprop' in methods:
            try:
                gb_results = self.guided_backprop.generate_guided_gradients(
                    x, target_class=target_class
                )
                explanations['guided_backprop'] = gb_results
            except Exception as e:
                logger.warning(f"Guided Backpropagation failed: {e}")
                explanations['guided_backprop'] = {'error': str(e)}
        
        # SmoothGrad
        if 'smooth_grad' in methods:
            try:
                sg_results = self.smooth_grad.generate_smooth_gradients(
                    x, target_class=target_class, **kwargs.get('sg_params', {})
                )
                explanations['smooth_grad'] = sg_results
            except Exception as e:
                logger.warning(f"SmoothGrad failed: {e}")
                explanations['smooth_grad'] = {'error': str(e)}
        
        # Grad-CAM
        if 'grad_cam' in methods and self.grad_cam:
            try:
                cam_results = self.grad_cam.generate_cam(
                    x, class_idx=target_class, **kwargs.get('cam_params', {})
                )
                explanations['grad_cam'] = cam_results
            except Exception as e:
                logger.warning(f"Grad-CAM failed: {e}")
                explanations['grad_cam'] = {'error': str(e)}
        
        # Layer activations
        if 'activations' in methods:
            try:
                activation_results = self.activation_analyzer.analyze_activations(
                    x, **kwargs.get('activation_params', {})
                )
                explanations['activations'] = activation_results
            except Exception as e:
                logger.warning(f"Activation analysis failed: {e}")
                explanations['activations'] = {'error': str(e)}
        
        return explanations
    
    def visualize_explanations(self,
                             explanations: Dict[str, Any],
                             eeg_data: torch.Tensor,
                             save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create visualizations for gradient-based explanations.
        
        Args:
            explanations: Explanation results from explain()
            eeg_data: Original EEG data for reference
            save_path: Optional path to save visualization
            
        Returns:
            Matplotlib figure
        """
        n_methods = len([k for k in explanations.keys() if 'error' not in explanations[k]])
        fig, axes = plt.subplots(n_methods + 1, 1, figsize=(15, 4 * (n_methods + 1)))
        
        if n_methods == 0:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot original EEG data
        eeg_np = eeg_data.cpu().numpy().squeeze()
        im = axes[plot_idx].imshow(eeg_np, aspect='auto', cmap='viridis')
        axes[plot_idx].set_title('Original EEG Data')
        axes[plot_idx].set_ylabel('Channels')
        axes[plot_idx].set_xlabel('Time')
        plt.colorbar(im, ax=axes[plot_idx])
        plot_idx += 1
        
        # Plot explanations
        for method_name, result in explanations.items():
            if 'error' in result:
                continue
                
            if method_name == 'integrated_gradients':
                attribution = result['integrated_gradients'].cpu().numpy().squeeze()
                im = axes[plot_idx].imshow(attribution, aspect='auto', cmap='RdBu_r', 
                                         vmin=-np.max(np.abs(attribution)), 
                                         vmax=np.max(np.abs(attribution)))
                axes[plot_idx].set_title('Integrated Gradients Attribution')
                
            elif method_name == 'guided_backprop':
                gradients = result['guided_gradients'].cpu().numpy().squeeze()
                im = axes[plot_idx].imshow(gradients, aspect='auto', cmap='RdBu_r',
                                         vmin=-np.max(np.abs(gradients)), 
                                         vmax=np.max(np.abs(gradients)))
                axes[plot_idx].set_title('Guided Backpropagation')
                
            elif method_name == 'smooth_grad':
                smooth_grads = result['smooth_gradients'].cpu().numpy().squeeze()
                im = axes[plot_idx].imshow(smooth_grads, aspect='auto', cmap='RdBu_r',
                                         vmin=-np.max(np.abs(smooth_grads)), 
                                         vmax=np.max(np.abs(smooth_grads)))
                axes[plot_idx].set_title('SmoothGrad')
                
            elif method_name == 'grad_cam':
                # Plot first available CAM
                for layer_name, cam in result.items():
                    if isinstance(cam, torch.Tensor):
                        cam_np = cam.cpu().numpy().squeeze()
                        if len(cam_np.shape) == 1:
                            # 1D CAM - plot as line
                            axes[plot_idx].plot(cam_np)
                            axes[plot_idx].set_title(f'Grad-CAM: {layer_name}')
                        else:
                            # 2D CAM - plot as heatmap
                            im = axes[plot_idx].imshow(cam_np, aspect='auto', cmap='jet')
                            axes[plot_idx].set_title(f'Grad-CAM: {layer_name}')
                        break
            
            if 'im' in locals():
                plt.colorbar(im, ax=axes[plot_idx])
            axes[plot_idx].set_ylabel('Channels')
            axes[plot_idx].set_xlabel('Time')
            plot_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compare_methods(self,
                       explanations: Dict[str, Any],
                       metrics: List[str] = None) -> Dict[str, Any]:
        """
        Compare different gradient explanation methods.
        
        Args:
            explanations: Results from explain()
            metrics: Metrics to compute for comparison
            
        Returns:
            Comparison results
        """
        if metrics is None:
            metrics = ['sparsity', 'sensitivity', 'consistency']
        
        comparison = {}
        
        # Extract gradients from different methods
        gradients = {}
        for method_name, result in explanations.items():
            if 'error' in result:
                continue
                
            if method_name == 'integrated_gradients':
                gradients[method_name] = result['integrated_gradients']
            elif method_name == 'guided_backprop':
                gradients[method_name] = result['guided_gradients']
            elif method_name == 'smooth_grad':
                gradients[method_name] = result['smooth_gradients']
        
        # Compute comparison metrics
        for method_name, grad in gradients.items():
            method_metrics = {}
            grad_np = grad.cpu().numpy()
            
            if 'sparsity' in metrics:
                # Proportion of near-zero attributions
                threshold = 0.01 * np.max(np.abs(grad_np))
                sparsity = np.mean(np.abs(grad_np) < threshold)
                method_metrics['sparsity'] = sparsity
            
            if 'sensitivity' in metrics:
                # Total variation of attributions
                sensitivity = np.sum(np.abs(grad_np))
                method_metrics['sensitivity'] = sensitivity
            
            if 'consistency' in metrics:
                # Consistency across channels (coefficient of variation)
                channel_means = np.mean(np.abs(grad_np), axis=1)
                if np.mean(channel_means) > 0:
                    consistency = np.std(channel_means) / np.mean(channel_means)
                else:
                    consistency = 0
                method_metrics['consistency'] = consistency
            
            comparison[method_name] = method_metrics 