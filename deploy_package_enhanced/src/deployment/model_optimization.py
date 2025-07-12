"""
Model Optimization Module for HMS EEG Classification

This module implements various optimization techniques for model inference:
- Quantization (INT8, FP16)
- Model pruning
- Knowledge distillation
- ONNX conversion
- TensorRT optimization
- Dynamic batching
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
import torch.nn.utils.prune as prune
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import logging
import time
from dataclasses import dataclass
import json

try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False
    
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for model optimization."""
    quantization_backend: str = "fbgemm"  # fbgemm, qnnpack
    quantization_dtype: str = "int8"  # int8, fp16
    pruning_amount: float = 0.3
    distillation_temperature: float = 5.0
    distillation_alpha: float = 0.7
    onnx_opset_version: int = 14
    tensorrt_precision: str = "fp16"  # fp32, fp16, int8
    dynamic_batch_size: bool = True
    max_batch_size: int = 32
    optimization_level: int = 1  # 0: no opt, 1: basic, 2: aggressive, 3: maximum


class ModelQuantizer:
    """Handles model quantization for reduced memory and faster inference."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
    def quantize_dynamic(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to model."""
        logger.info("Applying dynamic quantization...")
        
        # Set quantization backend
        torch.backends.quantized.engine = self.config.quantization_backend
        
        # Dynamic quantization for LSTM/GRU and Linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec={
                nn.LSTM: torch.quantization.default_dynamic_qconfig,
                nn.GRU: torch.quantization.default_dynamic_qconfig,
                nn.Linear: torch.quantization.default_dynamic_qconfig,
            },
            dtype=torch.qint8 if self.config.quantization_dtype == "int8" else torch.float16
        )
        
        # Calculate compression ratio
        original_size = self._get_model_size(model)
        quantized_size = self._get_model_size(quantized_model)
        compression_ratio = original_size / quantized_size
        
        logger.info(f"Quantization complete. Compression ratio: {compression_ratio:.2f}x")
        logger.info(f"Original size: {original_size/1e6:.2f} MB")
        logger.info(f"Quantized size: {quantized_size/1e6:.2f} MB")
        
        return quantized_model
    
    def quantize_static(self, model: nn.Module, calibration_data: torch.utils.data.DataLoader) -> nn.Module:
        """Apply static quantization with calibration."""
        logger.info("Applying static quantization...")
        
        # Prepare model for quantization
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig(self.config.quantization_backend)
        
        # Fuse modules
        model = self._fuse_modules(model)
        
        # Prepare for static quantization
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate with representative data
        logger.info("Calibrating model...")
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_data):
                if batch_idx >= 100:  # Use 100 batches for calibration
                    break
                model(data)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model, inplace=True)
        
        return quantized_model
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse conv-bn-relu modules for better quantization."""
        # This is model-specific, implement based on architecture
        # Example for ResNet-like architectures
        modules_to_fuse = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                # Look for conv-bn-relu patterns
                for i in range(len(module) - 2):
                    if (isinstance(module[i], nn.Conv2d) and
                        isinstance(module[i+1], nn.BatchNorm2d) and
                        isinstance(module[i+2], nn.ReLU)):
                        modules_to_fuse.append([f"{name}.{i}", f"{name}.{i+1}", f"{name}.{i+2}"])
        
        if modules_to_fuse:
            model = torch.quantization.fuse_modules(model, modules_to_fuse)
            
        return model
    
    def _get_model_size(self, model: nn.Module) -> int:
        """Calculate model size in bytes."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        return param_size + buffer_size


class ModelPruner:
    """Handles model pruning for reduced parameters."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
    def prune_model(self, model: nn.Module, importance_scores: Optional[Dict[str, torch.Tensor]] = None) -> nn.Module:
        """Apply structured or unstructured pruning to model."""
        logger.info(f"Applying pruning with {self.config.pruning_amount*100:.1f}% sparsity...")
        
        # Get all conv and linear layers
        layers_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                layers_to_prune.append((module, 'weight'))
        
        # Apply pruning
        if importance_scores:
            # Custom importance-based pruning
            self._importance_based_pruning(layers_to_prune, importance_scores)
        else:
            # L1 unstructured pruning
            prune.global_unstructured(
                layers_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.config.pruning_amount,
            )
        
        # Remove pruning reparameterization
        for module, param_name in layers_to_prune:
            prune.remove(module, param_name)
        
        # Calculate sparsity
        total_params = 0
        pruned_params = 0
        for module, _ in layers_to_prune:
            total_params += module.weight.nelement()
            pruned_params += (module.weight == 0).sum().item()
        
        actual_sparsity = pruned_params / total_params
        logger.info(f"Pruning complete. Actual sparsity: {actual_sparsity*100:.1f}%")
        
        return model
    
    def _importance_based_pruning(self, layers: List[Tuple[nn.Module, str]], 
                                  importance_scores: Dict[str, torch.Tensor]):
        """Apply importance-based structured pruning."""
        for module, param_name in layers:
            if hasattr(module, 'weight'):
                weight = getattr(module, param_name)
                
                # Get importance scores for this layer
                layer_importance = importance_scores.get(str(module), None)
                if layer_importance is None:
                    # Fall back to L2 norm
                    if len(weight.shape) > 1:
                        layer_importance = weight.norm(2, dim=1)
                    else:
                        layer_importance = weight.abs()
                
                # Calculate threshold
                k = int(layer_importance.numel() * (1 - self.config.pruning_amount))
                threshold = torch.topk(layer_importance.view(-1), k, largest=True)[0][-1]
                
                # Create mask
                mask = layer_importance >= threshold
                
                # Apply mask
                with torch.no_grad():
                    weight.mul_(mask.float().view(-1, 1) if len(weight.shape) > 1 else mask.float())


class KnowledgeDistiller:
    """Implements knowledge distillation for model compression."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
    def create_student_model(self, teacher_model: nn.Module, compression_ratio: float = 0.5) -> nn.Module:
        """Create a smaller student model from teacher architecture."""
        # This is model-specific, implement based on architecture
        # Example: reduce channels/layers by compression_ratio
        student_model = self._compress_architecture(teacher_model, compression_ratio)
        return student_model
    
    def distillation_loss(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor,
                         labels: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
        """Calculate distillation loss combining student loss and knowledge transfer."""
        # Soften the outputs
        student_soft = nn.functional.log_softmax(student_outputs / self.config.distillation_temperature, dim=1)
        teacher_soft = nn.functional.softmax(teacher_outputs / self.config.distillation_temperature, dim=1)
        
        # Distillation loss
        distillation_loss = nn.functional.kl_div(student_soft, teacher_soft, reduction='batchmean')
        distillation_loss *= self.config.distillation_temperature ** 2
        
        # Student loss
        student_loss = criterion(student_outputs, labels)
        
        # Combined loss
        total_loss = (self.config.distillation_alpha * distillation_loss + 
                     (1 - self.config.distillation_alpha) * student_loss)
        
        return total_loss
    
    def _compress_architecture(self, model: nn.Module, ratio: float) -> nn.Module:
        """Compress model architecture by reducing channels/layers."""
        # This is a placeholder - implement based on specific architecture
        # For now, return a copy of the model
        import copy
        return copy.deepcopy(model)


class ONNXConverter:
    """Handles ONNX conversion for cross-platform deployment."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
    def convert_to_onnx(self, model: nn.Module, dummy_input: torch.Tensor, 
                       output_path: Path, input_names: List[str] = ["input"],
                       output_names: List[str] = ["output"]) -> Path:
        """Convert PyTorch model to ONNX format."""
        logger.info("Converting model to ONNX...")
        
        model.eval()
        
        # Dynamic axes for variable batch size
        dynamic_axes = {}
        if self.config.dynamic_batch_size:
            for name in input_names:
                dynamic_axes[name] = {0: 'batch_size'}
            for name in output_names:
                dynamic_axes[name] = {0: 'batch_size'}
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=self.config.onnx_opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        
        # Optimize ONNX model
        self._optimize_onnx(output_path)
        
        logger.info(f"ONNX conversion complete: {output_path}")
        return output_path
    
    def _optimize_onnx(self, model_path: Path):
        """Apply ONNX-specific optimizations."""
        from onnxruntime.transformers import optimizer
        
        optimized_path = model_path.parent / f"{model_path.stem}_optimized.onnx"
        
        optimizer.optimize_model(
            str(model_path),
            str(optimized_path),
            optimization_level=self.config.optimization_level,
            num_heads=0,  # Set based on model architecture
            hidden_size=0  # Set based on model architecture
        )
        
        # Replace original with optimized
        optimized_path.replace(model_path)


class TensorRTOptimizer:
    """Handles TensorRT optimization for NVIDIA GPU inference."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        if not HAS_TENSORRT:
            logger.warning("TensorRT not available. Skipping TRT optimization.")
            
    def optimize_with_tensorrt(self, onnx_path: Path, output_path: Path,
                              calibration_data: Optional[np.ndarray] = None) -> Optional[Path]:
        """Convert ONNX model to TensorRT engine."""
        if not HAS_TENSORRT:
            return None
            
        logger.info("Optimizing with TensorRT...")
        
        # Create builder
        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
        
        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                return None
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        # Set precision
        if self.config.tensorrt_precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.config.tensorrt_precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            if calibration_data is not None:
                # Set up INT8 calibration
                calibrator = self._create_int8_calibrator(calibration_data)
                config.int8_calibrator = calibrator
        
        # Build engine
        engine = builder.build_engine(network, config)
        
        if engine is None:
            logger.error("Failed to build TensorRT engine")
            return None
        
        # Save engine
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
            
        logger.info(f"TensorRT optimization complete: {output_path}")
        return output_path
    
    def _create_int8_calibrator(self, calibration_data: np.ndarray):
        """Create INT8 calibrator for TensorRT."""
        # Implement calibrator based on IInt8EntropyCalibrator2
        # This is a placeholder
        return None


class DynamicBatcher:
    """Handles dynamic batching for efficient inference."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.pending_requests = []
        self.batch_timeout = 0.05  # 50ms
        
    def add_request(self, request_id: str, data: torch.Tensor, callback: callable):
        """Add request to pending batch."""
        self.pending_requests.append({
            'id': request_id,
            'data': data,
            'callback': callback,
            'timestamp': time.time()
        })
        
        # Check if batch is ready
        if len(self.pending_requests) >= self.config.max_batch_size:
            return self._process_batch()
        
        # Check timeout
        if self.pending_requests and (time.time() - self.pending_requests[0]['timestamp']) > self.batch_timeout:
            return self._process_batch()
            
        return None
    
    def _process_batch(self) -> List[Dict]:
        """Process accumulated batch."""
        if not self.pending_requests:
            return []
            
        # Take up to max_batch_size requests
        batch_requests = self.pending_requests[:self.config.max_batch_size]
        self.pending_requests = self.pending_requests[self.config.max_batch_size:]
        
        # Stack data
        batch_data = torch.stack([req['data'] for req in batch_requests])
        
        return batch_requests, batch_data


class OptimizedModelWrapper:
    """Wrapper for optimized model with various backends."""
    
    def __init__(self, model_path: Path, config: OptimizationConfig):
        self.config = config
        self.model_path = model_path
        self.model = None
        self.model_type = self._detect_model_type()
        self._load_model()
        
    def _detect_model_type(self) -> str:
        """Detect model type from file extension."""
        suffix = self.model_path.suffix.lower()
        if suffix == '.onnx':
            return 'onnx'
        elif suffix == '.trt' or suffix == '.engine':
            return 'tensorrt'
        elif suffix in ['.pth', '.pt']:
            return 'pytorch'
        else:
            raise ValueError(f"Unknown model type: {suffix}")
            
    def _load_model(self):
        """Load model based on type."""
        if self.model_type == 'onnx':
            self.model = ort.InferenceSession(
                str(self.model_path),
                providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            )
        elif self.model_type == 'tensorrt':
            # Load TensorRT engine
            if HAS_TENSORRT:
                # Implementation for TensorRT runtime
                pass
        elif self.model_type == 'pytorch':
            self.model = torch.load(self.model_path)
            self.model.eval()
            
    def predict(self, inputs: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Run inference with appropriate backend."""
        if self.model_type == 'onnx':
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.numpy()
            
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: inputs})
            return outputs[0]
            
        elif self.model_type == 'pytorch':
            with torch.no_grad():
                outputs = self.model(inputs)
                return outputs.numpy()
                
        else:
            raise NotImplementedError(f"Inference for {self.model_type} not implemented")


def optimize_model_pipeline(model: nn.Module, config: OptimizationConfig,
                          calibration_data: Optional[torch.utils.data.DataLoader] = None,
                          output_dir: Path = Path("models/optimized")) -> Dict[str, Path]:
    """Complete optimization pipeline for a model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    # 1. Quantization
    quantizer = ModelQuantizer(config)
    if config.quantization_dtype:
        if calibration_data:
            quantized_model = quantizer.quantize_static(model, calibration_data)
        else:
            quantized_model = quantizer.quantize_dynamic(model)
        
        quantized_path = output_dir / "model_quantized.pth"
        torch.save(quantized_model, quantized_path)
        results['quantized'] = quantized_path
    
    # 2. Pruning
    pruner = ModelPruner(config)
    if config.pruning_amount > 0:
        pruned_model = pruner.prune_model(model)
        pruned_path = output_dir / "model_pruned.pth"
        torch.save(pruned_model, pruned_path)
        results['pruned'] = pruned_path
    
    # 3. ONNX conversion
    converter = ONNXConverter(config)
    dummy_input = torch.randn(1, 19, 10000)  # Adjust based on model input
    onnx_path = output_dir / "model.onnx"
    converter.convert_to_onnx(model, dummy_input, onnx_path)
    results['onnx'] = onnx_path
    
    # 4. TensorRT optimization
    if HAS_TENSORRT:
        trt_optimizer = TensorRTOptimizer(config)
        trt_path = output_dir / "model.trt"
        trt_result = trt_optimizer.optimize_with_tensorrt(onnx_path, trt_path)
        if trt_result:
            results['tensorrt'] = trt_result
    
    return results 