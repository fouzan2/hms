"""Model optimization utilities for deployment."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple, Any
import onnx
import onnxruntime as ort
from torch.quantization import quantize_dynamic, prepare_qat, convert

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Optimize models for efficient inference."""
    
    def __init__(self, config: dict):
        """Initialize model optimizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.optimization_config = config.get('deployment', {}).get('optimization', {})
    
    def export_to_onnx(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        output_path: str,
        input_names: list = ['input'],
        output_names: list = ['output'],
        dynamic_axes: Optional[dict] = None
    ) -> str:
        """Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model to export
            sample_input: Sample input tensor
            output_path: Path to save ONNX model
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dynamic axes specification
            
        Returns:
            Path to exported ONNX model
        """
        logger.info(f"Exporting model to ONNX format: {output_path}")
        
        model.eval()
        
        # Default dynamic axes for batch size
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export model
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification successful")
        
        # Simplify if possible
        try:
            import onnxsim
            simplified_model, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(simplified_model, output_path)
                logger.info("ONNX model simplified successfully")
        except ImportError:
            logger.warning("onnx-simplifier not installed, skipping simplification")
        
        return output_path
    
    def quantize_model_dynamic(
        self,
        model: nn.Module,
        qconfig_spec: Optional[dict] = None
    ) -> nn.Module:
        """Apply dynamic quantization to model.
        
        Args:
            model: PyTorch model to quantize
            qconfig_spec: Quantization configuration
            
        Returns:
            Quantized model
        """
        logger.info("Applying dynamic quantization...")
        
        # Default quantization config
        if qconfig_spec is None:
            qconfig_spec = {
                nn.Linear: torch.quantization.default_dynamic_qconfig,
                nn.GRU: torch.quantization.default_dynamic_qconfig,
                nn.LSTM: torch.quantization.default_dynamic_qconfig,
            }
        
        # Apply dynamic quantization
        quantized_model = quantize_dynamic(
            model,
            qconfig_spec,
            dtype=torch.qint8
        )
        
        return quantized_model
    
    def prepare_qat_model(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor
    ) -> nn.Module:
        """Prepare model for Quantization Aware Training (QAT).
        
        Args:
            model: PyTorch model
            example_inputs: Example inputs for calibration
            
        Returns:
            QAT-prepared model
        """
        logger.info("Preparing model for QAT...")
        
        # Set quantization config
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare model
        model_prepared = prepare_qat(model.train())
        
        # Calibrate with example inputs
        with torch.no_grad():
            model_prepared(example_inputs)
        
        return model_prepared
    
    def convert_qat_model(self, model_prepared: nn.Module) -> nn.Module:
        """Convert QAT prepared model to quantized model.
        
        Args:
            model_prepared: QAT prepared model
            
        Returns:
            Quantized model
        """
        logger.info("Converting QAT model to quantized model...")
        model_prepared.eval()
        model_quantized = convert(model_prepared)
        return model_quantized
    
    def optimize_onnx_with_tensorrt(
        self,
        onnx_path: str,
        output_path: str,
        fp16: bool = True,
        int8: bool = False,
        workspace_size: int = 1 << 30
    ) -> Optional[str]:
        """Optimize ONNX model with TensorRT.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save TensorRT engine
            fp16: Enable FP16 precision
            int8: Enable INT8 precision
            workspace_size: GPU memory workspace size
            
        Returns:
            Path to TensorRT engine or None if TensorRT not available
        """
        try:
            import tensorrt as trt
            
            logger.info("Optimizing ONNX model with TensorRT...")
            
            # Create builder
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return None
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = workspace_size
            
            if fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            if int8:
                config.set_flag(trt.BuilderFlag.INT8)
                # Would need calibration dataset for INT8
            
            # Build engine
            engine = builder.build_engine(network, config)
            
            # Save engine
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"TensorRT engine saved to: {output_path}")
            return output_path
            
        except ImportError:
            logger.warning("TensorRT not available, skipping TensorRT optimization")
            return None
    
    def benchmark_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """Benchmark model inference performance.
        
        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Benchmark results
        """
        import time
        
        device = next(model.parameters()).device
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(device)
        
        # Warmup
        logger.info(f"Running {warmup_runs} warmup iterations...")
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Synchronize if using CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        logger.info(f"Running {num_runs} benchmark iterations...")
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = num_runs / total_time
        
        results = {
            'total_time': total_time,
            'average_time': avg_time,
            'throughput': throughput,
            'latency_ms': avg_time * 1000
        }
        
        logger.info(f"Benchmark results:")
        logger.info(f"  Average inference time: {avg_time:.4f}s ({avg_time*1000:.2f}ms)")
        logger.info(f"  Throughput: {throughput:.2f} inferences/second")
        
        return results
    
    def optimize_for_mobile(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        output_path: str
    ) -> str:
        """Optimize model for mobile deployment using TorchScript.
        
        Args:
            model: PyTorch model
            example_input: Example input tensor
            output_path: Path to save optimized model
            
        Returns:
            Path to optimized model
        """
        logger.info("Optimizing model for mobile deployment...")
        
        model.eval()
        
        # Trace model
        traced_model = torch.jit.trace(model, example_input)
        
        # Optimize for mobile
        from torch.utils.mobile_optimizer import optimize_for_mobile
        optimized_model = optimize_for_mobile(traced_model)
        
        # Save model
        optimized_model._save_for_lite_interpreter(output_path)
        
        logger.info(f"Mobile-optimized model saved to: {output_path}")
        return output_path
    
    def prune_model(
        self,
        model: nn.Module,
        amount: float = 0.3
    ) -> nn.Module:
        """Apply structured pruning to model.
        
        Args:
            model: PyTorch model
            amount: Fraction of weights to prune
            
        Returns:
            Pruned model
        """
        import torch.nn.utils.prune as prune
        
        logger.info(f"Applying structured pruning (amount={amount})...")
        
        # Get all conv and linear layers
        parameters_to_prune = []
        for module in model.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply global pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        
        # Remove pruning reparameterization
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        # Calculate sparsity
        total_params = 0
        pruned_params = 0
        for module, param_name in parameters_to_prune:
            param = getattr(module, param_name)
            total_params += param.numel()
            pruned_params += (param == 0).sum().item()
        
        sparsity = pruned_params / total_params
        logger.info(f"Model sparsity after pruning: {sparsity:.2%}")
        
        return model
    
    def create_inference_session(
        self,
        onnx_path: str,
        providers: Optional[list] = None
    ) -> ort.InferenceSession:
        """Create ONNX Runtime inference session.
        
        Args:
            onnx_path: Path to ONNX model
            providers: Execution providers
            
        Returns:
            ONNX Runtime session
        """
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        session = ort.InferenceSession(onnx_path, providers=providers)
        logger.info(f"Created ONNX Runtime session with providers: {session.get_providers()}")
        
        return session


def optimize_models_for_deployment(config: dict, model_dir: str):
    """Optimize all models for deployment.
    
    Args:
        config: Configuration dictionary
        model_dir: Directory containing trained models
    """
    optimizer = ModelOptimizer(config)
    model_dir = Path(model_dir)
    
    # Create output directory
    output_dir = model_dir / 'optimized'
    output_dir.mkdir(exist_ok=True)
    
    # Optimize each model
    for model_path in model_dir.glob('*.pth'):
        logger.info(f"Optimizing model: {model_path}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        model = checkpoint['model']
        
        # Get input shape based on model type
        if 'resnet' in model_path.name:
            input_shape = (1, config['preprocessing']['num_channels'], 10000)  # 50s at 200Hz
        else:  # EfficientNet
            input_shape = (1, 3, 224, 224)  # Spectrogram input
        
        sample_input = torch.randn(input_shape)
        
        # Export to ONNX
        onnx_path = output_dir / f"{model_path.stem}.onnx"
        optimizer.export_to_onnx(model, sample_input, str(onnx_path))
        
        # Quantize model
        quantized_model = optimizer.quantize_model_dynamic(model)
        quantized_path = output_dir / f"{model_path.stem}_quantized.pth"
        torch.save(quantized_model, quantized_path)
        
        # Benchmark
        logger.info("Original model benchmark:")
        optimizer.benchmark_model(model, input_shape)
        
        logger.info("Quantized model benchmark:")
        optimizer.benchmark_model(quantized_model, input_shape)
        
        # Mobile optimization
        mobile_path = output_dir / f"{model_path.stem}_mobile.ptl"
        optimizer.optimize_for_mobile(model, sample_input, str(mobile_path)) 