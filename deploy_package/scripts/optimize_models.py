#!/usr/bin/env python3
"""
Model Optimization Script for HMS EEG Classification System

This script applies various optimization techniques from Phase 8:
- Model quantization and pruning
- ONNX/TensorRT conversion
- Memory optimization
- Performance profiling
"""

import argparse
import logging
import yaml
import torch
from pathlib import Path
import sys
import time
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.deployment.model_optimization import (
    OptimizationConfig, 
    optimize_model_pipeline,
    OptimizedModelWrapper
)
from src.deployment.memory_optimization import (
    MemoryConfig,
    MemoryOptimizer,
    create_memory_optimized_training_loop
)
from src.deployment.performance_monitoring import (
    MonitoringConfig,
    create_monitoring_system,
    PerformanceProfiler
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def optimize_single_model(model_path: Path, optimization_config: OptimizationConfig, 
                         output_dir: Path) -> Dict[str, Path]:
    """Optimize a single model with all techniques."""
    logger.info(f"Optimizing model: {model_path}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    else:
        model_state = checkpoint
    
    # Load model architecture (you'll need to import the actual model classes)
    from src.models import ResNet1D_GRU, EfficientNetSpectrogram, HMSEnsembleModel
    
    # Determine model type from checkpoint
    if 'resnet' in str(model_path).lower():
        model = ResNet1D_GRU(num_channels=19, num_classes=6)
    elif 'efficientnet' in str(model_path).lower():
        model = EfficientNetSpectrogram(num_classes=6)
    else:
        # Load ensemble model
        config = load_config('config/config.yaml')
        model = HMSEnsembleModel(config)
    
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    # Apply optimization pipeline
    optimization_results = optimize_model_pipeline(
        model=model,
        config=optimization_config,
        output_dir=output_dir / model_path.stem
    )
    
    logger.info(f"Optimization complete for {model_path.name}")
    for opt_type, opt_path in optimization_results.items():
        logger.info(f"  - {opt_type}: {opt_path}")
    
    return optimization_results


def profile_optimized_models(optimized_models: Dict[str, Path], 
                           profiler: PerformanceProfiler) -> Dict[str, Dict]:
    """Profile performance of optimized models."""
    logger.info("Profiling optimized models...")
    
    profile_results = {}
    input_shape = (19, 10000)  # EEG input shape
    
    for model_type, model_path in optimized_models.items():
        logger.info(f"Profiling {model_type} model...")
        
        # Load optimized model
        if model_path.suffix == '.onnx':
            # ONNX models are handled differently
            wrapper = OptimizedModelWrapper(
                model_path, 
                OptimizationConfig(optimization_level=1)
            )
            # Profile using wrapper
            continue
        
        # Load PyTorch model
        model = torch.load(model_path)
        if hasattr(model, 'eval'):
            model.eval()
        
        # Profile model
        profile = profiler.profile_model(model, input_shape)
        profile_results[model_type] = profile
        
        # Profile with different batch sizes
        batch_profiles = profiler.profile_batch_sizes(model, input_shape)
        profile_results[f"{model_type}_batch"] = batch_profiles
    
    return profile_results


def setup_monitoring(config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup performance monitoring system."""
    logger.info("Setting up monitoring system...")
    
    monitoring_config = MonitoringConfig(
        enable_metrics=True,
        enable_alerts=True,
        metrics_port=config.get('deployment', {}).get('monitoring', {}).get('metrics_port', 8001),
        latency_threshold_ms=config.get('deployment', {}).get('monitoring', {}).get('alert_thresholds', {}).get('latency_ms', 1000),
        drift_threshold=0.15
    )
    
    monitoring_system = create_monitoring_system(monitoring_config)
    
    logger.info("Monitoring system initialized")
    return monitoring_system


def create_optimization_report(optimization_results: Dict, profile_results: Dict, 
                             output_path: Path):
    """Create comprehensive optimization report."""
    logger.info("Creating optimization report...")
    
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'optimization_results': {},
        'performance_profiles': profile_results,
        'recommendations': []
    }
    
    # Analyze optimization results
    for model_name, results in optimization_results.items():
        model_report = {}
        
        # Check file sizes
        for opt_type, opt_path in results.items():
            if opt_path.exists():
                size_mb = opt_path.stat().st_size / 1024 / 1024
                model_report[opt_type] = {
                    'path': str(opt_path),
                    'size_mb': round(size_mb, 2)
                }
        
        report['optimization_results'][model_name] = model_report
    
    # Add recommendations
    if profile_results:
        # Find best performing configuration
        best_config = min(
            profile_results.items(),
            key=lambda x: x[1].get('mean_latency_ms', float('inf'))
        )
        report['recommendations'].append(
            f"Best latency: {best_config[0]} with {best_config[1]['mean_latency_ms']:.2f}ms"
        )
    
    # Save report
    import json
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Optimization report saved to: {output_path}")


def main():
    """Main optimization pipeline."""
    parser = argparse.ArgumentParser(description='Optimize HMS EEG Classification Models')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model-dir', type=str, default='models/final',
                        help='Directory containing trained models')
    parser.add_argument('--output-dir', type=str, default='models/optimized',
                        help='Output directory for optimized models')
    parser.add_argument('--optimization-level', type=int, default=2,
                        help='Optimization level (0-3)')
    parser.add_argument('--enable-tensorrt', action='store_true',
                        help='Enable TensorRT optimization (requires NVIDIA GPU)')
    parser.add_argument('--profile-only', action='store_true',
                        help='Only profile existing optimized models')
    parser.add_argument('--monitoring', action='store_true',
                        help='Enable performance monitoring')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup monitoring if requested
    monitoring_system = None
    if args.monitoring:
        monitoring_system = setup_monitoring(config)
    
    # Create optimization configuration
    opt_config = OptimizationConfig(
        quantization_backend="fbgemm",
        quantization_dtype="int8",
        pruning_amount=config.get('optimization', {}).get('model_compression', {}).get('pruning_sparsity', 0.3),
        onnx_opset_version=14,
        tensorrt_precision="fp16" if args.enable_tensorrt else "fp32",
        dynamic_batch_size=True,
        max_batch_size=config.get('deployment', {}).get('model_serving', {}).get('max_batch_size', 32),
        optimization_level=args.optimization_level
    )
    
    if not args.profile_only:
        # Find models to optimize
        model_dir = Path(args.model_dir)
        model_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pt"))
        
        if not model_files:
            logger.error(f"No model files found in {model_dir}")
            return
        
        logger.info(f"Found {len(model_files)} models to optimize")
        
        # Optimize each model
        optimization_results = {}
        for model_file in model_files:
            try:
                results = optimize_single_model(model_file, opt_config, output_dir)
                optimization_results[model_file.stem] = results
            except Exception as e:
                logger.error(f"Failed to optimize {model_file}: {e}")
                continue
    
    # Profile optimized models
    profiler = PerformanceProfiler(MonitoringConfig())
    
    # Find optimized models
    optimized_models = {}
    for opt_dir in output_dir.iterdir():
        if opt_dir.is_dir():
            # Look for optimized model files
            for model_file in opt_dir.iterdir():
                if model_file.suffix in ['.onnx', '.pth', '.pt', '.trt']:
                    key = f"{opt_dir.name}_{model_file.stem}"
                    optimized_models[key] = model_file
    
    if optimized_models:
        profile_results = profile_optimized_models(optimized_models, profiler)
        
        # Create report
        report_path = output_dir / "optimization_report.json"
        create_optimization_report(
            optimization_results if not args.profile_only else {},
            profile_results,
            report_path
        )
    
    # Cleanup
    if monitoring_system:
        monitoring_system['performance_monitor'].stop()
    
    logger.info("Model optimization complete!")


if __name__ == "__main__":
    main() 