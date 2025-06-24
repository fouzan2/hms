#!/usr/bin/env python3
"""
Model Benchmarking Script for HMS EEG Classification System

This script benchmarks the performance of different model versions:
- Original models
- Quantized models
- ONNX models
- TensorRT models
"""

import argparse
import logging
import yaml
import torch
import numpy as np
from pathlib import Path
import sys
import time
import json
from typing import Dict, List, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.deployment.model_optimization import OptimizedModelWrapper, OptimizationConfig
from src.deployment.performance_monitoring import PerformanceProfiler, MonitoringConfig
from src.models import ResNet1D_GRU, EfficientNetSpectrogram, HMSEnsembleModel

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


def generate_dummy_data(batch_size: int, input_shape: Tuple[int, ...], 
                       device: str = 'cuda') -> torch.Tensor:
    """Generate dummy data for benchmarking."""
    data = torch.randn(batch_size, *input_shape)
    if device == 'cuda' and torch.cuda.is_available():
        data = data.cuda()
    return data


def benchmark_single_model(model_path: Path, batch_sizes: List[int], 
                          input_shape: Tuple[int, ...], 
                          num_runs: int = 100) -> Dict[str, Any]:
    """Benchmark a single model with different batch sizes."""
    logger.info(f"Benchmarking model: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {
        'model_path': str(model_path),
        'model_type': model_path.suffix,
        'device': str(device),
        'batch_results': {}
    }
    
    # Load model based on type
    if model_path.suffix == '.onnx':
        opt_config = OptimizationConfig(optimization_level=1)
        model = OptimizedModelWrapper(model_path, opt_config)
        is_optimized = True
    else:
        # Load PyTorch model
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            model_state = checkpoint
            
        # Create model architecture
        if 'resnet' in str(model_path).lower():
            model = ResNet1D_GRU(num_channels=19, num_classes=6)
        elif 'efficientnet' in str(model_path).lower():
            model = EfficientNetSpectrogram(num_classes=6)
        else:
            config = load_config('config/config.yaml')
            model = HMSEnsembleModel(config)
            
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        is_optimized = False
    
    # Benchmark each batch size
    for batch_size in batch_sizes:
        logger.info(f"Testing batch size: {batch_size}")
        
        try:
            # Generate dummy data
            dummy_data = generate_dummy_data(batch_size, input_shape, str(device))
            
            # Warmup
            for _ in range(10):
                if is_optimized:
                    _ = model.predict(dummy_data.cpu().numpy())
                else:
                    with torch.no_grad():
                        _ = model(dummy_data)
            
            # Benchmark
            latencies = []
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            for _ in range(num_runs):
                start_time = time.perf_counter()
                
                if is_optimized:
                    _ = model.predict(dummy_data.cpu().numpy())
                else:
                    with torch.no_grad():
                        _ = model(dummy_data)
                        
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                latencies.append(time.perf_counter() - start_time)
            
            # Calculate statistics
            batch_results = {
                'batch_size': batch_size,
                'mean_latency_ms': np.mean(latencies) * 1000,
                'std_latency_ms': np.std(latencies) * 1000,
                'min_latency_ms': np.min(latencies) * 1000,
                'max_latency_ms': np.max(latencies) * 1000,
                'p50_latency_ms': np.percentile(latencies, 50) * 1000,
                'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                'p99_latency_ms': np.percentile(latencies, 99) * 1000,
                'throughput_samples_per_sec': batch_size / np.mean(latencies)
            }
            
            # Memory usage
            if torch.cuda.is_available() and not is_optimized:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                with torch.no_grad():
                    _ = model(dummy_data)
                    
                batch_results['gpu_memory_mb'] = torch.cuda.max_memory_allocated() / 1e6
                
            results['batch_results'][batch_size] = batch_results
            
        except Exception as e:
            logger.error(f"Failed to benchmark batch size {batch_size}: {e}")
            break
    
    # Calculate model size
    model_size_mb = model_path.stat().st_size / 1e6
    results['model_size_mb'] = model_size_mb
    
    return results


def compare_models(benchmark_results: List[Dict[str, Any]], output_dir: Path):
    """Compare and visualize benchmark results."""
    logger.info("Comparing model performance...")
    
    # Create comparison dataframe
    comparison_data = []
    
    for result in benchmark_results:
        model_name = Path(result['model_path']).stem
        
        for batch_size, batch_result in result['batch_results'].items():
            row = {
                'model': model_name,
                'model_type': result['model_type'],
                'batch_size': batch_size,
                'mean_latency_ms': batch_result['mean_latency_ms'],
                'throughput': batch_result['throughput_samples_per_sec'],
                'model_size_mb': result['model_size_mb']
            }
            
            if 'gpu_memory_mb' in batch_result:
                row['gpu_memory_mb'] = batch_result['gpu_memory_mb']
                
            comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    # Latency comparison
    ax = axes[0, 0]
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        ax.plot(model_data['batch_size'], model_data['mean_latency_ms'], 
                marker='o', label=model)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title('Latency vs Batch Size')
    ax.legend()
    ax.grid(True)
    
    # Throughput comparison
    ax = axes[0, 1]
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        ax.plot(model_data['batch_size'], model_data['throughput'], 
                marker='o', label=model)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (samples/sec)')
    ax.set_title('Throughput vs Batch Size')
    ax.legend()
    ax.grid(True)
    
    # Model size comparison
    ax = axes[1, 0]
    model_sizes = df[['model', 'model_size_mb']].drop_duplicates()
    ax.bar(model_sizes['model'], model_sizes['model_size_mb'])
    ax.set_xlabel('Model')
    ax.set_ylabel('Model Size (MB)')
    ax.set_title('Model Size Comparison')
    ax.tick_params(axis='x', rotation=45)
    
    # Memory usage comparison
    ax = axes[1, 1]
    if 'gpu_memory_mb' in df.columns:
        for batch_size in df['batch_size'].unique():
            batch_data = df[df['batch_size'] == batch_size]
            models = batch_data['model'].values
            memory = batch_data['gpu_memory_mb'].values
            
            ax.plot(models, memory, marker='o', label=f'Batch {batch_size}')
            
        ax.set_xlabel('Model')
        ax.set_ylabel('GPU Memory (MB)')
        ax.set_title('GPU Memory Usage')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_comparison.png', dpi=300, bbox_inches='tight')
    
    # Save detailed results
    df.to_csv(output_dir / 'benchmark_results.csv', index=False)
    
    # Create summary report
    summary = {
        'fastest_model': df.loc[df['mean_latency_ms'].idxmin()]['model'],
        'highest_throughput': df.loc[df['throughput'].idxmax()]['model'],
        'smallest_model': model_sizes.loc[model_sizes['model_size_mb'].idxmin()]['model'],
        'recommendations': []
    }
    
    # Add recommendations
    if 'quantized' in df['model'].str.lower().values.any():
        quantized_speedup = df[df['model'].str.contains('quantized', case=False)]['mean_latency_ms'].mean()
        original_latency = df[~df['model'].str.contains('quantized|onnx|trt', case=False)]['mean_latency_ms'].mean()
        speedup = original_latency / quantized_speedup
        summary['recommendations'].append(
            f"Quantized models provide {speedup:.2f}x speedup on average"
        )
    
    if 'onnx' in df['model_type'].values:
        summary['recommendations'].append(
            "ONNX models provide cross-platform compatibility with good performance"
        )
    
    # Save summary
    with open(output_dir / 'benchmark_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Benchmark results saved to {output_dir}")
    
    return df, summary


def main():
    """Main benchmarking pipeline."""
    parser = argparse.ArgumentParser(description='Benchmark HMS EEG Classification Models')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory containing models to benchmark')
    parser.add_argument('--batch-sizes', type=str, default='1,2,4,8,16,32',
                        help='Comma-separated list of batch sizes')
    parser.add_argument('--num-runs', type=int, default=100,
                        help='Number of runs per benchmark')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                        help='Output directory for results')
    parser.add_argument('--input-channels', type=int, default=19,
                        help='Number of input channels')
    parser.add_argument('--input-length', type=int, default=10000,
                        help='Input sequence length')
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    
    # Input shape for EEG data
    input_shape = (args.input_channels, args.input_length)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all models to benchmark
    model_dir = Path(args.model_dir)
    model_files = []
    
    # Original models
    model_files.extend(model_dir.glob('final/*.pth'))
    model_files.extend(model_dir.glob('final/*.pt'))
    
    # Optimized models
    if (model_dir / 'optimized').exists():
        model_files.extend((model_dir / 'optimized').rglob('*.pth'))
        model_files.extend((model_dir / 'optimized').rglob('*.onnx'))
        model_files.extend((model_dir / 'optimized').rglob('*.trt'))
    
    if not model_files:
        logger.error(f"No model files found in {model_dir}")
        return
    
    logger.info(f"Found {len(model_files)} models to benchmark")
    
    # Benchmark each model
    benchmark_results = []
    
    for model_file in model_files:
        try:
            result = benchmark_single_model(
                model_file, 
                batch_sizes, 
                input_shape,
                num_runs=args.num_runs
            )
            benchmark_results.append(result)
            
        except Exception as e:
            logger.error(f"Failed to benchmark {model_file}: {e}")
            continue
    
    # Compare and visualize results
    if benchmark_results:
        df, summary = compare_models(benchmark_results, output_dir)
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("BENCHMARK SUMMARY")
        logger.info("="*50)
        logger.info(f"Fastest model: {summary['fastest_model']}")
        logger.info(f"Highest throughput: {summary['highest_throughput']}")
        logger.info(f"Smallest model: {summary['smallest_model']}")
        
        if summary['recommendations']:
            logger.info("\nRecommendations:")
            for rec in summary['recommendations']:
                logger.info(f"  - {rec}")
    
    logger.info("\nBenchmarking complete!")


if __name__ == "__main__":
    main() 