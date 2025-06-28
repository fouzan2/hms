"""
Performance Monitoring and Analytics for HMS EEG Classification

This module implements comprehensive monitoring and analytics:
- Model performance monitoring dashboards
- Real-time inference latency tracking
- Model drift detection
- Resource utilization monitoring
- Predictive scaling
- Automated performance testing
- Alerting system
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import asyncio
from prometheus_client import Counter, Histogram, Gauge, Summary, Info, CollectorRegistry
import prometheus_client
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import psutil
import GPUtil
from collections import deque
import warnings
import threading
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import redis
import influxdb

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring."""
    enable_metrics: bool = True
    enable_logging: bool = True
    enable_alerts: bool = True
    metrics_port: int = 8001
    log_interval: int = 60  # seconds
    drift_check_interval: int = 3600  # 1 hour
    drift_threshold: float = 0.15
    latency_threshold_ms: float = 100
    memory_threshold_gb: float = 8.0
    gpu_utilization_threshold: float = 0.9
    alert_cooldown_minutes: int = 15
    retention_days: int = 30
    batch_size_for_monitoring: int = 32
    enable_profiling: bool = True


class MetricsCollector:
    """Collects and exposes metrics for Prometheus."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.registry = CollectorRegistry()
        
        # Define metrics
        self.prediction_counter = Counter(
            'model_predictions_total',
            'Total number of predictions made',
            ['model_name', 'class_label'],
            registry=self.registry
        )
        
        self.prediction_latency = Histogram(
            'model_prediction_latency_seconds',
            'Prediction latency in seconds',
            ['model_name'],
            buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0),
            registry=self.registry
        )
        
        self.batch_size_gauge = Gauge(
            'model_batch_size',
            'Current batch size being processed',
            ['model_name'],
            registry=self.registry
        )
        
        self.accuracy_gauge = Gauge(
            'model_accuracy',
            'Current model accuracy',
            ['model_name', 'metric_type'],
            registry=self.registry
        )
        
        self.memory_usage_gauge = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes',
            ['type'],  # cpu, gpu
            registry=self.registry
        )
        
        self.gpu_utilization_gauge = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id'],
            registry=self.registry
        )
        
        self.model_drift_gauge = Gauge(
            'model_drift_score',
            'Model drift detection score',
            ['model_name', 'drift_type'],
            registry=self.registry
        )
        
        self.error_counter = Counter(
            'model_errors_total',
            'Total number of errors',
            ['model_name', 'error_type'],
            registry=self.registry
        )
        
        # Model info
        self.model_info = Info(
            'model_info',
            'Model information',
            ['model_name'],
            registry=self.registry
        )
    
    def record_prediction(self, model_name: str, class_label: str, latency: float):
        """Record a prediction event."""
        self.prediction_counter.labels(
            model_name=model_name,
            class_label=class_label
        ).inc()
        
        self.prediction_latency.labels(
            model_name=model_name
        ).observe(latency)
    
    def update_batch_size(self, model_name: str, batch_size: int):
        """Update current batch size."""
        self.batch_size_gauge.labels(model_name=model_name).set(batch_size)
    
    def update_accuracy(self, model_name: str, metric_type: str, value: float):
        """Update model accuracy metric."""
        self.accuracy_gauge.labels(
            model_name=model_name,
            metric_type=metric_type
        ).set(value)
    
    def update_memory_usage(self):
        """Update memory usage metrics."""
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        self.memory_usage_gauge.labels(type='cpu').set(cpu_memory.used)
        
        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.memory_allocated(i)
                self.memory_usage_gauge.labels(type=f'gpu_{i}').set(gpu_memory)
    
    def update_gpu_utilization(self):
        """Update GPU utilization metrics."""
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                self.gpu_utilization_gauge.labels(gpu_id=gpu.id).set(gpu.load * 100)
        except:
            pass
    
    def update_drift_score(self, model_name: str, drift_type: str, score: float):
        """Update model drift score."""
        self.model_drift_gauge.labels(
            model_name=model_name,
            drift_type=drift_type
        ).set(score)
    
    def record_error(self, model_name: str, error_type: str):
        """Record an error event."""
        self.error_counter.labels(
            model_name=model_name,
            error_type=error_type
        ).inc()
    
    def start_metrics_server(self):
        """Start Prometheus metrics server."""
        if self.config.enable_metrics:
            prometheus_client.start_http_server(
                self.config.metrics_port,
                registry=self.registry
            )
            logger.info(f"Metrics server started on port {self.config.metrics_port}")


class PerformanceMonitor:
    """Monitors model performance in real-time."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.performance_history = deque(maxlen=1000)
        self.latency_history = deque(maxlen=1000)
        self.resource_history = deque(maxlen=1000)
        self.monitoring_thread = None
        self.stop_monitoring = False
        
    def start(self):
        """Start performance monitoring."""
        # Start metrics server
        self.metrics_collector.start_metrics_server()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop(self):
        """Stop performance monitoring."""
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join()
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_monitoring:
            try:
                # Update resource metrics
                self.metrics_collector.update_memory_usage()
                self.metrics_collector.update_gpu_utilization()
                
                # Sleep for interval
                time.sleep(self.config.log_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def record_inference(self, model_name: str, input_data: np.ndarray,
                        output: np.ndarray, ground_truth: Optional[np.ndarray] = None):
        """Record inference for monitoring."""
        start_time = time.time()
        
        # Record prediction
        predicted_class = np.argmax(output)
        latency = time.time() - start_time
        
        self.metrics_collector.record_prediction(
            model_name=model_name,
            class_label=str(predicted_class),
            latency=latency
        )
        
        # Store in history
        record = {
            'timestamp': datetime.now(),
            'model_name': model_name,
            'latency': latency,
            'predicted_class': predicted_class,
            'confidence': float(np.max(output)),
            'input_shape': input_data.shape,
            'output_shape': output.shape
        }
        
        if ground_truth is not None:
            record['ground_truth'] = ground_truth
            record['correct'] = predicted_class == ground_truth
            
        self.performance_history.append(record)
        self.latency_history.append(latency)
        
        # Check for performance issues
        self._check_performance_thresholds(model_name, latency)
    
    def _check_performance_thresholds(self, model_name: str, latency: float):
        """Check if performance exceeds thresholds."""
        # Check latency
        if latency > self.config.latency_threshold_ms / 1000:
            logger.warning(
                f"High latency detected for {model_name}: {latency*1000:.2f}ms"
            )
            
        # Check memory
        memory_gb = psutil.virtual_memory().used / 1e9
        if memory_gb > self.config.memory_threshold_gb:
            logger.warning(
                f"High memory usage: {memory_gb:.2f}GB"
            )
    
    def get_performance_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for the specified time window."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_records = [
            r for r in self.performance_history
            if r['timestamp'] > cutoff_time
        ]
        
        if not recent_records:
            return {}
        
        # Calculate metrics
        latencies = [r['latency'] for r in recent_records]
        
        summary = {
            'total_predictions': len(recent_records),
            'avg_latency_ms': np.mean(latencies) * 1000,
            'p50_latency_ms': np.percentile(latencies, 50) * 1000,
            'p95_latency_ms': np.percentile(latencies, 95) * 1000,
            'p99_latency_ms': np.percentile(latencies, 99) * 1000,
            'max_latency_ms': np.max(latencies) * 1000,
        }
        
        # Add accuracy if ground truth available
        records_with_truth = [r for r in recent_records if 'correct' in r]
        if records_with_truth:
            summary['accuracy'] = np.mean([r['correct'] for r in records_with_truth])
            
        return summary


class ModelDriftDetector:
    """Detects model drift and data distribution changes."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.reference_distributions = {}
        self.drift_scores = deque(maxlen=100)
        self.last_drift_check = datetime.now()
        
    def set_reference_distribution(self, model_name: str, 
                                 reference_data: np.ndarray,
                                 reference_predictions: np.ndarray):
        """Set reference distribution for drift detection."""
        self.reference_distributions[model_name] = {
            'data_stats': self._calculate_data_statistics(reference_data),
            'prediction_stats': self._calculate_prediction_statistics(reference_predictions),
            'timestamp': datetime.now()
        }
        
        logger.info(f"Reference distribution set for {model_name}")
    
    def check_drift(self, model_name: str, current_data: np.ndarray,
                   current_predictions: np.ndarray) -> Dict[str, float]:
        """Check for model drift."""
        if model_name not in self.reference_distributions:
            logger.warning(f"No reference distribution for {model_name}")
            return {}
            
        reference = self.reference_distributions[model_name]
        
        # Calculate drift scores
        data_drift = self._calculate_data_drift(
            reference['data_stats'],
            self._calculate_data_statistics(current_data)
        )
        
        prediction_drift = self._calculate_prediction_drift(
            reference['prediction_stats'],
            self._calculate_prediction_statistics(current_predictions)
        )
        
        drift_scores = {
            'data_drift': data_drift,
            'prediction_drift': prediction_drift,
            'overall_drift': (data_drift + prediction_drift) / 2
        }
        
        # Record drift scores
        self.drift_scores.append({
            'timestamp': datetime.now(),
            'model_name': model_name,
            'scores': drift_scores
        })
        
        # Check threshold
        if drift_scores['overall_drift'] > self.config.drift_threshold:
            logger.warning(
                f"Model drift detected for {model_name}: {drift_scores['overall_drift']:.3f}"
            )
            
        return drift_scores
    
    def _calculate_data_statistics(self, data: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics for data distribution."""
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'percentiles': np.percentile(data, [25, 50, 75], axis=0)
        }
    
    def _calculate_prediction_statistics(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics for prediction distribution."""
        # Class distribution
        if len(predictions.shape) > 1:
            class_probs = np.mean(predictions, axis=0)
            entropy = -np.sum(predictions * np.log(predictions + 1e-8), axis=1)
        else:
            class_counts = np.bincount(predictions.astype(int))
            class_probs = class_counts / len(predictions)
            entropy = None
            
        stats = {
            'class_distribution': class_probs,
            'confidence_mean': np.mean(np.max(predictions, axis=1)) if len(predictions.shape) > 1 else None,
            'confidence_std': np.std(np.max(predictions, axis=1)) if len(predictions.shape) > 1 else None,
        }
        
        if entropy is not None:
            stats['entropy_mean'] = np.mean(entropy)
            stats['entropy_std'] = np.std(entropy)
            
        return stats
    
    def _calculate_data_drift(self, reference_stats: Dict, current_stats: Dict) -> float:
        """Calculate data drift score using statistical tests."""
        drift_scores = []
        
        # Compare means
        mean_diff = np.mean(np.abs(reference_stats['mean'] - current_stats['mean']))
        drift_scores.append(mean_diff)
        
        # Compare standard deviations
        std_diff = np.mean(np.abs(reference_stats['std'] - current_stats['std']))
        drift_scores.append(std_diff)
        
        # KS test approximation
        for p_ref, p_curr in zip(reference_stats['percentiles'], current_stats['percentiles']):
            ks_score = np.mean(np.abs(p_ref - p_curr))
            drift_scores.append(ks_score)
            
        return np.mean(drift_scores)
    
    def _calculate_prediction_drift(self, reference_stats: Dict, current_stats: Dict) -> float:
        """Calculate prediction drift score."""
        drift_scores = []
        
        # Compare class distributions using KL divergence
        ref_dist = reference_stats['class_distribution']
        curr_dist = current_stats['class_distribution']
        
        # Ensure same shape
        if len(ref_dist) == len(curr_dist):
            kl_div = np.sum(ref_dist * np.log((ref_dist + 1e-8) / (curr_dist + 1e-8)))
            drift_scores.append(kl_div)
            
        # Compare confidence statistics
        if reference_stats.get('confidence_mean') and current_stats.get('confidence_mean'):
            conf_diff = abs(reference_stats['confidence_mean'] - current_stats['confidence_mean'])
            drift_scores.append(conf_diff)
            
        return np.mean(drift_scores) if drift_scores else 0.0


class ResourceMonitor:
    """Monitors system resource utilization."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.resource_history = deque(maxlen=1000)
        self.monitoring_active = False
        
    def get_current_resources(self) -> Dict[str, Any]:
        """Get current resource utilization."""
        resources = {
            'timestamp': datetime.now(),
            'cpu': {
                'percent': psutil.cpu_percent(interval=0.1),
                'count': psutil.cpu_count(),
                'freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0
            },
            'memory': {
                'total_gb': psutil.virtual_memory().total / 1e9,
                'used_gb': psutil.virtual_memory().used / 1e9,
                'available_gb': psutil.virtual_memory().available / 1e9,
                'percent': psutil.virtual_memory().percent
            },
            'disk': {
                'total_gb': psutil.disk_usage('/').total / 1e9,
                'used_gb': psutil.disk_usage('/').used / 1e9,
                'free_gb': psutil.disk_usage('/').free / 1e9,
                'percent': psutil.disk_usage('/').percent
            }
        }
        
        # GPU resources
        if torch.cuda.is_available():
            resources['gpu'] = []
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total_gb': torch.cuda.get_device_properties(i).total_memory / 1e9,
                    'memory_allocated_gb': torch.cuda.memory_allocated(i) / 1e9,
                    'memory_reserved_gb': torch.cuda.memory_reserved(i) / 1e9,
                }
                
                # Add GPU utilization if available
                try:
                    gpus = GPUtil.getGPUs()
                    if i < len(gpus):
                        gpu_info['utilization_percent'] = gpus[i].load * 100
                        gpu_info['temperature_c'] = gpus[i].temperature
                except:
                    pass
                    
                resources['gpu'].append(gpu_info)
                
        self.resource_history.append(resources)
        return resources
    
    def predict_resource_needs(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Predict future resource needs based on trends."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_resources = [
            r for r in self.resource_history
            if r['timestamp'] > cutoff_time
        ]
        
        if len(recent_resources) < 10:
            return {}
            
        # Extract time series
        timestamps = [r['timestamp'].timestamp() for r in recent_resources]
        cpu_usage = [r['cpu']['percent'] for r in recent_resources]
        memory_usage = [r['memory']['percent'] for r in recent_resources]
        
        # Simple linear regression for trend
        cpu_trend = np.polyfit(timestamps, cpu_usage, 1)[0]
        memory_trend = np.polyfit(timestamps, memory_usage, 1)[0]
        
        # Predict next hour
        current_time = datetime.now().timestamp()
        future_time = current_time + 3600  # 1 hour
        
        predictions = {
            'cpu_percent_1h': np.clip(cpu_usage[-1] + cpu_trend * 3600, 0, 100),
            'memory_percent_1h': np.clip(memory_usage[-1] + memory_trend * 3600, 0, 100),
            'cpu_trend': 'increasing' if cpu_trend > 0.01 else 'decreasing' if cpu_trend < -0.01 else 'stable',
            'memory_trend': 'increasing' if memory_trend > 0.01 else 'decreasing' if memory_trend < -0.01 else 'stable'
        }
        
        return predictions


class AlertingSystem:
    """Manages alerts and notifications."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alert_history = deque(maxlen=1000)
        self.alert_cooldowns = {}
        
    def check_and_alert(self, alert_type: str, severity: str, 
                       message: str, details: Dict[str, Any] = None):
        """Check if alert should be sent and send if needed."""
        if not self.config.enable_alerts:
            return
            
        # Check cooldown
        alert_key = f"{alert_type}:{severity}"
        if alert_key in self.alert_cooldowns:
            last_alert_time = self.alert_cooldowns[alert_key]
            if datetime.now() - last_alert_time < timedelta(minutes=self.config.alert_cooldown_minutes):
                return
                
        # Create alert
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'severity': severity,
            'message': message,
            'details': details or {}
        }
        
        # Send alert
        self._send_alert(alert)
        
        # Update cooldown
        self.alert_cooldowns[alert_key] = datetime.now()
        
        # Store in history
        self.alert_history.append(alert)
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert through configured channels."""
        # Log alert
        if alert['severity'] == 'critical':
            logger.critical(f"ALERT: {alert['message']}")
        elif alert['severity'] == 'warning':
            logger.warning(f"ALERT: {alert['message']}")
        else:
            logger.info(f"ALERT: {alert['message']}")
            
        # Send to external systems (webhook, email, etc.)
        # This would be implemented based on specific requirements
        pass


class PerformanceProfiler:
    """Profiles model and system performance."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.profiling_results = {}
        
    def profile_model(self, model: torch.nn.Module, input_shape: Tuple[int, ...],
                     num_runs: int = 100) -> Dict[str, Any]:
        """Profile model performance."""
        device = next(model.parameters()).device
        
        # Warmup
        dummy_input = torch.randn(1, *input_shape).to(device)
        for _ in range(10):
            _ = model(dummy_input)
            
        # Profile inference time
        latencies = []
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        for _ in range(num_runs):
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = model(dummy_input)
                
            torch.cuda.synchronize() if device.type == 'cuda' else None
            latencies.append(time.perf_counter() - start_time)
            
        # Calculate statistics
        profile = {
            'mean_latency_ms': np.mean(latencies) * 1000,
            'std_latency_ms': np.std(latencies) * 1000,
            'min_latency_ms': np.min(latencies) * 1000,
            'max_latency_ms': np.max(latencies) * 1000,
            'p50_latency_ms': np.percentile(latencies, 50) * 1000,
            'p95_latency_ms': np.percentile(latencies, 95) * 1000,
            'p99_latency_ms': np.percentile(latencies, 99) * 1000,
        }
        
        # Profile memory usage
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            _ = model(dummy_input)
            
            profile['gpu_memory_mb'] = torch.cuda.max_memory_allocated() / 1e6
            
        # Profile FLOPs (if possible)
        try:
            from thop import profile as thop_profile
            flops, params = thop_profile(model, inputs=(dummy_input,), verbose=False)
            profile['flops'] = flops
            profile['parameters'] = params
        except:
            pass
            
        return profile
    
    def profile_batch_sizes(self, model: torch.nn.Module, input_shape: Tuple[int, ...],
                          batch_sizes: List[int] = [1, 2, 4, 8, 16, 32, 64]) -> Dict[int, Dict]:
        """Profile model with different batch sizes."""
        results = {}
        
        for batch_size in batch_sizes:
            try:
                # Create batch input
                batch_shape = (batch_size,) + input_shape
                profile = self.profile_model(model, batch_shape[1:], num_runs=50)
                
                # Add throughput
                profile['throughput_samples_per_sec'] = batch_size / (profile['mean_latency_ms'] / 1000)
                
                results[batch_size] = profile
                
            except Exception as e:
                logger.warning(f"Failed to profile batch size {batch_size}: {e}")
                break
                
        return results


class MonitoringDashboard:
    """Creates monitoring visualizations."""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        
    def create_performance_dashboard(self, save_path: Optional[Path] = None) -> plt.Figure:
        """Create comprehensive performance dashboard."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Performance Dashboard', fontsize=16)
        
        # Latency distribution
        ax = axes[0, 0]
        if self.monitor.latency_history:
            latencies_ms = [l * 1000 for l in self.monitor.latency_history]
            ax.hist(latencies_ms, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(np.mean(latencies_ms), color='red', linestyle='--', label=f'Mean: {np.mean(latencies_ms):.2f}ms')
            ax.set_xlabel('Latency (ms)')
            ax.set_ylabel('Frequency')
            ax.set_title('Latency Distribution')
            ax.legend()
        
        # Latency over time
        ax = axes[0, 1]
        if self.monitor.performance_history:
            timestamps = [r['timestamp'] for r in self.monitor.performance_history]
            latencies = [r['latency'] * 1000 for r in self.monitor.performance_history]
            ax.plot(timestamps, latencies, alpha=0.7)
            ax.set_xlabel('Time')
            ax.set_ylabel('Latency (ms)')
            ax.set_title('Latency Over Time')
            ax.tick_params(axis='x', rotation=45)
        
        # Accuracy over time
        ax = axes[0, 2]
        records_with_truth = [r for r in self.monitor.performance_history if 'correct' in r]
        if records_with_truth:
            # Calculate rolling accuracy
            window_size = min(100, len(records_with_truth))
            accuracies = []
            timestamps = []
            
            for i in range(window_size, len(records_with_truth)):
                window = records_with_truth[i-window_size:i]
                accuracy = np.mean([r['correct'] for r in window])
                accuracies.append(accuracy)
                timestamps.append(window[-1]['timestamp'])
                
            if accuracies:
                ax.plot(timestamps, accuracies, alpha=0.7, color='green')
                ax.set_xlabel('Time')
                ax.set_ylabel('Accuracy')
                ax.set_title('Rolling Accuracy (100 samples)')
                ax.tick_params(axis='x', rotation=45)
                ax.set_ylim(0, 1.1)
        
        # Resource utilization
        ax = axes[1, 0]
        resource_monitor = ResourceMonitor(self.monitor.config)
        current_resources = resource_monitor.get_current_resources()
        
        categories = ['CPU', 'Memory', 'Disk']
        values = [
            current_resources['cpu']['percent'],
            current_resources['memory']['percent'],
            current_resources['disk']['percent']
        ]
        
        bars = ax.bar(categories, values, color=['blue', 'green', 'orange'])
        ax.set_ylabel('Usage %')
        ax.set_title('Current Resource Utilization')
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}%', ha='center', va='bottom')
        
        # GPU utilization
        ax = axes[1, 1]
        if 'gpu' in current_resources and current_resources['gpu']:
            gpu_names = []
            gpu_memory = []
            gpu_util = []
            
            for gpu in current_resources['gpu']:
                gpu_names.append(f"GPU {gpu['id']}")
                gpu_memory.append(gpu['memory_allocated_gb'] / gpu['memory_total_gb'] * 100)
                if 'utilization_percent' in gpu:
                    gpu_util.append(gpu['utilization_percent'])
                    
            x = np.arange(len(gpu_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, gpu_memory, width, label='Memory', color='blue')
            if gpu_util:
                bars2 = ax.bar(x + width/2, gpu_util, width, label='Utilization', color='green')
                
            ax.set_xlabel('GPU')
            ax.set_ylabel('Usage %')
            ax.set_title('GPU Resource Usage')
            ax.set_xticks(x)
            ax.set_xticklabels(gpu_names)
            ax.legend()
            ax.set_ylim(0, 100)
        
        # Drift scores
        ax = axes[1, 2]
        drift_detector = ModelDriftDetector(self.monitor.config)
        if drift_detector.drift_scores:
            recent_scores = list(drift_detector.drift_scores)[-20:]
            timestamps = [s['timestamp'] for s in recent_scores]
            overall_drifts = [s['scores']['overall_drift'] for s in recent_scores]
            
            ax.plot(timestamps, overall_drifts, marker='o', alpha=0.7)
            ax.axhline(y=self.monitor.config.drift_threshold, color='r', linestyle='--', 
                      label=f'Threshold: {self.monitor.config.drift_threshold}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Drift Score')
            ax.set_title('Model Drift Detection')
            ax.tick_params(axis='x', rotation=45)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


def create_monitoring_system(config: MonitoringConfig) -> Dict[str, Any]:
    """Create complete monitoring system."""
    
    # Initialize components
    performance_monitor = PerformanceMonitor(config)
    drift_detector = ModelDriftDetector(config)
    resource_monitor = ResourceMonitor(config)
    alerting_system = AlertingSystem(config)
    profiler = PerformanceProfiler(config)
    
    # Start monitoring
    performance_monitor.start()
    
    return {
        'performance_monitor': performance_monitor,
        'drift_detector': drift_detector,
        'resource_monitor': resource_monitor,
        'alerting_system': alerting_system,
        'profiler': profiler,
        'dashboard': MonitoringDashboard(performance_monitor)
    } 