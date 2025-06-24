"""
Training Progress Monitor for HMS EEG Classification System

This module provides real-time visualization of training progress including:
- Loss and accuracy curves
- Learning rate scheduling
- GPU/CPU utilization
- Estimated time remaining
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import psutil
import GPUtil
import threading
from queue import Queue
import logging

logger = logging.getLogger(__name__)


class TrainingProgressMonitor:
    """Real-time training progress monitoring and visualization."""
    
    def __init__(self, 
                 log_dir: str = "logs/training",
                 update_interval: int = 10,
                 save_interval: int = 100,
                 enable_realtime: bool = True):
        """
        Initialize training progress monitor.
        
        Args:
            log_dir: Directory to save logs and visualizations
            update_interval: Update interval in seconds for real-time monitoring
            save_interval: Save interval for plots (in iterations)
            enable_realtime: Enable real-time monitoring
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.update_interval = update_interval
        self.save_interval = save_interval
        self.enable_realtime = enable_realtime
        
        # Training metrics storage
        self.metrics = {
            'iteration': [],
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': [],
            'gpu_memory': [],
            'cpu_percent': [],
            'timestamp': []
        }
        
        # Real-time monitoring
        self.monitoring_queue = Queue()
        self.monitoring_thread = None
        self.stop_monitoring = False
        
        # Initialize plots
        self._setup_plots()
        
        if self.enable_realtime:
            self._start_monitoring()
    
    def _setup_plots(self):
        """Setup matplotlib figures for visualization."""
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (15, 10)
        plt.rcParams['font.size'] = 10
        
        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 3, figsize=(20, 12))
        self.fig.suptitle('HMS EEG Training Progress Monitor', fontsize=16)
        
        # Configure subplots
        self.loss_ax = self.axes[0, 0]
        self.acc_ax = self.axes[0, 1]
        self.lr_ax = self.axes[0, 2]
        self.gpu_ax = self.axes[1, 0]
        self.cpu_ax = self.axes[1, 1]
        self.time_ax = self.axes[1, 2]
        
        # Set labels
        self.loss_ax.set_title('Loss Curves')
        self.loss_ax.set_xlabel('Iteration')
        self.loss_ax.set_ylabel('Loss')
        
        self.acc_ax.set_title('Accuracy Curves')
        self.acc_ax.set_xlabel('Iteration')
        self.acc_ax.set_ylabel('Accuracy (%)')
        
        self.lr_ax.set_title('Learning Rate Schedule')
        self.lr_ax.set_xlabel('Iteration')
        self.lr_ax.set_ylabel('Learning Rate')
        
        self.gpu_ax.set_title('GPU Memory Usage')
        self.gpu_ax.set_xlabel('Time')
        self.gpu_ax.set_ylabel('Memory (MB)')
        
        self.cpu_ax.set_title('CPU Usage')
        self.cpu_ax.set_xlabel('Time')
        self.cpu_ax.set_ylabel('Usage (%)')
        
        self.time_ax.set_title('Training Progress')
        self.time_ax.set_xlabel('Progress')
        
        plt.tight_layout()
    
    def _start_monitoring(self):
        """Start real-time monitoring thread."""
        self.monitoring_thread = threading.Thread(target=self._monitor_resources)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info("Started real-time resource monitoring")
    
    def _monitor_resources(self):
        """Monitor system resources in background."""
        while not self.stop_monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # GPU usage
                gpu_memory = 0
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_memory = gpus[0].memoryUsed
                except:
                    pass
                
                # Add to queue
                self.monitoring_queue.put({
                    'cpu_percent': cpu_percent,
                    'gpu_memory': gpu_memory,
                    'timestamp': datetime.now()
                })
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
    
    def log_metrics(self, 
                   iteration: int,
                   epoch: int,
                   train_loss: float,
                   val_loss: Optional[float] = None,
                   train_acc: Optional[float] = None,
                   val_acc: Optional[float] = None,
                   learning_rate: Optional[float] = None):
        """
        Log training metrics.
        
        Args:
            iteration: Current iteration
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            train_acc: Training accuracy
            val_acc: Validation accuracy
            learning_rate: Current learning rate
        """
        # Get resource metrics
        resource_metrics = {'cpu_percent': 0, 'gpu_memory': 0}
        if not self.monitoring_queue.empty():
            resource_metrics = self.monitoring_queue.get()
        
        # Store metrics
        self.metrics['iteration'].append(iteration)
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss or np.nan)
        self.metrics['train_acc'].append(train_acc or np.nan)
        self.metrics['val_acc'].append(val_acc or np.nan)
        self.metrics['learning_rate'].append(learning_rate or np.nan)
        self.metrics['gpu_memory'].append(resource_metrics['gpu_memory'])
        self.metrics['cpu_percent'].append(resource_metrics['cpu_percent'])
        self.metrics['timestamp'].append(datetime.now())
        
        # Update plots
        if iteration % self.save_interval == 0:
            self.update_plots()
            self.save_plots()
        
        # Save metrics
        self._save_metrics()
    
    def update_plots(self):
        """Update all plots with current metrics."""
        if len(self.metrics['iteration']) == 0:
            return
        
        # Clear axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Loss curves
        self.loss_ax.plot(self.metrics['iteration'], self.metrics['train_loss'], 
                         label='Train Loss', color='blue', alpha=0.8)
        if not all(np.isnan(self.metrics['val_loss'])):
            self.loss_ax.plot(self.metrics['iteration'], self.metrics['val_loss'], 
                             label='Val Loss', color='red', alpha=0.8)
        self.loss_ax.set_xlabel('Iteration')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.set_title('Loss Curves')
        self.loss_ax.legend()
        self.loss_ax.grid(True, alpha=0.3)
        
        # Accuracy curves
        if not all(np.isnan(self.metrics['train_acc'])):
            self.acc_ax.plot(self.metrics['iteration'], self.metrics['train_acc'], 
                            label='Train Acc', color='blue', alpha=0.8)
        if not all(np.isnan(self.metrics['val_acc'])):
            self.acc_ax.plot(self.metrics['iteration'], self.metrics['val_acc'], 
                            label='Val Acc', color='red', alpha=0.8)
        self.acc_ax.set_xlabel('Iteration')
        self.acc_ax.set_ylabel('Accuracy (%)')
        self.acc_ax.set_title('Accuracy Curves')
        self.acc_ax.legend()
        self.acc_ax.grid(True, alpha=0.3)
        
        # Learning rate
        if not all(np.isnan(self.metrics['learning_rate'])):
            self.lr_ax.plot(self.metrics['iteration'], self.metrics['learning_rate'], 
                           color='green', alpha=0.8)
        self.lr_ax.set_xlabel('Iteration')
        self.lr_ax.set_ylabel('Learning Rate')
        self.lr_ax.set_title('Learning Rate Schedule')
        self.lr_ax.set_yscale('log')
        self.lr_ax.grid(True, alpha=0.3)
        
        # GPU memory
        if any(self.metrics['gpu_memory']):
            time_minutes = [(t - self.metrics['timestamp'][0]).total_seconds() / 60 
                           for t in self.metrics['timestamp']]
            self.gpu_ax.plot(time_minutes, self.metrics['gpu_memory'], 
                            color='orange', alpha=0.8)
            self.gpu_ax.set_xlabel('Time (minutes)')
            self.gpu_ax.set_ylabel('Memory (MB)')
            self.gpu_ax.set_title('GPU Memory Usage')
            self.gpu_ax.grid(True, alpha=0.3)
        
        # CPU usage
        if any(self.metrics['cpu_percent']):
            time_minutes = [(t - self.metrics['timestamp'][0]).total_seconds() / 60 
                           for t in self.metrics['timestamp']]
            self.cpu_ax.plot(time_minutes, self.metrics['cpu_percent'], 
                            color='purple', alpha=0.8)
            self.cpu_ax.set_xlabel('Time (minutes)')
            self.cpu_ax.set_ylabel('Usage (%)')
            self.cpu_ax.set_title('CPU Usage')
            self.cpu_ax.grid(True, alpha=0.3)
        
        # Training progress
        if len(self.metrics['epoch']) > 0:
            current_epoch = self.metrics['epoch'][-1]
            total_epochs = max(self.metrics['epoch']) + 10  # Estimate
            progress = (current_epoch / total_epochs) * 100
            
            # Time estimation
            elapsed = (self.metrics['timestamp'][-1] - self.metrics['timestamp'][0]).total_seconds()
            if progress > 0:
                total_time = elapsed / (progress / 100)
                remaining = total_time - elapsed
                eta = datetime.now() + timedelta(seconds=remaining)
                
                self.time_ax.text(0.5, 0.7, f'Progress: {progress:.1f}%', 
                                 transform=self.time_ax.transAxes,
                                 ha='center', fontsize=14)
                self.time_ax.text(0.5, 0.5, f'Elapsed: {timedelta(seconds=int(elapsed))}', 
                                 transform=self.time_ax.transAxes,
                                 ha='center', fontsize=12)
                self.time_ax.text(0.5, 0.3, f'ETA: {eta.strftime("%Y-%m-%d %H:%M:%S")}', 
                                 transform=self.time_ax.transAxes,
                                 ha='center', fontsize=12)
            
            # Progress bar
            self.time_ax.barh(0, progress, height=0.3, color='green', alpha=0.7)
            self.time_ax.barh(0, 100-progress, left=progress, height=0.3, 
                             color='lightgray', alpha=0.3)
            self.time_ax.set_xlim(0, 100)
            self.time_ax.set_ylim(-0.5, 0.5)
            self.time_ax.set_xticks([0, 25, 50, 75, 100])
            self.time_ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
            self.time_ax.set_yticks([])
        
        self.time_ax.set_title('Training Progress')
        
        plt.tight_layout()
    
    def save_plots(self, filename: Optional[str] = None):
        """Save current plots to file."""
        if filename is None:
            filename = f"training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        filepath = self.log_dir / filename
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
        logger.info(f"Saved training progress plot to {filepath}")
    
    def create_interactive_dashboard(self) -> go.Figure:
        """Create interactive Plotly dashboard."""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Loss Curves', 'Accuracy Curves', 
                          'Learning Rate', 'GPU Memory',
                          'CPU Usage', 'Training Progress'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Loss curves
        fig.add_trace(
            go.Scatter(x=self.metrics['iteration'], y=self.metrics['train_loss'],
                      name='Train Loss', mode='lines'),
            row=1, col=1
        )
        if not all(np.isnan(self.metrics['val_loss'])):
            fig.add_trace(
                go.Scatter(x=self.metrics['iteration'], y=self.metrics['val_loss'],
                          name='Val Loss', mode='lines'),
                row=1, col=1
            )
        
        # Accuracy curves
        if not all(np.isnan(self.metrics['train_acc'])):
            fig.add_trace(
                go.Scatter(x=self.metrics['iteration'], y=self.metrics['train_acc'],
                          name='Train Acc', mode='lines'),
                row=1, col=2
            )
        if not all(np.isnan(self.metrics['val_acc'])):
            fig.add_trace(
                go.Scatter(x=self.metrics['iteration'], y=self.metrics['val_acc'],
                          name='Val Acc', mode='lines'),
                row=1, col=2
            )
        
        # Learning rate
        if not all(np.isnan(self.metrics['learning_rate'])):
            fig.add_trace(
                go.Scatter(x=self.metrics['iteration'], y=self.metrics['learning_rate'],
                          name='Learning Rate', mode='lines'),
                row=2, col=1
            )
        
        # GPU memory
        if any(self.metrics['gpu_memory']):
            time_minutes = [(t - self.metrics['timestamp'][0]).total_seconds() / 60 
                           for t in self.metrics['timestamp']]
            fig.add_trace(
                go.Scatter(x=time_minutes, y=self.metrics['gpu_memory'],
                          name='GPU Memory', mode='lines'),
                row=2, col=2
            )
        
        # CPU usage
        if any(self.metrics['cpu_percent']):
            time_minutes = [(t - self.metrics['timestamp'][0]).total_seconds() / 60 
                           for t in self.metrics['timestamp']]
            fig.add_trace(
                go.Scatter(x=time_minutes, y=self.metrics['cpu_percent'],
                          name='CPU Usage', mode='lines'),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='HMS EEG Training Progress Dashboard',
            showlegend=True,
            height=1000,
            template='plotly_white'
        )
        
        # Update axes
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
        fig.update_yaxes(title_text="Learning Rate", type="log", row=2, col=1)
        fig.update_yaxes(title_text="Memory (MB)", row=2, col=2)
        fig.update_yaxes(title_text="Usage (%)", row=3, col=1)
        
        fig.update_xaxes(title_text="Iteration", row=1, col=1)
        fig.update_xaxes(title_text="Iteration", row=1, col=2)
        fig.update_xaxes(title_text="Iteration", row=2, col=1)
        fig.update_xaxes(title_text="Time (minutes)", row=2, col=2)
        fig.update_xaxes(title_text="Time (minutes)", row=3, col=1)
        
        return fig
    
    def export_to_tensorboard(self, log_dir: Optional[str] = None):
        """Export metrics to TensorBoard format."""
        from torch.utils.tensorboard import SummaryWriter
        
        if log_dir is None:
            log_dir = self.log_dir / "tensorboard"
        
        writer = SummaryWriter(log_dir)
        
        for i in range(len(self.metrics['iteration'])):
            iteration = self.metrics['iteration'][i]
            
            # Scalars
            writer.add_scalar('Loss/train', self.metrics['train_loss'][i], iteration)
            if not np.isnan(self.metrics['val_loss'][i]):
                writer.add_scalar('Loss/val', self.metrics['val_loss'][i], iteration)
            
            if not np.isnan(self.metrics['train_acc'][i]):
                writer.add_scalar('Accuracy/train', self.metrics['train_acc'][i], iteration)
            if not np.isnan(self.metrics['val_acc'][i]):
                writer.add_scalar('Accuracy/val', self.metrics['val_acc'][i], iteration)
            
            if not np.isnan(self.metrics['learning_rate'][i]):
                writer.add_scalar('Learning_Rate', self.metrics['learning_rate'][i], iteration)
            
            writer.add_scalar('Resources/GPU_Memory', self.metrics['gpu_memory'][i], iteration)
            writer.add_scalar('Resources/CPU_Percent', self.metrics['cpu_percent'][i], iteration)
        
        writer.close()
        logger.info(f"Exported metrics to TensorBoard at {log_dir}")
    
    def _save_metrics(self):
        """Save metrics to JSON file."""
        # Convert timestamps to strings for JSON serialization
        metrics_copy = self.metrics.copy()
        metrics_copy['timestamp'] = [t.isoformat() for t in self.metrics['timestamp']]
        
        filepath = self.log_dir / "training_metrics.json"
        with open(filepath, 'w') as f:
            json.dump(metrics_copy, f, indent=2)
    
    def stop(self):
        """Stop monitoring and save final results."""
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join()
        
        # Save final plots
        self.update_plots()
        self.save_plots("training_progress_final.png")
        
        # Export to various formats
        self.export_to_tensorboard()
        
        # Save interactive dashboard
        fig = self.create_interactive_dashboard()
        fig.write_html(self.log_dir / "training_dashboard.html")
        
        logger.info("Training progress monitoring stopped and results saved") 