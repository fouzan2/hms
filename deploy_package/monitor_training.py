#!/usr/bin/env python3
"""
HMS Training Monitor for Novita AI
Real-time monitoring of training progress, GPU usage, and system status
"""

import os
import sys
import time
import json
import psutil
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class TrainingMonitor:
    def __init__(self):
        self.log_dir = Path("logs")
        self.model_dir = Path("models")
        self.data_dir = Path("data")
        
    def get_gpu_info(self):
        """Get GPU information"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_util, mem_used, mem_total, temp = result.stdout.strip().split(', ')
                return {
                    'utilization': int(gpu_util),
                    'memory_used': int(mem_used),
                    'memory_total': int(mem_total), 
                    'memory_percent': round(int(mem_used) / int(mem_total) * 100, 1),
                    'temperature': int(temp)
                }
        except Exception as e:
            print(f"Error getting GPU info: {e}")
        return None
        
    def get_system_info(self):
        """Get system information"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': round(memory.used / (1024**3), 1),
            'memory_total_gb': round(memory.total / (1024**3), 1),
            'disk_used_gb': round(disk.used / (1024**3), 1),
            'disk_total_gb': round(disk.total / (1024**3), 1),
            'disk_percent': round(disk.used / disk.total * 100, 1)
        }
        
    def get_training_status(self):
        """Get training status from logs"""
        status = {
            'stage': 'Unknown',
            'progress': 0,
            'current_epoch': 0,
            'total_epochs': 0,
            'best_accuracy': 0.0,
            'current_loss': 0.0,
            'eta': 'Unknown'
        }
        
        # Check for log files
        log_files = list(self.log_dir.glob("*.log"))
        if not log_files:
            return status
            
        # Parse latest log file
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()[-50:]  # Last 50 lines
                
            for line in reversed(lines):
                if 'Epoch' in line and '/' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'Epoch' in part and i+1 < len(parts):
                            epoch_info = parts[i+1]
                            if '/' in epoch_info:
                                current, total = epoch_info.split('/')
                                status['current_epoch'] = int(current)
                                status['total_epochs'] = int(total)
                                status['progress'] = round(int(current) / int(total) * 100, 1)
                                break
                                
                if 'Accuracy:' in line:
                    acc_part = line.split('Accuracy:')[1].split()[0]
                    status['best_accuracy'] = float(acc_part.strip('%'))
                    
                if 'Loss:' in line:
                    loss_part = line.split('Loss:')[1].split()[0]
                    status['current_loss'] = float(loss_part)
                    
        except Exception as e:
            print(f"Error parsing logs: {e}")
            
        return status
        
    def check_data_status(self):
        """Check data preparation status"""
        status = {
            'raw_data_downloaded': False,
            'preprocessing_complete': False,
            'total_samples': 0,
            'processed_samples': 0
        }
        
        # Check raw data
        if (self.data_dir / "raw").exists():
            raw_files = list((self.data_dir / "raw").glob("*.csv"))
            if raw_files:
                status['raw_data_downloaded'] = True
                
        # Check processed data
        if (self.data_dir / "processed").exists():
            processed_files = list((self.data_dir / "processed").glob("*.npz"))
            if processed_files:
                status['preprocessing_complete'] = True
                status['processed_samples'] = len(processed_files)
                
        return status
        
    def check_model_status(self):
        """Check model training status"""
        status = {
            'models_trained': [],
            'best_model': None,
            'onnx_exported': False
        }
        
        # Check for trained models
        if (self.model_dir / "final").exists():
            model_files = list((self.model_dir / "final").glob("*.pth"))
            status['models_trained'] = [f.stem for f in model_files]
            
        # Check for ONNX export
        if (self.model_dir / "onnx").exists():
            onnx_files = list((self.model_dir / "onnx").glob("*.onnx"))
            if onnx_files:
                status['onnx_exported'] = True
                
        return status
        
    def display_status(self):
        """Display comprehensive status"""
        os.system('clear')
        print("=" * 80)
        print("🧠 HMS EEG Classification - Novita AI Training Monitor")
        print("=" * 80)
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # GPU Status
        gpu_info = self.get_gpu_info()
        if gpu_info:
            print("🔥 GPU Status (H100):")
            print(f"   Utilization: {gpu_info['utilization']}%")
            print(f"   Memory: {gpu_info['memory_used']}MB / {gpu_info['memory_total']}MB ({gpu_info['memory_percent']}%)")
            print(f"   Temperature: {gpu_info['temperature']}°C")
        else:
            print("❌ GPU information not available")
        print()
        
        # System Status
        sys_info = self.get_system_info()
        print("💻 System Status:")
        print(f"   CPU: {sys_info['cpu_percent']}%")
        print(f"   RAM: {sys_info['memory_used_gb']}GB / {sys_info['memory_total_gb']}GB ({sys_info['memory_percent']}%)")
        print(f"   Disk: {sys_info['disk_used_gb']}GB / {sys_info['disk_total_gb']}GB ({sys_info['disk_percent']}%)")
        print()
        
        # Training Status
        training_status = self.get_training_status()
        print("🎯 Training Status:")
        print(f"   Stage: {training_status['stage']}")
        print(f"   Epoch: {training_status['current_epoch']}/{training_status['total_epochs']} ({training_status['progress']}%)")
        print(f"   Best Accuracy: {training_status['best_accuracy']:.2f}%")
        print(f"   Current Loss: {training_status['current_loss']:.4f}")
        print()
        
        # Data Status
        data_status = self.check_data_status()
        print("📊 Data Status:")
        print(f"   Raw Data Downloaded: {'✅' if data_status['raw_data_downloaded'] else '❌'}")
        print(f"   Preprocessing Complete: {'✅' if data_status['preprocessing_complete'] else '❌'}")
        print(f"   Processed Samples: {data_status['processed_samples']}")
        print()
        
        # Model Status
        model_status = self.check_model_status()
        print("🤖 Model Status:")
        print(f"   Trained Models: {', '.join(model_status['models_trained']) if model_status['models_trained'] else 'None'}")
        print(f"   ONNX Exported: {'✅' if model_status['onnx_exported'] else '❌'}")
        print()
        
        print("=" * 80)
        print("Press Ctrl+C to exit")
        
    def run_monitor(self):
        """Run continuous monitoring"""
        try:
            while True:
                self.display_status()
                time.sleep(10)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

if __name__ == "__main__":
    monitor = TrainingMonitor()
    monitor.run_monitor()
