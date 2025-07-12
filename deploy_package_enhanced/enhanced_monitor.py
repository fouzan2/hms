#!/usr/bin/env python3
"""
Enhanced HMS Training Monitor for Novita AI
Real-time monitoring with resume capability and advanced features
"""

import os
import sys
import time
import json
import pickle
import psutil
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

class EnhancedTrainingMonitor:
    def __init__(self):
        self.workspace = Path("/workspace")
        self.log_dir = self.workspace / "logs"
        self.model_dir = self.workspace / "models"
        self.data_dir = self.workspace / "data"
        self.state_dir = self.workspace / "training_state"
        
    def get_gpu_info(self) -> Optional[Dict]:
        """Get comprehensive GPU information"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                return {
                    'name': values[0],
                    'utilization': int(values[1]),
                    'memory_used': int(values[2]),
                    'memory_total': int(values[3]), 
                    'memory_percent': round(int(values[2]) / int(values[3]) * 100, 1),
                    'temperature': int(values[4]),
                    'power_draw': float(values[5]),
                    'power_limit': float(values[6]),
                    'power_percent': round(float(values[5]) / float(values[6]) * 100, 1)
                }
        except Exception as e:
            print(f"Error getting GPU info: {e}")
        return None
        
    def get_training_state(self) -> Dict[str, Any]:
        """Get training state from pickle file"""
        state_file = self.state_dir / "training_state.pkl"
        
        if state_file.exists():
            try:
                with open(state_file, 'rb') as f:
                    state = pickle.load(f)
                return {
                    'current_stage': state.current_stage,
                    'stages_completed': state.stages_completed,
                    'models_trained': state.models_trained,
                    'best_accuracy': state.best_accuracy,
                    'data_downloaded': state.data_downloaded,
                    'preprocessing_completed': state.preprocessing_completed,
                    'foundation_pretrained': state.foundation_pretrained,
                    'ensemble_trained': state.ensemble_trained,
                    'export_completed': state.export_completed
                }
            except Exception as e:
                print(f"Error reading training state: {e}")
                
        return {
            'current_stage': 'Unknown',
            'stages_completed': [],
            'models_trained': {},
            'best_accuracy': 0.0,
            'data_downloaded': False,
            'preprocessing_completed': False,
            'foundation_pretrained': False,
            'ensemble_trained': False,
            'export_completed': False
        }
        
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get checkpoint information"""
        checkpoint_dir = self.model_dir / "checkpoints"
        
        info = {
            'available_checkpoints': [],
            'latest_checkpoint': None,
            'checkpoint_sizes': {}
        }
        
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pth"))
            
            for checkpoint in checkpoints:
                size_mb = checkpoint.stat().st_size / (1024 * 1024)
                info['checkpoint_sizes'][checkpoint.name] = f"{size_mb:.1f}MB"
                info['available_checkpoints'].append(checkpoint.name)
                
            if checkpoints:
                latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
                info['latest_checkpoint'] = latest.name
                
        return info
        
    def estimate_time_and_cost(self, gpu_info: Dict) -> Dict[str, Any]:
        """Estimate remaining time and cost"""
        # H100 cost per hour (approximate)
        h100_cost_per_hour = 3.35
        
        training_state = self.get_training_state()
        
        # Estimate based on current stage and progress
        stage_estimates = {
            'data_download': 1.0,  # hours
            'preprocessing': 2.0,
            'foundation_pretraining': 3.0,
            'model_training': 8.0,
            'ensemble_training': 2.0,
            'model_export': 0.5
        }
        
        completed_stages = training_state['stages_completed']
        current_stage = training_state['current_stage']
        
        # Calculate time spent and remaining
        time_spent = sum(stage_estimates[stage] for stage in completed_stages)
        
        if current_stage in stage_estimates:
            remaining_time = stage_estimates[current_stage]
            for stage, estimate in stage_estimates.items():
                if stage not in completed_stages and stage != current_stage:
                    remaining_time += estimate
        else:
            remaining_time = 0
            
        cost_spent = time_spent * h100_cost_per_hour
        estimated_total_cost = (time_spent + remaining_time) * h100_cost_per_hour
        
        return {
            'time_spent_hours': time_spent,
            'remaining_hours': remaining_time,
            'cost_spent': cost_spent,
            'estimated_total_cost': estimated_total_cost,
            'efficiency': gpu_info['utilization'] if gpu_info else 0
        }
        
    def display_enhanced_status(self):
        """Display comprehensive enhanced status"""
        os.system('clear')
        print("=" * 100)
        print("🧠 Enhanced HMS EEG Classification - Novita AI Training Monitor")
        print("=" * 100)
        print(f"🕐 Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # GPU Status
        gpu_info = self.get_gpu_info()
        if gpu_info:
            print("🔥 GPU Status (H100 80GB):")
            print(f"   Name: {gpu_info['name']}")
            print(f"   Utilization: {gpu_info['utilization']}%")
            print(f"   Memory: {gpu_info['memory_used']}MB / {gpu_info['memory_total']}MB ({gpu_info['memory_percent']}%)")
            print(f"   Temperature: {gpu_info['temperature']}°C")
            print(f"   Power: {gpu_info['power_draw']:.1f}W / {gpu_info['power_limit']:.1f}W ({gpu_info['power_percent']}%)")
        else:
            print("❌ GPU information not available")
        print()
        
        # System Status
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/workspace')
        
        print("💻 System Status:")
        print(f"   CPU: {cpu_percent}% ({psutil.cpu_count()} cores)")
        print(f"   RAM: {memory.used/(1024**3):.1f}GB / {memory.total/(1024**3):.1f}GB ({memory.percent}%)")
        print(f"   Disk: {disk.used/(1024**3):.1f}GB / {disk.total/(1024**3):.1f}GB ({disk.used/disk.total*100:.1f}%)")
        print()
        
        # Enhanced Training Status
        training_state = self.get_training_state()
        print("🎯 Enhanced Training Status:")
        print(f"   Current Stage: {training_state['current_stage']}")
        print(f"   Completed Stages: {', '.join(training_state['stages_completed']) if training_state['stages_completed'] else 'None'}")
        print(f"   Best Accuracy: {training_state['best_accuracy']:.2f}%")
        print(f"   Target (90%): {'✅ ACHIEVED' if training_state['best_accuracy'] >= 90 else '⏳ IN PROGRESS'}")
        print()
        
        # Models Status
        print("🤖 Models Status:")
        if training_state['models_trained']:
            for model_name, trained in training_state['models_trained'].items():
                status = "✅ Trained" if trained else "⏳ Pending"
                print(f"   {model_name}: {status}")
        else:
            print("   No models trained yet")
        print()
        
        # Advanced Features Status
        print("🚀 Advanced Features:")
        print(f"   Data Downloaded: {'✅' if training_state['data_downloaded'] else '❌'}")
        print(f"   Preprocessing: {'✅' if training_state['preprocessing_completed'] else '❌'}")
        print(f"   Foundation Model: {'✅' if training_state['foundation_pretrained'] else '❌'}")
        print(f"   Ensemble Training: {'✅' if training_state['ensemble_trained'] else '❌'}")
        print(f"   Model Export: {'✅' if training_state['export_completed'] else '❌'}")
        print()
        
        # Checkpoint Information
        checkpoint_info = self.get_checkpoint_info()
        print("💾 Checkpoint Status:")
        print(f"   Available Checkpoints: {len(checkpoint_info['available_checkpoints'])}")
        if checkpoint_info['latest_checkpoint']:
            print(f"   Latest: {checkpoint_info['latest_checkpoint']}")
        print(f"   Resume Capability: {'✅ READY' if checkpoint_info['available_checkpoints'] else '❌ NO CHECKPOINTS'}")
        print()
        
        # Time and Cost Estimation
        estimates = self.estimate_time_and_cost(gpu_info)
        print("💰 Time & Cost Estimation:")
        print(f"   Time Spent: {estimates['time_spent_hours']:.1f} hours")
        print(f"   Remaining: {estimates['remaining_hours']:.1f} hours")
        print(f"   Cost Spent: ${estimates['cost_spent']:.2f}")
        print(f"   Est. Total: ${estimates['estimated_total_cost']:.2f}")
        print(f"   Efficiency: {estimates['efficiency']}% GPU utilization")
        print()
        
        print("=" * 100)
        print("🔧 Commands: hms-status | hms-resume | backup-state | ./restart_training.sh")
        print("Press Ctrl+C to exit monitoring")
        
    def run_enhanced_monitor(self):
        """Run continuous enhanced monitoring"""
        try:
            while True:
                self.display_enhanced_status()
                time.sleep(15)  # Update every 15 seconds
        except KeyboardInterrupt:
            print("\n\n👋 Enhanced monitoring stopped.")
            print("💡 Use 'hms-monitor' to restart monitoring")

if __name__ == "__main__":
    monitor = EnhancedTrainingMonitor()
    monitor.run_enhanced_monitor()
