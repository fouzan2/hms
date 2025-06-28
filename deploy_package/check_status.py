#!/usr/bin/env python3
"""Quick status check for HMS training"""

import os
import subprocess
from pathlib import Path

def quick_status():
    print("🔍 HMS Training Quick Status Check")
    print("=" * 50)
    
    # Check GPU
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_name, utilization = result.stdout.strip().split(', ')
            print(f"GPU: {gpu_name} ({utilization}% utilization)")
        else:
            print("GPU: Not available")
    except:
        print("GPU: Error checking status")
    
    # Check processes
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        python_processes = [line for line in result.stdout.split('\n') if 'python' in line and 'train' in line]
        if python_processes:
            print(f"Training Processes: {len(python_processes)} running")
        else:
            print("Training Processes: None running")
    except:
        print("Training Processes: Error checking")
    
    # Check disk space
    data_dir = Path("data")
    models_dir = Path("models")
    
    if data_dir.exists():
        data_size = sum(f.stat().st_size for f in data_dir.rglob('*') if f.is_file()) / (1024**3)
        print(f"Data Size: {data_size:.1f}GB")
    else:
        print("Data Size: No data directory")
        
    if models_dir.exists():
        model_files = list(models_dir.rglob('*.pth'))
        print(f"Trained Models: {len(model_files)} found")
    else:
        print("Trained Models: No models directory")
    
    # Check logs
    logs_dir = Path("logs")
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            print(f"Latest Log: {latest_log.name}")
            
            # Show last few lines
            with open(latest_log, 'r') as f:
                lines = f.readlines()[-5:]
                if lines:
                    print("Recent log entries:")
                    for line in lines:
                        print(f"  {line.strip()}")
        else:
            print("Logs: No log files found")
    else:
        print("Logs: No logs directory")

if __name__ == "__main__":
    quick_status()
