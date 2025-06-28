#!/usr/bin/env python3
"""
HMS EEG Classification - Novita AI Deployment Script
Complete deployment to Novita AI with H100 GPU for >90% accuracy
"""

import os
import sys
import subprocess
import time
import json
import argparse
from pathlib import Path
import requests
import yaml
from typing import Dict, List, Optional

# ANSI colors for output
class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[0;33m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[0;37m'
    NC = '\033[0m'  # No Color

class NovitaDeployer:
    """Novita AI deployment manager for HMS EEG Classification"""
    
    def __init__(self, ssh_key_path: str = None, instance_id: str = None):
        self.project_root = Path(__file__).parent
        self.ssh_key_path = ssh_key_path
        self.instance_id = instance_id
        self.ssh_host = None
        self.ssh_user = "root"
        
    def log(self, message: str, color: str = Colors.NC):
        """Print colored log message"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"{color}[{timestamp}] {message}{Colors.NC}")
        
    def create_deployment_package(self):
        """Create optimized deployment package"""
        self.log("Creating deployment package for Novita AI...", Colors.BLUE)
        
        # Create deployment directory
        deploy_dir = Path("deploy_package")
        deploy_dir.mkdir(exist_ok=True)
        
        # Essential files to include
        essential_files = [
            "src/",
            "config/",
            "requirements.txt",
            "setup.py",
            "run_project.py",
            "prepare_data.py",
            "scripts/",
            "Makefile",
            ".env.example"
        ]
        
        # Copy essential files
        for file_path in essential_files:
            src_path = self.project_root / file_path
            dst_path = deploy_dir / file_path
            
            if src_path.exists():
                if src_path.is_dir():
                    subprocess.run(['cp', '-r', str(src_path), str(dst_path.parent)], check=True)
                else:
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(['cp', str(src_path), str(dst_path)], check=True)
        
        # Create optimized setup script
        self.create_novita_setup_script(deploy_dir)
        
        # Create monitoring scripts
        self.create_monitoring_scripts(deploy_dir)
        
        # Create deployment config
        self.create_deployment_config(deploy_dir)
        
        # Create archive
        archive_name = "hms_novita_deployment.tar.gz"
        subprocess.run(['tar', '-czf', archive_name, '-C', str(deploy_dir), '.'], check=True)
        
        self.log(f"✅ Deployment package created: {archive_name}", Colors.GREEN)
        return archive_name
        
    def create_novita_setup_script(self, deploy_dir: Path):
        """Create optimized setup script for Novita AI"""
        setup_script = deploy_dir / "novita_setup.sh"
        
        script_content = '''#!/bin/bash
set -e

# HMS EEG Classification - Novita AI Setup Script
# Optimized for H100 GPU with full dataset training

echo "🚀 Starting HMS EEG Classification setup on Novita AI..."

# Update system
apt-get update -qq
apt-get install -y git wget curl unzip htop nvtop screen tmux

# Set environment variables
export DEBIAN_FRONTEND=noninteractive
export PYTHONPATH=/workspace
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create workspace structure
mkdir -p /workspace/{data/{raw,processed,models},logs,models/{final,onnx,checkpoints},monitoring}
cd /workspace

# Install Miniconda for better package management
if [ ! -f "/opt/miniconda3/bin/conda" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p /opt/miniconda3
    rm miniconda.sh
    export PATH="/opt/miniconda3/bin:$PATH"
    echo 'export PATH="/opt/miniconda3/bin:$PATH"' >> ~/.bashrc
fi

source /opt/miniconda3/etc/profile.d/conda.sh

# Create optimized conda environment
conda create -n hms python=3.10 -y
conda activate hms

# Install PyTorch with CUDA 11.8 (optimized for H100)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install ML packages via conda (faster)
conda install -c conda-forge numpy pandas scikit-learn matplotlib seaborn plotly -y
conda install -c conda-forge scipy pyyaml tqdm joblib -y

# Install specialized packages via pip
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir \
    kaggle \
    wandb \
    mlflow \
    optuna \
    mne \
    pyedflib \
    antropy \
    efficientnet-pytorch \
    timm \
    transformers \
    accelerate \
    flash-attn \
    xgboost \
    lightgbm \
    shap \
    lime \
    onnx \
    onnxruntime-gpu \
    fastapi \
    uvicorn \
    dash \
    dash-bootstrap-components

# Create optimized tmux session
cat > ~/.tmux.conf << 'EOF'
set -g default-terminal "screen-256color"
set -g history-limit 10000
set -g mouse on
set -g status-bg colour235
set -g status-fg colour246
EOF

# Create monitoring aliases
cat >> ~/.bashrc << 'EOF'
alias gpu='watch -n 1 nvidia-smi'
alias hms-monitor='cd /workspace && python monitor_training.py'
alias hms-logs='tail -f logs/training.log'
alias hms-status='cd /workspace && python check_status.py'
EOF

echo "✅ Novita AI setup complete!"
echo "Next steps:"
echo "1. Upload your HMS code to /workspace"
echo "2. Set up Kaggle credentials"
echo "3. Run the training pipeline"
'''
        
        setup_script.write_text(script_content)
        setup_script.chmod(0o755)
        
    def create_monitoring_scripts(self, deploy_dir: Path):
        """Create monitoring and status scripts"""
        
        # Training monitor script
        monitor_script = deploy_dir / "monitor_training.py"
        monitor_content = '''#!/usr/bin/env python3
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
            print("\\nMonitoring stopped.")

if __name__ == "__main__":
    monitor = TrainingMonitor()
    monitor.run_monitor()
'''
        
        monitor_script.write_text(monitor_content)
        monitor_script.chmod(0o755)
        
        # Status check script
        status_script = deploy_dir / "check_status.py"
        status_content = '''#!/usr/bin/env python3
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
        python_processes = [line for line in result.stdout.split('\\n') if 'python' in line and 'train' in line]
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
'''
        
        status_script.write_text(status_content)
        status_script.chmod(0o755)
        
    def create_deployment_config(self, deploy_dir: Path):
        """Create deployment configuration"""
        config = {
            "deployment_info": {
                "target": "Novita AI",
                "gpu": "H100 80GB",
                "optimization_level": "production",
                "expected_accuracy": ">90%",
                "full_dataset": True
            },
            "environment": {
                "python_version": "3.10",
                "cuda_version": "11.8",
                "pytorch_version": "2.1+",
                "memory_gb": 80,
                "storage_gb": 500
            },
            "training_config": {
                "config_file": "config/novita_production_config.yaml",
                "batch_size_resnet": 64,
                "batch_size_efficientnet": 32,
                "epochs_resnet": 150,
                "epochs_efficientnet": 120,
                "use_mixed_precision": True,
                "use_gradient_checkpointing": True
            },
            "monitoring": {
                "check_interval_seconds": 30,
                "log_level": "INFO",
                "save_checkpoints": True,
                "early_stopping": True
            }
        }
        
        config_file = deploy_dir / "deployment_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
    def create_ssh_config(self):
        """Create SSH configuration for easier access"""
        ssh_config = f"""
# HMS Novita AI Instance
Host hms-novita
    HostName {self.ssh_host}
    User {self.ssh_user}
    IdentityFile {self.ssh_key_path}
    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive yes
    Compression yes
"""
        
        ssh_config_path = Path.home() / ".ssh" / "config"
        
        # Backup existing config
        if ssh_config_path.exists():
            backup_path = ssh_config_path.with_suffix(".config.backup")
            ssh_config_path.copy(backup_path)
            self.log(f"SSH config backed up to {backup_path}", Colors.YELLOW)
        
        # Append new config
        with open(ssh_config_path, 'a') as f:
            f.write(ssh_config)
            
        self.log("SSH config updated", Colors.GREEN)
        
    def upload_and_deploy(self, archive_path: str):
        """Upload code and deploy to Novita AI"""
        if not self.ssh_host:
            self.log("SSH host not configured. Please set up instance first.", Colors.RED)
            return False
            
        self.log(f"Uploading {archive_path} to Novita AI...", Colors.BLUE)
        
        # Upload archive
        scp_cmd = [
            'scp', '-i', self.ssh_key_path, '-o', 'StrictHostKeyChecking=no',
            archive_path, f'{self.ssh_user}@{self.ssh_host}:/tmp/'
        ]
        
        try:
            subprocess.run(scp_cmd, check=True)
            self.log("Archive uploaded successfully", Colors.GREEN)
        except subprocess.CalledProcessError as e:
            self.log(f"Upload failed: {e}", Colors.RED)
            return False
            
        # Extract and setup
        ssh_cmd = f"""
            cd /workspace &&
            tar -xzf /tmp/{Path(archive_path).name} &&
            chmod +x novita_setup.sh &&
            ./novita_setup.sh
        """
        
        ssh_full_cmd = [
            'ssh', '-i', self.ssh_key_path, '-o', 'StrictHostKeyChecking=no',
            f'{self.ssh_user}@{self.ssh_host}', ssh_cmd
        ]
        
        try:
            subprocess.run(ssh_full_cmd, check=True)
            self.log("Deployment setup completed", Colors.GREEN)
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"Deployment setup failed: {e}", Colors.RED)
            return False
            
    def setup_environment(self):
        """Setup environment variables and credentials"""
        self.log("Setting up environment on Novita AI...", Colors.BLUE)
        
        # Environment setup commands
        env_setup = '''
            # Set up Kaggle credentials
            mkdir -p ~/.kaggle
            echo "Please upload your kaggle.json to ~/.kaggle/"
            
            # Set up conda environment
            source /opt/miniconda3/etc/profile.d/conda.sh
            conda activate hms
            
            # Set environment variables
            export PYTHONPATH=/workspace
            export CUDA_VISIBLE_DEVICES=0
            echo 'export PYTHONPATH=/workspace' >> ~/.bashrc
            echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc
            echo 'conda activate hms' >> ~/.bashrc
        '''
        
        ssh_cmd = [
            'ssh', '-i', self.ssh_key_path, '-o', 'StrictHostKeyChecking=no',
            f'{self.ssh_user}@{self.ssh_host}', env_setup
        ]
        
        try:
            subprocess.run(ssh_cmd, check=True)
            self.log("Environment setup completed", Colors.GREEN)
        except subprocess.CalledProcessError as e:
            self.log(f"Environment setup failed: {e}", Colors.YELLOW)
            
    def start_training(self):
        """Start the training pipeline"""
        self.log("Starting HMS training pipeline...", Colors.BLUE)
        
        training_cmd = '''
            cd /workspace &&
            source /opt/miniconda3/etc/profile.d/conda.sh &&
            conda activate hms &&
            screen -dmS hms-training python run_project.py --config config/novita_production_config.yaml --full-dataset
        '''
        
        ssh_cmd = [
            'ssh', '-i', self.ssh_key_path, '-o', 'StrictHostKeyChecking=no',
            f'{self.ssh_user}@{self.ssh_host}', training_cmd
        ]
        
        try:
            subprocess.run(ssh_cmd, check=True)
            self.log("Training started in screen session 'hms-training'", Colors.GREEN)
            self.log("Connect via SSH and run: screen -r hms-training", Colors.CYAN)
        except subprocess.CalledProcessError as e:
            self.log(f"Training start failed: {e}", Colors.RED)
            
    def connect_ssh(self):
        """Connect to instance via SSH"""
        if not self.ssh_host:
            self.log("SSH host not configured", Colors.RED)
            return
            
        self.log(f"Connecting to {self.ssh_host}...", Colors.BLUE)
        
        ssh_cmd = [
            'ssh', '-i', self.ssh_key_path, '-o', 'StrictHostKeyChecking=no',
            f'{self.ssh_user}@{self.ssh_host}'
        ]
        
        os.execvp('ssh', ssh_cmd)
        
    def show_instructions(self):
        """Show deployment instructions"""
        instructions = f"""
{Colors.CYAN}🚀 HMS Novita AI Deployment Instructions{Colors.NC}
{'='*60}

{Colors.YELLOW}1. Launch Novita AI Instance:{Colors.NC}
   - Go to https://novita.ai/
   - Launch H100 80GB instance
   - Choose Ubuntu 22.04
   - Storage: 500GB+
   - Copy SSH connection details

{Colors.YELLOW}2. Configure SSH:{Colors.NC}
   - Save your SSH private key
   - Update this script with host details:
     python deploy_novita.py --ssh-host YOUR_HOST --ssh-key YOUR_KEY_PATH

{Colors.YELLOW}3. Upload Kaggle Credentials:{Colors.NC}
   - Upload kaggle.json to ~/.kaggle/ on the instance
   - Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables

{Colors.YELLOW}4. Deploy and Train:{Colors.NC}
   - Run: python deploy_novita.py --deploy
   - Monitor: python deploy_novita.py --monitor
   - SSH: python deploy_novita.py --ssh

{Colors.YELLOW}5. Expected Results:{Colors.NC}
   - Training time: 8-12 hours
   - Expected accuracy: >90%
   - Cost: ~$25-40 total
   - Output: Production-ready ONNX model

{Colors.GREEN}Commands:{Colors.NC}
   --create-package    Create deployment package
   --deploy           Full deployment to Novita AI
   --ssh              Connect via SSH
   --monitor          Monitor training progress
   --start-training   Start training pipeline
   --instructions     Show these instructions

{Colors.PURPLE}Monitoring URLs (via SSH tunnel):{Colors.NC}
   - MLflow: http://localhost:5000
   - TensorBoard: http://localhost:6006
   - Training Monitor: python monitor_training.py
        """
        
        print(instructions)

def main():
    parser = argparse.ArgumentParser(description='HMS Novita AI Deployment Manager')
    parser.add_argument('--ssh-host', help='SSH host address')
    parser.add_argument('--ssh-key', help='SSH private key path')
    parser.add_argument('--instance-id', help='Instance ID for reference')
    
    # Actions
    parser.add_argument('--create-package', action='store_true', help='Create deployment package')
    parser.add_argument('--deploy', action='store_true', help='Deploy to Novita AI')
    parser.add_argument('--ssh', action='store_true', help='Connect via SSH')
    parser.add_argument('--monitor', action='store_true', help='Monitor training')
    parser.add_argument('--start-training', action='store_true', help='Start training')
    parser.add_argument('--instructions', action='store_true', help='Show instructions')
    
    args = parser.parse_args()
    
    deployer = NovitaDeployer(args.ssh_key, args.instance_id)
    
    if args.ssh_host:
        deployer.ssh_host = args.ssh_host
        
    if args.instructions or not any(vars(args).values()):
        deployer.show_instructions()
        return
        
    if args.create_package:
        deployer.create_deployment_package()
        
    if args.deploy:
        # Full deployment process
        archive = deployer.create_deployment_package()
        if deployer.upload_and_deploy(archive):
            deployer.setup_environment()
            deployer.log("Deployment completed! Use --start-training to begin.", Colors.GREEN)
            
    if args.ssh:
        deployer.connect_ssh()
        
    if args.start_training:
        deployer.start_training()
        
    if args.monitor:
        deployer.log("Monitoring feature requires SSH access. Connect first.", Colors.YELLOW)

if __name__ == '__main__':
    main() 