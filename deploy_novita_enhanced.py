#!/usr/bin/env python3
"""
Enhanced HMS EEG Classification - Novita AI Deployment Script
Complete deployment with advanced features and robust resume functionality
Supports: EEG Foundation Model, Ensemble Training, Distributed Training, Resume Capability
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
import pickle

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

class EnhancedNovitaDeployer:
    """Enhanced Novita AI deployment manager with resume capability"""
    
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
        
    def create_enhanced_deployment_package(self):
        """Create optimized deployment package with all advanced features"""
        self.log("Creating enhanced deployment package for Novita AI...", Colors.BLUE)
        
        # Create deployment directory
        deploy_dir = Path("deploy_package_enhanced")
        deploy_dir.mkdir(exist_ok=True)
        
        # Essential files to include
        essential_files = [
            "src/",
            "config/",
            "requirements.txt",
            "setup.py",
            "run_novita_training_enhanced.py",
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
        
        # Create enhanced setup script
        self.create_enhanced_setup_script(deploy_dir)
        
        # Create enhanced monitoring scripts
        self.create_enhanced_monitoring_scripts(deploy_dir)
        
        # Create resume management scripts
        self.create_resume_management_scripts(deploy_dir)
        
        # Create enhanced deployment config
        self.create_enhanced_deployment_config(deploy_dir)
        
        # Create archive
        archive_name = "hms_enhanced_novita_deployment.tar.gz"
        subprocess.run(['tar', '-czf', archive_name, '-C', str(deploy_dir), '.'], check=True)
        
        self.log(f"✅ Enhanced deployment package created: {archive_name}", Colors.GREEN)
        return archive_name
        
    def create_enhanced_setup_script(self, deploy_dir: Path):
        """Create enhanced setup script for Novita AI with all features"""
        setup_script = deploy_dir / "novita_enhanced_setup.sh"
        
        script_content = '''#!/bin/bash
set -e

# Enhanced HMS EEG Classification - Novita AI Setup Script
# Optimized for H100 GPU with all advanced features

echo "🚀 Starting Enhanced HMS EEG Classification setup on Novita AI..."

# Update system and install dependencies
apt-get update -qq
apt-get install -y git wget curl unzip htop nvtop screen tmux tree \
    build-essential cmake ninja-build libfftw3-dev \
    libopenblas-dev liblapack-dev libhdf5-dev

# Set environment variables for H100 optimization
export DEBIAN_FRONTEND=noninteractive
export PYTHONPATH=/workspace
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Create enhanced workspace structure
mkdir -p /workspace/{data/{raw,processed,models,cache},logs,models/{final,onnx,checkpoints},monitoring,training_state,backups}
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

# Create optimized conda environment with all dependencies
conda create -n hms python=3.10 -y
conda activate hms

# Install PyTorch with CUDA 11.8 (optimized for H100)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install ML packages via conda (faster)
conda install -c conda-forge numpy pandas scikit-learn matplotlib seaborn plotly -y
conda install -c conda-forge scipy pyyaml tqdm joblib -y
conda install -c conda-forge h5py hdf5 pytables -y

# Install specialized packages via pip
pip install --no-cache-dir --upgrade pip setuptools wheel

# Core ML packages
pip install --no-cache-dir \
    kaggle wandb mlflow optuna \
    mne pyedflib antropy \
    efficientnet-pytorch timm transformers accelerate \
    xgboost lightgbm catboost \
    shap lime captum \
    onnx onnxruntime-gpu \
    fastapi uvicorn dash dash-bootstrap-components

# Advanced packages for new features
pip install --no-cache-dir \
    einops flash-attn \
    ray[tune] ray[train] \
    tensorboard tensorboardX \
    pytorch-lightning lightning \
    hydra-core omegaconf \
    rich typer click

# Development and monitoring tools
pip install --no-cache-dir \
    jupyter ipywidgets \
    psutil GPUtil py3nvml \
    prometheus-client grafana-api

# Create optimized tmux configuration
cat > ~/.tmux.conf << 'EOF'
set -g default-terminal "screen-256color"
set -g history-limit 50000
set -g mouse on
set -g status-bg colour235
set -g status-fg colour246
set -g status-interval 1
set -g status-left '#[fg=green]#H#[default] '
set -g status-right '#[fg=cyan]#(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)%% GPU #[fg=yellow]#(free -h | grep Mem | awk "{print $3}")#[default] %H:%M'
bind-key r source-file ~/.tmux.conf \; display-message "Config reloaded!"
EOF

# Create enhanced monitoring aliases
cat >> ~/.bashrc << 'EOF'
# HMS Enhanced Aliases
alias gpu='watch -n 1 nvidia-smi'
alias hms-monitor='cd /workspace && python enhanced_monitor.py'
alias hms-logs='tail -f logs/novita_enhanced_training.log'
alias hms-status='cd /workspace && python run_novita_training_enhanced.py --status'
alias hms-resume='cd /workspace && python run_novita_training_enhanced.py --resume'
alias hms-stage='cd /workspace && python run_novita_training_enhanced.py --stage'
alias backup-state='cd /workspace && cp -r training_state/ models/checkpoints/ backups/backup_$(date +%Y%m%d_%H%M%S)/'
alias restore-state='cd /workspace && ls -la backups/ | tail -10'

# Environment setup
export PYTHONPATH=/workspace
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
conda activate hms
EOF

# Create auto-backup cron job for training state
echo "*/30 * * * * cd /workspace && cp -r training_state/ models/checkpoints/ backups/auto_backup_$(date +\%Y\%m\%d_\%H\%M\%S)/ 2>/dev/null" | crontab -

# Create restart script for resume capability
cat > /workspace/restart_training.sh << 'EOF'
#!/bin/bash
cd /workspace
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate hms

echo "🔄 Restarting HMS Enhanced Training..."
echo "📊 Checking current status..."
python run_novita_training_enhanced.py --status

echo "🚀 Resuming training..."
screen -dmS hms-enhanced python run_novita_training_enhanced.py --resume

echo "✅ Training resumed in screen session 'hms-enhanced'"
echo "📺 Attach with: screen -r hms-enhanced"
EOF
chmod +x /workspace/restart_training.sh

echo "✅ Enhanced Novita AI setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Upload your HMS enhanced code to /workspace"
echo "2. Set up Kaggle credentials"
echo "3. Run the enhanced training pipeline"
echo "4. Use ./restart_training.sh if training stops"
echo ""
echo "🔧 Useful commands:"
echo "  hms-status    - Check training status"
echo "  hms-monitor   - Real-time monitoring"
echo "  hms-resume    - Resume training"
echo "  backup-state  - Manual backup"
'''
        
        setup_script.write_text(script_content)
        setup_script.chmod(0o755)
        
    def create_enhanced_monitoring_scripts(self, deploy_dir: Path):
        """Create enhanced monitoring scripts with resume support"""
        
        # Enhanced training monitor
        monitor_script = deploy_dir / "enhanced_monitor.py"
        monitor_content = '''#!/usr/bin/env python3
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
            print("\\n\\n👋 Enhanced monitoring stopped.")
            print("💡 Use 'hms-monitor' to restart monitoring")

if __name__ == "__main__":
    monitor = EnhancedTrainingMonitor()
    monitor.run_enhanced_monitor()
'''
        
        monitor_script.write_text(monitor_content)
        monitor_script.chmod(0o755)
        
    def create_resume_management_scripts(self, deploy_dir: Path):
        """Create scripts for managing training resume"""
        
        # Resume manager script
        resume_script = deploy_dir / "resume_manager.py"
        resume_content = '''#!/usr/bin/env python3
"""
HMS Training Resume Manager
Manages training state and resume functionality
"""

import os
import sys
import json
import pickle
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

class ResumeManager:
    def __init__(self):
        self.workspace = Path("/workspace")
        self.state_dir = self.workspace / "training_state"
        self.backup_dir = self.workspace / "backups"
        self.checkpoint_dir = self.workspace / "models" / "checkpoints"
        
        # Ensure directories exist
        for dir_path in [self.state_dir, self.backup_dir, self.checkpoint_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self, name: str = None) -> str:
        """Create a backup of current training state"""
        if name is None:
            name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / name
        backup_path.mkdir(exist_ok=True)
        
        # Backup training state
        if self.state_dir.exists():
            shutil.copytree(self.state_dir, backup_path / "training_state", dirs_exist_ok=True)
        
        # Backup checkpoints
        if self.checkpoint_dir.exists():
            shutil.copytree(self.checkpoint_dir, backup_path / "checkpoints", dirs_exist_ok=True)
        
        # Create backup info
        info = {
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "files_backed_up": len(list(backup_path.rglob("*"))),
            "size_mb": sum(f.stat().st_size for f in backup_path.rglob("*") if f.is_file()) / (1024*1024)
        }
        
        with open(backup_path / "backup_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✅ Backup created: {backup_path}")
        print(f"📊 Files: {info['files_backed_up']}, Size: {info['size_mb']:.1f}MB")
        
        return str(backup_path)
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups"""
        backups = []
        
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir():
                info_file = backup_dir / "backup_info.json"
                if info_file.exists():
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    backups.append(info)
                else:
                    # Create basic info for old backups
                    backups.append({
                        "name": backup_dir.name,
                        "timestamp": datetime.fromtimestamp(backup_dir.stat().st_mtime).isoformat(),
                        "files_backed_up": len(list(backup_dir.rglob("*"))),
                        "size_mb": sum(f.stat().st_size for f in backup_dir.rglob("*") if f.is_file()) / (1024*1024)
                    })
        
        return sorted(backups, key=lambda x: x['timestamp'], reverse=True)
    
    def restore_backup(self, backup_name: str) -> bool:
        """Restore from a backup"""
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            print(f"❌ Backup not found: {backup_name}")
            return False
        
        try:
            # Restore training state
            state_backup = backup_path / "training_state"
            if state_backup.exists():
                if self.state_dir.exists():
                    shutil.rmtree(self.state_dir)
                shutil.copytree(state_backup, self.state_dir)
            
            # Restore checkpoints
            checkpoint_backup = backup_path / "checkpoints"
            if checkpoint_backup.exists():
                if self.checkpoint_dir.exists():
                    shutil.rmtree(self.checkpoint_dir)
                shutil.copytree(checkpoint_backup, self.checkpoint_dir)
            
            print(f"✅ Backup restored: {backup_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to restore backup: {e}")
            return False
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current training status"""
        state_file = self.state_dir / "training_state.pkl"
        
        if not state_file.exists():
            return {"status": "No training state found"}
        
        try:
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
            
            return {
                "current_stage": state.current_stage,
                "stages_completed": state.stages_completed,
                "models_trained": state.models_trained,
                "best_accuracy": state.best_accuracy,
                "can_resume": True
            }
        except Exception as e:
            return {"status": f"Error reading state: {e}"}
    
    def clean_old_backups(self, keep_count: int = 10):
        """Clean old backups, keeping only the most recent ones"""
        backups = self.list_backups()
        
        if len(backups) > keep_count:
            to_remove = backups[keep_count:]
            
            for backup in to_remove:
                backup_path = self.backup_dir / backup['name']
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                    print(f"🗑️  Removed old backup: {backup['name']}")
            
            print(f"✅ Cleaned {len(to_remove)} old backups, kept {keep_count} most recent")

def main():
    parser = argparse.ArgumentParser(description='HMS Training Resume Manager')
    parser.add_argument('--backup', metavar='NAME', help='Create backup with optional name')
    parser.add_argument('--list', action='store_true', help='List all backups')
    parser.add_argument('--restore', metavar='NAME', help='Restore from backup')
    parser.add_argument('--status', action='store_true', help='Show current training status')
    parser.add_argument('--clean', action='store_true', help='Clean old backups')
    parser.add_argument('--auto-backup', action='store_true', help='Create automatic backup')
    
    args = parser.parse_args()
    manager = ResumeManager()
    
    if args.backup is not None:
        manager.create_backup(args.backup if args.backup else None)
    elif args.list:
        backups = manager.list_backups()
        print("\\n📋 Available Backups:")
        print("=" * 80)
        for backup in backups:
            print(f"Name: {backup['name']}")
            print(f"Date: {backup['timestamp']}")
            print(f"Size: {backup['size_mb']:.1f}MB")
            print("-" * 40)
    elif args.restore:
        manager.restore_backup(args.restore)
    elif args.status:
        status = manager.get_current_status()
        print("\\n📊 Current Training Status:")
        print(json.dumps(status, indent=2))
    elif args.clean:
        manager.clean_old_backups()
    elif args.auto_backup:
        manager.create_backup()
        manager.clean_old_backups()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
'''
        
        resume_script.write_text(resume_content)
        resume_script.chmod(0o755)
        
    def create_enhanced_deployment_config(self, deploy_dir: Path):
        """Create enhanced deployment configuration"""
        config = {
            "deployment_info": {
                "target": "Novita AI",
                "gpu": "H100 80GB",
                "optimization_level": "production_enhanced",
                "expected_accuracy": ">90%",
                "full_dataset": True,
                "advanced_features": [
                    "EEG Foundation Model",
                    "Ensemble Training", 
                    "Resume Capability",
                    "Memory Optimization",
                    "Mixed Precision Training",
                    "Distributed Training Support"
                ]
            },
            "environment": {
                "python_version": "3.10",
                "cuda_version": "11.8",
                "pytorch_version": "2.1+",
                "memory_gb": 80,
                "storage_gb": 500,
                "auto_resume": True,
                "backup_interval_minutes": 30
            },
            "training_features": {
                "eeg_foundation_model": True,
                "ensemble_training": True,
                "cross_validation": True,
                "hyperparameter_optimization": True,
                "advanced_augmentation": True,
                "memory_optimization": True,
                "gradient_checkpointing": True,
                "mixed_precision": True
            },
            "resume_config": {
                "auto_backup_enabled": True,
                "backup_interval": 30,  # minutes
                "max_backups": 10,
                "checkpoint_every_epoch": True,
                "state_save_frequency": "after_each_stage"
            },
            "monitoring": {
                "check_interval_seconds": 15,
                "log_level": "INFO",
                "save_checkpoints": True,
                "early_stopping": True,
                "wandb_enabled": True,
                "mlflow_enabled": True
            }
        }
        
        config_file = deploy_dir / "enhanced_deployment_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
    def upload_and_deploy_enhanced(self, archive_path: str):
        """Upload enhanced code and deploy to Novita AI"""
        if not self.ssh_host:
            self.log("SSH host not configured. Please set up instance first.", Colors.RED)
            return False
            
        self.log(f"Uploading enhanced package {archive_path} to Novita AI...", Colors.BLUE)
        
        # Upload archive
        scp_cmd = [
            'scp', '-i', self.ssh_key_path, '-o', 'StrictHostKeyChecking=no',
            archive_path, f'{self.ssh_user}@{self.ssh_host}:/tmp/'
        ]
        
        try:
            subprocess.run(scp_cmd, check=True)
            self.log("Enhanced package uploaded successfully", Colors.GREEN)
        except subprocess.CalledProcessError as e:
            self.log(f"Upload failed: {e}", Colors.RED)
            return False
            
        # Extract and setup
        ssh_cmd = f"""
            cd /workspace &&
            tar -xzf /tmp/{Path(archive_path).name} &&
            chmod +x novita_enhanced_setup.sh &&
            chmod +x enhanced_monitor.py &&
            chmod +x resume_manager.py &&
            chmod +x restart_training.sh &&
            ./novita_enhanced_setup.sh
        """
        
        ssh_full_cmd = [
            'ssh', '-i', self.ssh_key_path, '-o', 'StrictHostKeyChecking=no',
            f'{self.ssh_user}@{self.ssh_host}', ssh_cmd
        ]
        
        try:
            subprocess.run(ssh_full_cmd, check=True)
            self.log("Enhanced deployment setup completed", Colors.GREEN)
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"Enhanced deployment setup failed: {e}", Colors.RED)
            return False
            
    def start_enhanced_training(self):
        """Start the enhanced training pipeline"""
        self.log("Starting HMS Enhanced Training Pipeline...", Colors.BLUE)
        
        training_cmd = '''
            cd /workspace &&
            source /opt/miniconda3/etc/profile.d/conda.sh &&
            conda activate hms &&
            screen -dmS hms-enhanced python run_novita_training_enhanced.py
        '''
        
        ssh_cmd = [
            'ssh', '-i', self.ssh_key_path, '-o', 'StrictHostKeyChecking=no',
            f'{self.ssh_user}@{self.ssh_host}', training_cmd
        ]
        
        try:
            subprocess.run(ssh_cmd, check=True)
            self.log("Enhanced training started in screen session 'hms-enhanced'", Colors.GREEN)
            self.log("Connect via SSH and run: screen -r hms-enhanced", Colors.CYAN)
            self.log("Monitor with: hms-monitor", Colors.CYAN)
        except subprocess.CalledProcessError as e:
            self.log(f"Enhanced training start failed: {e}", Colors.RED)
            
    def check_training_status(self):
        """Check training status on remote instance"""
        if not self.ssh_host:
            self.log("SSH host not configured", Colors.RED)
            return
            
        status_cmd = '''
            cd /workspace &&
            source /opt/miniconda3/etc/profile.d/conda.sh &&
            conda activate hms &&
            python run_novita_training_enhanced.py --status
        '''
        
        ssh_cmd = [
            'ssh', '-i', self.ssh_key_path, '-o', 'StrictHostKeyChecking=no',
            f'{self.ssh_user}@{self.ssh_host}', status_cmd
        ]
        
        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.log("Current training status:", Colors.CYAN)
                print(result.stdout)
            else:
                self.log(f"Status check failed: {result.stderr}", Colors.RED)
        except subprocess.CalledProcessError as e:
            self.log(f"Status check failed: {e}", Colors.RED)
            
    def show_enhanced_instructions(self):
        """Show enhanced deployment instructions"""
        instructions = f"""
{Colors.CYAN}🚀 Enhanced HMS Novita AI Deployment Instructions{Colors.NC}
{'='*80}

{Colors.YELLOW}🎯 This enhanced version includes:{Colors.NC}
✅ EEG Foundation Model pre-training
✅ Advanced ensemble methods
✅ Robust resume capability 
✅ Memory optimization
✅ Comprehensive monitoring
✅ Automatic backups

{Colors.YELLOW}1. Launch Novita AI Instance:{Colors.NC}
   - Go to https://novita.ai/
   - Launch H100 80GB instance (REQUIRED)
   - Choose Ubuntu 22.04
   - Storage: 500GB+ SSD
   - Copy SSH connection details

{Colors.YELLOW}2. Configure SSH:{Colors.NC}
   python deploy_novita_enhanced.py --ssh-host YOUR_HOST --ssh-key YOUR_KEY_PATH

{Colors.YELLOW}3. Deploy Enhanced Package:{Colors.NC}
   python deploy_novita_enhanced.py --deploy-enhanced

{Colors.YELLOW}4. Start Enhanced Training:{Colors.NC}
   python deploy_novita_enhanced.py --start-enhanced

{Colors.YELLOW}5. Monitor & Resume:{Colors.NC}
   python deploy_novita_enhanced.py --status
   python deploy_novita_enhanced.py --monitor
   
   # If training stops (credits out):
   python deploy_novita_enhanced.py --resume

{Colors.YELLOW}6. SSH Commands (on instance):{Colors.NC}
   hms-status         - Check training status
   hms-monitor        - Real-time monitoring  
   hms-resume         - Resume training
   backup-state       - Manual backup
   ./restart_training.sh - Quick restart

{Colors.GREEN}Enhanced Features:{Colors.NC}
🧠 Foundation Model: Self-supervised pre-training
🔗 Advanced Ensemble: Stacking + Bayesian averaging
💾 Auto-Resume: Automatic checkpoint recovery
📊 Enhanced Monitoring: Real-time GPU/cost tracking
🔄 Smart Backups: Automatic state preservation

{Colors.PURPLE}Expected Results:{Colors.NC}
🎯 Accuracy: >92% (enhanced models)
⏱️  Time: 10-14 hours (with all features)
💰 Cost: $30-45 total
🎁 Output: Production-ready ONNX models + Foundation model

{Colors.RED}⚠️  IMPORTANT - Resume Capability:{Colors.NC}
If your credits run out and training stops:
1. Top up credits and restart instance
2. SSH into instance  
3. Run: ./restart_training.sh
4. Training will resume from last checkpoint!

{Colors.CYAN}Ready to deploy? Run:{Colors.NC}
python deploy_novita_enhanced.py --deploy-enhanced
        """
        
        print(instructions)

def main():
    parser = argparse.ArgumentParser(description='Enhanced HMS Novita AI Deployment Manager')
    parser.add_argument('--ssh-host', help='SSH host address')
    parser.add_argument('--ssh-key', help='SSH private key path')
    parser.add_argument('--instance-id', help='Instance ID for reference')
    
    # Enhanced Actions
    parser.add_argument('--create-enhanced', action='store_true', help='Create enhanced deployment package')
    parser.add_argument('--deploy-enhanced', action='store_true', help='Full enhanced deployment')
    parser.add_argument('--start-enhanced', action='store_true', help='Start enhanced training')
    parser.add_argument('--status', action='store_true', help='Check training status')
    parser.add_argument('--monitor', action='store_true', help='Remote monitoring')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--ssh', action='store_true', help='Connect via SSH')
    parser.add_argument('--instructions', action='store_true', help='Show enhanced instructions')
    
    args = parser.parse_args()
    
    deployer = EnhancedNovitaDeployer(args.ssh_key, args.instance_id)
    
    if args.ssh_host:
        deployer.ssh_host = args.ssh_host
        
    if args.instructions or not any(vars(args).values()):
        deployer.show_enhanced_instructions()
        return
        
    if args.create_enhanced:
        deployer.create_enhanced_deployment_package()
        
    if args.deploy_enhanced:
        # Full enhanced deployment process
        archive = deployer.create_enhanced_deployment_package()
        if deployer.upload_and_deploy_enhanced(archive):
            deployer.log("Enhanced deployment completed! Use --start-enhanced to begin.", Colors.GREEN)
            
    if args.start_enhanced:
        deployer.start_enhanced_training()
        
    if args.status:
        deployer.check_training_status()
        
    if args.ssh:
        deployer.connect_ssh()
        
    if args.monitor:
        deployer.log("Use SSH to access enhanced monitoring: hms-monitor", Colors.YELLOW)
        
    if args.resume:
        deployer.log("Resuming training on remote instance...", Colors.BLUE)
        resume_cmd = '''
            cd /workspace &&
            source /opt/miniconda3/etc/profile.d/conda.sh &&
            conda activate hms &&
            ./restart_training.sh
        '''
        
        ssh_cmd = [
            'ssh', '-i', deployer.ssh_key_path, '-o', 'StrictHostKeyChecking=no',
            f'{deployer.ssh_user}@{deployer.ssh_host}', resume_cmd
        ]
        
        try:
            subprocess.run(ssh_cmd, check=True)
            deployer.log("Training resume initiated", Colors.GREEN)
        except subprocess.CalledProcessError as e:
            deployer.log(f"Resume failed: {e}", Colors.RED)

if __name__ == '__main__':
    main() 