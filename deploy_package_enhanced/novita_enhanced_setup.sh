#!/bin/bash
set -e

# Enhanced HMS EEG Classification - Novita AI Setup Script
# Optimized for H100 GPU with all advanced features

echo "🚀 Starting Enhanced HMS EEG Classification setup on Novita AI..."

# Update system and install dependencies
apt-get update -qq
apt-get install -y git wget curl unzip htop nvtop screen tmux tree     build-essential cmake ninja-build libfftw3-dev     libopenblas-dev liblapack-dev libhdf5-dev

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
pip install --no-cache-dir     kaggle wandb mlflow optuna     mne pyedflib antropy     efficientnet-pytorch timm transformers accelerate     xgboost lightgbm catboost     shap lime captum     onnx onnxruntime-gpu     fastapi uvicorn dash dash-bootstrap-components

# Advanced packages for new features
pip install --no-cache-dir     einops flash-attn     ray[tune] ray[train]     tensorboard tensorboardX     pytorch-lightning lightning     hydra-core omegaconf     rich typer click

# Development and monitoring tools
pip install --no-cache-dir     jupyter ipywidgets     psutil GPUtil py3nvml     prometheus-client grafana-api

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
