#!/bin/bash
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
pip install --no-cache-dir     kaggle     wandb     mlflow     optuna     mne     pyedflib     antropy     efficientnet-pytorch     timm     transformers     accelerate     flash-attn     xgboost     lightgbm     shap     lime     onnx     onnxruntime-gpu     fastapi     uvicorn     dash     dash-bootstrap-components

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
