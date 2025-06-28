#!/bin/bash
set -e

echo "🚀 Starting ultra-fast HMS pipeline..."

# 1. Quick environment setup (5 minutes)
echo "⚙️ Setting up environment..."
export PYTHONPATH=/workspace
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 2. Install only essential packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets
pip install onnx onnxruntime-gpu
pip install pandas numpy scikit-learn tqdm
pip install mne scipy pywavelets

# 3. Create minimal directory structure
mkdir -p data/{raw,processed} models/{final,onnx} logs checkpoints

echo "✅ Environment ready!"