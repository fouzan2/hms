# HMS EEG Classification - Novita AI Deployment Guide

## 🎯 Goal: Achieve >90% Accuracy on Full HMS Dataset

This guide provides complete instructions for deploying the HMS EEG Classification system on Novita AI using H100 GPU with the full dataset to achieve >90% accuracy.

## 📋 Prerequisites

### 1. Accounts & Credentials
- **Novita AI Account**: [Sign up at novita.ai](https://novita.ai)
- **Kaggle Account**: For dataset access
- **Weights & Biases** (optional): For advanced tracking
- **MLflow** (optional): For experiment management

### 2. Local Requirements
- Python 3.8+
- SSH client
- 50GB+ local storage for deployment package

## 🚀 Quick Start (5-Step Process)

### Step 1: Prepare Deployment Package
```bash
# Clone your HMS repository
git clone <your-hms-repo>
cd hms

# Create deployment package
python deploy_novita.py --create-package
```

### Step 2: Launch Novita AI Instance
1. Go to [Novita AI Dashboard](https://novita.ai)
2. Click "Create Instance"
3. Select configuration:
   - **GPU**: H100 80GB (Required)
   - **OS**: Ubuntu 22.04
   - **Storage**: 500GB+ SSD
   - **Region**: Choose closest to you

4. Launch instance and note SSH details

### Step 3: Configure SSH Access
```bash
# Save your SSH private key
vim ~/.ssh/novita_hms_key
chmod 600 ~/.ssh/novita_hms_key

# Update deployment script with SSH details
python deploy_novita.py --ssh-host YOUR_INSTANCE_IP --ssh-key ~/.ssh/novita_hms_key
```

### Step 4: Deploy Code
```bash
# Deploy to Novita AI
python deploy_novita.py --deploy

# SSH into instance
python deploy_novita.py --ssh
```

### Step 5: Start Training
```bash
# On Novita AI instance:
cd /workspace

# Set up Kaggle credentials
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Start training
conda activate hms
python run_novita_training.py --config config/novita_production_config.yaml
```

## 📊 Expected Results

| Metric | Target | Expected Time |
|--------|--------|---------------|
| **Accuracy** | >90% | 8-12 hours |
| **Total Cost** | $25-40 | - |
| **Dataset Size** | 106,800 samples | Full dataset |
| **Models Trained** | ResNet1D-GRU + EfficientNet + Ensemble | - |
| **Output** | ONNX models ready for deployment | - |

## 🔧 Detailed Configuration

### Production Configuration (`config/novita_production_config.yaml`)

Key optimizations for >90% accuracy:

```yaml
# Enhanced model architectures
models:
  resnet1d_gru:
    resnet:
      initial_filters: 128      # Increased capacity
      num_blocks: [3, 4, 6, 3]  # ResNet50-like structure
      kernel_size: 9            # Larger receptive field
    gru:
      hidden_size: 512          # Increased capacity
      num_layers: 3             # Deeper network
    training:
      epochs: 150               # More epochs for full dataset
      batch_size: 64            # Optimized for H100

  efficientnet:
    model_name: "efficientnet-b5"  # Larger model
    training:
      epochs: 120
      batch_size: 32
      mixup_alpha: 0.4
      cutmix_alpha: 1.2

# Advanced training features
advanced_training:
  gradient_clipping: 1.0
  label_smoothing: 0.1
  stochastic_weight_averaging: true
  test_time_augmentation: true

# H100 optimizations
system:
  max_memory_gb: 70
  gpu_memory_fraction: 0.95
  compile_model: true
  flash_attention: true
```

### Monitoring Commands

```bash
# Real-time monitoring
python monitor_training.py

# Quick status check
python check_status.py

# Check GPU usage
nvidia-smi -l 1

# View logs
tail -f logs/novita_training.log

# Screen sessions
screen -r hms-training  # Attach to training session
screen -ls              # List sessions
```

## 📈 Training Pipeline Details

### Phase 1: Data Download & Preprocessing (2-3 hours)
- Download 106,800 EEG samples from Kaggle
- Advanced preprocessing with ICA artifact removal
- High-resolution spectrogram generation
- Parallel processing on all CPU cores

### Phase 2: Model Training (6-8 hours)
- **ResNet1D-GRU**: Raw EEG signal processing
- **EfficientNet-B5**: Spectrogram analysis
- **Ensemble**: Stacking both models
- Mixed precision training with gradient scaling
- Advanced augmentation techniques

### Phase 3: Optimization & Export (1 hour)
- Model pruning and quantization
- ONNX export for deployment
- Performance validation

## 🔍 Monitoring & Troubleshooting

### Real-time Monitoring Dashboard
```bash
# Launch comprehensive monitor
python monitor_training.py
```

Shows:
- GPU utilization and memory
- Training progress and metrics
- System resources
- ETA and cost estimation

### Common Issues & Solutions

#### 1. Out of Memory Errors
```bash
# Reduce batch size in config
# Edit config/novita_production_config.yaml
training:
  batch_size: 32  # Reduce from 64
```

#### 2. Slow Training
```bash
# Check GPU utilization
nvidia-smi

# Verify mixed precision is enabled
grep "mixed_precision" config/novita_production_config.yaml
```

#### 3. Network Issues
```bash
# Monitor network usage
iftop

# Resume from checkpoint if interrupted
python run_novita_training.py --resume models/checkpoints/resnet1d_gru_best.pth
```

## 💡 Optimization Tips for >90% Accuracy

### 1. Data Quality
- Use full 106,800 sample dataset
- Enable advanced preprocessing
- Increase spectrogram resolution

### 2. Model Architecture
- Use EfficientNet-B5 (larger model)
- Increase ResNet capacity
- Enable attention mechanisms

### 3. Training Strategy
- More epochs (150+ for ResNet, 120+ for EfficientNet)
- Advanced augmentation (mixup, cutmix)
- Label smoothing and test-time augmentation
- Ensemble multiple models

### 4. Hardware Utilization
- Use mixed precision training
- Enable torch.compile for speed
- Optimize batch sizes for H100
- Use persistent data workers

## 🔐 Security & Best Practices

### SSH Security
```bash
# Use SSH key authentication only
# Disable password authentication
# Set up SSH config for easy access
```

### Data Protection
```bash
# Secure Kaggle credentials
chmod 600 ~/.kaggle/kaggle.json

# Regular backups
rsync -av models/ backup_models/
```

### Cost Management
```bash
# Monitor costs regularly
python deploy_novita.py --monitor

# Auto-shutdown after training
echo "sudo shutdown -h +30" | at now + 12 hours
```

## 📤 Model Export & Deployment

### ONNX Export
```bash
# Models automatically exported to models/onnx/
ls models/onnx/
# resnet1d_gru.onnx
# efficientnet.onnx
```

### Model Validation
```bash
# Test ONNX models
python test_onnx_export.py --model models/onnx/resnet1d_gru.onnx
```

### Download Models
```bash
# From local machine:
scp -r -i ~/.ssh/novita_hms_key root@YOUR_IP:/workspace/models/onnx/ ./trained_models/
scp -r -i ~/.ssh/novita_hms_key root@YOUR_IP:/workspace/training_results.json ./
```

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Novita AI account created
- [ ] H100 instance launched
- [ ] SSH access configured
- [ ] Kaggle credentials ready
- [ ] Local deployment package created

### During Training
- [ ] Monitor GPU utilization (>90%)
- [ ] Check training progress regularly
- [ ] Verify accuracy improvements
- [ ] Monitor cost accumulation
- [ ] Backup checkpoints periodically

### Post-Training
- [ ] Validate >90% accuracy achieved
- [ ] Export ONNX models
- [ ] Download trained models
- [ ] Save training results
- [ ] Terminate instance to stop billing

## 💰 Cost Estimation

| Component | Time | H100 Rate | Cost |
|-----------|------|-----------|------|
| Setup | 1 hour | $3.35/hr | $3.35 |
| Data Prep | 2 hours | $3.35/hr | $6.70 |
| Training | 8 hours | $3.35/hr | $26.80 |
| Export | 1 hour | $3.35/hr | $3.35 |
| **Total** | **12 hours** | **$3.35/hr** | **~$40** |

## 🆘 Support & Troubleshooting

### Log Files
- Training logs: `logs/novita_training.log`
- System logs: `logs/system.log`
- Error logs: `logs/errors.log`

### Useful Commands
```bash
# View training progress
tail -f logs/novita_training.log | grep "Accuracy"

# Check disk space
df -h

# Monitor memory usage
free -h

# List running processes
ps aux | grep python

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Contact Information
- **GitHub Issues**: [Your repo issues page]
- **Email**: [Your support email]
- **Discord**: [Your support server]

## 🎉 Success Indicators

When training is complete, you should see:

```
🎉 TRAINING COMPLETED SUCCESSFULLY!
🎯 Best Accuracy: 92.3%
⏱️  Total Time: 10.2 hours
💰 Estimated Cost: $34.17
🎯 Target Achieved: ✅ YES
```

**Congratulations!** You now have production-ready HMS EEG classification models achieving >90% accuracy!

## 📚 Additional Resources

- [HMS Kaggle Competition](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)
- [Novita AI Documentation](https://docs.novita.ai)
- [PyTorch H100 Optimization Guide](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)
- [EEG Signal Processing Best Practices](https://mne.tools/stable/overview/index.html)

---

**Happy Training!** 🧠✨ 