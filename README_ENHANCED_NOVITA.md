# 🧠 Enhanced HMS EEG Classification - Novita AI Deployment

## 🎯 Complete System with Advanced Features & Resume Capability

This enhanced deployment system provides a production-ready HMS EEG Classification pipeline optimized for Novita AI H100 GPU with **robust resume functionality** and advanced ML features.

### 🚀 New Advanced Features

✅ **EEG Foundation Model** - Self-supervised pre-training  
✅ **Advanced Ensemble Methods** - Stacking + Bayesian averaging  
✅ **Robust Resume Capability** - Automatic checkpoint recovery  
✅ **Memory Optimization** - H100-optimized training  
✅ **Enhanced Monitoring** - Real-time GPU/cost tracking  
✅ **Smart Backups** - Automatic state preservation  
✅ **Stage-wise Training** - Independent recoverable stages  

## 📦 What's Included

### Core Files
- **`run_novita_training_enhanced.py`** - Main enhanced training pipeline
- **`deploy_novita_enhanced.py`** - Enhanced deployment manager
- **`config/novita_enhanced_config.yaml`** - Production configuration
- **Enhanced monitoring and resume scripts** (auto-generated)

### Advanced Components
- **EEG Foundation Model** with masked language modeling
- **Advanced Ensemble** with uncertainty quantification
- **Memory Optimization** for large model training
- **Fault-tolerant Training** with automatic checkpointing
- **Enhanced Preprocessing** with ICA artifact removal

## 🚀 Quick Start Guide

### 1. Prerequisites
```bash
# Accounts needed:
# - Novita AI account (novita.ai)
# - Kaggle account for dataset
# - SSH key for instance access
```

### 2. Launch Novita AI Instance
1. Go to [novita.ai](https://novita.ai)
2. Launch **H100 80GB** instance (Required!)
3. Choose Ubuntu 22.04, 500GB+ storage
4. Note SSH details (IP, key, etc.)

### 3. Deploy Enhanced System
```bash
# Configure deployment
python deploy_novita_enhanced.py --ssh-host YOUR_IP --ssh-key ~/.ssh/your_key

# Deploy complete enhanced system
python deploy_novita_enhanced.py --deploy-enhanced
```

### 4. Start Enhanced Training
```bash
# Start training with all features
python deploy_novita_enhanced.py --start-enhanced

# Or SSH and start manually
python deploy_novita_enhanced.py --ssh
# Then on instance:
cd /workspace
./restart_training.sh
```

## 📊 Expected Enhanced Results

| Feature | Value |
|---------|-------|
| **🎯 Accuracy** | >92% (enhanced models) |
| **⏱️ Training Time** | 10-14 hours (all features) |
| **💰 Total Cost** | $30-45 |
| **📦 Dataset** | Full 106,800 samples |
| **🤖 Models** | Foundation + ResNet + EfficientNet + Ensemble |
| **🎁 Output** | Production ONNX + Foundation model |

## 🔄 Resume Capability (IMPORTANT!)

### If Credits Run Out During Training:

1. **Training automatically saves state every 30 minutes**
2. **All checkpoints preserved in multiple locations**
3. **Resume process:**

```bash
# When you restart the instance:
ssh root@YOUR_IP
cd /workspace

# Quick restart (automatic resume)
./restart_training.sh

# Or manual resume
python run_novita_training_enhanced.py --resume
```

### Resume Features:
- ✅ **Stage-wise resume** - Each major stage (data, preprocessing, training, etc.) is recoverable
- ✅ **Model-wise resume** - Each model can resume from its last checkpoint
- ✅ **Epoch-wise resume** - Training resumes from exact epoch where it stopped
- ✅ **State preservation** - All training state, metrics, and progress preserved
- ✅ **Automatic backups** - Multiple backup points created automatically

## 🎯 Training Pipeline Stages

The enhanced pipeline consists of 6 independent, resumable stages:

### Stage 1: Data Download & Setup (1-2 hours)
```bash
# Downloads full HMS dataset from Kaggle
# ✅ Resumable: Skipped if data already exists
```

### Stage 2: Advanced Preprocessing (2-3 hours)
```bash
# - Advanced EEG filtering and artifact removal
# - High-resolution spectrogram generation
# - Feature extraction with parallel processing
# ✅ Resumable: Uses batch processing with temp saves
```

### Stage 3: Foundation Model Pre-training (3-4 hours)
```bash
# - Self-supervised EEG foundation model training
# - Masked EEG modeling + contrastive learning
# ✅ Resumable: Checkpoint-based training
```

### Stage 4: Main Model Training (6-8 hours)
```bash
# - Enhanced ResNet1D-GRU (using foundation features)
# - Advanced EfficientNet-B5 with progressive training
# ✅ Resumable: Per-model checkpointing
```

### Stage 5: Ensemble Training (1-2 hours)
```bash
# - Advanced stacking ensemble
# - Uncertainty quantification
# ✅ Resumable: Meta-learner checkpointing
```

### Stage 6: Model Export (30 mins)
```bash
# - ONNX export for all models
# - Model validation and optimization
# ✅ Resumable: Per-model export tracking
```

## 🔧 Enhanced Commands

### On Your Local Machine:
```bash
# Deploy enhanced system
python deploy_novita_enhanced.py --deploy-enhanced

# Check training status remotely
python deploy_novita_enhanced.py --status

# Resume training remotely
python deploy_novita_enhanced.py --resume

# Connect to instance
python deploy_novita_enhanced.py --ssh
```

### On Novita AI Instance:
```bash
# Enhanced monitoring (real-time dashboard)
hms-monitor

# Check detailed status
hms-status

# Resume training (smart resume)
hms-resume

# Manual backup
backup-state

# Quick restart
./restart_training.sh

# View specific stage
python run_novita_training_enhanced.py --stage preprocessing
```

## 📈 Enhanced Monitoring

### Real-time Dashboard (`hms-monitor`)
```
🧠 Enhanced HMS EEG Classification - Novita AI Training Monitor
================================================================================
🕐 Time: 2024-01-15 14:30:45

🔥 GPU Status (H100 80GB):
   Name: NVIDIA H100 80GB HBM2e
   Utilization: 98%
   Memory: 72,456MB / 81,920MB (88.4%)
   Temperature: 68°C
   Power: 650.2W / 700.0W (92.9%)

🎯 Enhanced Training Status:
   Current Stage: model_training
   Completed Stages: data_download, preprocessing, foundation_pretraining
   Best Accuracy: 91.2%
   Target (90%): ✅ ACHIEVED

🤖 Models Status:
   resnet1d_gru: ✅ Trained (91.2%)
   efficientnet: ⏳ Training (Epoch 45/120)
   ensemble: ⏳ Pending

💰 Time & Cost Estimation:
   Time Spent: 8.5 hours
   Remaining: 3.2 hours
   Cost Spent: $28.48
   Est. Total: $39.22
   Efficiency: 98% GPU utilization
```

## 🛠️ Configuration

### Enhanced Features Configuration
The enhanced config (`config/novita_enhanced_config.yaml`) includes:

```yaml
# EEG Foundation Model
models:
  eeg_foundation:
    enabled: true
    pretraining:
      methods: ["masked_eeg", "contrastive", "temporal_prediction"]
      
# Advanced Ensemble
ensemble:
  method: "stacking_plus"
  uncertainty_estimation:
    enabled: true
    method: "monte_carlo_dropout"

# Resume Configuration
checkpointing:
  save_every_epoch: true
  auto_backup_enabled: true
  backup_interval_minutes: 30
```

## 🚨 Troubleshooting

### If Training Stops Unexpectedly:

1. **Check if it's a credit issue:**
   ```bash
   # Check Novita AI dashboard for credit balance
   ```

2. **If credits ran out:**
   ```bash
   # Top up credits, restart instance
   # SSH back in and run:
   ./restart_training.sh
   ```

3. **If it's a different issue:**
   ```bash
   # Check logs
   hms-logs
   
   # Check status
   hms-status
   
   # Try manual resume
   python run_novita_training_enhanced.py --resume
   ```

### Common Issues:

**"No training state found"**
```bash
# Check for backups
python resume_manager.py --list

# Restore from backup
python resume_manager.py --restore backup_name
```

**"CUDA out of memory"**
```bash
# The enhanced config is optimized for H100
# If issues persist, reduce batch sizes in config
```

**"Kaggle credentials not found"**
```bash
# Set up credentials
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USER","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## 🎯 Achieving >90% Accuracy

### Enhanced Model Architecture:
- **Foundation Model**: Pre-trained on large EEG corpus
- **ResNet1D-GRU**: Enhanced with foundation features
- **EfficientNet-B5**: Larger model with progressive training
- **Advanced Ensemble**: Stacking + uncertainty quantification

### Training Optimizations:
- **Full Dataset**: All 106,800 samples
- **Advanced Augmentation**: Time + frequency domain
- **Mixed Precision**: FP16 training on H100
- **Memory Optimization**: Gradient checkpointing
- **Enhanced Preprocessing**: ICA artifact removal

### Expected Accuracy Progression:
- **ResNet1D-GRU**: ~89% (baseline)
- **EfficientNet-B5**: ~90% (improved)
- **Ensemble**: ~92%+ (target exceeded!)

## 📞 Support

### Automated Recovery:
The system is designed to handle most issues automatically through:
- Automatic checkpointing every epoch
- Stage-wise progress tracking
- Smart resume capability
- Multiple backup points

### Manual Support:
If you encounter issues:
1. Check the monitoring dashboard: `hms-monitor`
2. Review logs: `hms-logs`
3. Try the restart script: `./restart_training.sh`
4. Use the resume manager: `python resume_manager.py --help`

## 🏁 Success Criteria

When training completes successfully, you'll see:

```
🎉 ENHANCED TRAINING COMPLETED SUCCESSFULLY!
🎯 Best Accuracy: 92.7%
⏱️  Total Time: 11h 23m
🤖 Models Trained: resnet1d_gru, efficientnet, ensemble
🎯 Target Achieved: ✅ YES (>90%)

🎁 Outputs Available:
   📁 models/final/ - Trained PyTorch models
   📁 models/onnx/ - Production ONNX models
   📁 training_results.json - Complete results
   🧠 EEG Foundation Model - Pre-trained for transfer learning
```

## 🚀 Ready to Deploy?

Start your enhanced HMS training with full resume capability:

```bash
python deploy_novita_enhanced.py --instructions
python deploy_novita_enhanced.py --deploy-enhanced
```

**Happy Training with Enhanced Features!** 🧠✨

---

*This enhanced system ensures your training investment is protected with robust resume capability and advanced ML features for maximum accuracy.* 