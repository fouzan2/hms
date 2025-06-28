# HMS EEG Classification - Novita AI Deployment

## 🎯 Quick Deploy to Novita AI for >90% Accuracy

This package contains everything you need to deploy the HMS EEG Classification system on Novita AI H100 GPU and achieve >90% accuracy on the full dataset.

## 📦 What's Included

### Core Deployment Files
- **`deploy_novita.py`** - Main deployment script with SSH automation
- **`run_novita_training.py`** - Optimized training pipeline for H100
- **`config/novita_production_config.yaml`** - Production configuration for >90% accuracy
- **`NOVITA_DEPLOYMENT_GUIDE.md`** - Complete step-by-step instructions

### Supporting Files
- **`monitor_training.py`** - Real-time training monitor (auto-generated)
- **`check_status.py`** - Quick status checker (auto-generated)
- **`novita_setup.sh`** - Environment setup script (auto-generated)

## 🚀 Quick Start (5 Minutes to Launch)

### 1. Create Deployment Package
```bash
python deploy_novita.py --create-package
```

### 2. Launch Novita AI Instance
- Go to [novita.ai](https://novita.ai)
- Launch **H100 80GB** instance
- Ubuntu 22.04, 500GB+ storage
- Note SSH details

### 3. Deploy & Connect
```bash
# Configure SSH
python deploy_novita.py --ssh-host YOUR_IP --ssh-key YOUR_SSH_KEY

# Deploy code
python deploy_novita.py --deploy

# Connect via SSH
python deploy_novita.py --ssh
```

### 4. Start Training
```bash
# On Novita AI instance:
cd /workspace
conda activate hms

# Set Kaggle credentials
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USER","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Start full dataset training
python run_novita_training.py
```

## 📊 Expected Results

| Metric | Value |
|--------|-------|
| **Accuracy** | >90% |
| **Training Time** | 8-12 hours |
| **Cost** | ~$25-40 |
| **Dataset** | Full 106,800 samples |
| **Output** | ONNX models ready for production |

## 🔧 Key Optimizations for >90% Accuracy

### Model Enhancements
- **EfficientNet-B5**: Larger pre-trained model
- **ResNet1D-GRU**: Increased capacity (512 hidden units, 3 layers)
- **Advanced Ensemble**: Stacking with LightGBM meta-learner
- **Attention Mechanisms**: 16-head multi-head attention

### Training Optimizations
- **150 epochs** for ResNet1D-GRU
- **120 epochs** for EfficientNet
- **Mixed Precision** training (FP16)
- **Advanced Augmentation**: Mixup, CutMix, SpecAugment
- **Label Smoothing**: 0.1 smoothing factor
- **Test-Time Augmentation**: Multiple inference passes

### H100 Utilizations
- **Torch Compile**: PyTorch 2.0 compilation
- **Flash Attention**: Memory-efficient attention
- **Optimized Batch Sizes**: 64 for ResNet, 32 for EfficientNet
- **Persistent Workers**: Faster data loading

## 📈 Monitoring

### Real-Time Dashboard
```bash
# Comprehensive monitoring
python monitor_training.py
```

Shows:
- GPU utilization and temperature
- Training progress and ETA
- System resources and costs
- Model performance metrics

### Quick Commands
```bash
# Status check
python check_status.py

# GPU usage
watch nvidia-smi

# Training logs
tail -f logs/novita_training.log

# Screen sessions
screen -r hms-training
```

## 💡 Pro Tips

### Cost Management
- Monitor every 2-3 hours
- Use screen sessions for persistence
- Set auto-shutdown after training
- Download models immediately after completion

### Performance Optimization
- Ensure GPU utilization >90%
- Monitor for memory issues
- Use gradient checkpointing if needed
- Enable all mixed precision features

### Troubleshooting
- Check logs for any errors
- Verify Kaggle credentials
- Monitor disk space usage
- Use checkpoints for resume capability

## 📋 Success Checklist

### Pre-Training
- [ ] H100 instance launched
- [ ] SSH access working
- [ ] Code deployed successfully
- [ ] Kaggle credentials configured
- [ ] Environment activated

### During Training
- [ ] GPU utilization >90%
- [ ] Training progressing normally
- [ ] Accuracy improving each epoch
- [ ] No memory errors
- [ ] Logs being written

### Post-Training
- [ ] >90% accuracy achieved
- [ ] ONNX models exported
- [ ] Results downloaded locally
- [ ] Instance terminated

## 🎯 Target Achievement

When successful, you'll see:
```
🎉 TRAINING COMPLETED SUCCESSFULLY!
🎯 Best Accuracy: 92.3% ✅
⏱️  Total Time: 10.2 hours  
💰 Estimated Cost: $34.17
🎯 Target Achieved: ✅ YES
```

## 📞 Support

- **Full Guide**: See `NOVITA_DEPLOYMENT_GUIDE.md`
- **Configuration**: Check `config/novita_production_config.yaml`
- **Troubleshooting**: Review log files in `/workspace/logs/`

## 🏁 Ready to Deploy?

Run this command to start:
```bash
python deploy_novita.py --instructions
```

**Good luck achieving >90% accuracy!** 🧠🚀 