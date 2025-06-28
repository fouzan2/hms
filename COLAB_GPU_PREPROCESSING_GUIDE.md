# Google Colab GPU Preprocessing Guide

This guide helps you run comprehensive EEG preprocessing on Google Colab with GPU acceleration, which is **10-50x faster** than CPU preprocessing.

## 📋 Prerequisites

1. **Google Account** with access to Google Colab
2. **Google Drive** with at least 50GB free space
3. **Kaggle Account** (no API file needed - kagglehub handles authentication)
4. **Google Colab Pro** (recommended for longer sessions and better GPUs)

> **Note**: The download script automatically deletes temporary files after extraction to save space on your Google Drive.

## 🚀 Quick Start

### Option 1: Using the Setup Script (Recommended)

1. **Open Google Colab**: https://colab.research.google.com/

2. **Enable GPU Runtime**:
   - Go to `Runtime` → `Change runtime type`
   - Select `GPU` as Hardware accelerator
   - Choose `High-RAM` if available

3. **Upload the script**:
   - Upload `colab_setup_and_preprocessing.py` to Colab
   - Copy and paste the code sections into separate cells

4. **Run the cells in order**:

```python
# Cell 1: Setup environment
gpu_available = setup_environment()

# Cell 2: Mount Drive (no Kaggle setup needed)
setup_data_access()

# Cell 3: Download data (will prompt for Kaggle login)
download_data()

# Cell 4: Test with small subset
DATA_PATH = "/content/drive/MyDrive/hms-data"
OUTPUT_PATH = "/content/drive/MyDrive/hms-processed"
summary = process_data_gpu(DATA_PATH, OUTPUT_PATH, max_samples=50)

# Cell 5: Visualize results
visualize_results(OUTPUT_PATH)

# Cell 6: Monitor GPU
monitor_gpu()
```

### Option 2: Using the Comprehensive Script

Use `colab_preprocessing_gpu.py` for more control:

```python
# Upload and run the script
!python colab_preprocessing_gpu.py
```

## 📊 Performance Optimization

### GPU Memory Management

Adjust batch size based on your GPU:

| GPU Type | Recommended Batch Size | Est. Speed |
|----------|----------------------|------------|
| T4 (Free) | 16-32 | ~2 samples/sec |
| P100 | 32-64 | ~4 samples/sec |
| V100 | 64-128 | ~8 samples/sec |
| A100 | 128-256 | ~15 samples/sec |

### Processing Time Estimates

For the full dataset (106,800 samples):

- **CPU**: 8-12 hours
- **T4 GPU**: 12-15 hours  
- **V100 GPU**: 3-4 hours
- **A100 GPU**: 2-3 hours

## 🛠️ Configuration Options

### Preprocessing Parameters

```python
CONFIG = {
    'sampling_rate': 200,      # EEG sampling rate
    'window_size': 256,        # STFT window size
    'hop_length': 128,         # STFT hop length
    'n_fft': 512,             # FFT points
    'freq_min': 0.5,          # Min frequency (Hz)
    'freq_max': 50.0,         # Max frequency (Hz)
    'lowcut': 0.5,            # Bandpass low cut
    'highcut': 50.0,          # Bandpass high cut
    'notch_freq': 60.0,       # Notch filter frequency
    'batch_size': 32          # GPU batch size
}
```

### GPU Features Used

1. **FFT-based Filtering**: 10x faster than time-domain filtering
2. **Batch STFT**: Process multiple channels simultaneously
3. **Mixed Precision**: FP16 computation for 2x speedup
4. **Parallel Feature Extraction**: Extract all features on GPU
5. **Optimized Memory Transfer**: Minimize CPU-GPU transfers

## 📁 Output Structure

```
/content/drive/MyDrive/hms-processed/
├── spectrograms/         # Spectrograms (channels × freq × time)
│   ├── 1000088191_spec.npy
│   └── ...
├── features/             # Extracted features  
│   ├── 1000088191_feat.npy
│   └── ...
├── filtered/             # Filtered signals
│   ├── 1000088191_filt.npy
│   └── ...
└── processing_summary.csv # Processing metadata
```

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory Error**:
   - Reduce batch_size
   - Clear GPU cache: `torch.cuda.empty_cache()`
   - Restart runtime

2. **Session Timeout**:
   - Use Colab Pro for 24h sessions
   - Process in chunks and save checkpoints
   - Use `colab-alive` extension

3. **Slow Processing**:
   - Check GPU utilization: `!nvidia-smi`
   - Ensure mixed precision is enabled
   - Verify data is on GPU

4. **Data Download Issues**:
   - **FileNotFoundError: train.csv not found**:
     ```python
     # Run the debugging function to check kagglehub
     debug_kaggle_setup()
     ```
   - **Common causes**:
     - Competition rules not accepted
     - Account not verified
     - kagglehub authentication issues
   - **Solutions**:
     1. Go to https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/rules
     2. Click "I Understand and Accept" to accept the rules
     3. Verify your Kaggle account with phone number
     4. Run `debug_kaggle_setup()` to test authentication
     5. When prompted, authenticate with kagglehub

### Memory Monitoring

```python
# Check GPU memory usage
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

## 🎯 Best Practices

1. **Test First**: Always test with 50-100 samples before full run
2. **Monitor Progress**: Use tqdm progress bars
3. **Save Regularly**: Process and save in batches
4. **Use Checkpoints**: Save processing state periodically
5. **Clean Cache**: Clear GPU memory between batches

## 📈 Advanced Features

### Multi-GPU Processing (Colab Pro+)

```python
# Check available GPUs
!nvidia-smi -L

# Use DataParallel for multi-GPU
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

### Custom Preprocessing

Add your own GPU-accelerated functions:

```python
@torch.cuda.amp.autocast()
def custom_gpu_function(data):
    # Your GPU code here
    return processed_data
```

## 🤝 Tips for Kaggle Competition

1. **Feature Engineering**: The GPU allows testing more features quickly
2. **Augmentation**: Apply real-time augmentation during preprocessing
3. **Ensemble Features**: Generate multiple spectrogram types
4. **Quality Filtering**: Use GPU for fast quality assessment

## 📚 Additional Resources

- [Google Colab GPU Tips](https://colab.research.google.com/notebooks/gpu.ipynb)
- [PyTorch GPU Best Practices](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [CuPy Documentation](https://docs.cupy.dev/en/stable/)

## 🚨 Important Notes

1. **Data Privacy**: Ensure your Kaggle data usage complies with competition rules
2. **Resource Limits**: Free Colab has GPU usage limits
3. **Saving Results**: Always save to Google Drive to prevent data loss
4. **Version Control**: Keep track of preprocessing versions for reproducibility

---

**Happy GPU Processing! 🚀**

If you encounter any issues, refer to the troubleshooting section or modify the batch size and parameters based on your GPU capabilities. 