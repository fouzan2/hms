"""
Google Colab Setup and GPU Preprocessing Script
==============================================

This script sets up the environment and runs GPU-accelerated preprocessing on Google Colab.

Usage:
1. Upload this file to Google Colab
2. Ensure GPU runtime is enabled (Runtime > Change runtime type > GPU)
3. Run the script in cells
"""

# ============================
# CELL 1: Environment Setup
# ============================

def setup_environment():
    """Setup Google Colab environment."""
    import os
    import sys
    
    print("🚀 Setting up Google Colab environment...")
    
    # Install packages
    packages = [
        "kagglehub",           # Modern Kaggle API
        "torch torchvision torchaudio",
        "librosa scipy scikit-learn",
        "pandas numpy pywt",
        "h5py pyarrow fastparquet",
        "matplotlib seaborn",
        "pyyaml joblib tqdm"
    ]
    
    for pkg in packages:
        print(f"📦 Installing {pkg}...")
        os.system(f'pip install -q {pkg}')
    
    # Import and check GPU
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("❌ No GPU detected! Please enable GPU runtime.")
        return False

# Run this in first cell:
# gpu_available = setup_environment()


# ============================
# CELL 2: Mount Drive & Setup Kaggle
# ============================

def setup_data_access():
    """Setup Google Drive and Kaggle API."""
    from google.colab import drive
    import os
    
    # Mount Google Drive
    print("📁 Mounting Google Drive...")
    drive.mount('/content/drive')
    
    # Create directories
    os.makedirs('/content/drive/MyDrive/hms-processed', exist_ok=True)
    os.makedirs('/content/drive/MyDrive/hms-data', exist_ok=True)
    
    print("✅ Google Drive mounted and directories created!")
    print("📋 Note: Kaggle authentication will be handled automatically by kagglehub")
    print("   You'll be prompted to authenticate when downloading data.")

# Run this in second cell:
# setup_data_access()


# ============================
# CELL 3: Download Data
# ============================

def download_data(data_path='/content/drive/MyDrive/hms-data'):
    """Download HMS competition data using kagglehub."""
    import os
    import shutil
    
    if not os.path.exists(f'{data_path}/train.csv'):
        print("📥 Downloading data using kagglehub...")
        
        # Ensure the directory exists
        os.makedirs(data_path, exist_ok=True)
        
        try:
            # Import kagglehub
            import kagglehub
            
            # Login to Kaggle (will prompt for authentication if needed)
            print("🔑 Authenticating with Kaggle...")
            kagglehub.login()
            
            # Download competition data
            print("📦 Downloading HMS competition data...")
            source_path = kagglehub.competition_download('hms-harmful-brain-activity-classification')
            
            print(f"✅ Data downloaded to: {source_path}")
            
            # Copy data to our desired location
            print(f"📁 Moving data to: {data_path}")
            
            # List what was downloaded
            print("📋 Files downloaded:")
            for item in os.listdir(source_path):
                item_path = os.path.join(source_path, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path) / (1024*1024)  # MB
                    print(f"   📄 {item} ({size:.1f} MB)")
                else:
                    print(f"   📁 {item}/ (folder)")
            
            # Copy all files to our target directory
            for item in os.listdir(source_path):
                source_item = os.path.join(source_path, item)
                target_item = os.path.join(data_path, item)
                
                if os.path.isfile(source_item):
                    shutil.copy2(source_item, target_item)
                    print(f"   ✅ Copied: {item}")
                elif os.path.isdir(source_item):
                    if os.path.exists(target_item):
                        shutil.rmtree(target_item)
                    shutil.copytree(source_item, target_item)
                    print(f"   ✅ Copied folder: {item}")
            
            # Cleanup: Remove the original download cache
            print("🗑️  Cleaning up - removing temporary download cache...")
            try:
                # kagglehub stores data in cache, let's clean up if possible
                # Keep our copied data, remove any zip files
                for item in os.listdir(data_path):
                    if item.endswith('.zip'):
                        zip_path = os.path.join(data_path, item)
                        os.remove(zip_path)
                        print(f"   🗑️  Removed: {item}")
                
                # Optional: clean kagglehub cache (comment out if you want to keep it)
                # shutil.rmtree(source_path, ignore_errors=True)
                # print("   🗑️  Removed kagglehub cache")
                
            except Exception as e:
                print(f"   ⚠️  Cleanup warning: {e}")
            
            # Verify the download
            if os.path.exists(f'{data_path}/train.csv'):
                print("✅ Data successfully downloaded and verified!")
                
                # Show final directory contents
                print(f"\n📊 Final data directory contents:")
                os.system(f'ls -lh {data_path}/')
                
                return True
            else:
                print("❌ train.csv not found after download!")
                return False
                
        except ImportError:
            print("❌ kagglehub not installed! Installing...")
            os.system('pip install -q kagglehub')
            print("✅ kagglehub installed. Please run this function again.")
            return False
            
        except Exception as e:
            print(f"❌ Download failed: {str(e)}")
            print("   Please check your Kaggle authentication and competition access.")
            return False
    else:
        print("✅ Data already exists!")
        # Show what we have
        print(f"📊 Existing data:")
        os.system(f'ls -lh {data_path}/')
        return True

# Run this in third cell:
# download_data()


# ============================
# DEBUGGING: Kaggle API Issues
# ============================

def debug_kaggle_setup():
    """Debug kagglehub setup and authentication."""
    import os
    
    print("🔍 Debugging kagglehub setup...")
    
    # Check if kagglehub is installed
    try:
        import kagglehub
        print("✅ kagglehub is installed")
        print(f"   Version: {kagglehub.__version__}")
    except ImportError:
        print("❌ kagglehub not installed!")
        print("   Installing kagglehub...")
        os.system('pip install -q kagglehub')
        return
    
    # Test authentication
    print("\n🔗 Testing kagglehub authentication...")
    try:
        # This will prompt for login if not authenticated
        kagglehub.login()
        print("✅ kagglehub authentication successful")
    except Exception as e:
        print(f"❌ kagglehub authentication failed: {e}")
        print("   You'll be prompted to authenticate when downloading data")
    
    # Test competition access
    print("\n🏆 Testing HMS competition access...")
    try:
        # Try to get competition info (this is a lightweight check)
        info = kagglehub.competition_download('hms-harmful-brain-activity-classification', force_download=False)
        print("✅ Competition access verified")
        print(f"   Data available at: {info}")
    except Exception as e:
        print(f"❌ Competition access issue: {e}")
        print("   Make sure you've accepted the competition rules at:")
        print("   https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/rules")

# Run this if download fails:
# debug_kaggle_setup()


# ============================
# CELL 4: GPU Preprocessing Pipeline with ID Mapping Fix
# ============================

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# Global mapping cache
EEG_FILE_MAPPING = {}


def build_eeg_file_mapping(data_path):
    """Build mapping between eeg_id and actual file paths."""
    global EEG_FILE_MAPPING
    
    if EEG_FILE_MAPPING:
        return EEG_FILE_MAPPING
    
    print("🔍 Building EEG file mapping...")
    
    # Load train.csv
    train_df = pd.read_csv(f"{data_path}/train.csv")
    print(f"📊 Found {len(train_df)} entries in train.csv")
    
    # Get all parquet files
    eeg_dir = Path(data_path) / 'train_eegs'
    eeg_files = {f.stem: f for f in eeg_dir.glob('*.parquet')}
    print(f"📁 Found {len(eeg_files)} parquet files")
    
    # Build mapping - HMS dataset uses patient_id as filename
    for _, row in train_df.iterrows():
        eeg_id = str(row['eeg_id'])
        patient_id = str(row['patient_id'])
        
        # Primary strategy: Use patient_id
        if patient_id in eeg_files:
            EEG_FILE_MAPPING[eeg_id] = eeg_files[patient_id]
        # Fallback: Try direct eeg_id match
        elif eeg_id in eeg_files:
            EEG_FILE_MAPPING[eeg_id] = eeg_files[eeg_id]
    
    print(f"✅ Mapped {len(EEG_FILE_MAPPING)} EEG files")
    
    if len(EEG_FILE_MAPPING) == 0:
        print("❌ WARNING: No files could be mapped!")
        print("   Sample eeg_ids:", list(train_df['eeg_id'].head()))
        print("   Sample patient_ids:", list(train_df['patient_id'].head()))
        print("   Sample files:", list(eeg_files.keys())[:5])
    
    return EEG_FILE_MAPPING


class GPUBatchProcessor:
    """Optimized GPU batch processor for EEG data with complete preprocessing."""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        self.sampling_rate = 200
        
        # Processing parameters
        self.params = {
            'window_size': 256,
            'hop_length': 128,
            'n_fft': 512,
            'freq_min': 0.5,
            'freq_max': 50.0,
            'lowcut': 0.5,
            'highcut': 50.0,
            'notch_freq': 60.0,
            'batch_size': 32
        }
    
    @torch.cuda.amp.autocast()
    def process_batch(self, batch_tensor):
        """Process a batch of EEG signals with mixed precision."""
        
        # 1. Filtering
        filtered = self.gpu_filter_batch(batch_tensor)
        
        # 2. Quality check and artifact removal
        cleaned = self.remove_artifacts_gpu(filtered)
        
        # 3. Generate spectrograms
        spectrograms = self.generate_spectrograms_gpu(cleaned)
        
        # 4. Extract features
        features = self.extract_features_gpu(cleaned, spectrograms)
        
        return {
            'filtered': cleaned,
            'spectrograms': spectrograms,
            'features': features
        }
    
    def gpu_filter_batch(self, signals):
        """Apply filters using FFT on GPU."""
        # FFT-based filtering
        fft = torch.fft.fft(signals, dim=-1)
        freqs = torch.fft.fftfreq(signals.shape[-1], 1/self.sampling_rate).to(self.device)
        
        # Create filter mask
        bandpass = (torch.abs(freqs) >= self.params['lowcut']) & \
                   (torch.abs(freqs) <= self.params['highcut'])
        
        # Notch filter
        notch_width = 2.0
        notch = ~((torch.abs(freqs - self.params['notch_freq']) < notch_width/2) | 
                  (torch.abs(freqs + self.params['notch_freq']) < notch_width/2))
        
        # Apply combined filter
        mask = bandpass & notch
        fft *= mask.unsqueeze(0).unsqueeze(0)
        
        # Inverse FFT
        return torch.fft.ifft(fft, dim=-1).real
    
    def remove_artifacts_gpu(self, signals):
        """Remove artifacts using statistical thresholding."""
        # Calculate statistics across time dimension
        mean = signals.mean(dim=-1, keepdim=True)
        std = signals.std(dim=-1, keepdim=True)
        
        # Clip outliers
        threshold = 4.0
        clipped = torch.clamp(signals, 
                             mean - threshold * std,
                             mean + threshold * std)
        
        # Smooth transitions
        # Reshape for conv1d: [batch*channels, 1, time]
        batch_size, n_channels, time_len = clipped.shape
        reshaped = clipped.view(batch_size * n_channels, 1, time_len)
        
        kernel = torch.ones(1, 1, 5).to(self.device) / 5
        smoothed = F.conv1d(reshaped, kernel, padding=2)
        
        # Reshape back: [batch, channels, time]
        smoothed = smoothed.view(batch_size, n_channels, time_len)
        
        return smoothed
    
    def generate_spectrograms_gpu(self, signals):
        """Generate spectrograms using STFT."""
        batch_size, n_channels, time_len = signals.shape
        
        # Handle short signals
        if time_len < self.params['window_size']:
            # Pad signals that are too short
            pad_len = self.params['window_size'] - time_len
            signals = F.pad(signals, (0, pad_len), mode='replicate')
            time_len = signals.shape[-1]
        
        # Window for STFT
        window = torch.hann_window(self.params['window_size']).to(self.device)
        
        all_spectrograms = []
        
        # Process each sample in batch
        for i in range(batch_size):
            channel_specs = []
            
            for ch in range(n_channels):
                try:
                    # Compute STFT
                    stft = torch.stft(
                        signals[i, ch],
                        n_fft=self.params['n_fft'],
                        hop_length=self.params['hop_length'],
                        win_length=self.params['window_size'],
                        window=window,
                        return_complex=True
                    )
                    
                    # Power spectrum in dB
                    power = torch.abs(stft) ** 2
                    db_spec = 10 * torch.log10(power + 1e-10)
                    
                    # Frequency filtering
                    freqs = torch.linspace(0, self.sampling_rate/2, stft.shape[0]).to(self.device)
                    freq_mask = (freqs >= self.params['freq_min']) & (freqs <= self.params['freq_max'])
                    db_spec = db_spec[freq_mask, :]
                    
                    # Ensure consistent output size
                    if db_spec.shape[0] < 64:
                        db_spec = F.pad(db_spec, (0, 0, 0, 64 - db_spec.shape[0]))
                    else:
                        db_spec = db_spec[:64, :]
                    
                    if db_spec.shape[1] < 128:
                        db_spec = F.pad(db_spec, (0, 128 - db_spec.shape[1], 0, 0))
                    else:
                        db_spec = db_spec[:, :128]
                    
                    channel_specs.append(db_spec)
                    
                except Exception as e:
                    # Fallback for problematic signals
                    print(f"⚠️  STFT failed for signal {i}, channel {ch}: {str(e)[:50]}")
                    fallback_spec = torch.zeros(64, 128).to(self.device)
                    channel_specs.append(fallback_spec)
            
            all_spectrograms.append(torch.stack(channel_specs))
        
        return torch.stack(all_spectrograms)
    
    def extract_features_gpu(self, signals, spectrograms):
        """Extract comprehensive features on GPU."""
        batch_size = signals.shape[0]
        
        features = []
        
        # Time domain features
        features.extend([
            signals.mean(dim=-1),           # Mean
            signals.std(dim=-1),            # Std
            signals.var(dim=-1),            # Variance
            signals.min(dim=-1)[0],         # Min
            signals.max(dim=-1)[0],         # Max
            signals.abs().mean(dim=-1),     # Mean absolute
            (signals ** 2).mean(dim=-1),    # RMS
        ])
        
        # Zero crossing rate
        zero_cross = (torch.diff(torch.sign(signals)) != 0).sum(dim=-1).float() / signals.shape[-1]
        features.append(zero_cross)
        
        # Frequency features from spectrograms
        # Band powers
        n_freqs = spectrograms.shape[2]
        freqs = torch.linspace(self.params['freq_min'], self.params['freq_max'], n_freqs).to(self.device)
        
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        for band_name, (fmin, fmax) in bands.items():
            mask = (freqs >= fmin) & (freqs <= fmax)
            if mask.any():
                band_power = spectrograms[:, :, mask, :].mean(dim=(2, 3))
            else:
                band_power = torch.zeros(batch_size, spectrograms.shape[1]).to(self.device)
            features.append(band_power)
        
        # Spectral centroid
        power = 10 ** (spectrograms / 10)
        freq_weights = freqs.view(1, 1, -1, 1)
        centroid = (freq_weights * power).sum(dim=2) / (power.sum(dim=2) + 1e-10)
        features.append(centroid.mean(dim=-1))
        
        # Concatenate all features
        return torch.cat([f.view(batch_size, -1) for f in features], dim=1)


def process_data_gpu(data_path, output_path, max_samples=None):
    """Main processing function with ID mapping fix."""
    
    print("\n🚀 Starting GPU-accelerated preprocessing with ID mapping fix...")
    
    # Build file mapping first
    file_mapping = build_eeg_file_mapping(data_path)
    
    if not file_mapping:
        print("❌ Failed to build file mapping! Cannot proceed.")
        return pd.DataFrame()
    
    # Load metadata
    metadata = pd.read_csv(f"{data_path}/train.csv")
    
    # Filter to only mapped entries
    metadata['eeg_id_str'] = metadata['eeg_id'].astype(str)
    metadata_mapped = metadata[metadata['eeg_id_str'].isin(file_mapping.keys())]
    
    print(f"📊 Found {len(metadata_mapped)} mapped samples out of {len(metadata)} total")
    
    if max_samples:
        metadata_mapped = metadata_mapped.head(max_samples)
    
    print(f"📊 Processing {len(metadata_mapped)} samples")
    
    # Initialize processor
    processor = GPUBatchProcessor()
    
    # Create output directories
    import os
    for subdir in ['spectrograms', 'features', 'filtered']:
        os.makedirs(f"{output_path}/{subdir}", exist_ok=True)
    
    # Process in batches
    batch_size = processor.params['batch_size']
    n_batches = len(metadata_mapped) // batch_size + (1 if len(metadata_mapped) % batch_size else 0)
    
    results = []
    
    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        # Get batch metadata
        start = batch_idx * batch_size
        end = min(start + batch_size, len(metadata_mapped))
        batch_meta = metadata_mapped.iloc[start:end]
        
        # Load batch data
        batch_data = []
        valid_meta = []
        
        for _, row in batch_meta.iterrows():
            try:
                eeg_id = str(row['eeg_id'])
                file_path = file_mapping[eeg_id]
                
                # Load parquet file
                df = pd.read_parquet(file_path)
                
                # Get EEG channels (all numeric columns)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                exclude_cols = ['time', 'timestamp', 'index']
                eeg_cols = [col for col in numeric_cols if col not in exclude_cols]
                
                if len(eeg_cols) == 0:
                    print(f"⚠️  No EEG channels found in {file_path.name}")
                    continue
                
                eeg_data = df[eeg_cols].values.T  # Shape: (channels, time)
                
                # Limit to 20 channels
                if eeg_data.shape[0] > 20:
                    eeg_data = eeg_data[:20]
                
                batch_data.append(eeg_data)
                valid_meta.append(row)
                
            except Exception as e:
                print(f"Error loading {row['eeg_id']}: {str(e)[:100]}")
                continue
        
        if not batch_data:
            continue
        
        # Pad to same length
        max_len = max(data.shape[1] for data in batch_data)
        padded = []
        for data in batch_data:
            if data.shape[1] < max_len:
                pad_width = max_len - data.shape[1]
                data = np.pad(data, ((0, 0), (0, pad_width)), mode='edge')
            padded.append(data)
        
        # Convert to tensor
        batch_tensor = torch.tensor(np.stack(padded), dtype=torch.float32).to(processor.device)
        
        # Process batch
        try:
            with torch.cuda.amp.autocast():
                outputs = processor.process_batch(batch_tensor)
            
            # Save results
            for i, row in enumerate(valid_meta):
                file_id = str(row['eeg_id'])
                
                # Convert to numpy and save
                np.save(f"{output_path}/spectrograms/{file_id}_spec.npy", 
                       outputs['spectrograms'][i].cpu().numpy())
                np.save(f"{output_path}/features/{file_id}_feat.npy",
                       outputs['features'][i].cpu().numpy())
                np.save(f"{output_path}/filtered/{file_id}_filt.npy",
                       outputs['filtered'][i].cpu().numpy())
                
                results.append({
                    'file_id': file_id,
                    'label': row['expert_consensus'],
                    'patient_id': row['patient_id']
                })
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)[:100]}")
            continue
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    # Save summary
    summary_df = pd.DataFrame(results)
    if len(summary_df) > 0:
        summary_df.to_csv(f"{output_path}/processing_summary.csv", index=False)
    
    print(f"\n✅ Processing complete!")
    print(f"   Processed: {len(summary_df)} samples")
    print(f"   Saved to: {output_path}")
    
    return summary_df


# ============================
# CELL 5: Run Processing
# ============================

# Configuration
DATA_PATH = "/content/drive/MyDrive/hms-data"
OUTPUT_PATH = "/content/drive/MyDrive/hms-processed"

# Test with small subset first
TEST_SAMPLES = 50  # Set to None for all data

# Run this in a cell:
# summary = process_data_gpu(DATA_PATH, OUTPUT_PATH, max_samples=TEST_SAMPLES)


# ============================
# CELL 6: Visualization
# ============================

def visualize_results(output_path, sample_idx=0):
    """Visualize preprocessing results."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load summary
    summary = pd.read_csv(f"{output_path}/processing_summary.csv")
    
    if len(summary) == 0:
        print("No processed data found!")
        return
    
    # Get sample ID
    sample_id = summary.iloc[sample_idx]['file_id']
    
    # Load and visualize spectrogram
    spec = np.load(f"{output_path}/spectrograms/{sample_id}_spec.npy")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot first 4 channels
    for i in range(min(4, spec.shape[0])):
        ax = axes[i//2, i%2]
        im = ax.imshow(spec[i], aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'Channel {i+1} - {sample_id}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        plt.colorbar(im, ax=ax, label='Power (dB)')
    
    plt.tight_layout()
    plt.show()
    
    # Plot label distribution
    plt.figure(figsize=(10, 6))
    summary['label'].value_counts().plot(kind='bar')
    plt.title('Label Distribution in Processed Data')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Run this in a cell:
# visualize_results(OUTPUT_PATH)


# ============================
# CELL 7: Performance Monitoring
# ============================

def monitor_gpu():
    """Monitor GPU usage and performance."""
    import torch
    
    if torch.cuda.is_available():
        print("🖥️  GPU Performance Metrics:")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Allocated Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"   Cached Memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"   Free Memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9:.2f} GB")
        
        # Run nvidia-smi
        print("\n📊 NVIDIA-SMI Output:")
        os.system('nvidia-smi')
    else:
        print("No GPU available!")

# Run this in a cell:
# monitor_gpu()


# ============================
# ADDITIONAL UTILITIES
# ============================

def estimate_processing_time(n_samples, samples_per_second=2.5):
    """Estimate processing time based on GPU performance."""
    time_seconds = n_samples / samples_per_second
    hours = int(time_seconds // 3600)
    minutes = int((time_seconds % 3600) // 60)
    seconds = int(time_seconds % 60)
    
    print(f"⏱️  Estimated processing time for {n_samples} samples:")
    print(f"   {hours}h {minutes}m {seconds}s")
    print(f"   (assuming ~{samples_per_second} samples/second on GPU)")

# Example usage:
# estimate_processing_time(106800)  # Full dataset


def verify_preprocessing_outputs(output_path, n_samples=5):
    """Verify that preprocessing outputs are correct."""
    import os
    
    print("🔍 Verifying preprocessing outputs...")
    
    # Check directories
    subdirs = ['filtered', 'spectrograms', 'features']
    for subdir in subdirs:
        path = f"{output_path}/{subdir}"
        if os.path.exists(path):
            files = os.listdir(path)
            print(f"✅ {subdir}: {len(files)} files")
            
            # Check a few files
            for f in files[:n_samples]:
                file_path = f"{path}/{f}"
                data = np.load(file_path)
                print(f"   - {f}: shape {data.shape}")
        else:
            print(f"❌ {subdir} directory not found!")
    
    # Check summary
    summary_path = f"{output_path}/processing_summary.csv"
    if os.path.exists(summary_path):
        summary = pd.read_csv(summary_path)
        print(f"\n✅ Summary file: {len(summary)} entries")
        print(f"   Labels: {summary['label'].value_counts().to_dict()}")
    else:
        print(f"❌ Summary file not found!")

# Run this after preprocessing:
# verify_preprocessing_outputs(OUTPUT_PATH) 