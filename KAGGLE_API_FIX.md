# Kaggle API Compatibility Fix & Data Structure Update

## Problem
The project was experiencing two issues:
1. **Kaggle API compatibility issue** with the error:
   ```
   TypeError: call() got an unexpected keyword argument 'headers'
   ```
2. **Data structure mismatch** - Script expected ZIP files but data is stored as individual Parquet files

## Solution

### 1. Fixed Kaggle API Version Issue ✅ 
- **Updated `requirements.txt`** to pin Kaggle API to version `1.5.16`
- This version is stable and doesn't have the `headers` compatibility issue
- **Added fallback mechanism** for remaining API compatibility issues

### 2. Updated Data Download Strategy ✅
- **Discovered data structure** - Competition uses individual `.parquet` files, not ZIP archives
- **Completely rewrote download script** to handle individual file downloads
- **Improved progress tracking** and verification for thousands of files

### 3. Enhanced Download Script Features
- **Real-time file discovery** - Fetches actual available files from competition
- **Parallel-friendly structure** for downloading thousands of parquet files  
- **Better error handling** and retry mechanisms
- **Progress tracking** with resumable downloads
- **Folder-by-folder verification** with detailed statistics

## Current Data Structure

The HMS competition contains:
```
📁 data/raw/
├── 📄 train.csv (metadata)
├── 📄 test.csv (metadata) 
├── 📄 sample_submission.csv
├── 📁 train_eegs/
│   ├── 1000913311.parquet
│   ├── 1001369401.parquet
│   └── ... (thousands of parquet files)
├── 📁 train_spectrograms/
│   ├── *.parquet files
├── 📁 test_eegs/
│   ├── 3911565283.parquet
│   └── ... (parquet files)
├── 📁 test_spectrograms/
│   ├── 853520.parquet  
│   └── ... (parquet files)
└── 📁 example_figures/
    ├── Sample01.pdf
    └── ... (example visualizations)
```

## How to Use

### Quick Fix
```bash
./fix_kaggle_download.sh
```

### Manual Steps
1. **Rebuild Docker container:**
   ```bash
   docker-compose build data-downloader
   ```

2. **Run download:**
   ```bash
   make download
   ```

### Download Options
```bash
# Download everything
make download

# Download specific folders only
docker-compose run --rm -v ~/.kaggle:/root/.kaggle:ro data-downloader --folders train_eegs

# Download multiple specific folders  
docker-compose run --rm -v ~/.kaggle:/root/.kaggle:ro data-downloader --folders train_eegs test_eegs

# Verify existing downloads
docker-compose run --rm -v ~/.kaggle:/root/.kaggle:ro data-downloader --verify-only
```

## Prerequisites
Ensure you have valid Kaggle credentials:
1. Go to https://www.kaggle.com/account
2. Click "Create New API Token"
3. Save `kaggle.json` to `~/.kaggle/`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`
5. **Accept competition rules** at: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification

## What Was Changed

### requirements.txt
```diff
- kaggle
+ kaggle==1.5.16
```

### scripts/download_dataset.py - Major Rewrite
- ✅ **Removed ZIP file logic** - No longer tries to download non-existent ZIP files
- ✅ **Added individual file discovery** - Uses Kaggle API to fetch real file list
- ✅ **Implemented folder-based downloads** - Downloads all parquet files within each folder
- ✅ **Enhanced progress tracking** - Shows download progress per folder with file counts
- ✅ **Better verification** - Compares local vs remote file counts
- ✅ **Improved error handling** - Handles API compatibility issues gracefully
- ✅ **Directory structure creation** - Automatically creates proper folder hierarchy

### New Files
- `fix_kaggle_download.sh` - Automated fix script
- `KAGGLE_API_FIX.md` - This updated documentation

## Expected Download Output
After running the fix, you should see:
```
🎉 Dataset download completed successfully!
```

With detailed statistics like:
```
Download Summary
============================================================
Downloaded: 7/7 components
  ✓ train_eegs: 1234/1234 files, 2.45 GB
  ✓ train_spectrograms: 5678/5678 files, 1.23 GB  
  ✓ test_eegs: 890/890 files, 456.78 MB
  ✓ test_spectrograms: 567/567 files, 234.56 MB
  ✓ train.csv: 1.66 MB
  ✓ test.csv: 0.06 MB
  ✓ sample_submission.csv: 0.20 MB
```

## Performance Notes
- **Large dataset**: Thousands of individual parquet files
- **Download time**: Expect 30-60 minutes depending on connection
- **Storage space**: ~4-5 GB total
- **Resumable**: Script tracks progress and can resume interrupted downloads

## Troubleshooting
If you encounter issues:

1. **Check Kaggle access:**
   ```bash
   kaggle competitions list
   ```

2. **Verify competition access:**
   - Ensure you've accepted the competition rules
   - Check your API quota hasn't been exceeded

3. **Test single folder download:**
   ```bash
   docker-compose run --rm -v ~/.kaggle:/root/.kaggle:ro data-downloader --folders train_eegs
   ```

4. **Check container logs:**
   ```bash
   docker-compose logs data-downloader
   ```

5. **Manual verification:**
   ```bash
   docker-compose run --rm -v ~/.kaggle:/root/.kaggle:ro data-downloader --verify-only
   ```

## Technical Details
- **API Version**: Fixed at kaggle==1.5.16 for stability
- **Download Strategy**: Individual file downloads with folder organization
- **Error Handling**: Automatic retries with exponential backoff
- **Progress Tracking**: JSON-based progress file for resumability
- **File Discovery**: Dynamic fetching of available files from competition 