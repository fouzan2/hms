#!/usr/bin/env python3
"""
Download HMS Harmful Brain Activity Classification dataset from Kaggle.

This module handles:
- Folder-by-folder download for large datasets
- Progress tracking and resumable downloads
- Automatic retry on failures
- Data integrity verification
"""

import os
import sys
import json
import time
import hashlib
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)


class KaggleDatasetDownloader:
    """Handle Kaggle dataset download with folder-by-folder approach."""
    
    def __init__(self, competition_name: str, output_dir: str, batch_size: int = 10):
        self.competition_name = competition_name
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.raw_dir = self.output_dir / "kaggle"
        self.raw_dir.mkdir(exist_ok=True)
        
        # Progress tracking
        self.progress_file = self.output_dir / "download_progress.json"
        self.progress = self._load_progress()
        
    def _load_progress(self) -> Dict[str, any]:
        """Load download progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "downloaded_files": [],
            "failed_files": [],
            "total_size_mb": 0,
            "download_timestamp": {}
        }
    
    def _save_progress(self):
        """Save download progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def _check_kaggle_auth(self) -> bool:
        """Check if Kaggle API is properly configured."""
        try:
            result = subprocess.run(
                ["kaggle", "config", "view"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info("Kaggle API authenticated ✓")
                return True
            else:
                logger.error("Kaggle API not configured properly")
                return False
        except FileNotFoundError:
            logger.error("Kaggle CLI not found. Install with: pip install kaggle")
            return False
    
    def _get_file_list(self) -> List[Dict[str, any]]:
        """Get list of files in the competition."""
        logger.info(f"Fetching file list for {self.competition_name}...")
        
        try:
            # Get file list using Kaggle API
            result = subprocess.run(
                ["kaggle", "competitions", "files", self.competition_name, "-v"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"Failed to get file list: {result.stderr}")
            
            # Parse the output
            files = []
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 2:
                        filename = parts[0]
                        size = parts[1]
                        files.append({
                            'name': filename,
                            'size': size,
                            'downloaded': filename in self.progress['downloaded_files']
                        })
            
            return files
            
        except Exception as e:
            logger.error(f"Error getting file list: {str(e)}")
            return []
    
    def _download_file(self, filename: str, retry_count: int = 3) -> bool:
        """Download a single file with retry logic."""
        if filename in self.progress['downloaded_files']:
            logger.info(f"Skipping {filename} (already downloaded)")
            return True
        
        for attempt in range(retry_count):
            try:
                logger.info(f"Downloading {filename} (attempt {attempt + 1}/{retry_count})...")
                
                # Download using Kaggle CLI
                result = subprocess.run(
                    [
                        "kaggle", "competitions", "download",
                        self.competition_name,
                        "-f", filename,
                        "-p", str(self.raw_dir)
                    ],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    # Extract if it's a zip file
                    file_path = self.raw_dir / filename
                    if file_path.suffix == '.zip':
                        logger.info(f"Extracting {filename}...")
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(self.raw_dir)
                        # Remove zip file to save space
                        file_path.unlink()
                    
                    # Update progress
                    self.progress['downloaded_files'].append(filename)
                    self.progress['download_timestamp'][filename] = time.time()
                    self._save_progress()
                    
                    logger.info(f"Successfully downloaded {filename} ✓")
                    return True
                else:
                    logger.error(f"Download failed: {result.stderr}")
                    
            except Exception as e:
                logger.error(f"Error downloading {filename}: {str(e)}")
            
            if attempt < retry_count - 1:
                time.sleep(5)  # Wait before retry
        
        # Mark as failed after all retries
        if filename not in self.progress['failed_files']:
            self.progress['failed_files'].append(filename)
            self._save_progress()
        
        return False
    
    def _organize_downloaded_files(self):
        """Organize downloaded files into proper directory structure."""
        logger.info("Organizing downloaded files...")
        
        # Expected file patterns
        file_patterns = {
            'train_eegs': self.output_dir / 'train_eegs',
            'train_spectrograms': self.output_dir / 'train_spectrograms',
            'test_spectrograms': self.output_dir / 'test_spectrograms'
        }
        
        # Create directories
        for pattern, target_dir in file_patterns.items():
            target_dir.mkdir(exist_ok=True)
            
            # Move matching files
            for file_path in self.raw_dir.glob(f"*{pattern}*"):
                if file_path.is_file():
                    target_path = target_dir / file_path.name
                    if not target_path.exists():
                        file_path.rename(target_path)
                        logger.info(f"Moved {file_path.name} to {target_dir}")
        
        # Move metadata files
        for meta_file in ['train.csv', 'test.csv', 'sample_submission.csv']:
            source = self.raw_dir / meta_file
            if source.exists():
                target = self.output_dir / meta_file
                if not target.exists():
                    source.rename(target)
                    logger.info(f"Moved {meta_file} to {self.output_dir}")
    
    def _verify_download_integrity(self) -> Tuple[bool, List[str]]:
        """Verify the integrity of downloaded files."""
        logger.info("Verifying download integrity...")
        
        missing_files = []
        corrupted_files = []
        
        # Check metadata files
        required_metadata = ['train.csv', 'test.csv']
        for meta_file in required_metadata:
            file_path = self.output_dir / meta_file
            if not file_path.exists():
                missing_files.append(meta_file)
            else:
                try:
                    # Try to read CSV to check integrity
                    pd.read_csv(file_path, nrows=5)
                except Exception:
                    corrupted_files.append(meta_file)
        
        # Check data directories
        data_dirs = ['train_eegs', 'train_spectrograms', 'test_spectrograms']
        for data_dir in data_dirs:
            dir_path = self.output_dir / data_dir
            if not dir_path.exists() or not any(dir_path.iterdir()):
                missing_files.append(data_dir)
        
        is_valid = len(missing_files) == 0 and len(corrupted_files) == 0
        
        if not is_valid:
            logger.error(f"Missing files: {missing_files}")
            logger.error(f"Corrupted files: {corrupted_files}")
        else:
            logger.info("All files verified successfully ✓")
        
        return is_valid, missing_files + corrupted_files
    
    def download_dataset(self) -> bool:
        """Download the complete dataset folder by folder."""
        logger.info(f"Starting download of {self.competition_name} dataset...")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Check Kaggle authentication
        if not self._check_kaggle_auth():
            logger.error("Please configure Kaggle API credentials first.")
            logger.error("Visit: https://www.kaggle.com/docs/api")
            return False
        
        # Get file list
        files = self._get_file_list()
        if not files:
            logger.error("No files found or unable to get file list")
            return False
        
        logger.info(f"Found {len(files)} files in the competition")
        
        # Group files by type for folder-by-folder download
        file_groups = {
            'metadata': [],
            'train_eegs': [],
            'train_spectrograms': [],
            'test_spectrograms': [],
            'other': []
        }
        
        for file_info in files:
            filename = file_info['name']
            if filename.endswith('.csv'):
                file_groups['metadata'].append(file_info)
            elif 'train_eegs' in filename:
                file_groups['train_eegs'].append(file_info)
            elif 'train_spectrograms' in filename:
                file_groups['train_spectrograms'].append(file_info)
            elif 'test_spectrograms' in filename:
                file_groups['test_spectrograms'].append(file_info)
            else:
                file_groups['other'].append(file_info)
        
        # Download files group by group
        download_order = ['metadata', 'train_eegs', 'train_spectrograms', 
                         'test_spectrograms', 'other']
        
        total_files = len(files)
        downloaded_count = len(self.progress['downloaded_files'])
        
        with tqdm(total=total_files, initial=downloaded_count, 
                 desc="Overall Progress") as pbar:
            
            for group_name in download_order:
                group_files = file_groups[group_name]
                if not group_files:
                    continue
                
                logger.info(f"\nDownloading {group_name} files ({len(group_files)} files)...")
                
                # Download files in batches
                for i in range(0, len(group_files), self.batch_size):
                    batch = group_files[i:i + self.batch_size]
                    logger.info(f"Processing batch {i//self.batch_size + 1} "
                              f"({len(batch)} files)")
                    
                    for file_info in batch:
                        if not file_info['downloaded']:
                            success = self._download_file(file_info['name'])
                            if success:
                                pbar.update(1)
                            else:
                                logger.warning(f"Failed to download {file_info['name']}")
                    
                    # Brief pause between batches
                    time.sleep(2)
        
        # Organize files
        self._organize_downloaded_files()
        
        # Verify integrity
        is_valid, problematic_files = self._verify_download_integrity()
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("Download Summary:")
        logger.info(f"Total files: {total_files}")
        logger.info(f"Downloaded: {len(self.progress['downloaded_files'])}")
        logger.info(f"Failed: {len(self.progress['failed_files'])}")
        
        if self.progress['failed_files']:
            logger.warning(f"Failed files: {self.progress['failed_files']}")
            logger.info("To retry failed downloads, run the download command again.")
        
        if is_valid:
            logger.info("✅ Dataset download completed successfully!")
            return True
        else:
            logger.error("❌ Dataset download incomplete or corrupted")
            logger.error(f"Problematic files: {problematic_files}")
            return False
    
    def get_download_stats(self) -> Dict[str, any]:
        """Get download statistics."""
        stats = {
            'total_downloaded': len(self.progress['downloaded_files']),
            'failed_downloads': len(self.progress['failed_files']),
            'data_size_mb': 0
        }
        
        # Calculate total size
        for data_dir in ['train_eegs', 'train_spectrograms', 'test_spectrograms']:
            dir_path = self.output_dir / data_dir
            if dir_path.exists():
                size_bytes = sum(f.stat().st_size for f in dir_path.rglob('*') 
                               if f.is_file())
                stats['data_size_mb'] += size_bytes / (1024 * 1024)
        
        return stats


def download_dataset(output_dir: str, dataset_name: str, batch_size: int = 10) -> bool:
    """
    Download HMS dataset from Kaggle.
    
    Args:
        output_dir: Directory to save the dataset
        dataset_name: Kaggle competition name
        batch_size: Number of files to download in each batch
        
    Returns:
        bool: True if download successful, False otherwise
    """
    downloader = KaggleDatasetDownloader(dataset_name, output_dir, batch_size)
    return downloader.download_dataset()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Download HMS dataset from Kaggle")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                       help="Output directory for dataset")
    parser.add_argument("--competition", type=str, 
                       default="hms-harmful-brain-activity-classification",
                       help="Kaggle competition name")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Number of files to download per batch")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Download dataset
    success = download_dataset(
        output_dir=args.output_dir,
        dataset_name=args.competition,
        batch_size=args.batch_size
    )
    
    sys.exit(0 if success else 1) 