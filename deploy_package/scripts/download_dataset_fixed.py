#!/usr/bin/env python3
"""
Fixed HMS Harmful Brain Activity Classification dataset downloader.
Downloads individual parquet files since the data is stored as separate files, not ZIP archives.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import logging
import argparse
from tqdm import tqdm

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("Error: kaggle package not installed. Please run: pip install kaggle==1.5.16")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Competition and dataset info
COMPETITION_NAME = "hms-harmful-brain-activity-classification"

class FixedKaggleDatasetDownloader:
    """Download Kaggle dataset with proper handling for individual parquet files."""
    
    def __init__(self, output_dir: str = "data/raw", max_retries: int = 3, batch_size: int = 50):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.progress_file = self.output_dir / ".download_progress_fixed.json"
        self.api = None
        self._authenticate()
        
    def _authenticate(self):
        """Authenticate with Kaggle API."""
        try:
            self.api = KaggleApi()
            self.api.authenticate()
            logger.info("Successfully authenticated with Kaggle API")
        except Exception as e:
            logger.error(f"Failed to authenticate with Kaggle: {e}")
            logger.error("Please ensure your kaggle.json is in ~/.kaggle/")
            sys.exit(1)
            
    def _load_progress(self) -> Dict[str, bool]:
        """Load download progress from file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}
        
    def _save_progress(self, progress: Dict[str, bool]):
        """Save download progress to file."""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
            
    def _get_all_files(self) -> List[Dict[str, str]]:
        """Get list of all available files in the competition using kaggle CLI."""
        try:
            logger.info("Fetching complete file list from competition...")
            
            # Use kaggle CLI to get all files
            result = subprocess.run(
                ['kaggle', 'competitions', 'files', COMPETITION_NAME],
                capture_output=True,
                text=True,
                check=True
            )
            
            files = []
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            
            for line in lines:
                if line.strip():
                    # Parse the kaggle CLI output
                    parts = line.split()
                    if len(parts) >= 3:
                        filename = parts[0]
                        size = parts[1]
                        files.append({
                            'name': filename,
                            'size': size,
                            'type': self._classify_file(filename)
                        })
            
            logger.info(f"Found {len(files)} files in competition")
            return files
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get file list: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting file list: {e}")
            return []
    
    def _classify_file(self, filename: str) -> str:
        """Classify file type based on its path."""
        if filename.endswith('.csv'):
            return 'metadata'
        elif filename.startswith('train_eegs/'):
            return 'train_eegs'
        elif filename.startswith('train_spectrograms/'):
            return 'train_spectrograms'
        elif filename.startswith('test_eegs/'):
            return 'test_eegs'
        elif filename.startswith('test_spectrograms/'):
            return 'test_spectrograms'
        elif filename.startswith('example_figures/'):
            return 'example_figures'
        else:
            return 'other'
    
    def _download_file(self, filename: str, retry_count: int = 0) -> bool:
        """Download a single file from the competition."""
        file_path = self.output_dir / filename
        
        # Skip if already exists and has content
        if file_path.exists() and file_path.stat().st_size > 0:
            return True
            
        try:
            # Create directory structure if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download using kaggle CLI (more reliable for individual files)
            result = subprocess.run(
                ['kaggle', 'competitions', 'download', COMPETITION_NAME, 
                 '-f', filename, '-p', str(file_path.parent), '--force'],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Verify the file exists and has content
            if file_path.exists() and file_path.stat().st_size > 0:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.debug(f"✅ Downloaded {filename} ({size_mb:.2f} MB)")
                return True
            else:
                logger.error(f"Downloaded file {filename} is missing or empty")
                return False
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error downloading {filename}: {e.stderr}")
            
            if retry_count < self.max_retries:
                logger.info(f"Retrying download ({retry_count + 1}/{self.max_retries})...")
                time.sleep(2 * (retry_count + 1))  # Exponential backoff
                return self._download_file(filename, retry_count + 1)
                
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {filename}: {e}")
            return False

    def _download_file_group(self, files: List[Dict[str, str]], group_name: str, progress: Dict[str, bool]) -> int:
        """Download a group of files with progress tracking."""
        if not files:
            logger.info(f"No files found for {group_name}")
            return 0
            
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading {group_name}: {len(files)} files")
        logger.info(f"{'='*60}")
        
        success_count = 0
        
        # Process files in batches to avoid overwhelming the API
        for i in range(0, len(files), self.batch_size):
            batch = files[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(files) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} files)")
            
            batch_progress = tqdm(batch, desc=f"{group_name} batch {batch_num}")
            
            for file_info in batch_progress:
                filename = file_info['name']
                
                if progress.get(filename, False):
                    batch_progress.set_description(f"Skip {filename}")
                    success_count += 1
                    continue
                
                batch_progress.set_description(f"Downloading {filename}")
                
                if self._download_file(filename):
                    progress[filename] = True
                    success_count += 1
                else:
                    logger.warning(f"Failed to download {filename}")
                
                # Save progress after each file
                self._save_progress(progress)
                
                # Small delay to be respectful to the API
                time.sleep(0.2)
            
            # Longer delay between batches
            if i + self.batch_size < len(files):
                logger.info(f"Completed batch {batch_num}, pausing before next batch...")
                time.sleep(2)
        
        success_rate = success_count / len(files) if files else 0
        logger.info(f"Completed {group_name}: {success_count}/{len(files)} files ({success_rate*100:.1f}%)")
        
        return success_count
            
    def download_dataset(self, file_types: Optional[List[str]] = None):
        """Download the complete dataset."""
        logger.info("Starting HMS dataset download with individual file handling...")
        
        # Get list of all available files
        all_files = self._get_all_files()
        if not all_files:
            logger.error("Could not retrieve file list. Exiting.")
            return
        
        # Load progress
        progress = self._load_progress()
        
        # Group files by type
        file_groups = {}
        for file_info in all_files:
            file_type = file_info['type']
            if file_type not in file_groups:
                file_groups[file_type] = []
            file_groups[file_type].append(file_info)
        
        # Determine what to download
        if file_types:
            download_groups = {k: v for k, v in file_groups.items() if k in file_types}
        else:
            download_groups = file_groups
        
        # Show summary
        logger.info("\nDataset Summary:")
        total_files = 0
        for group_name, files in file_groups.items():
            status = "✓ Will download" if group_name in download_groups else "- Will skip"
            logger.info(f"  {status} {group_name}: {len(files)} files")
            if group_name in download_groups:
                total_files += len(files)
        
        logger.info(f"\nTotal files to download: {total_files}")
        
        # Download each group
        download_order = ['metadata', 'example_figures', 'train_eegs', 'train_spectrograms', 'test_eegs', 'test_spectrograms', 'other']
        total_downloaded = 0
        
        for group_name in download_order:
            if group_name in download_groups:
                downloaded = self._download_file_group(download_groups[group_name], group_name, progress)
                total_downloaded += downloaded
            
        # Final verification
        self._verify_download(progress, file_groups)
        
        logger.info(f"\nDownload completed! Successfully downloaded {total_downloaded}/{total_files} files")
        
    def _verify_download(self, progress: Dict[str, bool], file_groups: Dict[str, List[Dict[str, str]]]):
        """Verify that all components were downloaded successfully."""
        logger.info("\n" + "="*60)
        logger.info("Download Summary")
        logger.info("="*60)
        
        for group_name, files in file_groups.items():
            downloaded_count = sum(1 for f in files if progress.get(f['name'], False))
            total_count = len(files)
            
            if downloaded_count == total_count:
                # Calculate total size
                group_dir = self.output_dir / group_name.replace('_', '_').split('_')[0]
                if group_name.startswith('train_') or group_name.startswith('test_'):
                    group_dir = self.output_dir / group_name
                
                total_size = 0
                if group_dir.exists():
                    total_size = sum(f.stat().st_size for f in group_dir.rglob('*') if f.is_file())
                    total_size_mb = total_size / (1024 * 1024)
                    logger.info(f"  ✓ {group_name}: {downloaded_count}/{total_count} files ({total_size_mb:.1f} MB)")
                else:
                    logger.info(f"  ✓ {group_name}: {downloaded_count}/{total_count} files")
            else:
                logger.warning(f"  ⚠ {group_name}: {downloaded_count}/{total_count} files (incomplete)")
        
        # Check if all major components are complete
        critical_groups = ['metadata', 'train_eegs', 'train_spectrograms', 'test_eegs', 'test_spectrograms']
        complete_groups = []
        
        for group in critical_groups:
            if group in file_groups:
                files = file_groups[group]
                downloaded = sum(1 for f in files if progress.get(f['name'], False))
                if downloaded == len(files):
                    complete_groups.append(group)
        
        if len(complete_groups) == len([g for g in critical_groups if g in file_groups]):
            logger.info("\n🎉 Dataset download completed successfully!")
            
            # Clean up progress file
            if self.progress_file.exists():
                self.progress_file.unlink()
        else:
            logger.warning(f"\n⚠️  Some components failed to download. Run again to retry.")
            missing = [g for g in critical_groups if g in file_groups and g not in complete_groups]
            logger.warning(f"Incomplete: {missing}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download HMS Harmful Brain Activity Classification dataset (Fixed Version)"
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Output directory for dataset (default: data/raw)"
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=['metadata', 'train_eegs', 'train_spectrograms', 'test_eegs', 'test_spectrograms', 'example_figures', 'other'],
        help="Download only specific file types"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of files to download per batch (default: 50)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing downloads"
    )
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = FixedKaggleDatasetDownloader(
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
        
    # Verify only if requested
    if args.verify_only:
        progress = downloader._load_progress()
        all_files = downloader._get_all_files()
        file_groups = {}
        for file_info in all_files:
            file_type = file_info['type']
            if file_type not in file_groups:
                file_groups[file_type] = []
            file_groups[file_type].append(file_info)
        downloader._verify_download(progress, file_groups)
    else:
        # Download dataset
        downloader.download_dataset(file_types=args.types)
        
    logger.info("\nDownload script completed!")


if __name__ == "__main__":
    main() 