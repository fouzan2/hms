#!/usr/bin/env python3
"""
Download HMS Harmful Brain Activity Classification dataset from Kaggle.
Downloads individual files since the data is stored as separate parquet files, not ZIP archives.
"""

import os
import sys
import time
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Set
import logging
import argparse
from tqdm import tqdm
import requests

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
DATASET_FOLDERS = [
    "train_eegs",
    "train_spectrograms", 
    "test_spectrograms",
    "test_eegs"
]

# Additional files to download
DATASET_FILES = [
    "train.csv",
    "test.csv",
    "sample_submission.csv"
]


class KaggleDatasetDownloader:
    """Download Kaggle dataset with proper handling for individual parquet files."""
    
    def __init__(self, output_dir: str = "data/raw", max_retries: int = 3):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.progress_file = self.output_dir / ".download_progress.json"
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
            
    def _get_available_files(self) -> List[str]:
        """Get list of all available files in the competition."""
        try:
            logger.info("Fetching available files from competition...")
            files_response = self.api.competition_list_files(COMPETITION_NAME)
            
            # Handle different response formats from Kaggle API
            files = []
            if hasattr(files_response, 'files'):
                files = files_response.files
            elif isinstance(files_response, list):
                files = files_response
            else:
                # Try to iterate directly
                try:
                    files = list(files_response)
                except:
                    logger.warning("Could not retrieve file list from Kaggle API")
                    return []
            
            file_names = []
            for file_info in files:
                if hasattr(file_info, 'name'):
                    file_names.append(file_info.name)
                elif isinstance(file_info, dict) and 'name' in file_info:
                    file_names.append(file_info['name'])
                else:
                    # Try to convert to string and use as filename
                    file_names.append(str(file_info))
                
            logger.info(f"Found {len(file_names)} files in competition")
            return file_names
            
        except Exception as e:
            logger.error(f"Could not retrieve file list: {e}")
            logger.info("Falling back to kaggle CLI for file list...")
            
            # Fallback: use kaggle CLI to get file list
            try:
                result = subprocess.run(
                    ['kaggle', 'competitions', 'files', COMPETITION_NAME, '--csv'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Parse CSV output
                file_names = []
                for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                    if line.strip():
                        filename = line.split(',')[0]  # First column is filename
                        file_names.append(filename)
                
                logger.info(f"Found {len(file_names)} files via kaggle CLI")
                return file_names
                
            except subprocess.CalledProcessError as cli_error:
                logger.error(f"Kaggle CLI also failed: {cli_error}")
                logger.info("Using minimal known file list...")
                return DATASET_FILES
            except Exception as cli_error:
                logger.error(f"Error parsing CLI output: {cli_error}")
                logger.info("Using minimal known file list...")
                return DATASET_FILES
        
    def _download_file_with_fallback(self, filename: str) -> bool:
        """Download file with fallback for API compatibility issues."""
        try:
            # Primary method using Kaggle API
            self.api.competition_download_file(
                COMPETITION_NAME,
                filename,
                path=str(self.output_dir),
                quiet=False
            )
            return True
        except TypeError as e:
            if "headers" in str(e):
                logger.warning(f"Kaggle API compatibility issue detected: {e}")
                logger.info("Trying alternative download method...")
                
                # Alternative method using direct download URL
                try:
                    import kaggle
                    # Get download URL without using problematic call method
                    url = f"https://www.kaggle.com/api/v1/competitions/data/download/{COMPETITION_NAME}/{filename}"
                    
                    # Use requests to download directly
                    response = requests.get(url, auth=(kaggle.api.get_default_api().username, kaggle.api.get_default_api().key))
                    if response.status_code == 200:
                        file_path = self.output_dir / filename
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                        logger.info(f"Successfully downloaded {filename} using fallback method")
                        return True
                    else:
                        logger.error(f"Fallback download failed with status: {response.status_code}")
                        return False
                        
                except Exception as fallback_error:
                    logger.error(f"Fallback download method also failed: {fallback_error}")
                    return False
            else:
                # Re-raise if it's not the headers issue
                raise e
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def _download_file(self, filename: str, retry_count: int = 0) -> bool:
        """Download a single file from the competition."""
        try:
            logger.info(f"Downloading {filename}...")
            
            # Create directory structure if needed
            file_path = self.output_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use the fallback-enabled download method
            success = self._download_file_with_fallback(filename)
            
            if not success:
                raise Exception("Download failed")
            
            # Verify the file exists and has content
            if file_path.exists() and file_path.stat().st_size > 0:
                logger.info(f"✅ Successfully downloaded {filename} ({file_path.stat().st_size/1024/1024:.2f} MB)")
                return True
            else:
                logger.error(f"Downloaded file {filename} is missing or empty")
                return False
            
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            
            if retry_count < self.max_retries:
                logger.info(f"Retrying download ({retry_count + 1}/{self.max_retries})...")
                time.sleep(5 * (retry_count + 1))  # Exponential backoff
                return self._download_file(filename, retry_count + 1)
                
            return False

    def _download_folder_files(self, folder: str, available_files: List[str]) -> bool:
        """Download all files in a specific folder."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading folder: {folder}")
        logger.info(f"{'='*60}")
        
        # Find all files in this folder
        folder_files = [f for f in available_files if f.startswith(f"{folder}/")]
        
        if not folder_files:
            logger.warning(f"No files found in folder {folder}")
            return False
            
        logger.info(f"Found {len(folder_files)} files in {folder}")
        
        # Create folder directory
        folder_path = self.output_dir / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Download each file
        success_count = 0
        for filename in tqdm(folder_files, desc=f"Downloading {folder}"):
            if self._download_file(filename):
                success_count += 1
            else:
                logger.error(f"Failed to download {filename}")
                
            # Small delay between downloads
            time.sleep(0.5)
        
        success_rate = success_count / len(folder_files)
        if success_rate >= 0.9:  # 90% success rate threshold
            logger.info(f"✅ Successfully downloaded {folder} ({success_count}/{len(folder_files)} files)")
            return True
        else:
            logger.warning(f"⚠️ Partial download for {folder} ({success_count}/{len(folder_files)} files)")
            return False
            
    def download_dataset(self, folders_only: Optional[List[str]] = None):
        """Download the complete dataset."""
        logger.info("Starting HMS dataset download...")
        
        # Get list of available files
        available_files = self._get_available_files()
        if not available_files:
            logger.error("Could not retrieve file list. Exiting.")
            return
        
        # Load progress
        progress = self._load_progress()
        
        # Determine what to download
        folders_to_download = folders_only or DATASET_FOLDERS
        files_to_download = [] if folders_only else DATASET_FILES
        
        # Download individual CSV files first
        logger.info("\n" + "="*60)
        logger.info("Downloading CSV files")
        logger.info("="*60)
        
        for filename in files_to_download:
            if progress.get(filename, False):
                logger.info(f"File {filename} already downloaded, skipping...")
                continue
                
            if filename in available_files:
                success = self._download_file(filename)
                if success:
                    progress[filename] = True
                    self._save_progress(progress)
                else:
                    logger.error(f"Failed to download {filename}")
            else:
                logger.warning(f"File {filename} not found in competition")
                
        # Download folder contents
        for folder in folders_to_download:
            if progress.get(folder, False):
                logger.info(f"Folder {folder} already downloaded, skipping...")
                continue
                
            success = self._download_folder_files(folder, available_files)
            if success:
                progress[folder] = True
                self._save_progress(progress)
            else:
                logger.error(f"Failed to download complete {folder}")
                
            # Add delay between large downloads
            time.sleep(2)
            
        # Final verification
        self._verify_download(progress, available_files)
        
    def _verify_download(self, progress: Dict[str, bool], available_files: List[str]):
        """Verify that all components were downloaded successfully."""
        logger.info("\n" + "="*60)
        logger.info("Download Summary")
        logger.info("="*60)
        
        all_items = DATASET_FOLDERS + DATASET_FILES
        downloaded = sum(1 for item in all_items if progress.get(item, False))
        
        logger.info(f"Downloaded: {downloaded}/{len(all_items)} components")
        
        # Check specific folders
        for folder in DATASET_FOLDERS:
            folder_path = self.output_dir / folder
            if folder_path.exists():
                folder_files = [f for f in available_files if f.startswith(f"{folder}/")]
                local_files = list(folder_path.glob('**/*'))
                local_file_count = len([f for f in local_files if f.is_file()])
                total_size_mb = sum(f.stat().st_size for f in local_files if f.is_file()) / (1024 * 1024)
                logger.info(f"  ✓ {folder}: {local_file_count}/{len(folder_files)} files, {total_size_mb:.2f} MB")
            else:
                logger.warning(f"  ✗ {folder}: Not found")
                
        # Check CSV files
        for filename in DATASET_FILES:
            file_path = self.output_dir / filename
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"  ✓ {filename}: {size_mb:.2f} MB")
            else:
                logger.warning(f"  ✗ {filename}: Not found")
                
        if downloaded == len(all_items):
            logger.info("\n🎉 Dataset download completed successfully!")
            
            # Clean up progress file
            if self.progress_file.exists():
                self.progress_file.unlink()
        else:
            logger.warning(f"\n⚠️  Some components failed to download. Run again to retry.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download HMS Harmful Brain Activity Classification dataset"
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Output directory for dataset (default: data/raw)"
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        choices=DATASET_FOLDERS,
        help="Download only specific folders"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing downloads"
    )
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = KaggleDatasetDownloader(output_dir=args.output_dir)
        
    # Verify only if requested
    if args.verify_only:
        progress = downloader._load_progress()
        available_files = downloader._get_available_files()
        downloader._verify_download(progress, available_files)
    else:
        # Download dataset
        downloader.download_dataset(folders_only=args.folders)
        
    logger.info("\nDownload script completed!")


if __name__ == "__main__":
    main() 