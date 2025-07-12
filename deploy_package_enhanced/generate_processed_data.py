#!/usr/bin/env python3
"""
Enterprise-Level Processed Data Generation Script
Handles disk space issues and generates final processed data from temporary batch files
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import argparse
import psutil
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import gc
import time
from tqdm import tqdm
import yaml
import json
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/processed_data_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DataGenerationConfig:
    """Configuration for data generation process"""
    processed_data_path: Path
    temp_batch_pattern: str = "temp_batch_*.npz"
    min_disk_space_gb: float = 10.0
    max_memory_usage_gb: float = 50.0
    batch_size: int = 1000
    compression_level: int = 1  # Lower compression for speed
    cleanup_temp_files: bool = True
    verify_data_integrity: bool = True
    backup_enabled: bool = False  # Disabled by default to save space
    backup_path: Optional[Path] = None

class DiskSpaceManager:
    """Manages disk space and provides cleanup utilities"""
    
    def __init__(self, min_space_gb: float = 10.0):
        self.min_space_gb = min_space_gb
        
    def get_available_space_gb(self, path: Path) -> float:
        """Get available disk space in GB"""
        try:
            stat = shutil.disk_usage(path)
            return stat.free / (1024**3)
        except Exception as e:
            logger.warning(f"Could not get disk space for {path}: {e}")
            return 0.0
    
    def check_disk_space(self, path: Path) -> bool:
        """Check if sufficient disk space is available"""
        available_gb = self.get_available_space_gb(path)
        logger.info(f"Available disk space: {available_gb:.2f} GB")
        
        if available_gb < self.min_space_gb:
            logger.error(f"Insufficient disk space: {available_gb:.2f} GB < {self.min_space_gb} GB")
            return False
        return True
    
    def cleanup_temp_files(self, temp_dir: Path, pattern: str = "temp_batch_*.npz") -> int:
        """Clean up temporary files and return number of files removed"""
        removed_count = 0
        try:
            for temp_file in temp_dir.glob(pattern):
                try:
                    temp_file.unlink()
                    removed_count += 1
                    logger.debug(f"Removed temp file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove {temp_file}: {e}")
        except Exception as e:
            logger.error(f"Error during temp file cleanup: {e}")
        
        logger.info(f"Cleaned up {removed_count} temporary files")
        return removed_count
    
    def get_largest_files(self, directory: Path, n: int = 10) -> List[Tuple[Path, int]]:
        """Get the largest files in a directory"""
        files = []
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    try:
                        size = file_path.stat().st_size
                        files.append((file_path, size))
                    except Exception:
                        continue
            
            # Sort by size (largest first) and return top n
            files.sort(key=lambda x: x[1], reverse=True)
            return files[:n]
        except Exception as e:
            logger.error(f"Error getting largest files: {e}")
            return []

class MemoryManager:
    """Manages memory usage during data processing"""
    
    def __init__(self, max_usage_gb: float = 50.0):
        self.max_usage_gb = max_usage_gb
        
    def get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024**3)
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return 0.0
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits"""
        usage_gb = self.get_memory_usage_gb()
        logger.info(f"Current memory usage: {usage_gb:.2f} GB")
        
        if usage_gb > self.max_usage_gb:
            logger.warning(f"High memory usage: {usage_gb:.2f} GB > {self.max_usage_gb} GB")
            return False
        return True
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory"""
        gc.collect()
        logger.debug("Forced garbage collection")

class DataIntegrityChecker:
    """Checks data integrity and consistency"""
    
    @staticmethod
    def verify_array_consistency(arrays: List[np.ndarray], name: str) -> bool:
        """Verify that all arrays have consistent shapes"""
        if not arrays:
            logger.error(f"No {name} arrays to verify")
            return False
        
        shapes = [arr.shape for arr in arrays]
        unique_shapes = set(shapes)
        
        if len(unique_shapes) > 1:
            logger.error(f"Inconsistent {name} shapes: {unique_shapes}")
            return False
        
        logger.info(f"All {name} arrays have consistent shape: {shapes[0]}")
        return True
    
    @staticmethod
    def check_for_nan_inf(arrays: List[np.ndarray], name: str) -> bool:
        """Check for NaN or infinite values"""
        for i, arr in enumerate(arrays):
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                logger.error(f"Found NaN/Inf values in {name} array {i}")
                return False
        
        logger.info(f"No NaN/Inf values found in {name} arrays")
        return True
    
    @staticmethod
    def verify_data_types(arrays: List[np.ndarray], expected_dtype: np.dtype, name: str) -> bool:
        """Verify data types are consistent"""
        for i, arr in enumerate(arrays):
            if arr.dtype != expected_dtype:
                logger.warning(f"{name} array {i} has dtype {arr.dtype}, expected {expected_dtype}")
                return False
        
        logger.info(f"All {name} arrays have correct dtype: {expected_dtype}")
        return True

class ProcessedDataGenerator:
    """Main class for generating processed data from temporary batch files"""
    
    def __init__(self, config: DataGenerationConfig):
        self.config = config
        self.disk_manager = DiskSpaceManager(config.min_disk_space_gb)
        self.memory_manager = MemoryManager(config.max_memory_usage_gb)
        self.integrity_checker = DataIntegrityChecker()
        
        # Ensure directories exist
        self.config.processed_data_path.mkdir(parents=True, exist_ok=True)
        if self.config.backup_path:
            self.config.backup_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
    def find_temp_batch_files(self) -> List[Path]:
        """Find all temporary batch files"""
        temp_files = list(self.config.processed_data_path.glob(self.config.temp_batch_pattern))
        temp_files.sort(key=lambda x: int(x.stem.split('_')[-1]))  # Sort by batch number
        
        logger.info(f"Found {len(temp_files)} temporary batch files")
        return temp_files
    
    def load_temp_batch(self, file_path: Path) -> Dict[str, List]:
        """Load a single temporary batch file with proper error handling"""
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Safely extract data with proper type checking
            batch_data = {
                'eeg': [],
                'spectrograms': [],
                'labels': []
            }
            
            # Handle EEG data
            if 'eeg' in data.files:
                eeg_data = data['eeg']
                if isinstance(eeg_data, np.ndarray):
                    if len(eeg_data) > 0:
                        batch_data['eeg'] = list(eeg_data)
                    else:
                        batch_data['eeg'] = []
                else:
                    batch_data['eeg'] = []
            
            # Handle spectrogram data
            if 'spectrograms' in data.files:
                spec_data = data['spectrograms']
                if isinstance(spec_data, np.ndarray):
                    if len(spec_data) > 0:
                        batch_data['spectrograms'] = list(spec_data)
                    else:
                        batch_data['spectrograms'] = []
                else:
                    batch_data['spectrograms'] = []
            
            # Handle labels
            if 'labels' in data.files:
                label_data = data['labels']
                if isinstance(label_data, np.ndarray):
                    if len(label_data) > 0:
                        batch_data['labels'] = list(label_data)
                    else:
                        batch_data['labels'] = []
                else:
                    batch_data['labels'] = []
            
            return batch_data
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return {'eeg': [], 'spectrograms': [], 'labels': []}
    
    def combine_batches_incremental(self, temp_files: List[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Combine batches incrementally to manage memory"""
        logger.info("Starting incremental batch combination...")
        
        all_eeg_data = []
        all_spectrograms = []
        all_labels = []
        
        # Process batches with progress bar
        for i, temp_file in enumerate(tqdm(temp_files, desc="Loading batches")):
            try:
                # Check memory usage
                if not self.memory_manager.check_memory_usage():
                    logger.warning("High memory usage detected, forcing garbage collection")
                    self.memory_manager.force_garbage_collection()
                
                # Load batch
                batch_data = self.load_temp_batch(temp_file)
                
                # Safely extend lists with proper length checking
                if batch_data['eeg'] and len(batch_data['eeg']) > 0:
                    all_eeg_data.extend(batch_data['eeg'])
                if batch_data['spectrograms'] and len(batch_data['spectrograms']) > 0:
                    all_spectrograms.extend(batch_data['spectrograms'])
                if batch_data['labels'] and len(batch_data['labels']) > 0:
                    all_labels.extend(batch_data['labels'])
                
                # Periodic memory cleanup
                if (i + 1) % 10 == 0:
                    self.memory_manager.force_garbage_collection()
                    
            except Exception as e:
                logger.error(f"Error processing batch {temp_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(all_eeg_data)} EEG samples, {len(all_spectrograms)} spectrograms, {len(all_labels)} labels")
        
        # Convert to numpy arrays with proper error handling
        try:
            if len(all_eeg_data) > 0:
                eeg_array = np.array(all_eeg_data)
                logger.info(f"EEG array shape: {eeg_array.shape}")
            else:
                logger.error("No EEG data loaded")
                raise ValueError("No EEG data available")
                
            if len(all_spectrograms) > 0:
                spectrogram_array = np.array(all_spectrograms)
                logger.info(f"Spectrogram array shape: {spectrogram_array.shape}")
            else:
                logger.error("No spectrogram data loaded")
                raise ValueError("No spectrogram data available")
                
            if len(all_labels) > 0:
                labels_array = np.array(all_labels)
                logger.info(f"Labels array shape: {labels_array.shape}")
            else:
                logger.error("No labels loaded")
                raise ValueError("No labels available")
            
            logger.info(f"Final array shapes - EEG: {eeg_array.shape}, Spectrograms: {spectrogram_array.shape}, Labels: {labels_array.shape}")
            
            return eeg_array, spectrogram_array, labels_array
            
        except Exception as e:
            logger.error(f"Failed to convert to numpy arrays: {e}")
            raise
    
    def normalize_array_shapes(self, arrays: List[np.ndarray], name: str) -> List[np.ndarray]:
        """Normalize array shapes to ensure consistency"""
        if not arrays:
            return []
        
        # Find the most common shape
        shapes = [arr.shape for arr in arrays]
        shape_counts = {}
        for shape in shapes:
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        
        target_shape = max(shape_counts.items(), key=lambda x: x[1])[0]
        logger.info(f"Normalizing {name} arrays to shape: {target_shape}")
        
        normalized_arrays = []
        for i, arr in enumerate(arrays):
            if arr.shape != target_shape:
                # Create new array with target shape
                normalized_arr = np.zeros(target_shape, dtype=arr.dtype)
                # Copy as much as possible
                slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(arr.shape, target_shape))
                normalized_arr[slices] = arr[slices]
                normalized_arrays.append(normalized_arr)
                logger.debug(f"Normalized {name} array {i} from {arr.shape} to {target_shape}")
            else:
                normalized_arrays.append(arr)
        
        return normalized_arrays
    
    def clean_nan_inf_values(self, eeg_array: np.ndarray, spectrogram_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Clean NaN and Inf values from arrays using advanced interpolation methods"""
        logger.info("Cleaning NaN/Inf values from data using advanced interpolation...")
        
        # Clean EEG data with interpolation
        eeg_cleaned = eeg_array.copy()
        nan_count_eeg = np.isnan(eeg_cleaned).sum()
        inf_count_eeg = np.isinf(eeg_cleaned).sum()
        
        if nan_count_eeg > 0 or inf_count_eeg > 0:
            logger.warning(f"Found {nan_count_eeg} NaN and {inf_count_eeg} Inf values in EEG data, using interpolation...")
            
            # Replace inf with nan first
            eeg_cleaned = np.where(np.isinf(eeg_cleaned), np.nan, eeg_cleaned)
            
            # Clean each sample individually
            for sample_idx in range(eeg_cleaned.shape[0]):
                for channel_idx in range(eeg_cleaned.shape[1]):
                    channel_data = eeg_cleaned[sample_idx, channel_idx, :]
                    
                    if np.any(np.isnan(channel_data)):
                        # Use efficient interpolation without scipy
                        channel_data = self._interpolate_channel(channel_data)
                        eeg_cleaned[sample_idx, channel_idx, :] = channel_data
        
        # Clean spectrogram data (use simpler method for images)
        spec_cleaned = spectrogram_array.copy()
        nan_count_spec = np.isnan(spec_cleaned).sum()
        inf_count_spec = np.isinf(spec_cleaned).sum()
        
        if nan_count_spec > 0 or inf_count_spec > 0:
            logger.warning(f"Found {nan_count_spec} NaN and {inf_count_spec} Inf values in spectrogram data, using median replacement...")
            
            # Replace inf with nan first
            spec_cleaned = np.where(np.isinf(spec_cleaned), np.nan, spec_cleaned)
            
            # For spectrograms, use median of valid pixels in the same position across samples
            for i in range(spec_cleaned.shape[1]):  # height
                for j in range(spec_cleaned.shape[2]):  # width
                    for k in range(spec_cleaned.shape[3]):  # channels
                        pixel_values = spec_cleaned[:, i, j, k]
                        valid_pixels = ~np.isnan(pixel_values)
                        
                        if np.any(valid_pixels):
                            # Use median of valid pixels
                            median_val = np.median(pixel_values[valid_pixels])
                            spec_cleaned[:, i, j, k] = np.where(np.isnan(pixel_values), 
                                                              median_val, 
                                                              pixel_values)
                        else:
                            # If no valid pixels, use 0
                            spec_cleaned[:, i, j, k] = 0
            
            # Ensure uint8 type
            spec_cleaned = spec_cleaned.astype(np.uint8)
        
        logger.info("✅ NaN/Inf values cleaned successfully using advanced interpolation")
        return eeg_cleaned, spec_cleaned
    
    def _interpolate_channel(self, channel_data: np.ndarray) -> np.ndarray:
        """Efficient channel interpolation without scipy dependency"""
        if not np.any(np.isnan(channel_data)):
            return channel_data
        
        # Get valid indices
        valid_indices = ~np.isnan(channel_data)
        
        if np.sum(valid_indices) == 0:
            # If no valid data, use zeros
            return np.zeros_like(channel_data)
        
        if np.sum(valid_indices) == 1:
            # If only one valid point, use that value
            valid_value = channel_data[valid_indices][0]
            return np.full_like(channel_data, valid_value)
        
        # Get valid data and positions
        valid_data = channel_data[valid_indices]
        valid_positions = np.where(valid_indices)[0]
        
        # Create interpolated array
        interpolated = channel_data.copy()
        
        # Handle edge cases (before first valid point)
        if valid_positions[0] > 0:
            interpolated[:valid_positions[0]] = valid_data[0]
        
        # Handle edge cases (after last valid point)
        if valid_positions[-1] < len(channel_data) - 1:
            interpolated[valid_positions[-1]+1:] = valid_data[-1]
        
        # Interpolate gaps between valid points
        for i in range(len(valid_positions) - 1):
            start_pos = valid_positions[i]
            end_pos = valid_positions[i + 1]
            start_val = valid_data[i]
            end_val = valid_data[i + 1]
            
            if end_pos > start_pos + 1:
                # Linear interpolation for the gap
                gap_size = end_pos - start_pos
                for j in range(start_pos + 1, end_pos):
                    alpha = (j - start_pos) / gap_size
                    interpolated[j] = start_val * (1 - alpha) + end_val * alpha
        
        return interpolated
    
    def _forward_backward_fill(self, data: np.ndarray) -> np.ndarray:
        """Forward fill then backward fill for handling edge cases"""
        # Forward fill
        mask = np.isnan(data)
        idx = np.where(~mask, np.arange(mask.shape[0])[:, None], 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        data = data[idx, np.arange(data.shape[1])]
        
        # Backward fill for remaining NaN
        mask = np.isnan(data)
        idx = np.where(~mask, np.arange(mask.shape[0])[:, None], mask.shape[0] - 1)
        idx = np.minimum.accumulate(idx[::-1], axis=0)[::-1]
        data = data[idx, np.arange(data.shape[1])]
        
        # Replace any remaining NaN with 0
        data = np.where(np.isnan(data), 0, data)
        return data
    
    def verify_data_integrity(self, eeg_array: np.ndarray, spectrogram_array: np.ndarray, labels_array: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        """Verify data integrity before saving and return cleaned arrays"""
        if not self.config.verify_data_integrity:
            return True, eeg_array, spectrogram_array
        
        logger.info("Verifying data integrity...")
        
        # Check array lengths
        if len(eeg_array) != len(spectrogram_array) or len(eeg_array) != len(labels_array):
            logger.error(f"Array length mismatch: EEG={len(eeg_array)}, Spectrograms={len(spectrogram_array)}, Labels={len(labels_array)}")
            return False, eeg_array, spectrogram_array
        
        # Check for NaN/Inf values and clean them
        if np.any(np.isnan(eeg_array)) or np.any(np.isinf(eeg_array)):
            logger.warning("Found NaN/Inf values in EEG data, cleaning...")
            eeg_array, spectrogram_array = self.clean_nan_inf_values(eeg_array, spectrogram_array)
        
        if np.any(np.isnan(spectrogram_array)) or np.any(np.isinf(spectrogram_array)):
            logger.warning("Found NaN/Inf values in spectrogram data, cleaning...")
            eeg_array, spectrogram_array = self.clean_nan_inf_values(eeg_array, spectrogram_array)
        
        # Final check after cleaning
        if np.any(np.isnan(eeg_array)) or np.any(np.isinf(eeg_array)):
            logger.error("Still found NaN/Inf values in EEG data after cleaning")
            return False, eeg_array, spectrogram_array
        
        if np.any(np.isnan(spectrogram_array)) or np.any(np.isinf(spectrogram_array)):
            logger.error("Still found NaN/Inf values in spectrogram data after cleaning")
            return False, eeg_array, spectrogram_array
        
        # Check data types
        if eeg_array.dtype != np.float32:
            logger.warning(f"EEG data type is {eeg_array.dtype}, expected float32")
        
        if spectrogram_array.dtype != np.uint8:
            logger.warning(f"Spectrogram data type is {spectrogram_array.dtype}, expected uint8")
        
        logger.info("✅ Data integrity verification passed")
        return True, eeg_array, spectrogram_array
    
    def save_processed_data(self, eeg_array: np.ndarray, spectrogram_array: np.ndarray, labels_array: np.ndarray):
        """Save processed data with error handling and backup"""
        logger.info("Saving processed data...")
        
        # Check disk space before saving
        if not self.disk_manager.check_disk_space(self.config.processed_data_path):
            raise RuntimeError("Insufficient disk space for saving processed data")
        
        # Create backup if enabled and sufficient space
        if self.config.backup_enabled and self.config.backup_path:
            # Check if we have enough space for backup
            estimated_size_gb = (eeg_array.nbytes + spectrogram_array.nbytes) / (1024**3) * 2  # Rough estimate
            available_gb = self.disk_manager.get_available_space_gb(self.config.processed_data_path)
            
            if available_gb > estimated_size_gb + 5:  # 5GB buffer
                backup_eeg_path = self.config.backup_path / "eeg_processed_backup.npz"
                backup_spec_path = self.config.backup_path / "spectrograms_backup.npz"
                
                logger.info("Creating backup files...")
                np.savez_compressed(
                    backup_eeg_path,
                    data=eeg_array,
                    labels=labels_array
                )
                np.savez_compressed(
                    backup_spec_path,
                    data=spectrogram_array,
                    labels=labels_array
                )
            else:
                logger.warning(f"Insufficient space for backup (need ~{estimated_size_gb:.1f}GB, have {available_gb:.1f}GB). Skipping backup.")
        
        # Save main files
        eeg_path = self.config.processed_data_path / "eeg_processed.npz"
        spec_path = self.config.processed_data_path / "spectrograms.npz"
        
        logger.info(f"Saving EEG data to {eeg_path}")
        np.savez_compressed(
            eeg_path,
            data=eeg_array,
            labels=labels_array
        )
        
        logger.info(f"Saving spectrogram data to {spec_path}")
        np.savez_compressed(
            spec_path,
            data=spectrogram_array,
            labels=labels_array
        )
        
        # Verify saved files
        self.verify_saved_files(eeg_path, spec_path, eeg_array, spectrogram_array, labels_array)
        
        logger.info("✅ Processed data saved successfully")
    
    def verify_saved_files(self, eeg_path: Path, spec_path: Path, 
                          original_eeg: np.ndarray, original_spec: np.ndarray, original_labels: np.ndarray):
        """Verify that saved files can be loaded correctly"""
        logger.info("Verifying saved files...")
        
        try:
            # Load EEG data
            loaded_eeg = np.load(eeg_path)
            loaded_eeg_data = loaded_eeg['data']
            loaded_eeg_labels = loaded_eeg['labels']
            
            # Load spectrogram data
            loaded_spec = np.load(spec_path)
            loaded_spec_data = loaded_spec['data']
            loaded_spec_labels = loaded_spec['labels']
            
            # Verify shapes
            if loaded_eeg_data.shape != original_eeg.shape:
                raise ValueError(f"EEG shape mismatch: saved {loaded_eeg_data.shape} vs original {original_eeg.shape}")
            
            if loaded_spec_data.shape != original_spec.shape:
                raise ValueError(f"Spectrogram shape mismatch: saved {loaded_spec_data.shape} vs original {original_spec.shape}")
            
            if not np.array_equal(loaded_eeg_labels, original_labels):
                raise ValueError("EEG labels mismatch")
            
            if not np.array_equal(loaded_spec_labels, original_labels):
                raise ValueError("Spectrogram labels mismatch")
            
            logger.info("✅ Saved files verified successfully")
            
        except Exception as e:
            logger.error(f"❌ File verification failed: {e}")
            raise
    
    def cleanup_temp_files(self):
        """Clean up temporary batch files"""
        if not self.config.cleanup_temp_files:
            logger.info("Skipping temp file cleanup (disabled in config)")
            return
        
        logger.info("Cleaning up temporary batch files...")
        removed_count = self.disk_manager.cleanup_temp_files(
            self.config.processed_data_path, 
            self.config.temp_batch_pattern
        )
        
        # Check disk space after cleanup
        available_gb = self.disk_manager.get_available_space_gb(self.config.processed_data_path)
        logger.info(f"Available disk space after cleanup: {available_gb:.2f} GB")
    
    def generate_summary_report(self, eeg_array: np.ndarray, spectrogram_array: np.ndarray, 
                               labels_array: np.ndarray, temp_files: List[Path]) -> Dict:
        """Generate a summary report of the data generation process"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(eeg_array),
            'eeg_shape': eeg_array.shape,
            'spectrogram_shape': spectrogram_array.shape,
            'labels_shape': labels_array.shape,
            'temp_files_processed': len(temp_files),
            'disk_space_available_gb': self.disk_manager.get_available_space_gb(self.config.processed_data_path),
            'memory_usage_gb': self.memory_manager.get_memory_usage_gb(),
            'unique_labels': list(np.unique(labels_array)),
            'label_distribution': {str(label): int(np.sum(labels_array == label)) for label in np.unique(labels_array)},
            'data_types': {
                'eeg_dtype': str(eeg_array.dtype),
                'spectrogram_dtype': str(spectrogram_array.dtype),
                'labels_dtype': str(labels_array.dtype)
            },
            'file_sizes_mb': {
                'eeg_processed_npz': (self.config.processed_data_path / "eeg_processed.npz").stat().st_size / (1024**2) if (self.config.processed_data_path / "eeg_processed.npz").exists() else 0,
                'spectrograms_npz': (self.config.processed_data_path / "spectrograms.npz").stat().st_size / (1024**2) if (self.config.processed_data_path / "spectrograms.npz").exists() else 0
            }
        }
        
        # Save report
        report_path = self.config.processed_data_path / "data_generation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("📊 Data generation summary report saved")
        return report
    
    def run(self) -> bool:
        """Main execution method"""
        try:
            logger.info("🚀 Starting processed data generation...")
            
            # Check initial conditions
            if not self.disk_manager.check_disk_space(self.config.processed_data_path):
                logger.error("❌ Insufficient disk space to start processing")
                return False
            
            # Find temp batch files
            temp_files = self.find_temp_batch_files()
            if not temp_files:
                logger.error("❌ No temporary batch files found")
                return False
            
            # Combine batches
            eeg_array, spectrogram_array, labels_array = self.combine_batches_incremental(temp_files)
            
            # Verify data integrity
            is_valid, cleaned_eeg_array, cleaned_spectrogram_array = self.verify_data_integrity(eeg_array, spectrogram_array, labels_array)
            if not is_valid:
                logger.error("❌ Data integrity check failed")
                return False
            
            # Save processed data
            self.save_processed_data(cleaned_eeg_array, cleaned_spectrogram_array, labels_array)
            
            # Generate summary report
            report = self.generate_summary_report(cleaned_eeg_array, cleaned_spectrogram_array, labels_array, temp_files)
            
            # Clean up temp files
            self.cleanup_temp_files()
            
            logger.info("✅ Processed data generation completed successfully!")
            logger.info(f"📊 Summary: {report['total_samples']} samples processed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Processed data generation failed: {e}")
            return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate processed data from temporary batch files")
    parser.add_argument("--config", type=str, default="config/novita_enhanced_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--processed-data-path", type=str, default="data/processed",
                       help="Path to processed data directory")
    parser.add_argument("--min-disk-space", type=float, default=10.0,
                       help="Minimum required disk space in GB")
    parser.add_argument("--max-memory", type=float, default=50.0,
                       help="Maximum memory usage in GB")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Don't clean up temporary files")
    parser.add_argument("--no-verify", action="store_true",
                       help="Skip data integrity verification")
    parser.add_argument("--backup-path", type=str, default="data/backup",
                       help="Path for backup files")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Extract paths from YAML config
        processed_data_path = Path(yaml_config.get('dataset', {}).get('processed_data_path', args.processed_data_path))
    else:
        processed_data_path = Path(args.processed_data_path)
    
    # Create configuration
    config = DataGenerationConfig(
        processed_data_path=processed_data_path,
        min_disk_space_gb=args.min_disk_space,
        max_memory_usage_gb=args.max_memory,
        cleanup_temp_files=not args.no_cleanup,
        verify_data_integrity=not args.no_verify,
        backup_path=Path(args.backup_path) if args.backup_path else None
    )
    
    # Create and run generator
    generator = ProcessedDataGenerator(config)
    success = generator.run()
    
    if success:
        logger.info("🎉 Processed data generation completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Processed data generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 