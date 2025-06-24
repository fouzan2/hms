#!/usr/bin/env python3
"""
Multi-format EEG file readers for HMS Brain Activity Classification.

Supports:
- EDF (European Data Format)
- BDF (BioSemi Data Format)
- CSV (Comma-Separated Values)
- Parquet (Columnar storage format)
- HDF5 (Hierarchical Data Format)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import h5py
import pyedflib
import mne
from scipy import signal
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseEEGReader(ABC):
    """Abstract base class for EEG file readers."""
    
    @abstractmethod
    def read(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Read EEG data from file.
        
        Returns:
            data: numpy array of shape (n_channels, n_samples)
            metadata: dictionary containing sampling rate, channel names, etc.
        """
        pass
    
    @abstractmethod
    def validate(self, filepath: Union[str, Path]) -> bool:
        """Validate if file can be read by this reader."""
        pass


class EDFReader(BaseEEGReader):
    """Reader for European Data Format (EDF) files."""
    
    def __init__(self, preload: bool = True):
        self.preload = preload
        
    def validate(self, filepath: Union[str, Path]) -> bool:
        """Check if file is valid EDF."""
        filepath = Path(filepath)
        if not filepath.exists() or not filepath.suffix.lower() in ['.edf']:
            return False
        
        try:
            # Try to read header
            with pyedflib.EdfReader(str(filepath)) as f:
                _ = f.getHeader()
            return True
        except:
            return False
    
    def read(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Read EDF file."""
        filepath = Path(filepath)
        logger.info(f"Reading EDF file: {filepath}")
        
        with pyedflib.EdfReader(str(filepath)) as f:
            # Get metadata
            n_channels = f.signals_in_file
            channel_names = f.getSignalLabels()
            sample_rates = [f.getSampleFrequency(i) for i in range(n_channels)]
            
            # Check if all channels have same sampling rate
            if len(set(sample_rates)) > 1:
                logger.warning("Different sampling rates detected across channels")
            
            # Read signals
            data = []
            for i in range(n_channels):
                data.append(f.readSignal(i))
            
            data = np.array(data)
            
            metadata = {
                'sampling_rate': sample_rates[0],  # Use first channel's rate
                'channel_names': channel_names,
                'duration': f.getFileDuration(),
                'start_datetime': f.getStartdatetime(),
                'patient_info': f.getPatientCode(),
                'recording_info': f.getEquipment(),
                'n_samples': data.shape[1],
                'n_channels': n_channels
            }
            
        return data, metadata


class BDFReader(BaseEEGReader):
    """Reader for BioSemi Data Format (BDF) files."""
    
    def validate(self, filepath: Union[str, Path]) -> bool:
        """Check if file is valid BDF."""
        filepath = Path(filepath)
        if not filepath.exists() or not filepath.suffix.lower() in ['.bdf']:
            return False
        
        try:
            # Use MNE to validate
            raw = mne.io.read_raw_bdf(str(filepath), preload=False)
            return True
        except:
            return False
    
    def read(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Read BDF file using MNE."""
        filepath = Path(filepath)
        logger.info(f"Reading BDF file: {filepath}")
        
        # Read with MNE
        raw = mne.io.read_raw_bdf(str(filepath), preload=True, verbose='ERROR')
        
        # Extract data and metadata
        data = raw.get_data()
        
        metadata = {
            'sampling_rate': raw.info['sfreq'],
            'channel_names': raw.ch_names,
            'duration': raw.times[-1],
            'n_samples': data.shape[1],
            'n_channels': data.shape[0],
            'mne_info': raw.info
        }
        
        return data, metadata


class CSVReader(BaseEEGReader):
    """Reader for CSV format EEG files."""
    
    def __init__(self, sampling_rate: float = 200.0, 
                 time_column: Optional[str] = None,
                 channel_prefix: str = 'ch'):
        self.sampling_rate = sampling_rate
        self.time_column = time_column
        self.channel_prefix = channel_prefix
    
    def validate(self, filepath: Union[str, Path]) -> bool:
        """Check if file is valid CSV with EEG data."""
        filepath = Path(filepath)
        if not filepath.exists() or not filepath.suffix.lower() in ['.csv']:
            return False
        
        try:
            # Try to read first few rows
            df = pd.read_csv(filepath, nrows=5)
            # Check if has channel columns
            channel_cols = [col for col in df.columns if col.startswith(self.channel_prefix)]
            return len(channel_cols) > 0
        except:
            return False
    
    def read(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Read CSV file."""
        filepath = Path(filepath)
        logger.info(f"Reading CSV file: {filepath}")
        
        # Read data
        df = pd.read_csv(filepath)
        
        # Identify channel columns
        if self.time_column and self.time_column in df.columns:
            channel_columns = [col for col in df.columns if col != self.time_column]
        else:
            channel_columns = [col for col in df.columns if col.startswith(self.channel_prefix)]
        
        # Extract data
        data = df[channel_columns].values.T  # Transpose to (n_channels, n_samples)
        
        # Calculate duration
        if self.time_column and self.time_column in df.columns:
            duration = df[self.time_column].iloc[-1] - df[self.time_column].iloc[0]
        else:
            duration = len(df) / self.sampling_rate
        
        metadata = {
            'sampling_rate': self.sampling_rate,
            'channel_names': channel_columns,
            'duration': duration,
            'n_samples': data.shape[1],
            'n_channels': data.shape[0],
            'source_columns': list(df.columns)
        }
        
        return data, metadata


class ParquetReader(BaseEEGReader):
    """Reader for Parquet format EEG files (HMS dataset format)."""
    
    def __init__(self, sampling_rate: float = 200.0):
        self.sampling_rate = sampling_rate
        self.expected_channels = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
            'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
            'Fz', 'Cz', 'Pz'
        ]
    
    def validate(self, filepath: Union[str, Path]) -> bool:
        """Check if file is valid Parquet with EEG data."""
        filepath = Path(filepath)
        if not filepath.exists() or not filepath.suffix.lower() in ['.parquet']:
            return False
        
        try:
            df = pd.read_parquet(filepath, columns=['Fp1'])  # Just check one column
            return True
        except:
            return False
    
    def read(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Read Parquet file."""
        filepath = Path(filepath)
        logger.info(f"Reading Parquet file: {filepath}")
        
        # Read data
        df = pd.read_parquet(filepath)
        
        # Find available channels
        available_channels = [ch for ch in self.expected_channels if ch in df.columns]
        missing_channels = [ch for ch in self.expected_channels if ch not in df.columns]
        
        if missing_channels:
            logger.warning(f"Missing channels: {missing_channels}")
        
        # Extract data
        data = df[available_channels].values.T  # Transpose to (n_channels, n_samples)
        
        # Calculate duration
        duration = len(df) / self.sampling_rate
        
        metadata = {
            'sampling_rate': self.sampling_rate,
            'channel_names': available_channels,
            'duration': duration,
            'n_samples': data.shape[1],
            'n_channels': data.shape[0],
            'missing_channels': missing_channels,
            'all_columns': list(df.columns)
        }
        
        return data, metadata


class HDF5Reader(BaseEEGReader):
    """Reader for HDF5 format EEG files."""
    
    def __init__(self, data_path: str = '/eeg/data',
                 metadata_path: str = '/eeg/metadata'):
        self.data_path = data_path
        self.metadata_path = metadata_path
    
    def validate(self, filepath: Union[str, Path]) -> bool:
        """Check if file is valid HDF5 with EEG data."""
        filepath = Path(filepath)
        if not filepath.exists() or not filepath.suffix.lower() in ['.h5', '.hdf5']:
            return False
        
        try:
            with h5py.File(filepath, 'r') as f:
                return self.data_path in f
        except:
            return False
    
    def read(self, filepath: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Read HDF5 file."""
        filepath = Path(filepath)
        logger.info(f"Reading HDF5 file: {filepath}")
        
        with h5py.File(filepath, 'r') as f:
            # Read data
            data = f[self.data_path][:]
            
            # Read metadata
            metadata = {}
            if self.metadata_path in f:
                meta_group = f[self.metadata_path]
                for key in meta_group.attrs:
                    metadata[key] = meta_group.attrs[key]
            
            # Ensure required metadata
            if 'sampling_rate' not in metadata:
                metadata['sampling_rate'] = 200.0  # Default
            if 'channel_names' not in metadata:
                metadata['channel_names'] = [f'ch_{i}' for i in range(data.shape[0])]
            
            metadata.update({
                'n_samples': data.shape[1],
                'n_channels': data.shape[0],
                'duration': data.shape[1] / metadata['sampling_rate']
            })
        
        return data, metadata


class MultiFormatEEGReader:
    """Unified reader supporting multiple EEG file formats."""
    
    def __init__(self, sampling_rate: float = 200.0):
        self.sampling_rate = sampling_rate
        
        # Initialize format-specific readers
        self.readers = {
            '.edf': EDFReader(),
            '.bdf': BDFReader(),
            '.csv': CSVReader(sampling_rate=sampling_rate),
            '.parquet': ParquetReader(sampling_rate=sampling_rate),
            '.h5': HDF5Reader(),
            '.hdf5': HDF5Reader()
        }
    
    def read(self, filepath: Union[str, Path], 
             format: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Read EEG data from file with automatic format detection.
        
        Args:
            filepath: Path to EEG file
            format: Optional format override
            
        Returns:
            data: numpy array of shape (n_channels, n_samples)
            metadata: dictionary containing file metadata
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Determine format
        if format:
            file_format = format.lower() if not format.startswith('.') else format
        else:
            file_format = filepath.suffix.lower()
        
        # Get appropriate reader
        if file_format not in self.readers:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        reader = self.readers[file_format]
        
        # Validate file
        if not reader.validate(filepath):
            raise ValueError(f"Invalid {file_format} file: {filepath}")
        
        # Read data
        data, metadata = reader.read(filepath)
        
        # Add file information to metadata
        metadata['filepath'] = str(filepath)
        metadata['format'] = file_format
        
        # Standardize sampling rate if needed
        if metadata['sampling_rate'] != self.sampling_rate:
            logger.info(f"Resampling from {metadata['sampling_rate']}Hz to {self.sampling_rate}Hz")
            data = self._resample_data(data, metadata['sampling_rate'], self.sampling_rate)
            metadata['original_sampling_rate'] = metadata['sampling_rate']
            metadata['sampling_rate'] = self.sampling_rate
            metadata['n_samples'] = data.shape[1]
        
        return data, metadata
    
    def _resample_data(self, data: np.ndarray, 
                      orig_rate: float, 
                      target_rate: float) -> np.ndarray:
        """Resample data to target sampling rate."""
        if orig_rate == target_rate:
            return data
        
        # Calculate resampling factor
        resample_factor = target_rate / orig_rate
        n_samples_new = int(data.shape[1] * resample_factor)
        
        # Resample each channel
        resampled_data = np.zeros((data.shape[0], n_samples_new))
        for ch in range(data.shape[0]):
            resampled_data[ch, :] = signal.resample(data[ch, :], n_samples_new)
        
        return resampled_data
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return list(self.readers.keys())


def read_eeg_file(filepath: Union[str, Path], 
                  sampling_rate: float = 200.0,
                  format: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to read EEG file.
    
    Args:
        filepath: Path to EEG file
        sampling_rate: Target sampling rate (will resample if needed)
        format: Optional format override
        
    Returns:
        data: numpy array of shape (n_channels, n_samples)
        metadata: dictionary containing file metadata
    """
    reader = MultiFormatEEGReader(sampling_rate=sampling_rate)
    return reader.read(filepath, format=format) 