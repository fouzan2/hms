"""Preprocessing utilities for EEG and spectrogram data."""

from .eeg_preprocessor import EEGPreprocessor
from .spectrogram_generator import SpectrogramGenerator
from .file_readers import (
    MultiFormatEEGReader,
    EDFReader,
    BDFReader,
    CSVReader,
    ParquetReader,
    HDF5Reader,
    read_eeg_file
)
from .signal_quality import (
    SignalQualityAssessor,
    QualityMetrics,
    assess_signal_quality
)
from .signal_filters import (
    EEGFilter,
    AdvancedDenoiser,
    create_eeg_filter
)
from .feature_extraction import (
    EEGFeatureExtractor,
    FeatureSelector,
    FeatureSet,
    extract_features
)

__all__ = [
    # Main classes
    'EEGPreprocessor',
    'SpectrogramGenerator',
    
    # File readers
    'MultiFormatEEGReader',
    'EDFReader',
    'BDFReader',
    'CSVReader',
    'ParquetReader',
    'HDF5Reader',
    'read_eeg_file',
    
    # Signal quality
    'SignalQualityAssessor',
    'QualityMetrics',
    'assess_signal_quality',
    
    # Signal filtering
    'EEGFilter',
    'AdvancedDenoiser',
    'create_eeg_filter',
    
    # Feature extraction
    'EEGFeatureExtractor',
    'FeatureSelector',
    'FeatureSet',
    'extract_features'
] 