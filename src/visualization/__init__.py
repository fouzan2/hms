"""
HMS EEG Classification System - Visualization Module

This module provides comprehensive visualization capabilities for:
- Training progress monitoring
- Model performance analysis
- Clinical decision support
- Interactive dashboards
"""

from .training.progress_monitor import TrainingProgressMonitor
from .training.learning_curves import LearningCurveVisualizer
from .training.hyperparameter_viz import HyperparameterVisualizer

from .performance.confusion_matrix import ConfusionMatrixVisualizer
from .performance.roc_curves import ROCCurveVisualizer
from .performance.feature_importance import FeatureImportanceVisualizer

from .clinical.patient_report import PatientReportGenerator
from .clinical.eeg_viewer import EEGSignalViewer
from .clinical.alert_visualizer import ClinicalAlertVisualizer

from .dashboard.app import DashboardApp
from .dashboard.real_time import RealTimeMonitor

__all__ = [
    'TrainingProgressMonitor',
    'LearningCurveVisualizer',
    'HyperparameterVisualizer',
    'ConfusionMatrixVisualizer',
    'ROCCurveVisualizer',
    'FeatureImportanceVisualizer',
    'PatientReportGenerator',
    'EEGSignalViewer',
    'ClinicalAlertVisualizer',
    'DashboardApp',
    'RealTimeMonitor'
]

__version__ = '1.0.0' 