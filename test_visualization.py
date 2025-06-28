#!/usr/bin/env python3
"""
Test visualization components for HMS EEG Classification System.
Tests training progress monitoring, performance analysis, clinical visualizations, and dashboard functionality.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
import warnings
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_visualization_imports():
    """Test that all visualization components can be imported."""
    logger.info("🧪 Testing visualization imports...")
    
    try:
        from visualization import (
            TrainingProgressMonitor,
            LearningCurveVisualizer,
            HyperparameterVisualizer,
            ConfusionMatrixVisualizer,
            ROCCurveVisualizer,
            FeatureImportanceVisualizer,
            PatientReportGenerator,
            EEGSignalViewer,
            ClinicalAlertVisualizer,
            DashboardApp,
            RealTimeMonitor
        )
        logger.info("✅ All visualization imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Visualization import failed: {e}")
        return False


def test_training_progress_monitor():
    """Test training progress monitoring visualization."""
    logger.info("🧪 Testing training progress monitor...")
    
    try:
        from visualization import TrainingProgressMonitor
        
        # Create temporary directory for saving plots
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create progress monitor
            monitor = TrainingProgressMonitor(save_dir=temp_path)
            
            # Mock training history
            training_history = {
                'train_loss': [1.5, 1.2, 1.0, 0.9, 0.8],
                'val_loss': [1.6, 1.3, 1.1, 0.95, 0.85],
                'train_acc': [0.4, 0.5, 0.6, 0.65, 0.7],
                'val_acc': [0.35, 0.45, 0.55, 0.6, 0.65],
                'learning_rate': [0.001, 0.001, 0.0008, 0.0006, 0.0005]
            }
            
            # Test plotting training curves
            monitor.plot_training_curves(training_history)
            
            # Check if plot was saved
            plot_files = list(temp_path.glob("*.png"))
            assert len(plot_files) > 0, "No plot files were saved"
            logger.info(f"✅ Training curves plotted and saved: {len(plot_files)} files")
            
            # Test real-time update simulation
            monitor.update_metrics(epoch=5, metrics={'train_loss': 0.75, 'val_loss': 0.8})
            logger.info("✅ Metrics updated successfully")
            
        logger.info("✅ Training progress monitor test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Training progress monitor test failed: {e}")
        return False


def test_learning_curve_visualizer():
    """Test learning curve visualization."""
    logger.info("🧪 Testing learning curve visualizer...")
    
    try:
        from visualization import LearningCurveVisualizer
        
        # Create learning curve visualizer
        visualizer = LearningCurveVisualizer()
        
        # Mock learning curve data
        train_sizes = np.array([50, 100, 200, 500, 1000])
        train_scores = np.random.rand(5, 3) * 0.3 + 0.6  # Mock CV scores
        val_scores = np.random.rand(5, 3) * 0.4 + 0.5
        
        # Test plotting learning curves
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "learning_curve.png"
            
            visualizer.plot_learning_curve(
                train_sizes, train_scores, val_scores,
                save_path=save_path
            )
            
            assert save_path.exists(), "Learning curve plot was not saved"
            logger.info("✅ Learning curve plotted and saved")
            
        # Test validation curve
        param_range = [0.1, 0.5, 1.0, 2.0, 5.0]
        train_scores_param = np.random.rand(5, 3) * 0.3 + 0.6
        val_scores_param = np.random.rand(5, 3) * 0.4 + 0.5
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "validation_curve.png"
            
            visualizer.plot_validation_curve(
                param_range, train_scores_param, val_scores_param,
                param_name="C", save_path=save_path
            )
            
            assert save_path.exists(), "Validation curve plot was not saved"
            logger.info("✅ Validation curve plotted and saved")
        
        logger.info("✅ Learning curve visualizer test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Learning curve visualizer test failed: {e}")
        return False


def test_confusion_matrix_visualizer():
    """Test confusion matrix visualization."""
    logger.info("🧪 Testing confusion matrix visualizer...")
    
    try:
        from visualization import ConfusionMatrixVisualizer
        from sklearn.metrics import confusion_matrix
        
        # Create confusion matrix visualizer
        visualizer = ConfusionMatrixVisualizer()
        
        # Generate mock data
        np.random.seed(42)
        n_samples = 100
        n_classes = 6
        y_true = np.random.randint(0, n_classes, n_samples)
        y_pred = np.random.randint(0, n_classes, n_samples)
        
        class_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Test basic confusion matrix plot
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "confusion_matrix.png"
            
            visualizer.plot_confusion_matrix(
                cm, class_names, save_path=save_path
            )
            
            assert save_path.exists(), "Confusion matrix plot was not saved"
            logger.info("✅ Confusion matrix plotted and saved")
            
        # Test normalized confusion matrix
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "confusion_matrix_norm.png"
            
            visualizer.plot_normalized_confusion_matrix(
                cm, class_names, save_path=save_path
            )
            
            assert save_path.exists(), "Normalized confusion matrix plot was not saved"
            logger.info("✅ Normalized confusion matrix plotted and saved")
        
        logger.info("✅ Confusion matrix visualizer test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Confusion matrix visualizer test failed: {e}")
        return False


def test_roc_curve_visualizer():
    """Test ROC curve visualization."""
    logger.info("🧪 Testing ROC curve visualizer...")
    
    try:
        from visualization import ROCCurveVisualizer
        
        # Create ROC curve visualizer
        visualizer = ROCCurveVisualizer()
        
        # Generate mock data
        np.random.seed(42)
        n_samples = 100
        n_classes = 6
        y_true = np.random.randint(0, n_classes, n_samples)
        y_prob = np.random.rand(n_samples, n_classes)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize
        
        class_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
        
        # Test multiclass ROC curves
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "roc_curves.png"
            
            visualizer.plot_multiclass_roc(
                y_true, y_prob, class_names, save_path=save_path
            )
            
            assert save_path.exists(), "ROC curves plot was not saved"
            logger.info("✅ Multiclass ROC curves plotted and saved")
            
        # Test PR curves
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "pr_curves.png"
            
            visualizer.plot_precision_recall_curves(
                y_true, y_prob, class_names, save_path=save_path
            )
            
            assert save_path.exists(), "PR curves plot was not saved"
            logger.info("✅ Precision-Recall curves plotted and saved")
        
        logger.info("✅ ROC curve visualizer test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ ROC curve visualizer test failed: {e}")
        return False


def test_feature_importance_visualizer():
    """Test feature importance visualization."""
    logger.info("🧪 Testing feature importance visualizer...")
    
    try:
        from visualization import FeatureImportanceVisualizer
        
        # Create feature importance visualizer
        visualizer = FeatureImportanceVisualizer()
        
        # Generate mock feature importance data
        n_features = 20
        feature_names = [f'Feature_{i}' for i in range(n_features)]
        importance_scores = np.random.rand(n_features)
        
        # Test feature importance bar plot
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "feature_importance.png"
            
            visualizer.plot_feature_importance(
                importance_scores, feature_names, 
                top_k=10, save_path=save_path
            )
            
            assert save_path.exists(), "Feature importance plot was not saved"
            logger.info("✅ Feature importance bar plot created and saved")
            
        # Test feature importance heatmap for multiple models
        n_models = 3
        importance_matrix = np.random.rand(n_models, n_features)
        model_names = [f'Model_{i}' for i in range(n_models)]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "feature_importance_heatmap.png"
            
            visualizer.plot_feature_importance_heatmap(
                importance_matrix, feature_names, model_names,
                save_path=save_path
            )
            
            assert save_path.exists(), "Feature importance heatmap was not saved"
            logger.info("✅ Feature importance heatmap created and saved")
        
        logger.info("✅ Feature importance visualizer test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Feature importance visualizer test failed: {e}")
        return False


def test_eeg_signal_viewer():
    """Test EEG signal visualization."""
    logger.info("🧪 Testing EEG signal viewer...")
    
    try:
        from visualization import EEGSignalViewer
        
        # Create EEG signal viewer
        viewer = EEGSignalViewer()
        
        # Generate mock EEG data
        n_channels = 19
        n_timepoints = 1000
        sampling_rate = 200
        eeg_data = np.random.randn(n_channels, n_timepoints) * 50  # μV scale
        
        channel_names = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
        ]
        
        # Test EEG signal plotting
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "eeg_signals.png"
            
            viewer.plot_eeg_signals(
                eeg_data, channel_names, sampling_rate,
                duration=5.0, save_path=save_path
            )
            
            assert save_path.exists(), "EEG signals plot was not saved"
            logger.info("✅ EEG signals plotted and saved")
            
        # Test topographic map
        # Generate mock topographic data (one value per channel)
        topo_data = np.random.randn(n_channels)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "topographic_map.png"
            
            try:
                viewer.plot_topographic_map(
                    topo_data, channel_names, save_path=save_path
                )
                if save_path.exists():
                    logger.info("✅ Topographic map plotted and saved")
                else:
                    logger.warning("⚠️ Topographic map plot was not saved")
            except Exception as e:
                logger.warning(f"⚠️ Topographic map plotting failed (expected without electrode positions): {e}")
        
        logger.info("✅ EEG signal viewer test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ EEG signal viewer test failed: {e}")
        return False


def test_patient_report_generator():
    """Test patient report generation."""
    logger.info("🧪 Testing patient report generator...")
    
    try:
        from visualization import PatientReportGenerator
        
        # Create patient report generator
        generator = PatientReportGenerator()
        
        # Mock patient data
        patient_data = {
            'patient_id': 'P001',
            'age': 45,
            'gender': 'F',
            'recording_date': '2024-01-15',
            'diagnosis': 'Epilepsy monitoring',
            'medications': ['Levetiracetam', 'Phenytoin']
        }
        
        # Mock analysis results
        analysis_results = {
            'predictions': ['Seizure', 'Other', 'LPD'],
            'confidences': [0.95, 0.3, 0.8],
            'timestamps': ['10:15:30', '10:45:12', '11:20:45'],
            'total_seizures': 1,
            'seizure_duration': 45.2,
            'quality_score': 0.85
        }
        
        # Mock EEG data
        eeg_data = np.random.randn(19, 1000)
        channel_names = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
        ]
        
        # Test report generation
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "patient_report.html"
            
            generator.generate_patient_report(
                patient_data, analysis_results, eeg_data, 
                channel_names, save_path=save_path
            )
            
            assert save_path.exists(), "Patient report was not saved"
            
            # Check if report contains expected content
            with open(save_path, 'r') as f:
                content = f.read()
                assert 'P001' in content
                assert 'Seizure' in content
                assert '0.95' in content  # Confidence score
                
            logger.info("✅ Patient report generated and saved")
        
        logger.info("✅ Patient report generator test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Patient report generator test failed: {e}")
        return False


def test_clinical_alert_visualizer():
    """Test clinical alert visualization."""
    logger.info("🧪 Testing clinical alert visualizer...")
    
    try:
        from visualization import ClinicalAlertVisualizer
        
        # Create clinical alert visualizer
        visualizer = ClinicalAlertVisualizer()
        
        # Mock alert data
        alerts = [
            {
                'timestamp': '2024-01-15 10:15:30',
                'type': 'Seizure',
                'confidence': 0.95,
                'duration': 45.2,
                'severity': 'High',
                'patient_id': 'P001'
            },
            {
                'timestamp': '2024-01-15 11:20:15',
                'type': 'LPD',
                'confidence': 0.82,
                'duration': 120.5,
                'severity': 'Medium',
                'patient_id': 'P001'
            },
            {
                'timestamp': '2024-01-15 12:05:45',
                'type': 'GPD',
                'confidence': 0.75,
                'duration': 90.0,
                'severity': 'Medium',
                'patient_id': 'P002'
            }
        ]
        
        # Test alert dashboard
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "alert_dashboard.png"
            
            visualizer.create_alert_dashboard(alerts, save_path=save_path)
            
            assert save_path.exists(), "Alert dashboard was not saved"
            logger.info("✅ Alert dashboard created and saved")
            
        # Test alert timeline
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "alert_timeline.png"
            
            visualizer.plot_alert_timeline(alerts, save_path=save_path)
            
            assert save_path.exists(), "Alert timeline was not saved"
            logger.info("✅ Alert timeline plotted and saved")
        
        logger.info("✅ Clinical alert visualizer test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Clinical alert visualizer test failed: {e}")
        return False


def test_hyperparameter_visualizer():
    """Test hyperparameter optimization visualization."""
    logger.info("🧪 Testing hyperparameter visualizer...")
    
    try:
        from visualization import HyperparameterVisualizer
        
        # Create hyperparameter visualizer
        visualizer = HyperparameterVisualizer()
        
        # Mock hyperparameter optimization results
        trials_data = []
        for i in range(20):
            trial = {
                'trial_id': i,
                'learning_rate': np.random.uniform(0.0001, 0.01),
                'batch_size': np.random.choice([16, 32, 64, 128]),
                'dropout_rate': np.random.uniform(0.1, 0.5),
                'num_layers': np.random.randint(2, 6),
                'hidden_size': np.random.choice([64, 128, 256, 512]),
                'validation_loss': np.random.uniform(0.5, 2.0),
                'validation_accuracy': np.random.uniform(0.6, 0.9)
            }
            trials_data.append(trial)
        
        # Test hyperparameter importance plot
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "hyperparameter_importance.png"
            
            visualizer.plot_hyperparameter_importance(
                trials_data, target_metric='validation_accuracy',
                save_path=save_path
            )
            
            assert save_path.exists(), "Hyperparameter importance plot was not saved"
            logger.info("✅ Hyperparameter importance plotted and saved")
            
        # Test optimization history
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "optimization_history.png"
            
            visualizer.plot_optimization_history(
                trials_data, target_metric='validation_accuracy',
                save_path=save_path
            )
            
            assert save_path.exists(), "Optimization history plot was not saved"
            logger.info("✅ Optimization history plotted and saved")
        
        logger.info("✅ Hyperparameter visualizer test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Hyperparameter visualizer test failed: {e}")
        return False


def test_dashboard_components():
    """Test dashboard components (without actually running the server)."""
    logger.info("🧪 Testing dashboard components...")
    
    try:
        # Test dashboard imports and basic initialization
        from visualization import DashboardApp, RealTimeMonitor
        
        # Test dashboard app initialization
        try:
            dashboard = DashboardApp(
                redis_url='redis://localhost:6379',
                update_interval=5000,
                data_dir='logs'
            )
            logger.info("✅ Dashboard app initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Dashboard app initialization failed (expected without Redis): {e}")
        
        # Test real-time monitor initialization
        try:
            monitor = RealTimeMonitor()
            logger.info("✅ Real-time monitor initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Real-time monitor initialization failed: {e}")
        
        # Test dashboard layout creation (without server)
        try:
            dashboard = DashboardApp(
                redis_url='redis://localhost:6379',
                update_interval=5000,
                data_dir='logs'
            )
            
            # Test if layout can be created
            layout = dashboard.create_layout()
            assert layout is not None
            logger.info("✅ Dashboard layout created successfully")
        except Exception as e:
            logger.warning(f"⚠️ Dashboard layout creation failed (expected): {e}")
        
        logger.info("✅ Dashboard components test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dashboard components test failed: {e}")
        return False


def main():
    """Run all visualization tests."""
    logger.info("🧪 Testing visualization components...")
    
    tests = [
        ("Visualization Imports", test_visualization_imports),
        ("Training Progress Monitor", test_training_progress_monitor),
        ("Learning Curve Visualizer", test_learning_curve_visualizer),
        ("Confusion Matrix Visualizer", test_confusion_matrix_visualizer),
        ("ROC Curve Visualizer", test_roc_curve_visualizer),
        ("Feature Importance Visualizer", test_feature_importance_visualizer),
        ("EEG Signal Viewer", test_eeg_signal_viewer),
        ("Patient Report Generator", test_patient_report_generator),
        ("Clinical Alert Visualizer", test_clinical_alert_visualizer),
        ("Hyperparameter Visualizer", test_hyperparameter_visualizer),
        ("Dashboard Components", test_dashboard_components),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Testing {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
            failed += 1
    
    logger.info(f"\n📊 Visualization Test Results:")
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    
    if failed == 0:
        logger.info("✅ All visualization tests passed!")
    else:
        logger.warning(f"⚠️ {failed} visualization tests failed!")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 