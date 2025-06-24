#!/usr/bin/env python3
"""
HMS EEG Visualization System - Demo Script

This script demonstrates the various visualization capabilities
of the HMS EEG Classification System.
"""

import sys
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.visualization.training.progress_monitor import TrainingProgressMonitor
from src.visualization.training.learning_curves import LearningCurveVisualizer
from src.visualization.performance.confusion_matrix import ConfusionMatrixVisualizer


def demo_training_progress():
    """Demonstrate training progress monitoring."""
    print("="*60)
    print("Training Progress Monitoring Demo")
    print("="*60)
    
    # Create monitor
    monitor = TrainingProgressMonitor(
        log_dir="logs/demo_training",
        update_interval=5,
        save_interval=10
    )
    
    # Simulate training progress
    print("Simulating training progress...")
    for epoch in range(5):
        for iteration in range(20):
            # Simulate metrics
            train_loss = 1.0 * np.exp(-0.01 * (epoch * 20 + iteration)) + np.random.normal(0, 0.05)
            val_loss = train_loss + 0.1 + np.random.normal(0, 0.05)
            train_acc = min(99, 50 + (epoch * 20 + iteration) * 0.5 + np.random.normal(0, 2))
            val_acc = train_acc - 5 + np.random.normal(0, 2)
            lr = 0.001 * (0.95 ** epoch)
            
            # Log metrics
            monitor.log_metrics(
                iteration=epoch * 20 + iteration,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                learning_rate=lr
            )
    
    # Create interactive dashboard
    dashboard = monitor.create_interactive_dashboard()
    dashboard.write_html("logs/demo_training/training_dashboard.html")
    print("✅ Training dashboard saved to: logs/demo_training/training_dashboard.html")
    
    # Generate report
    monitor.stop()
    print("✅ Training monitoring complete!")


def demo_learning_curves():
    """Demonstrate learning curve analysis."""
    print("\n" + "="*60)
    print("Learning Curve Analysis Demo")
    print("="*60)
    
    # Create analyzer
    analyzer = LearningCurveVisualizer(log_dir="logs/demo_training")
    
    # Load metrics (from previous demo)
    try:
        analyzer.load_metrics()
        
        # Perform analyses
        overfit_results = analyzer.detect_overfitting()
        conv_results = analyzer.analyze_convergence()
        
        print("\nAnalysis Results:")
        print(f"- Overfitting detected: {overfit_results['detected']}")
        if overfit_results['detected']:
            print(f"  Started at iteration: {overfit_results['start_iteration']}")
            print(f"  Best model at iteration: {overfit_results['best_iteration']}")
        
        print(f"- Convergence: {conv_results['converged']}")
        if conv_results['converged']:
            print(f"  Converged at iteration: {conv_results['convergence_iteration']}")
        
        # Create visualizations
        fig = analyzer.plot_learning_curves(save_path="logs/demo_training/learning_curves.png")
        interactive_fig = analyzer.create_interactive_plot()
        interactive_fig.write_html("logs/demo_training/learning_curves.html")
        
        # Generate report
        report = analyzer.generate_report("logs/demo_training/analysis_report.txt")
        print("\n✅ Learning curve analysis complete!")
        
    except Exception as e:
        print(f"⚠️  Could not load metrics: {e}")


def demo_confusion_matrix():
    """Demonstrate confusion matrix visualization."""
    print("\n" + "="*60)
    print("Confusion Matrix Visualization Demo")
    print("="*60)
    
    # Generate sample predictions
    classes = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
    n_samples = 1000
    
    # Create realistic confusion pattern
    y_true = np.random.choice(range(6), n_samples, p=[0.15, 0.15, 0.15, 0.15, 0.15, 0.25])
    y_pred = y_true.copy()
    
    # Add some realistic errors
    error_rate = 0.15
    error_indices = np.random.choice(n_samples, int(n_samples * error_rate), replace=False)
    
    for idx in error_indices:
        true_class = y_true[idx]
        # More likely to confuse with similar classes
        if true_class == 0:  # Seizure
            y_pred[idx] = np.random.choice([1, 2, 5], p=[0.4, 0.3, 0.3])
        elif true_class in [1, 2]:  # LPD, GPD
            y_pred[idx] = np.random.choice([0, 1, 2, 5], p=[0.2, 0.3, 0.3, 0.2])
        else:
            y_pred[idx] = np.random.choice([i for i in range(6) if i != true_class])
    
    # Create visualizer
    cm_viz = ConfusionMatrixVisualizer(clinical_mode=True)
    
    # Compute and visualize
    cm_viz.compute_matrix(y_true, y_pred, normalize='true')
    
    # Static visualization
    fig = cm_viz.plot_matrix(save_path="logs/demo_training/confusion_matrix.png")
    
    # Interactive visualization
    interactive_cm = cm_viz.create_interactive_matrix()
    interactive_cm.write_html("logs/demo_training/confusion_matrix.html")
    
    # Class performance
    class_perf_fig = cm_viz.plot_class_performance(
        save_path="logs/demo_training/class_performance.png"
    )
    
    # Generate report
    report = cm_viz.generate_report("logs/demo_training/confusion_matrix_report.txt")
    
    print("\nClass Performance Summary:")
    metrics_df = cm_viz.get_class_metrics()
    print(metrics_df[['precision', 'recall', 'f1-score']].round(3))
    
    print("\n✅ Confusion matrix analysis complete!")


def demo_dashboard_data():
    """Generate sample data for dashboard demo."""
    print("\n" + "="*60)
    print("Generating Dashboard Demo Data")
    print("="*60)
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Generate sample alerts
    alerts = []
    severities = ["critical", "warning", "info"]
    alert_titles = [
        "Seizure Detected",
        "High Model Uncertainty",
        "System Resource Warning",
        "Data Quality Issue"
    ]
    
    for i in range(10):
        alerts.append({
            "id": f"alert_{i}",
            "timestamp": (datetime.now() - timedelta(minutes=i*5)).isoformat(),
            "severity": np.random.choice(severities, p=[0.2, 0.5, 0.3]),
            "title": np.random.choice(alert_titles),
            "message": f"Sample alert message {i}",
            "patient_id": f"P{np.random.randint(1000, 9999)}"
        })
    
    with open(log_dir / "alerts.json", 'w') as f:
        json.dump(alerts, f, indent=2)
    
    # Generate sample predictions
    predictions = []
    classes = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
    
    for i in range(100):
        predictions.append({
            "id": f"pred_{i}",
            "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
            "patient_id": f"P{np.random.randint(1000, 9999)}",
            "predicted_class": np.random.choice(classes),
            "confidence": np.random.uniform(0.6, 0.99),
            "processing_time": np.random.exponential(50)
        })
    
    with open(log_dir / "predictions.json", 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print("✅ Dashboard demo data generated!")
    print("\nTo view the dashboard, run:")
    print("  python -m src.visualization.run_dashboard")
    print("\nOr with Docker:")
    print("  make visualize")


def main():
    """Run all visualization demos."""
    print("\nHMS EEG Visualization System Demo")
    print("="*60)
    
    # Create output directory
    Path("logs/demo_training").mkdir(parents=True, exist_ok=True)
    
    # Run demos
    demo_training_progress()
    demo_learning_curves()
    demo_confusion_matrix()
    demo_dashboard_data()
    
    print("\n" + "="*60)
    print("All demos complete! Check the logs/demo_training directory for outputs.")
    print("="*60)


if __name__ == "__main__":
    main() 