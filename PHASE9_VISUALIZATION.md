# Phase 9: Visualization and Reporting System

This document describes the comprehensive visualization and reporting system implemented for the HMS EEG Classification System.

## 🎨 Overview

Phase 9 adds a powerful visualization and reporting system that provides:
- Real-time training progress monitoring
- Interactive model performance analysis
- Clinical decision support visualizations
- System resource monitoring
- Comprehensive reporting capabilities

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Start the visualization dashboard
make visualize

# Or start all dashboards
make dashboard
```

### Direct Python

```bash
# Run the dashboard
python -m src.visualization.run_dashboard

# Run demo
python examples/visualization_demo.py
```

## 📊 Components Implemented

### 1. Training Visualization

**Location**: `src/visualization/training/`

- **TrainingProgressMonitor**: Real-time training monitoring
  - Loss and accuracy curves
  - Learning rate scheduling
  - GPU/CPU utilization tracking
  - Estimated time remaining
  - TensorBoard export support

- **LearningCurveVisualizer**: Advanced learning analysis
  - Overfitting detection
  - Convergence analysis
  - Early stopping recommendations
  - Interactive Plotly dashboards

### 2. Performance Visualization

**Location**: `src/visualization/performance/`

- **ConfusionMatrixVisualizer**: Clinical-aware confusion matrices
  - Per-class performance metrics
  - Clinical significance highlighting
  - Interactive heatmaps
  - Critical misclassification alerts

- **ROCCurveVisualizer**: ROC and PR curve analysis
- **FeatureImportanceVisualizer**: Model interpretability

### 3. Clinical Visualization

**Location**: `src/visualization/clinical/`

- **PatientReportGenerator**: Automated clinical reports
- **EEGSignalViewer**: Interactive EEG visualization
- **ClinicalAlertVisualizer**: Real-time alert management

### 4. Interactive Dashboard

**Location**: `src/visualization/dashboard/`

- **DashboardApp**: Comprehensive web dashboard
  - Real-time monitoring
  - Multiple visualization tabs
  - Alert management
  - Resource monitoring
  - Prediction analysis

## 🌐 Dashboard Features

### Overview Tab
- Key metrics cards (predictions, accuracy, alerts, status)
- Predictions timeline
- Class distribution charts
- Recent activity feed

### Training Monitor Tab
- Real-time loss curves
- Learning rate visualization
- Training progress with ETA
- Resource utilization

### Model Performance Tab
- Interactive confusion matrix
- ROC curves for all classes
- Per-class metrics table
- Model comparison tools

### Clinical Analysis Tab
- Clinical alerts timeline
- Seizure detection analysis
- Critical findings summary
- Patient-specific insights

### System Resources Tab
- CPU and GPU usage graphs
- Memory consumption tracking
- API response time monitoring
- System health indicators

### Predictions Tab
- Filterable predictions table
- Date range selection
- Class and confidence filtering
- Export capabilities

## 🔧 Configuration

### Environment Variables

```bash
# Dashboard configuration
DASH_DEBUG=false           # Enable debug mode
REDIS_URL=redis://...      # Redis for real-time data
DATABASE_URL=postgresql://... # PostgreSQL connection
```

### Dashboard Settings

Edit `src/visualization/dashboard/app.py`:

```python
dashboard = DashboardApp(
    redis_url="redis://localhost:6379",
    update_interval=5000,  # milliseconds
    data_dir="logs"
)
```

## 📈 Usage Examples

### 1. Training Monitoring

```python
from src.visualization.training.progress_monitor import TrainingProgressMonitor

# Create monitor
monitor = TrainingProgressMonitor(log_dir="logs/training")

# During training loop
for epoch in range(epochs):
    for batch in dataloader:
        # Training step...
        
        # Log metrics
        monitor.log_metrics(
            iteration=iteration,
            epoch=epoch,
            train_loss=loss.item(),
            val_loss=val_loss,
            train_acc=accuracy,
            learning_rate=optimizer.param_groups[0]['lr']
        )

# Generate final report
monitor.stop()
```

### 2. Performance Analysis

```python
from src.visualization.performance.confusion_matrix import ConfusionMatrixVisualizer

# Create visualizer
cm_viz = ConfusionMatrixVisualizer(clinical_mode=True)

# Compute matrix
cm_viz.compute_matrix(y_true, y_pred)

# Generate visualizations
fig = cm_viz.plot_matrix()
interactive_fig = cm_viz.create_interactive_matrix()

# Generate report
report = cm_viz.generate_report("confusion_report.txt")
```

### 3. Dashboard Integration

The dashboard automatically connects to:
- Redis for real-time metrics
- PostgreSQL for historical data
- Log files for fallback data

## 🖼️ Generated Outputs

### Static Reports
- Training progress plots (PNG)
- Confusion matrices (PNG)
- Learning curve analysis (PNG)
- PDF/Word clinical reports

### Interactive Dashboards
- HTML dashboards with Plotly
- Real-time Dash application
- TensorBoard logs
- Exportable visualizations

## 🔌 Integration Points

### With Training Pipeline
- Automatic metric logging
- Checkpoint visualization
- Hyperparameter tracking

### With API
- Real-time prediction monitoring
- Performance metrics collection
- Alert generation

### With Monitoring Stack
- Prometheus metrics export
- Grafana dashboard integration
- Alert manager connectivity

## 🚨 Clinical Features

### Alert Priorities
1. **Critical**: Seizure detection, high-risk patterns
2. **Warning**: Model uncertainty, data quality issues
3. **Info**: System updates, routine notifications

### Clinical Validations
- Seizure detection recall monitoring
- False positive rate tracking
- Clinical impact scoring
- Expert annotation comparison

## 📝 Generated Reports

### Training Report
- Convergence analysis
- Overfitting detection
- Performance progression
- Resource utilization

### Clinical Report
- Patient demographics
- EEG analysis results
- Confidence scores
- Clinical recommendations

### System Report
- Model performance metrics
- Resource usage statistics
- Error analysis
- Optimization suggestions

## 🛠️ Troubleshooting

### Dashboard Not Loading
```bash
# Check if service is running
docker ps | grep hms-dashboard

# View logs
docker logs hms-dashboard

# Restart service
docker-compose restart visualization-dashboard
```

### Missing Visualizations
- Ensure Redis is running for real-time data
- Check log directory permissions
- Verify data files exist

### Performance Issues
- Reduce update interval in dashboard
- Enable data sampling for large datasets
- Use pagination for tables

## 📚 Dependencies Added

- `dash>=2.14.0` - Interactive web applications
- `dash-bootstrap-components>=1.5.0` - UI components
- `plotly>=5.15.0` - Interactive plotting
- `kaleido>=0.2.1` - Static image export
- `reportlab>=4.0.0` - PDF generation

## 🎉 Summary

Phase 9 successfully implements a comprehensive visualization and reporting system that:

1. **Monitors Training**: Real-time progress tracking with advanced analytics
2. **Analyzes Performance**: Interactive confusion matrices and metrics
3. **Supports Clinical Decisions**: Medical-grade visualizations and alerts
4. **Tracks Resources**: System performance monitoring
5. **Generates Reports**: Automated clinical and technical reports

The system is fully dockerized and integrates seamlessly with the existing pipeline, accessible at http://localhost:8050 after deployment. 