# HMS Brain Activity Classification System - Backend API

A comprehensive machine learning backend system for automated classification of harmful brain activities using EEG signals from the Kaggle HMS Harmful Brain Activity Classification dataset. This repository contains the backend API, with the frontend to be developed separately.

## 🐳 Quick Start with Docker (Recommended)

Avoid Python version compatibility issues by running everything in Docker:

```bash
# One-command start (skip download & training)
./docker-run.sh start --quick

# OR full setup with data download and training
make all

# OR using Python script
python run_docker.py --skip-download --skip-train
```

Access the API at:
- **API Docs**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health
- **MLflow**: http://localhost:5000
- **Grafana**: http://localhost:3001
- **Dashboard**: http://localhost:8050

See [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md) for detailed Docker instructions.

## Overview

This system classifies six types of brain activities from EEG recordings:
- **Seizures**: Abnormal, excessive neuronal activity
- **LPD (Lateralized Periodic Discharges)**: Unilateral periodic patterns
- **GPD (Generalized Periodic Discharges)**: Bilateral periodic patterns  
- **LRDA (Lateralized Rhythmic Delta Activity)**: Unilateral rhythmic slow waves
- **GRDA (Generalized Rhythmic Delta Activity)**: Bilateral rhythmic slow waves
- **Other**: Background activity or artifacts

## Features

- **Multi-modal Processing**: Handles both raw EEG (50-second, 200 Hz) and spectrograms (10-minute segments)
- **Advanced Preprocessing**: Medical-grade signal processing with artifact removal
- **State-of-the-art Models**: 
  - ResNet1D-GRU for raw EEG
  - EfficientNet for spectrograms
  - Ensemble model with attention fusion
- **Clinical Interpretability**: SHAP analysis and attention visualization
- **Production Ready**: FastAPI deployment with real-time inference
- **Optimization**: ONNX export, quantization, and TensorRT support
- **Comprehensive Monitoring**: Prometheus + Grafana for system monitoring
- **Single Command Execution**: Run entire pipeline with one command
- **Dockerized Deployment**: Fully containerized architecture

## Installation

### Requirements
- Docker Engine 20.10+
- Docker Compose v2.0+
- 32GB RAM recommended
- 100GB+ disk space for dataset
- (Optional) NVIDIA GPU with CUDA 11.0+ for faster training

### Docker Installation (Recommended)

1. Install Docker:
   - [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/Mac)
   - [Docker Engine](https://docs.docker.com/engine/install/) (Linux)

2. Clone the repository:
```bash
git clone https://github.com/yourusername/hms-brain-activity-classification.git
cd hms-brain-activity-classification
```

3. Run with Docker:
```bash
# Quick start (skip download & training)
make quick

# OR full installation with dataset download and training
make all
```

### Manual Installation (Alternative)

⚠️ **Note**: Manual installation requires Python 3.8-3.9. Python 3.13 is not supported due to TensorFlow compatibility.

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hms-brain-activity-classification.git
cd hms-brain-activity-classification
```

2. Run comprehensive setup:
```bash
python setup.py --full
```

3. Activate virtual environment:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Configure Kaggle API:
```bash
# Create ~/.kaggle/kaggle.json with your credentials
mkdir -p ~/.kaggle
echo '{"username":"your_username","key":"your_api_key"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Usage

### Docker Commands (Recommended)

```bash
# Start all services
make up

# Stop all services
make down

# View logs
make logs

# Train models
make train

# Run tests
make test

# Open shell in container
make shell

# See all commands
make help
```

### Using the Docker Run Script

```bash
# Start services
./docker-run.sh start

# Stop services
./docker-run.sh stop

# View logs
./docker-run.sh logs [service_name]

# Open shell
./docker-run.sh shell
```

### Manual Usage (Alternative)

```bash
# Download dataset
python train.py --download-only

# Train models
python train.py --config config/config.yaml

# Start API
uvicorn src.deployment.api:app --reload
```

## Project Structure

```
hms/
├── config/
│   ├── config.yaml          # Main configuration
│   ├── logging.yaml         # Logging configuration
│   └── prometheus.yml       # Prometheus monitoring config
├── data/
│   ├── raw/                 # Downloaded raw data
│   │   ├── train_eegs/      # Training EEG files
│   │   ├── train_spectrograms/  # Training spectrograms
│   │   └── test_spectrograms/   # Test spectrograms
│   └── processed/           # Processed data
├── src/
│   ├── preprocessing/       # EEG and spectrogram preprocessing
│   ├── models/              # Model architectures
│   ├── training/            # Training logic
│   ├── evaluation/          # Evaluation metrics
│   ├── deployment/          # API and optimization
│   └── utils/               # Utilities and dataset
├── webapp/                  # Web interface (Next.js)
│   ├── frontend/           # React frontend
│   └── backend/            # Backend services
├── experiments/            # Experiment tracking
│   ├── mlflow/            # MLflow artifacts
│   └── tensorboard/       # TensorBoard logs
├── tests/                  # Unit and integration tests
├── notebooks/              # Jupyter notebooks
├── logs/                   # Application logs
├── docker-compose.yml     # Docker services
├── Dockerfile             # Container definition
├── Makefile               # Docker commands
├── docker-run.sh          # Docker helper script
├── run_docker.py          # Python Docker runner
├── setup.py               # Manual setup script
└── run_project.py         # Manual runner
```

## Configuration

The system is highly configurable via `config/config.yaml`. Key sections include:

- **Dataset**: Data paths, splits, and preprocessing settings
- **Models**: Architecture parameters for each model type
- **Training**: Hyperparameters, optimization, and augmentation
- **Deployment**: API settings and monitoring configuration
- **System**: Hardware utilization and performance settings

### Environment Variables

Create a `.env` file with:
```bash
# Kaggle API (for dataset download)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# GPU Configuration
CUDA_VISIBLE_DEVICES=0

# Other settings (optional)
POSTGRES_PASSWORD=secure_password
REDIS_PASSWORD=redis_password
```

## Monitoring

### Service Endpoints

After deployment, access the following backend services:

- **API Documentation**: http://localhost:8000/docs (Interactive API docs)
- **API Health Check**: http://localhost:8000/health
- **MLflow UI**: http://localhost:5000
- **Grafana**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Visualization Dashboard**: http://localhost:8050
- **Jupyter**: http://localhost:8888 (in dev mode)

### Frontend Integration (Planned)

The frontend will be developed separately and will connect to this backend API. The API is designed with:
- CORS support for cross-origin requests
- WebSocket support for real-time features
- RESTful endpoints for all operations
- JWT authentication ready

### Visualization Dashboard Features

The comprehensive visualization dashboard (Phase 9) provides:

- **Real-time Training Monitoring**: Track loss curves, accuracy, learning rate schedules
- **Model Performance Analysis**: Interactive confusion matrices, ROC curves, per-class metrics
- **Clinical Decision Support**: Patient-specific visualizations, seizure detection analysis
- **System Resource Monitoring**: CPU/GPU usage, memory consumption, API latency
- **Alert Management**: Real-time clinical alerts and notifications

Access the dashboard at http://localhost:8050 after deployment.

### Health Monitoring

Check system health:

```bash
# Using Make
make health

# API health check
curl http://localhost:8000/health

# View all service statuses
make status

# Check resource usage
docker stats
```

## API Usage

### Single Prediction

```python
import requests
import numpy as np

# Prepare data
eeg_data = np.random.randn(19, 10000).tolist()
spectrogram_data = np.random.randn(3, 224, 224).tolist()

# Make request
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "eeg_data": eeg_data,
        "spectrogram_data": spectrogram_data
    }
)

print(response.json())
```

### Batch Prediction

```python
response = requests.post(
    "http://localhost:8000/predict_batch",
    json={
        "samples": [
            {"eeg_data": eeg1, "spectrogram_data": spec1},
            {"eeg_data": eeg2, "spectrogram_data": spec2}
        ]
    }
)
```

## Model Performance

### Expected Performance
| Model | Accuracy | F1 Score | Inference Time |
|-------|----------|----------|----------------|
| ResNet1D-GRU | 0.85 | 0.83 | 50ms |
| EfficientNet | 0.87 | 0.85 | 40ms |
| Ensemble | 0.91 | 0.89 | 100ms |

### Hardware Requirements
- **Training**: NVIDIA GPU with 16GB+ VRAM
- **Inference**: 
  - GPU: 10-20ms latency
  - CPU: 100-200ms latency
  - Quantized: 30-50ms latency

## Development

### Running Tests

```bash
# With Docker
make test

# Manual
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Type checking
mypy src/
```

### Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Troubleshooting

### Common Issues

1. **Python Version Issues**
   - Use Docker to avoid compatibility problems
   - Manual install requires Python 3.8-3.9

2. **Kaggle Download Fails**
   - Verify API credentials in `.env` file
   - Check internet connection
   - Run with `make download`

3. **Out of Memory**
   - Ensure Docker has enough RAM allocated
   - Reduce batch size in config
   - Use `make clean` to free space

4. **Docker Issues**
   - Ensure Docker daemon is running
   - Check available disk space
   - Run `docker system prune` to clean up

5. **GPU Not Detected**
   - Install NVIDIA Docker runtime
   - Verify with `nvidia-smi`
   - Set `CUDA_VISIBLE_DEVICES` in `.env`

## Phase-wise Development Status

- [x] **Phase 1**: Project Setup and Environment Configuration
- [x] **Phase 2**: Comprehensive Data Preprocessing Pipeline
- [x] **Phase 3**: Deep Learning Model Architecture Development
- [x] **Phase 4**: Training Strategy and Optimization
- [x] **Phase 5**: Model Interpretability and Explainability
- [x] **Phase 6**: Comprehensive Evaluation and Validation
- [x] **Phase 7**: Deployment Architecture and API Development
- [x] **Phase 8**: Performance Optimization and Scalability
- [x] **Phase 9**: Visualization and Reporting System
- [x] **Phase 10**: Frontend Development with Next.js
- [ ] **Phase 11**: Integration and System Testing

## Citation

If you use this system in your research, please cite:

```bibtex
@software{hms_brain_activity_classification,
  title={HMS Brain Activity Classification System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/hms-brain-activity-classification}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kaggle HMS competition organizers
- MNE-Python for EEG processing tools
- PyTorch team for the deep learning framework
- Clinical advisors for domain expertise

## Contact

For questions or support:
- Open an issue on GitHub
- Email: your.email@example.com
- Discord: [Join our server](https://discord.gg/example) 