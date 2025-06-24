# HMS EEG Classification System - Backend API Docker Quick Start 🐳

## Overview
This guide shows you how to run the HMS EEG Classification System Backend API using Docker, avoiding Python version compatibility issues. The frontend will be developed separately and integrated later.

## Prerequisites
- Docker Engine (20.10+)
- Docker Compose (v2.0+)
- At least 16GB RAM
- 50GB free disk space
- (Optional) NVIDIA GPU with CUDA support for faster training

## Quick Start

### 1. One-Command Start (Skip Download & Training)
```bash
# Quick start with pre-built models
./docker-run.sh start --quick

# OR using make
make quick
```

### 2. Full Setup (Download, Train, Deploy)
```bash
# Complete setup including dataset download and model training
./docker-run.sh start

# OR using make
make all
```

## Available Commands

### Using docker-run.sh Script
```bash
./docker-run.sh start           # Start all services
./docker-run.sh stop            # Stop all services
./docker-run.sh restart         # Restart services
./docker-run.sh build           # Build containers
./docker-run.sh logs [service]  # View logs
./docker-run.sh shell [service] # Open shell in container
./docker-run.sh status          # Show service status
./docker-run.sh clean           # Clean up
```

### Using Make Commands
```bash
# Core commands
make build          # Build all containers
make up             # Start services
make down           # Stop services
make logs           # View all logs
make logs-api       # View API logs only
make status         # Service status

# Data & Training
make download       # Download Kaggle dataset
make prepare        # Prepare data
make train          # Train models
make optimize       # Optimize models

# Development
make dev            # Start with Jupyter Lab
make shell          # Open container shell
make test           # Run tests

# Utilities
make backup         # Backup DB & models
make clean          # Clean containers
make help           # Show all commands
```

### Using Python Script
```bash
# Full Docker-based runner
python run_docker.py --skip-download --skip-train

# With options
python run_docker.py --dev  # Development mode
```

## Service URLs
Once running, access these backend services:

| Service | URL | Description |
|---------|-----|-------------|
| API Docs | http://localhost:8000/docs | FastAPI interactive documentation |
| API Health | http://localhost:8000/health | Health check endpoint |
| MLflow | http://localhost:5000 | Model tracking |
| Grafana | http://localhost:3001 | Monitoring (admin/admin) |
| EEG Dashboard | http://localhost:8050 | Visualization dashboard |
| Jupyter Lab | http://localhost:8888 | Development (token: jupyter_token) |

## API Endpoints

The backend provides the following main endpoints:

- `POST /predict` - Single EEG prediction
- `POST /predict_batch` - Batch predictions
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `GET /metrics` - Prometheus metrics

## Environment Setup

### 1. Kaggle Dataset Access
Edit `.env` file to add your Kaggle credentials:
```bash
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

### 2. GPU Configuration
```bash
# Use specific GPU
make gpu-0  # Use GPU 0
make gpu-1  # Use GPU 1

# Or set in .env
CUDA_VISIBLE_DEVICES=0
```

## Common Workflows

### Development Workflow
```bash
# Start development environment
make dev

# Open Jupyter Lab
# http://localhost:8888 (token: jupyter_token)

# Run tests
make test

# Format code
make format

# Lint code
make lint
```

### Production Deployment
```bash
# Build and optimize
make build
make train
make optimize

# Deploy production
make prod

# Monitor
make logs-api
make status
```

### Data Science Workflow
```bash
# Download and prepare data
make download
make prepare

# Train models
make train

# Benchmark performance
make benchmark

# Export models
make export
```

## Testing the API

### Using curl
```bash
# Health check
curl http://localhost:8000/health

# Prediction (example)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"eeg_data": [[...]], "spectrogram_data": [[...]]}'
```

### Using Python
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Prediction
data = {
    "eeg_data": [...],  # Your EEG data
    "spectrogram_data": [...]  # Your spectrogram data
}
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
```

## Troubleshooting

### Docker Issues
```bash
# Check Docker status
docker info

# Clean up Docker
docker system prune -a

# Rebuild from scratch
make deep-clean
make build
```

### Service Issues
```bash
# Check service health
make health

# View specific logs
make logs-api
make logs-trainer

# Restart specific service
docker-compose restart api
```

### GPU Issues
```bash
# Check GPU availability
nvidia-smi

# Monitor GPU usage
make gpu
```

## Advanced Usage

### Custom Training
```bash
# Run with custom config
docker-compose run --rm trainer python train.py --config custom_config.yaml

# Distributed training
docker-compose run --rm -e WORLD_SIZE=4 trainer
```

### Database Operations
```bash
# Access database
make db-shell

# Run migrations
make db-migrate

# Reset database
make db-reset
```

### Performance Profiling
```bash
# Profile training
make profile

# Memory profiling
make memory

# Benchmark models
make benchmark
```

## Tips

1. **First Run**: The first run will take 30-60 minutes to download data and train models
2. **GPU**: Training is 10x faster with GPU. CPU training works but is slow
3. **Memory**: Ensure Docker has at least 8GB RAM allocated
4. **Storage**: Models and data require ~20GB space

## Quick Debug Commands
```bash
# If services won't start
docker-compose down -v
make build
make up

# If out of space
docker system prune -a

# View real-time logs
make logs

# Check service health
curl http://localhost:8000/health
```

## Frontend Integration (Future)

When the frontend is ready, it can connect to this backend API at `http://localhost:8000`. The API provides CORS support and WebSocket connections for real-time features.

## Next Steps
1. Access the API documentation at http://localhost:8000/docs
2. Test the API endpoints using the interactive documentation
3. View model metrics in MLflow at http://localhost:5000
4. Monitor system performance in Grafana at http://localhost:3001

For more details, see the full documentation in `README.md`. 