# HMS EEG Classification System - Docker Usage Guide

This guide explains how to run the HMS EEG Classification System entirely within Docker containers, with proper GPU support and fallback options.

## Quick Start

### 1. Fix Permissions
```bash
chmod +x fix_permissions.sh
./fix_permissions.sh
```

### 2. Check GPU Availability (Optional)
```bash
make check-gpu
```

### 3. Start the System

**Option A: CPU Only (Works on any system)**
```bash
make build
make up
```

**Option B: With GPU Support (Requires NVIDIA GPU + Docker)**
```bash
make build
make up-gpu
```

**Option C: Quick Start (Skip data download and training)**
```bash
make quick
```

## System Architecture

The system now has separate services for CPU and GPU execution:

### CPU Services (Default)
- `api` - Main API server (CPU)
- `trainer` - Model training (CPU) 
- `optimizer` - Model optimization (CPU)

### GPU Services (Optional)
- `api-gpu` - GPU-enabled API server
- `trainer-gpu` - GPU-accelerated training
- `optimizer-gpu` - GPU-enabled optimization

### Common Services
- `postgres` - Database
- `redis` - Caching
- `kafka` - Message streaming
- `mlflow` - Experiment tracking
- `grafana` - Monitoring dashboards
- `prometheus` - Metrics collection

## Available Commands

### Basic Operations
```bash
make help           # Show all commands
make build          # Build all containers
make up             # Start CPU-only services
make up-gpu         # Start services with GPU support
make down           # Stop all services
make restart        # Restart all services
make logs           # Show all logs
make status         # Show service status
```

### GPU Operations
```bash
make check-gpu      # Check GPU availability
make train-gpu      # Train with GPU acceleration
make optimize-gpu   # Optimize models with GPU
```

### Data Operations
```bash
make download       # Download dataset from Kaggle
make prepare        # Prepare and preprocess data
make train          # Train models (CPU)
make optimize       # Optimize models (CPU)
```

### Development
```bash
make dev           # Start with Jupyter notebook
make shell         # Open bash shell in container
make test          # Run tests
```

### Utilities
```bash
make clean         # Clean containers and volumes
make backup        # Backup database and models
make health        # Check service health
```

## Service Access URLs

After starting the services, you can access:

- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health
- **MLflow UI**: http://localhost:5000
- **Grafana Monitoring**: http://localhost:3001 (admin/admin)
- **EEG Dashboard**: http://localhost:8050
- **Jupyter Lab** (dev mode): http://localhost:8888

## Configuration

### Environment Variables

Create a `.env` file with your settings:

```bash
# Kaggle API (for dataset download)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# GPU Settings
CUDA_VISIBLE_DEVICES=0
USE_OPTIMIZED_MODELS=true

# Feature flags
ENABLE_MONITORING=true
ENABLE_MIXED_PRECISION=true
ENABLE_TENSORRT=false

# Database
POSTGRES_PASSWORD=secure_password
REDIS_PASSWORD=redis_password
```

### GPU Configuration

The system automatically detects GPU availability:

1. **With NVIDIA GPU**: Use `make up-gpu` for GPU acceleration
2. **Without GPU**: Use `make up` for CPU-only execution
3. **Mixed environments**: Services gracefully fall back to CPU

## Workflow Examples

### Complete Setup (First Time)
```bash
# 1. Fix permissions
./fix_permissions.sh

# 2. Build containers (installs all packages inside Docker)
make build

# 3. Download data (requires Kaggle credentials)
make download

# 4. Prepare data (runs entirely in Docker)
make prepare

# 5. Train models (choose GPU or CPU, runs in Docker)
make train-gpu    # With GPU
# OR
make train        # CPU only

# 6. Optimize models (runs in Docker)
make optimize-gpu # With GPU
# OR
make optimize     # CPU only

# 7. Start all services
make up-gpu       # With GPU
# OR
make up          # CPU only
```

### Quick Development Setup
```bash
# Skip data download and training, start services immediately
./fix_permissions.sh
make build   # Builds containers with all packages
make up      # Start services
# All packages are pre-installed in containers
```

### Production Deployment
```bash
./fix_permissions.sh
make build
make up-gpu  # Use GPU if available
make health  # Verify all services
```

## Troubleshooting

### Package Installation Issues
The system uses a **Docker-first approach**:
- ✅ **All packages are pre-installed in Docker containers** during `make build`
- ❌ **No local package installation required** for normal operation
- 💡 **`make install` is for local development only** (optional)

If you see package installation attempts:
```bash
# This is correct - runs entirely in Docker:
make prepare

# This installs locally (only for development):
make install
```

### Permission Issues
```bash
# Fix all permissions
./fix_permissions.sh

# Or manually:
chmod +x *.py *.sh scripts/*.py
chmod -R 755 data/ logs/ models/
```

### GPU Issues
```bash
# Check GPU availability
make check-gpu

# If GPU not available, use CPU mode
make up

# For GPU issues, check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

### Docker Issues
```bash
# Clean up everything
make clean

# Rebuild from scratch
make build

# Check Docker daemon
docker info
```

### Service Issues
```bash
# Check service status
make status

# View specific service logs
make logs-api
make logs-trainer
make logs-mlflow

# Health check
make health
```

## Performance Optimization

### With GPU
- Use `make up-gpu` for GPU-accelerated services
- Enable mixed precision: `ENABLE_MIXED_PRECISION=true`
- Use TensorRT: `ENABLE_TENSORRT=true` (requires GPU)

### CPU Only
- Use `make up` for standard CPU services
- Optimize batch sizes in config
- Enable model quantization for inference

## Data Management

The system uses Docker volumes for persistent data:

- `data/raw/` - Downloaded dataset
- `data/processed/` - Preprocessed data
- `models/` - Trained models
- `logs/` - Application logs
- `backups/` - Database backups

## Monitoring and Logging

- **Grafana**: Real-time dashboards at http://localhost:3001
- **Prometheus**: Metrics collection at http://localhost:9090
- **MLflow**: Experiment tracking at http://localhost:5000
- **Logs**: Use `make logs` or `make logs-<service>`

## Security Notes

- Change default passwords in `.env`
- Use proper Kaggle API credentials
- Secure access in production deployments
- Keep Docker images updated

## Support

For issues:
1. Check this documentation
2. Run `make help` for command reference
3. Use `make health` to diagnose issues
4. Check logs with `make logs`

The system is designed to work entirely within Docker containers, ensuring consistent behavior across different environments while providing both CPU and GPU execution options. 