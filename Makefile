# HMS EEG Classification System - Docker Makefile
# Run all project operations through Docker

.PHONY: help build up down start stop restart logs clean test deploy train prepare download optimize shell dev prod status backup restore install run

# Default target
help:
	@echo "HMS EEG Classification System - Docker Commands"
	@echo "=============================================="
	@echo ""
	@echo "🐳 Docker-First Approach (Recommended):"
	@echo "  make build        - Build all Docker containers"
	@echo "  make up           - Start all services (CPU only)"
	@echo "  make up-gpu       - Start all services with GPU support"
	@echo "  make prepare      - Prepare data (inside Docker)"
	@echo "  make train        - Train models (CPU, inside Docker)"
	@echo "  make train-gpu    - Train models (GPU, inside Docker)"
	@echo ""
	@echo "Quick Start:"
	@echo "  ./fix_permissions.sh && make build && make up"
	@echo ""
	@echo "Docker Management:"
	@echo "  make down         - Stop all services"
	@echo "  make restart      - Restart all services"
	@echo "  make logs         - Show logs for all services"
	@echo "  make status       - Show status of all services"
	@echo ""
	@echo "GPU Support:"
	@echo "  make check-gpu    - Check if GPU is available"
	@echo "  make optimize-gpu - Optimize models with GPU"
	@echo ""
	@echo "Data & Model Operations (Docker-based):"
	@echo "  make download     - Download dataset from Kaggle"
	@echo "  make optimize     - Optimize models (CPU)"
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Start in development mode (with Jupyter)"
	@echo "  make shell        - Open shell in main container"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Clean up containers and volumes"
	@echo ""
	@echo "Local Development (Optional):"
	@echo "  make install      - Install Python dependencies locally"
	@echo "  make run          - Run project setup locally (not recommended)"
	@echo ""
	@echo "Service-specific logs:"
	@echo "  make logs-api     - Show API logs"
	@echo "  make logs-trainer - Show trainer logs"
	@echo "  make logs-mlflow  - Show MLflow logs"
	@echo ""
	@echo "Utilities:"
	@echo "  make backup       - Backup database and models"
	@echo "  make restore      - Restore from backup"
	@echo "  make health       - Check service health"
	@echo ""

# Main run command - complete project setup and start
run:
	@echo "🚀 Starting HMS EEG Classification System..."
	@python run_project.py
	@echo "✅ HMS EEG System deployment complete!"

# Install Python dependencies (LOCAL DEVELOPMENT ONLY)
# Note: This is for local development. Docker containers have packages pre-installed.
install:
	@echo "📦 Installing Python dependencies locally (for development)..."
	@echo "⚠️  Note: Docker containers already have all packages installed"
	@python -m pip install --upgrade pip
	@echo "Installing core ML dependencies (excluding TensorFlow for Python 3.13+ compatibility)..."
	@python -c "import sys; exit(0) if sys.version_info < (3, 13) else exit(1)" && \
		python -m pip install -r requirements.txt || \
		(echo "Python 3.13+ detected - installing without TensorFlow..." && \
		 python -m pip install torch torchvision scikit-learn xgboost lightgbm && \
		 python -m pip install mne scipy pywavelets antropy yasa && \
		 python -m pip install pyedflib h5py pymatreader && \
		 python -m pip install numpy pandas librosa numba && \
		 python -m pip install matplotlib seaborn plotly ipympl altair && \
		 python -m pip install fastapi uvicorn pydantic redis prometheus-client && \
		 python -m pip install pyyaml tqdm joblib einops kaggle==1.5.16 python-dotenv rich psutil)
	@echo "✅ Dependencies installed locally!"
	@echo "💡 Tip: Use Docker commands (make build, make up) for containerized execution"

# Complete setup and run (Docker-based)
all: build download prepare train optimize up
	@echo "✅ HMS EEG Backend System is fully deployed!"
	@echo "Access the API at http://localhost:8000/docs"

# Quick start without download and training (Docker-based)
quick: build up
	@echo "✅ Quick start complete!"
	@echo "Access the API at http://localhost:8000/docs"
	@echo ""
	@echo "To add data and models later:"
	@echo "  make download  # Download dataset"
	@echo "  make prepare   # Prepare data"
	@echo "  make train     # Train models"

# Build all containers
build:
	@echo "🔨 Building Docker containers..."
	@docker-compose build --no-cache
	@docker build -t hms-runner:latest --target runner .
	@echo "✅ Build complete!"

# Start all services
up:
	@echo "🚀 Starting all services..."
	@docker-compose up -d
	@echo "✅ All services started!"
	@echo ""
	@echo "Access URLs:"
	@echo "  API Docs:     http://localhost:8000/docs"
	@echo "  API Health:   http://localhost:8000/health"
	@echo "  MLflow:       http://localhost:5000"
	@echo "  Grafana:      http://localhost:3001"
	@echo "  Dashboard:    http://localhost:8050"

# Start services with GPU support
up-gpu:
	@echo "🚀 Starting all services with GPU support..."
	@docker-compose --profile gpu up -d
	@echo "✅ All services started with GPU support!"
	@echo ""
	@echo "Access URLs:"
	@echo "  API Docs:     http://localhost:8000/docs"
	@echo "  API Health:   http://localhost:8000/health"
	@echo "  MLflow:       http://localhost:5000"
	@echo "  Grafana:      http://localhost:3001"
	@echo "  Dashboard:    http://localhost:8050"

# Check if GPU is available
check-gpu:
	@echo "🔍 Checking GPU availability..."
	@docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi || echo "❌ No GPU available"

# Stop all services
down:
	@echo "🛑 Stopping all services..."
	@docker-compose down
	@echo "✅ All services stopped!"

# Start services (if stopped)
start:
	@docker-compose start

# Stop services (keep containers)
stop:
	@docker-compose stop

# Restart all services
restart: down up

# Show logs
logs:
	@docker-compose logs -f

# Show specific service logs
logs-%:
	@docker-compose logs -f $*

# Download dataset
download:
	@echo "📥 Downloading dataset from Kaggle..."
	@docker run --rm \
		-v $(PWD)/data:/app/data \
		-v ~/.kaggle:/root/.kaggle:ro \
		--entrypoint python \
		hms-runner:latest \
		scripts/download_dataset.py --output-dir /app/data/raw
	@echo "✅ Dataset downloaded!"

# Prepare data
prepare:
	@echo "🔧 Preparing data..."
	@docker run --rm \
		-v $(PWD):/workspace \
		-w /workspace \
		--user root \
		hms-runner:latest \
		python prepare_data.py
	@echo "✅ Data prepared!"

# Alternative prepare using local Python (if Docker has issues)
prepare-local:
	@echo "🔧 Preparing data locally..."
	@python prepare_data.py
	@echo "✅ Data prepared!"

# Train models
train:
	@echo "🧠 Training models..."
	@docker-compose --profile training up --build trainer
	@echo "✅ Training complete!"

# Train models with GPU
train-gpu:
	@echo "🧠 Training models with GPU..."
	@docker-compose --profile training --profile gpu up --build trainer-gpu
	@echo "✅ GPU training complete!"

# Optimize models
optimize:
	@echo "⚡ Optimizing models..."
	@docker-compose --profile optimization up --build optimizer
	@echo "✅ Optimization complete!"

# Optimize models with GPU
optimize-gpu:
	@echo "⚡ Optimizing models with GPU..."
	@docker-compose --profile optimization --profile gpu up --build optimizer-gpu
	@echo "✅ GPU optimization complete!"

# Development mode
dev:
	@echo "🔧 Starting in development mode..."
	@docker-compose --profile development up -d
	@echo "✅ Development environment ready!"
	@echo ""
	@echo "Jupyter Lab: http://localhost:8888 (token: jupyter_token)"

# Production mode
prod:
	@echo "🚀 Starting in production mode..."
	@docker-compose --profile production up -d
	@echo "✅ Production environment ready!"

# Open shell in container
shell:
	@docker-compose run --rm api bash

# Run tests
test:
	@echo "🧪 Running tests..."
	@docker-compose run --rm api pytest tests/ -v
	@echo "✅ Tests complete!"

# Show service status
status:
	@echo "📊 Service Status:"
	@docker-compose ps

# Clean everything
clean:
	@echo "🧹 Cleaning up..."
	@docker-compose down -v --remove-orphans
	@docker system prune -f
	@echo "✅ Cleanup complete!"

# Deep clean (including images)
deep-clean: clean
	@docker-compose down --rmi all
	@echo "✅ Deep cleanup complete!"

# Backup database and models
backup:
	@echo "💾 Creating backup..."
	@mkdir -p backups
	@docker-compose exec postgres pg_dump -U hms_user hms_eeg > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
	@tar -czf backups/models_$(shell date +%Y%m%d_%H%M%S).tar.gz models/
	@echo "✅ Backup complete!"

# Restore from backup
restore:
	@echo "📥 Restoring from backup..."
	@echo "Available backups:"
	@ls -la backups/
	@echo "Please run: docker-compose exec postgres psql -U hms_user hms_eeg < backups/backup_TIMESTAMP.sql"

# Health check
health:
	@echo "🏥 Health Check:"
	@curl -s http://localhost:8000/health | jq . || echo "API not responding"
	@curl -s http://localhost:5000 > /dev/null && echo "MLflow: ✅" || echo "MLflow: ❌"
	@curl -s http://localhost:3001/api/health | jq . || echo "Grafana not responding"

# View specific service logs with tail
tail-%:
	@docker-compose logs --tail=100 -f $*

# Execute command in service
exec-%:
	@docker-compose exec $* bash

# Pull latest images
pull:
	@docker-compose pull

# Push to registry
push:
	@docker-compose push

# Run specific service
run-%:
	@docker-compose run --rm $*

# --- Shortcuts ---
b: build
u: up
d: down
l: logs
s: status
t: test

# Environment setup
env:
	@cp -n .env.example .env || true
	@echo "✅ Environment file ready. Please update .env with your credentials."

# Install pre-commit hooks
hooks:
	@docker-compose run --rm api pre-commit install

# Format code
format:
	@docker-compose run --rm api black src/ tests/
	@docker-compose run --rm api isort src/ tests/

# Lint code
lint:
	@docker-compose run --rm api flake8 src/ tests/
	@docker-compose run --rm api mypy src/

# Generate documentation
docs:
	@docker-compose run --rm api sphinx-build -b html docs/ docs/_build/html
	@echo "📚 Documentation generated at docs/_build/html/index.html"

# Monitor GPU usage
gpu:
	@watch -n 1 nvidia-smi

# Database operations
db-shell:
	@docker-compose exec postgres psql -U hms_user hms_eeg

db-migrate:
	@docker-compose run --rm api alembic upgrade head

db-reset:
	@docker-compose exec postgres psql -U hms_user -c "DROP DATABASE hms_eeg;"
	@docker-compose exec postgres psql -U hms_user -c "CREATE DATABASE hms_eeg;"
	@make db-migrate

# Run with specific GPU
gpu-%:
	@CUDA_VISIBLE_DEVICES=$* docker-compose up -d

# Performance profiling
profile:
	@docker-compose run --rm api python -m cProfile -o profile.stats train.py
	@echo "✅ Profile saved to profile.stats"

# Memory profiling
memory:
	@docker-compose run --rm api python -m memory_profiler train.py

# Benchmark models
benchmark:
	@docker-compose run --rm api python scripts/benchmark_models.py

# Export models
export:
	@docker-compose run --rm api python scripts/export_models.py

# --- Docker Compose Shortcuts ---
dc:
	@docker-compose $(filter-out $@,$(MAKECMDGOALS))

# Catch-all to allow dc shortcuts
%:
	@: 