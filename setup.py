#!/usr/bin/env python3
"""
HMS Brain Activity Classification System - Setup Script

This script handles:
1. Environment setup and validation
2. Directory structure creation
3. Dependency installation
4. GPU/CUDA configuration
5. Experiment tracking setup
6. Initial configuration validation
"""

import os
import sys
import platform
import subprocess
import json
import shutil
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProjectSetup:
    """Handle complete project setup and initialization."""
    
    def __init__(self, project_root: Path = Path.cwd()):
        self.project_root = project_root
        self.python_version = sys.version_info
        self.platform = platform.system()
        self.has_gpu = self._check_gpu()
        
    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            # Check NVIDIA SMI
            return shutil.which('nvidia-smi') is not None
    
    def check_python_version(self) -> bool:
        """Ensure Python version is compatible."""
        min_version = (3, 8)
        if self.python_version < min_version:
            logger.error(f"Python {min_version[0]}.{min_version[1]}+ required. "
                        f"Current: {self.python_version.major}.{self.python_version.minor}")
            return False
        logger.info(f"Python version: {self.python_version.major}.{self.python_version.minor} ✓")
        return True
    
    def create_directory_structure(self):
        """Create comprehensive project directory structure."""
        directories = [
            # Data directories
            "data/raw/kaggle",
            "data/raw/train_eegs",
            "data/raw/train_spectrograms", 
            "data/raw/test_spectrograms",
            "data/processed/cache",
            "data/processed/features",
            "data/processed/splits",
            "data/interim",
            "data/external",
            
            # Model directories
            "models/checkpoints",
            "models/final",
            "models/onnx",
            "models/tensorrt",
            
            # Results directories
            "results/figures",
            "results/predictions",
            "results/interpretability",
            "results/reports",
            
            # Experiment tracking
            "experiments/mlflow",
            "experiments/wandb",
            "experiments/tensorboard",
            "experiments/optuna",
            
            # Logs
            "logs/training",
            "logs/api",
            "logs/preprocessing",
            
            # Web application
            "webapp/frontend",
            "webapp/backend",
            "webapp/static",
            
            # Notebooks
            "notebooks/exploration",
            "notebooks/modeling",
            "notebooks/evaluation",
            
            # Tests
            "tests/unit",
            "tests/integration",
            "tests/performance",
            
            # Documentation
            "docs/api",
            "docs/guides",
            "docs/clinical",
            
            # Configuration backups
            "config/experiments",
            "config/deployments",
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep for empty directories
            gitkeep = dir_path / ".gitkeep"
            if not any(dir_path.iterdir()):
                gitkeep.touch()
        
        logger.info("Directory structure created successfully ✓")
    
    def setup_virtual_environment(self):
        """Create and activate virtual environment."""
        venv_path = self.project_root / "venv"
        
        if not venv_path.exists():
            logger.info("Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
            logger.info("Virtual environment created ✓")
        else:
            logger.info("Virtual environment already exists ✓")
        
        # Provide activation instructions
        if self.platform == "Windows":
            activate_cmd = f"{venv_path}\\Scripts\\activate"
        else:
            activate_cmd = f"source {venv_path}/bin/activate"
        
        logger.info(f"\nTo activate the virtual environment, run:\n{activate_cmd}")
        
        return venv_path
    
    def install_dependencies(self, venv_path: Path, cuda_version: Optional[str] = None):
        """Install all project dependencies."""
        pip_cmd = str(venv_path / "bin" / "pip") if self.platform != "Windows" else str(venv_path / "Scripts" / "pip")
        
        # Upgrade pip
        logger.info("Upgrading pip...")
        subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
        
        # Install PyTorch with CUDA support if available
        if self.has_gpu and cuda_version:
            logger.info(f"Installing PyTorch with CUDA {cuda_version} support...")
            torch_cmd = f"torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"
            subprocess.run([pip_cmd, "install"] + torch_cmd.split(), check=True)
        
        # Install requirements
        logger.info("Installing project requirements...")
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        
        # Install development dependencies
        dev_deps = ["ipykernel", "jupyter-contrib-nbextensions", "jupyter_nbextensions_configurator"]
        subprocess.run([pip_cmd, "install"] + dev_deps, check=True)
        
        logger.info("Dependencies installed successfully ✓")
    
    def setup_jupyter_kernel(self, venv_path: Path):
        """Set up Jupyter kernel for the virtual environment."""
        python_cmd = str(venv_path / "bin" / "python") if self.platform != "Windows" else str(venv_path / "Scripts" / "python")
        
        logger.info("Setting up Jupyter kernel...")
        subprocess.run([
            python_cmd, "-m", "ipykernel", "install",
            "--user", "--name", "hms-eeg", "--display-name", "HMS EEG Classification"
        ], check=True)
        logger.info("Jupyter kernel installed ✓")
    
    def setup_git_hooks(self):
        """Set up pre-commit hooks for code quality."""
        logger.info("Setting up Git hooks...")
        
        # Create .pre-commit-config.yaml
        pre_commit_config = """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=5000']
      - id: check-json
      - id: check-merge-conflict
      
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3
        
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
        
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100', '--extend-ignore=E203,W503']
        
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.4.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
"""
        
        with open(self.project_root / ".pre-commit-config.yaml", "w") as f:
            f.write(pre_commit_config)
        
        # Install pre-commit
        try:
            subprocess.run(["pre-commit", "install"], check=True)
            logger.info("Git hooks installed ✓")
        except subprocess.CalledProcessError:
            logger.warning("Could not install pre-commit hooks. Run 'pre-commit install' manually.")
    
    def setup_environment_files(self):
        """Create environment configuration files."""
        # Create .env template
        env_template = """# Environment variables for HMS EEG Classification

# API Keys
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
WANDB_API_KEY=your_wandb_api_key

# Database
DATABASE_URL=postgresql://user:password@localhost/hms_eeg

# Redis
REDIS_URL=redis://localhost:6379

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# API Configuration  
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model Serving
MODEL_CACHE_SIZE=5
MAX_BATCH_SIZE=32

# Monitoring
ENABLE_MONITORING=true
LOG_LEVEL=INFO

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
"""
        
        env_path = self.project_root / ".env.template"
        with open(env_path, "w") as f:
            f.write(env_template)
        
        # Create .gitignore if not exists
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
.venv

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data
data/raw/
data/processed/cache/
*.parquet
*.csv
!data/processed/splits/*.csv

# Models
models/checkpoints/
models/final/
*.pth
*.onnx
*.trt

# Logs
logs/
*.log

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# MLflow
mlruns/
mlflow.db

# Wandb
wandb/

# Testing
.coverage
htmlcov/
.pytest_cache/

# Documentation
docs/_build/
"""
        
        gitignore_path = self.project_root / ".gitignore"
        if not gitignore_path.exists():
            with open(gitignore_path, "w") as f:
                f.write(gitignore_content)
        
        logger.info("Environment files created ✓")
    
    def setup_mlflow(self):
        """Initialize MLflow tracking server."""
        logger.info("Setting up MLflow...")
        
        # Create MLflow database
        mlflow_dir = self.project_root / "experiments" / "mlflow"
        subprocess.run(["mlflow", "db", "upgrade", f"sqlite:///{mlflow_dir}/mlflow.db"], check=True)
        
        # Create systemd service file (Linux only)
        if self.platform == "Linux":
            service_content = f"""[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Type=simple
User={os.getenv('USER')}
WorkingDirectory={self.project_root}
ExecStart={self.project_root}/venv/bin/mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///{mlflow_dir}/mlflow.db --default-artifact-root {mlflow_dir}/artifacts
Restart=always

[Install]
WantedBy=multi-user.target
"""
            
            service_path = self.project_root / "mlflow.service"
            with open(service_path, "w") as f:
                f.write(service_content)
            
            logger.info(f"MLflow service file created at {service_path}")
            logger.info("To install: sudo cp mlflow.service /etc/systemd/system/ && sudo systemctl enable mlflow")
    
    def validate_cuda_setup(self) -> Optional[str]:
        """Validate CUDA installation and return version."""
        if not self.has_gpu:
            logger.info("No GPU detected, skipping CUDA validation")
            return None
        
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("NVIDIA GPU detected ✓")
                
                # Try to detect CUDA version
                nvcc_result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
                if nvcc_result.returncode == 0:
                    import re
                    match = re.search(r'release (\d+\.\d+)', nvcc_result.stdout)
                    if match:
                        cuda_version = match.group(1)
                        logger.info(f"CUDA version: {cuda_version} ✓")
                        return cuda_version
                
            return "11.8"  # Default CUDA version
                
        except FileNotFoundError:
            logger.warning("nvidia-smi not found. GPU support may not be available.")
            return None
    
    def create_initial_config(self):
        """Create initial configuration files if they don't exist."""
        config_dir = self.project_root / "config"
        config_dir.mkdir(exist_ok=True)
        
        # The config.yaml already exists, so we'll create additional configs
        
        # Create logging configuration
        logging_config = """version: 1
disable_existing_loggers: false

formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
    stream: ext://sys.stdout
    
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/hms_eeg.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  '':
    level: INFO
    handlers: [console, file]
    
  src:
    level: DEBUG
    handlers: [console, file]
    propagate: false
"""
        
        with open(config_dir / "logging.yaml", "w") as f:
            f.write(logging_config)
        
        logger.info("Configuration files created ✓")
    
    def setup_docker_files(self):
        """Create Docker configuration files."""
        # Create Dockerfile
        dockerfile_content = """# Multi-stage build for HMS EEG Classification System

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \\
    python3.10 \\
    python3-pip \\
    python3-dev \\
    build-essential \\
    wget \\
    git \\
    libhdf5-dev \\
    libopenblas-dev \\
    liblapack-dev \\
    libjpeg-dev \\
    zlib1g-dev \\
    && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Stage 2: Dependencies
FROM base AS dependencies

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \\
    pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies AS app

WORKDIR /app

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/processed models/final results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Expose ports
EXPOSE 8000 8501 5000

# Default command
CMD ["python", "-m", "uvicorn", "src.deployment.api:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        with open(self.project_root / "Dockerfile", "w") as f:
            f.write(dockerfile_content)
        
        # Create docker-compose.yml
        docker_compose_content = """version: '3.8'

services:
  # Main application API
  api:
    build:
      context: .
      dockerfile: Dockerfile
    image: hms-eeg-classifier:latest
    container_name: hms-api
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - API_WORKERS=4
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    networks:
      - hms-network

  # MLflow tracking server
  mlflow:
    image: python:3.10-slim
    container_name: hms-mlflow
    ports:
      - "5000:5000"
    volumes:
      - ./experiments/mlflow:/mlflow
    environment:
      - MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow/mlflow.db
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
    command: >
      sh -c "pip install mlflow==2.5.0 &&
             mlflow server --host 0.0.0.0 --port 5000
             --backend-store-uri $${MLFLOW_BACKEND_STORE_URI}
             --default-artifact-root $${MLFLOW_DEFAULT_ARTIFACT_ROOT}"
    restart: unless-stopped
    networks:
      - hms-network

  # Redis for caching
  redis:
    image: redis:7-alpine
    container_name: hms-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped
    networks:
      - hms-network

  # PostgreSQL database
  postgres:
    image: postgres:15-alpine
    container_name: hms-postgres
    environment:
      - POSTGRES_USER=hms_user
      - POSTGRES_PASSWORD=hms_password
      - POSTGRES_DB=hms_eeg
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped
    networks:
      - hms-network

  # Frontend application
  frontend:
    build:
      context: ./webapp/frontend
      dockerfile: Dockerfile
    container_name: hms-frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://api:8000
    depends_on:
      - api
    restart: unless-stopped
    networks:
      - hms-network

  # Monitoring with Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: hms-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    restart: unless-stopped
    networks:
      - hms-network

  # Grafana for visualization
  grafana:
    image: grafana/grafana:latest
    container_name: hms-grafana
    ports:
      - "3001:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./config/grafana:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - hms-network

networks:
  hms-network:
    driver: bridge

volumes:
  redis-data:
  postgres-data:
  prometheus-data:
  grafana-data:
"""
        
        with open(self.project_root / "docker-compose.yml", "w") as f:
            f.write(docker_compose_content)
        
        # Create .dockerignore
        dockerignore_content = """# Git
.git/
.gitignore

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
venv/
env/

# Jupyter
.ipynb_checkpoints/
notebooks/

# Data (only include processed)
data/raw/
data/interim/

# Logs
logs/

# Testing
.pytest_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/

# Documentation
docs/

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db
"""
        
        with open(self.project_root / ".dockerignore", "w") as f:
            f.write(dockerignore_content)
        
        logger.info("Docker configuration files created ✓")
    
    def create_makefile(self):
        """Create Makefile for common tasks."""
        makefile_content = """# HMS EEG Classification System - Makefile

.PHONY: help setup install clean train deploy test docs

help:
	@echo "HMS EEG Classification System"
	@echo "============================"
	@echo "Available commands:"
	@echo "  make setup      - Complete project setup"
	@echo "  make install    - Install dependencies"
	@echo "  make download   - Download dataset from Kaggle"
	@echo "  make train      - Train all models"
	@echo "  make deploy     - Deploy API and services"
	@echo "  make test       - Run all tests"
	@echo "  make docs       - Build documentation"
	@echo "  make clean      - Clean up generated files"

setup:
	python setup.py --full

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pre-commit install

download:
	python train.py --download-only

prepare-data:
	python prepare_data.py --config config/config.yaml

train:
	python train.py --config config/config.yaml

train-gpu:
	CUDA_VISIBLE_DEVICES=0 python train.py --config config/config.yaml

deploy:
	docker-compose up -d

deploy-dev:
	uvicorn src.deployment.api:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --cov=src --cov-report=html

test-parallel:
	pytest tests/ -v -n auto --cov=src --cov-report=html

lint:
	black src/ tests/
	isort src/ tests/
	flake8 src/ tests/
	mypy src/

docs:
	cd docs && make html

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf logs/*.log

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

tensorboard:
	tensorboard --logdir logs/tensorboard

jupyter:
	jupyter lab --ip 0.0.0.0 --no-browser

format:
	black src/ tests/ --line-length 100
	isort src/ tests/ --profile black

requirements:
	pip freeze > requirements-freeze.txt
"""
        
        with open(self.project_root / "Makefile", "w") as f:
            f.write(makefile_content)
        
        logger.info("Makefile created ✓")
    
    def run_full_setup(self, cuda_version: Optional[str] = None):
        """Run complete project setup."""
        logger.info("Starting HMS EEG Classification System setup...")
        logger.info("=" * 50)
        
        # Check Python version
        if not self.check_python_version():
            sys.exit(1)
        
        # Create directory structure
        self.create_directory_structure()
        
        # Setup virtual environment
        venv_path = self.setup_virtual_environment()
        
        # Install dependencies
        if cuda_version or self.has_gpu:
            cuda_ver = cuda_version or self.validate_cuda_setup()
            self.install_dependencies(venv_path, cuda_ver)
        else:
            self.install_dependencies(venv_path)
        
        # Setup Jupyter kernel
        self.setup_jupyter_kernel(venv_path)
        
        # Setup Git hooks
        self.setup_git_hooks()
        
        # Create environment files
        self.setup_environment_files()
        
        # Setup MLflow
        self.setup_mlflow()
        
        # Create configs
        self.create_initial_config()
        
        # Setup Docker
        self.setup_docker_files()
        
        # Create Makefile
        self.create_makefile()
        
        logger.info("\n" + "=" * 50)
        logger.info("✅ Setup completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Activate virtual environment:")
        if self.platform == "Windows":
            logger.info("   venv\\Scripts\\activate")
        else:
            logger.info("   source venv/bin/activate")
        logger.info("2. Configure Kaggle API:")
        logger.info("   - Copy your kaggle.json to ~/.kaggle/")
        logger.info("   - Or set KAGGLE_USERNAME and KAGGLE_KEY in .env")
        logger.info("3. Download the dataset:")
        logger.info("   make download")
        logger.info("4. Prepare data:")
        logger.info("   make prepare-data")
        logger.info("5. Start training:")
        logger.info("   make train")
        logger.info("\nFor more commands, run: make help")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Setup HMS EEG Classification System")
    parser.add_argument("--full", action="store_true", help="Run full setup")
    parser.add_argument("--cuda-version", type=str, help="CUDA version (e.g., 11.8)")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip GPU setup")
    
    args = parser.parse_args()
    
    setup = ProjectSetup()
    
    if args.full:
        setup.run_full_setup(cuda_version=args.cuda_version)
    else:
        # Interactive setup
        logger.info("HMS EEG Classification System - Setup")
        logger.info("=====================================")
        response = input("Run full setup? (y/n): ")
        if response.lower() == 'y':
            setup.run_full_setup(cuda_version=args.cuda_version)
        else:
            logger.info("Setup cancelled.")


if __name__ == "__main__":
    main() 