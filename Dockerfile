# Multi-stage Dockerfile for HMS EEG ClASsification System

# Stage 1: BASe dependencies
FROM python:3.9-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    libgfortran5 \
    libffi-dev \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy scipy && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Development environment
FROM base AS development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    pytest-asyncio \
    black \
    flake8 \
    mypy \
    ipython \
    jupyter

# Copy source code
COPY . .

# Set Python path to include src directory
ENV PYTHONPATH="/app/src:/app:$PYTHONPATH"

# Expose ports
EXPOSE 8000 8888

# Default command for development
CMD ["python", "-m", "uvicorn", "src.deployment.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 3: Model training environment
FROM base AS training

# Install additional ML dependencies
RUN pip install --no-cache-dir \
    tensorboard \
    wandb \
    optuna

# Copy source code
COPY . .

# Set Python path to include src directory
ENV PYTHONPATH="/app/src:/app:$PYTHONPATH"

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/logs /app/checkpoints

# Default command for training
CMD ["python", "train.py"]

# Stage 4: Production API server
FROM base AS production

# Install production server
RUN pip install --no-cache-dir \
    gunicorn \
    uvicorn[standard]

# Copy only necessary files
COPY src /app/src
COPY config /app/config
COPY README.md /app/

# Set Python path to include src directory
ENV PYTHONPATH="/app/src:/app:$PYTHONPATH"

# Create non-root user
RUN useradd -m -u 1000 eeg_user && \
    chown -R eeg_user:eeg_user /app

# Switch to non-root user
USER eeg_user

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command with gunicorn
CMD ["gunicorn", "src.deployment.api:app", \
     "-w", "4", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info"]

# Stage 5: Model optimization environment
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS gpu-optimization

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3.9-venv \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 AS default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies with GPU support
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy scipy && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    onnx \
    onnxruntime-gpu \
    tensorrt \
    torch-tensorrt

# Copy source code
COPY . .

# Set Python path to include src directory
ENV PYTHONPATH="/app/src:/app:$PYTHONPATH"

# Default command for optimization
CMD ["python", "-m", "src.deployment.optimization"]

# Stage 6: Monitoring and metrics
FROM prom/prometheus:latest AS prometheus

# Copy Prometheus configuration
COPY monitoring/prometheus.yml /etc/prometheus/prometheus.yml

# Stage 7: Web interface builder - TO BE ADDED WHEN FRONTEND IS READY
# FROM node:18-alpine AS web-builder
# 
# WORKDIR /app
# 
# # Copy web application source
# COPY webapp/frontend/package*.json ./
# RUN npm ci --only=production
# 
# COPY webapp/frontend/ ./
# RUN npm run build

# Stage 8: Nginx for serving web interface - TO BE CONFIGURED WHEN FRONTEND IS READY
# FROM nginx:alpine AS web-server
# 
# # Copy built web application
# COPY --from=web-builder /app/dist /usr/share/nginx/html
# 
# # Copy nginx configuration
# COPY webapp/nginx.conf /etc/nginx/nginx.conf
# 
# # Expose web port
# EXPOSE 80
# 
# # Health check
# HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
#     CMD wget --no-verbose --tries=1 --spider http://localhost || exit 1

# Stage 9: Project runner - NEW STAGE
FROM base AS runner

# Install additional tools needed for running the project
RUN apt-get update && apt-get install -y \
    docker.io \
    docker-compose \
    nodejs \
    npm \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy entire project
COPY . .

# Fix file permissions
RUN chmod +x *.py *.sh scripts/*.py 2>/dev/null || true && \
    find src/ -type f -name "*.py" -exec chmod 644 {} \; && \
    chmod -R 755 data/ logs/ models/ backups/ 2>/dev/null || true

# Set Python path to include src directory
ENV PYTHONPATH="/app/src:/app:$PYTHONPATH"

# Create necessary directories
RUN mkdir -p data/raw data/processed data/models logs backups ssl \
    monitoring/grafana/dashboards monitoring/grafana/datasources \
    models/registry models/deployments models/final models/optimized \
    checkpoints

# Set entrypoint to handle different commands
ENTRYPOINT ["python"]
CMD ["run_project.py"] 