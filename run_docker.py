#!/usr/bin/env python3
"""
HMS EEG Classification System - Docker-based Project Runner
Run the entire project using Docker containers
"""

import os
import sys
import subprocess
import time
import signal
import argparse
from pathlib import Path
import shutil
import webbrowser
from typing import List, Dict, Optional

# ANSI color codes
BLUE = '\033[0;34m'
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[0;33m'
NC = '\033[0m'  # No Color

class DockerProjectRunner:
    """Docker-based project runner"""
    
    def __init__(self, skip_download: bool = False, skip_train: bool = False, dev_mode: bool = False):
        self.project_root = Path(__file__).parent
        self.skip_download = skip_download
        self.skip_train = skip_train
        self.dev_mode = dev_mode
        
    def print_colored(self, message: str, color: str = NC):
        """Print colored message"""
        print(f"{color}{message}{NC}")
        
    def check_docker(self) -> bool:
        """Check if Docker and Docker Compose are installed"""
        self.print_colored("Checking Docker installation...", BLUE)
        
        requirements = {
            'docker': 'Docker',
            'docker-compose': 'Docker Compose',
        }
        
        missing = []
        for cmd, name in requirements.items():
            if shutil.which(cmd) is None:
                missing.append(name)
                
        if missing:
            self.print_colored(f"Missing requirements: {', '.join(missing)}", RED)
            self.print_colored("Please install Docker and Docker Compose:", RED)
            self.print_colored("  https://docs.docker.com/get-docker/", YELLOW)
            self.print_colored("  https://docs.docker.com/compose/install/", YELLOW)
            return False
            
        # Check if Docker daemon is running
        try:
            subprocess.run(['docker', 'info'], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            self.print_colored("Docker daemon is not running. Please start Docker.", RED)
            return False
            
        self.print_colored("✓ Docker is ready", GREEN)
        return True
        
    def setup_environment(self):
        """Setup the environment"""
        self.print_colored("Setting up environment...", BLUE)
        
        # Create necessary directories
        dirs = [
            'data/raw', 'data/processed', 'data/models',
            'logs', 'backups', 'ssl',
            'monitoring/grafana/dashboards',
            'monitoring/grafana/datasources',
            'models/registry', 'models/deployments',
            'models/final', 'models/optimized',
            'checkpoints'
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        # Create .env file if not exists
        env_file = self.project_root / '.env'
        if not env_file.exists():
            self.print_colored("Creating .env file...", YELLOW)
            with open(env_file, 'w') as f:
                f.write("""# HMS EEG Classification System Environment Variables

# Database
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=hms_eeg
POSTGRES_USER=hms_user

# Redis
REDIS_PASSWORD=redis_password

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000

# Grafana
GRAFANA_PASSWORD=admin

# Jupyter
JUPYTER_TOKEN=jupyter_token

# Kaggle (for dataset download)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key

# GPU Settings
CUDA_VISIBLE_DEVICES=0
USE_OPTIMIZED_MODELS=true

# Feature flags
ENABLE_MONITORING=true
ENABLE_DISTRIBUTED_SERVING=false
ENABLE_MIXED_PRECISION=true
ENABLE_GRADIENT_CHECKPOINTING=true
ENABLE_TENSORRT=false

# Debug
DASH_DEBUG=false
""")
            self.print_colored("Created .env file. Please update KAGGLE credentials if needed.", YELLOW)
            
        self.print_colored("✓ Environment setup complete", GREEN)
        
    def build_containers(self):
        """Build all Docker containers"""
        self.print_colored("Building Docker containers...", BLUE)
        
        # Build main containers
        subprocess.run(['docker-compose', 'build', '--no-cache'], check=True)
        
        self.print_colored("✓ All containers built", GREEN)
        
    def download_dataset(self):
        """Download dataset using Docker"""
        if self.skip_download:
            self.print_colored("Skipping dataset download (--skip-download flag)", YELLOW)
            return
            
        self.print_colored("Downloading dataset from Kaggle...", BLUE)
        
        # Check if Kaggle credentials are set
        if not os.path.exists('.env'):
            self.print_colored("Please create .env file with KAGGLE credentials", RED)
            return
            
        # Use docker-compose to download dataset
        subprocess.run([
            'docker-compose', 'run', '--rm', 
            '-v', f'{os.path.expanduser("~/.kaggle")}:/root/.kaggle:ro',
            'data-downloader'
        ], check=True)
        
        self.print_colored("✓ Dataset downloaded", GREEN)
        
    def prepare_data(self):
        """Prepare data using Docker"""
        self.print_colored("Preparing data...", BLUE)
        
        # Run data preparation in Docker
        subprocess.run([
            'docker', 'run', '--rm',
            '-v', f'{self.project_root}/data:/app/data',
            '-v', f'{self.project_root}/src:/app/src',
            '-v', f'{self.project_root}/config:/app/config',
            '-v', f'{self.project_root}/prepare_data.py:/app/prepare_data.py',
            '--entrypoint', 'python',
            'hms-runner:latest',
            'prepare_data.py'
        ], check=True)
        
        self.print_colored("✓ Data prepared", GREEN)
        
    def train_models(self):
        """Train models using Docker"""
        if self.skip_train:
            self.print_colored("Skipping model training (--skip-train flag)", YELLOW)
            return
            
        self.print_colored("Training models...", BLUE)
        
        # Check if GPU is available
        try:
            subprocess.run(['nvidia-smi'], capture_output=True, check=True)
            self.print_colored("✓ GPU detected", GREEN)
            gpu_flag = '--gpus all'
        except:
            self.print_colored("⚠ No GPU detected, training will be slower", YELLOW)
            gpu_flag = ''
            
        # Run training with docker-compose
        cmd = ['docker-compose', 'run', '--rm']
        if gpu_flag:
            cmd.extend(['--gpus', 'all'])
        cmd.append('trainer')
        
        subprocess.run(cmd, check=True)
        
        self.print_colored("✓ Model training complete", GREEN)
        
    def optimize_models(self):
        """Optimize models using Docker"""
        self.print_colored("Optimizing models...", BLUE)
        
        subprocess.run([
            'docker-compose', 'run', '--rm',
            'optimizer'
        ], check=True)
        
        self.print_colored("✓ Model optimization complete", GREEN)
        
    def start_services(self):
        """Start all services with Docker Compose"""
        self.print_colored("Starting all services...", BLUE)
        
        # Start core services first
        self.print_colored("Starting core services...", BLUE)
        subprocess.run([
            'docker-compose', 'up', '-d',
            'postgres', 'redis', 'kafka', 'zookeeper'
        ], check=True)
        
        # Wait for core services
        time.sleep(15)
        
        # Start remaining services
        self.print_colored("Starting application services...", BLUE)
        if self.dev_mode:
            subprocess.run(['docker-compose', '--profile', 'development', 'up', '-d'], check=True)
        else:
            subprocess.run(['docker-compose', 'up', '-d'], check=True)
            
        self.print_colored("✓ All services started", GREEN)
        
    def wait_for_services(self):
        """Wait for services to be ready"""
        self.print_colored("Waiting for services to be ready...", BLUE)
        
        services = [
            ('API', 'http://localhost:8000/health', 60),
            ('MLflow', 'http://localhost:5000', 30),
            ('Grafana', 'http://localhost:3001/api/health', 30),
            ('Dashboard', 'http://localhost:8050', 30),
        ]
        
        import requests
        
        for name, url, max_wait in services:
            start_time = time.time()
            while time.time() - start_time < max_wait:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code < 500:
                        self.print_colored(f"✓ {name} is ready", GREEN)
                        break
                except:
                    pass
                time.sleep(2)
            else:
                self.print_colored(f"⚠ {name} may not be ready", YELLOW)
                
    def show_urls(self):
        """Show access URLs"""
        self.print_colored("\nAccess URLs:", GREEN)
        urls = [
            ('API Documentation', 'http://localhost:8000/docs'),
            ('API Health Check', 'http://localhost:8000/health'),
            ('MLflow UI', 'http://localhost:5000'),
            ('Grafana Monitoring', 'http://localhost:3001 (admin/admin)'),
            ('EEG Dashboard', 'http://localhost:8050'),
        ]
        
        if self.dev_mode:
            urls.append(('Jupyter Lab', 'http://localhost:8888 (token: jupyter_token)'))
            
        for name, url in urls:
            print(f"  {name}: {url}")
            
    def show_logs(self, service: Optional[str] = None):
        """Show logs for services"""
        if service:
            subprocess.run(['docker-compose', 'logs', '-f', service])
        else:
            subprocess.run(['docker-compose', 'logs', '-f'])
            
    def stop_services(self):
        """Stop all services"""
        self.print_colored("\nStopping all services...", YELLOW)
        subprocess.run(['docker-compose', 'down'], check=True)
        self.print_colored("✓ All services stopped", GREEN)
        
    def cleanup(self, signum=None, frame=None):
        """Cleanup on exit"""
        self.stop_services()
        sys.exit(0)
        
    def run(self):
        """Run the complete project"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        
        try:
            # Check Docker
            if not self.check_docker():
                return
                
            # Setup environment
            self.setup_environment()
            
            # Build containers
            self.build_containers()
            
            # Download dataset
            self.download_dataset()
            
            # Prepare data
            if not self.skip_download:
                self.prepare_data()
                
            # Train models
            self.train_models()
            
            # Optimize models
            if not self.skip_train:
                self.optimize_models()
                
            # Start services
            self.start_services()
            
            # Wait for services
            self.wait_for_services()
            
            # Show URLs
            self.show_urls()
            
            self.print_colored("\n✅ HMS EEG Classification System is running!", GREEN)
            self.print_colored("Press Ctrl+C to stop all services", YELLOW)
            
            # Open browser
            webbrowser.open('http://localhost:8000/docs')
            
            # Keep running
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.cleanup()
        except Exception as e:
            self.print_colored(f"Error: {str(e)}", RED)
            self.cleanup()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='HMS EEG Classification System - Docker Runner'
    )
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip dataset download')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip model training')
    parser.add_argument('--dev', action='store_true',
                        help='Run in development mode')
    parser.add_argument('--logs', type=str, nargs='?', const='all',
                        help='Show logs for specific service or all')
    parser.add_argument('--stop', action='store_true',
                        help='Stop all services')
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "="*60)
    print("HMS EEG Classification System - Docker Edition")
    print("Harmful Brain Activity Detection")
    print("="*60 + "\n")
    
    if args.stop:
        runner = DockerProjectRunner()
        runner.stop_services()
    elif args.logs:
        runner = DockerProjectRunner()
        service = None if args.logs == 'all' else args.logs
        runner.show_logs(service)
    else:
        runner = DockerProjectRunner(
            skip_download=args.skip_download,
            skip_train=args.skip_train,
            dev_mode=args.dev
        )
        runner.run()


if __name__ == '__main__':
    main() 