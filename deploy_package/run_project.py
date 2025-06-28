#!/usr/bin/env python3
"""
HMS EEG Classification System - Complete Project Runner
Run the entire project with a single command
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

class HMSProjectRunner:
    """Main class to handle the complete project execution"""
    
    def __init__(self, skip_download: bool = False, skip_train: bool = False):
        self.project_root = Path(__file__).parent
        self.skip_download = skip_download
        self.skip_train = skip_train
        self.processes = []
        
    def print_colored(self, message: str, color: str = NC):
        """Print colored message"""
        print(f"{color}{message}{NC}")
        
    def check_requirements(self) -> bool:
        """Check if all requirements are installed"""
        self.print_colored("Checking system requirements...", BLUE)
        
        requirements = {
            'docker': 'Docker',
            'docker-compose': 'Docker Compose',
            'python3': 'Python 3.8+',
            'npm': 'Node.js/npm'
        }
        
        missing = []
        for cmd, name in requirements.items():
            if shutil.which(cmd) is None:
                missing.append(name)
                
        if missing:
            self.print_colored(f"Missing requirements: {', '.join(missing)}", RED)
            self.print_colored("Please install missing requirements and try again.", RED)
            return False
            
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            self.print_colored("Python 3.8+ is required", RED)
            return False
            
        self.print_colored("✓ All requirements satisfied", GREEN)
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
        env_example = self.project_root / '.env.example'
        
        if not env_file.exists() and env_example.exists():
            shutil.copy(env_example, env_file)
            self.print_colored("Created .env file. Please update with your credentials.", YELLOW)
            
        # Create frontend .env.local if not exists
        frontend_env = self.project_root / 'webapp/frontend/.env.local'
        frontend_env_example = self.project_root / 'webapp/frontend/.env.example'
        
        if not frontend_env.exists() and frontend_env_example.exists():
            shutil.copy(frontend_env_example, frontend_env)
            
        self.print_colored("✓ Environment setup complete", GREEN)
        
    def install_dependencies(self):
        """Install Python and Node.js dependencies"""
        self.print_colored("Installing dependencies...", BLUE)
        
        # Install Python dependencies
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True)
        
        # Install frontend dependencies
        frontend_dir = self.project_root / 'webapp/frontend'
        if frontend_dir.exists():
            subprocess.run(['npm', 'install'], cwd=frontend_dir, check=True)
            
        self.print_colored("✓ Dependencies installed", GREEN)
        
    def download_dataset(self):
        """Download dataset from Kaggle"""
        if self.skip_download:
            self.print_colored("Skipping dataset download (--skip-download flag)", YELLOW)
            return
            
        self.print_colored("Downloading dataset from Kaggle...", BLUE)
        
        # Check if Kaggle credentials are set
        env_vars = os.environ.copy()
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value
                    
        if not env_vars.get('KAGGLE_USERNAME') or not env_vars.get('KAGGLE_KEY'):
            self.print_colored("Please set KAGGLE_USERNAME and KAGGLE_KEY in .env file", RED)
            return
            
        # Run download script
        subprocess.run(['docker-compose', 'run', '--rm', 'data-downloader'], 
                      env=env_vars, check=True)
        
        self.print_colored("✓ Dataset downloaded", GREEN)
        
    def prepare_data(self):
        """Prepare and preprocess data"""
        self.print_colored("Preparing data...", BLUE)
        
        # Run data preparation script
        subprocess.run([sys.executable, 'prepare_data.py'], check=True)
        
        self.print_colored("✓ Data prepared", GREEN)
        
    def train_models(self):
        """Train the models"""
        if self.skip_train:
            self.print_colored("Skipping model training (--skip-train flag)", YELLOW)
            return
            
        self.print_colored("Training models...", BLUE)
        
        # Check if GPU is available
        try:
            subprocess.run(['nvidia-smi'], capture_output=True, check=True)
            self.print_colored("✓ GPU detected", GREEN)
        except:
            self.print_colored("⚠ No GPU detected, training will be slower", YELLOW)
            
        # Run training
        subprocess.run(['docker-compose', 'run', '--rm', 'trainer'], check=True)
        
        self.print_colored("✓ Model training complete", GREEN)
        
    def optimize_models(self):
        """Optimize models for deployment"""
        self.print_colored("Optimizing models...", BLUE)
        
        subprocess.run([
            sys.executable, 'scripts/optimize_models.py',
            '--config', 'config/config.yaml',
            '--model-dir', 'models/final',
            '--output-dir', 'models/optimized',
            '--optimization-level', '2'
        ], check=True)
        
        self.print_colored("✓ Model optimization complete", GREEN)
        
    def build_containers(self):
        """Build Docker containers"""
        self.print_colored("Building Docker containers...", BLUE)
        
        subprocess.run(['docker-compose', 'build'], check=True)
        
        self.print_colored("✓ Containers built", GREEN)
        
    def start_services(self):
        """Start all services"""
        self.print_colored("Starting all services...", BLUE)
        
        # Start databases first
        subprocess.run(['docker-compose', 'up', '-d', 'postgres', 'redis'], check=True)
        time.sleep(10)  # Wait for databases to initialize
        
        # Start all other services
        subprocess.run(['docker-compose', 'up', '-d'], check=True)
        
        self.print_colored("✓ All services started", GREEN)
        
    def wait_for_services(self):
        """Wait for all services to be ready"""
        self.print_colored("Waiting for services to be ready...", BLUE)
        
        services = [
            ('API', 'http://localhost:8000/health'),
            ('Frontend', 'http://localhost:3000'),
            ('MLflow', 'http://localhost:5000'),
            ('Grafana', 'http://localhost:3001/api/health'),
            ('Dashboard', 'http://localhost:8050'),
        ]
        
        import requests
        max_retries = 30
        
        for name, url in services:
            retries = 0
            while retries < max_retries:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code < 500:
                        self.print_colored(f"✓ {name} is ready", GREEN)
                        break
                except:
                    pass
                    
                retries += 1
                time.sleep(2)
                
            if retries >= max_retries:
                self.print_colored(f"⚠ {name} failed to start", YELLOW)
                
    def open_browser(self):
        """Open the main dashboard in browser"""
        self.print_colored("Opening dashboard in browser...", BLUE)
        
        urls = [
            ('Main Dashboard', 'http://localhost'),
            ('API Documentation', 'http://localhost:8000/docs'),
            ('MLflow', 'http://localhost:5000'),
            ('Grafana', 'http://localhost:3001'),
            ('Visualization Dashboard', 'http://localhost:8050'),
        ]
        
        # Open main dashboard
        webbrowser.open('http://localhost')
        
        self.print_colored("\nAccess URLs:", GREEN)
        for name, url in urls:
            print(f"  {name}: {url}")
            
    def cleanup(self, signum=None, frame=None):
        """Cleanup function"""
        self.print_colored("\nCleaning up...", YELLOW)
        
        # Stop all services
        subprocess.run(['docker-compose', 'down'], capture_output=True)
        
        self.print_colored("✓ Cleanup complete", GREEN)
        sys.exit(0)
        
    def run(self):
        """Run the complete project"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        
        try:
            # Check requirements
            if not self.check_requirements():
                return
                
            # Setup environment
            self.setup_environment()
            
            # Install dependencies
            self.install_dependencies()
            
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
                
            # Build containers
            self.build_containers()
            
            # Start services
            self.start_services()
            
            # Wait for services
            self.wait_for_services()
            
            # Open browser
            self.open_browser()
            
            self.print_colored("\n✅ HMS EEG Classification System is fully deployed!", GREEN)
            self.print_colored("Press Ctrl+C to stop all services", YELLOW)
            
            # Keep the script running
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
        description='HMS EEG Classification System - Complete Project Runner'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip dataset download (use existing data)'
    )
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip model training (use existing models)'
    )
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Run in development mode'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "="*60)
    print("HMS EEG Classification System")
    print("Harmful Brain Activity Detection")
    print("="*60 + "\n")
    
    runner = HMSProjectRunner(
        skip_download=args.skip_download,
        skip_train=args.skip_train
    )
    runner.run()


if __name__ == '__main__':
    main() 