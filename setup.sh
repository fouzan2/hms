#!/bin/bash

# HMS EEG Classification System - One Line Setup Script
# Run with: curl -sSL https://raw.githubusercontent.com/yourusername/hms/main/setup.sh | bash

set -e

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Print colored messages
print_message() {
    echo -e "${2}${1}${NC}"
}

# Banner
echo ""
echo "=============================================="
echo "HMS EEG Classification System"
echo "Automated Setup and Deployment"
echo "=============================================="
echo ""

# Check prerequisites
print_message "Checking prerequisites..." "$BLUE"

# Check Docker
if ! command -v docker &> /dev/null; then
    print_message "Docker is not installed. Please install Docker first:" "$RED"
    echo "  - Ubuntu/Debian: sudo apt install docker.io docker-compose"
    echo "  - macOS: brew install docker docker-compose"
    echo "  - Or visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_message "Docker Compose is not installed. Installing..." "$YELLOW"
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    print_message "Python 3 is not installed. Please install Python 3.8 or higher." "$RED"
    exit 1
fi

# Check npm
if ! command -v npm &> /dev/null; then
    print_message "Node.js/npm is not installed. Please install Node.js 18+." "$RED"
    echo "Visit: https://nodejs.org/"
    exit 1
fi

print_message "✓ All prerequisites satisfied" "$GREEN"

# Check if running in HMS directory
if [ ! -f "docker-compose.yml" ]; then
    print_message "Error: This script must be run from the HMS project directory." "$RED"
    echo "Please clone the repository first:"
    echo "  git clone <repository-url>"
    echo "  cd hms"
    echo "  ./setup.sh"
    exit 1
fi

# Setup Kaggle credentials if not exists
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
    print_message "Kaggle API credentials not found." "$YELLOW"
    echo "Please enter your Kaggle credentials (from https://www.kaggle.com/account):"
    read -p "Kaggle Username: " kaggle_username
    read -sp "Kaggle API Key: " kaggle_key
    echo ""
    
    mkdir -p "$HOME/.kaggle"
    echo "{\"username\":\"$kaggle_username\",\"key\":\"$kaggle_key\"}" > "$HOME/.kaggle/kaggle.json"
    chmod 600 "$HOME/.kaggle/kaggle.json"
    print_message "✓ Kaggle credentials saved" "$GREEN"
fi

# Create .env file if not exists
if [ ! -f ".env" ]; then
    print_message "Creating environment configuration..." "$BLUE"
    
    if [ -f ".env.example" ]; then
        cp .env.example .env
    else
        # Create basic .env file
        cat > .env << EOF
# Database Configuration
POSTGRES_PASSWORD=secure_password
POSTGRES_USER=hms_user
POSTGRES_DB=hms_eeg

# Redis Configuration
REDIS_PASSWORD=redis_password

# Grafana Configuration
GRAFANA_PASSWORD=admin

# Jupyter Configuration
JUPYTER_TOKEN=jupyter_token

# Kaggle API Credentials
KAGGLE_USERNAME=$(jq -r '.username' ~/.kaggle/kaggle.json 2>/dev/null || echo "")
KAGGLE_KEY=$(jq -r '.key' ~/.kaggle/kaggle.json 2>/dev/null || echo "")

# GPU Configuration
CUDA_VISIBLE_DEVICES=0

# Model Optimization
USE_OPTIMIZED_MODELS=true
ENABLE_MONITORING=true

# Environment
ENVIRONMENT=development
EOF
    fi
    
    print_message "✓ Environment configuration created" "$GREEN"
fi

# Create necessary directories
print_message "Creating project directories..." "$BLUE"
mkdir -p data/{raw,processed,models} logs backups ssl
mkdir -p monitoring/grafana/{dashboards,datasources}
mkdir -p models/{registry,deployments,final,optimized}
mkdir -p checkpoints webapp/frontend
print_message "✓ Directories created" "$GREEN"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        print_message "✓ GPU detected - Training will be accelerated" "$GREEN"
    else
        print_message "⚠ GPU drivers not properly installed" "$YELLOW"
    fi
else
    print_message "⚠ No GPU detected - Training will use CPU (slower)" "$YELLOW"
fi

# Ask user for quick start options
echo ""
print_message "Setup Options:" "$BLUE"
echo "1) Full setup (download data, train models) - ~60 minutes"
echo "2) Quick demo (skip download and training) - ~5 minutes"
echo "3) Custom setup"
echo ""
read -p "Select option [1-3]: " setup_option

case $setup_option in
    1)
        RUN_COMMAND="make run"
        ;;
    2)
        RUN_COMMAND="python run_project.py --skip-download --skip-train"
        ;;
    3)
        echo ""
        read -p "Skip dataset download? [y/N]: " skip_download
        read -p "Skip model training? [y/N]: " skip_train
        
        RUN_COMMAND="python run_project.py"
        if [[ $skip_download =~ ^[Yy]$ ]]; then
            RUN_COMMAND="$RUN_COMMAND --skip-download"
        fi
        if [[ $skip_train =~ ^[Yy]$ ]]; then
            RUN_COMMAND="$RUN_COMMAND --skip-train"
        fi
        ;;
    *)
        print_message "Invalid option. Using full setup." "$YELLOW"
        RUN_COMMAND="make run"
        ;;
esac

# Final confirmation
echo ""
print_message "Ready to start HMS EEG Classification System" "$GREEN"
echo "This will:"
echo "  ✓ Install all dependencies"
echo "  ✓ Download dataset from Kaggle (if needed)"
echo "  ✓ Train ML models (if needed)"
echo "  ✓ Start all services with Docker"
echo "  ✓ Launch the web interface"
echo ""
read -p "Continue? [Y/n]: " confirm

if [[ $confirm =~ ^[Nn]$ ]]; then
    print_message "Setup cancelled." "$YELLOW"
    exit 0
fi

# Run the project
echo ""
print_message "Starting HMS EEG Classification System..." "$BLUE"
print_message "This may take a while on first run..." "$YELLOW"
echo ""

# Execute the run command
$RUN_COMMAND

# If we get here, something went wrong
print_message "Setup failed. Please check the logs above for errors." "$RED"
exit 1 