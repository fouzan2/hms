#!/bin/bash
# HMS EEG Classification System - Docker Runner Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Print colored output
print_color() {
    color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Show help
show_help() {
    echo "HMS EEG Classification System - Docker Runner"
    echo "==========================================="
    echo ""
    echo "Usage: ./docker-run.sh [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start         Start all services"
    echo "  stop          Stop all services"
    echo "  restart       Restart all services"
    echo "  build         Build Docker containers"
    echo "  train         Train models"
    echo "  logs          Show logs"
    echo "  shell         Open shell in container"
    echo "  status        Show service status"
    echo "  clean         Clean up containers"
    echo ""
    echo "Quick start:"
    echo "  ./docker-run.sh start --quick    # Skip download & training"
    echo "  ./docker-run.sh start            # Full setup"
    echo ""
}

# Check Docker installation
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_color $RED "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_color $RED "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_color $RED "Docker daemon is not running. Please start Docker."
        exit 1
    fi
}

# Main command handler
case "${1:-help}" in
    start)
        check_docker
        if [[ "$2" == "--quick" ]]; then
            print_color $BLUE "Quick start (skipping download & training)..."
            make quick
        else
            print_color $BLUE "Starting full project..."
            make all
        fi
        ;;
    
    stop)
        print_color $YELLOW "Stopping all services..."
        make down
        ;;
    
    restart)
        print_color $YELLOW "Restarting all services..."
        make restart
        ;;
    
    build)
        check_docker
        print_color $BLUE "Building Docker containers..."
        make build
        ;;
    
    train)
        print_color $BLUE "Training models..."
        make train
        ;;
    
    logs)
        if [[ -n "$2" ]]; then
            make logs-$2
        else
            make logs
        fi
        ;;
    
    shell)
        service="${2:-api}"
        print_color $BLUE "Opening shell in $service container..."
        docker-compose run --rm $service bash
        ;;
    
    status)
        make status
        ;;
    
    clean)
        print_color $YELLOW "Cleaning up..."
        make clean
        ;;
    
    dev)
        print_color $BLUE "Starting development environment..."
        make dev
        ;;
    
    test)
        print_color $BLUE "Running tests..."
        make test
        ;;
    
    help|--help|-h|*)
        show_help
        ;;
esac 