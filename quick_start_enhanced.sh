#!/bin/bash

# Enhanced HMS EEG Classification - Novita AI Quick Start
# Complete deployment with advanced features and resume capability

set -e

echo "🧠 Enhanced HMS EEG Classification - Novita AI Quick Start"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SSH_KEY_PATH=""
SSH_HOST=""
SKIP_DEPLOY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ssh-key)
            SSH_KEY_PATH="$2"
            shift 2
            ;;
        --ssh-host)
            SSH_HOST="$2"
            shift 2
            ;;
        --skip-deploy)
            SKIP_DEPLOY=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --ssh-key PATH     Path to SSH private key"
            echo "  --ssh-host IP      Novita AI instance IP address"
            echo "  --skip-deploy      Skip deployment, just show status"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --ssh-key ~/.ssh/novita_key --ssh-host 192.168.1.100"
            echo "  $0 --skip-deploy  # Just show current status"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to print colored messages
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    print_message $BLUE "🔍 Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_message $RED "❌ Python3 not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check required files
    local required_files=(
        "run_novita_training_enhanced.py"
        "deploy_novita_enhanced.py" 
        "config/novita_enhanced_config.yaml"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            print_message $RED "❌ Required file not found: $file"
            exit 1
        fi
    done
    
    print_message $GREEN "✅ Prerequisites check passed"
}

# Function to show enhanced features
show_enhanced_features() {
    print_message $PURPLE "🚀 Enhanced Features Available:"
    echo ""
    echo "✅ EEG Foundation Model - Self-supervised pre-training"
    echo "✅ Advanced Ensemble Methods - Stacking + Bayesian averaging"
    echo "✅ Robust Resume Capability - Automatic checkpoint recovery"
    echo "✅ Memory Optimization - H100-optimized training"
    echo "✅ Enhanced Monitoring - Real-time GPU/cost tracking"
    echo "✅ Smart Backups - Automatic state preservation"
    echo "✅ Stage-wise Training - Independent recoverable stages"
    echo ""
}

# Function to show expected results
show_expected_results() {
    print_message $CYAN "📊 Expected Enhanced Results:"
    echo ""
    echo "🎯 Accuracy: >92% (enhanced models)"
    echo "⏱️  Training Time: 10-14 hours (all features)"
    echo "💰 Total Cost: \$30-45"
    echo "📦 Dataset: Full 106,800 samples"
    echo "🤖 Models: Foundation + ResNet + EfficientNet + Ensemble"
    echo "🎁 Output: Production ONNX + Foundation model"
    echo ""
}

# Function to deploy enhanced system
deploy_enhanced_system() {
    if [[ "$SKIP_DEPLOY" == true ]]; then
        print_message $YELLOW "⏭️  Skipping deployment (--skip-deploy flag set)"
        return
    fi
    
    if [[ -z "$SSH_HOST" ]] || [[ -z "$SSH_KEY_PATH" ]]; then
        print_message $YELLOW "⚠️  SSH configuration not provided"
        print_message $BLUE "Please provide SSH details for deployment:"
        echo ""
        echo "Usage: $0 --ssh-key ~/.ssh/your_key --ssh-host YOUR_IP"
        echo ""
        echo "Or run interactively:"
        read -p "SSH Host (Novita AI IP): " SSH_HOST
        read -p "SSH Key Path: " SSH_KEY_PATH
        echo ""
    fi
    
    if [[ -n "$SSH_HOST" ]] && [[ -n "$SSH_KEY_PATH" ]]; then
        print_message $BLUE "🚀 Deploying enhanced system to Novita AI..."
        
        # Configure deployment
        python3 deploy_novita_enhanced.py --ssh-host "$SSH_HOST" --ssh-key "$SSH_KEY_PATH"
        
        # Deploy enhanced package
        print_message $BLUE "📦 Creating and deploying enhanced package..."
        python3 deploy_novita_enhanced.py --deploy-enhanced
        
        if [[ $? -eq 0 ]]; then
            print_message $GREEN "✅ Enhanced deployment completed successfully!"
            echo ""
            print_message $CYAN "🎯 Next Steps:"
            echo "1. Start enhanced training:"
            echo "   python3 deploy_novita_enhanced.py --start-enhanced"
            echo ""
            echo "2. Monitor training:"
            echo "   python3 deploy_novita_enhanced.py --status"
            echo ""
            echo "3. SSH to instance:"
            echo "   python3 deploy_novita_enhanced.py --ssh"
            echo "   # Then run: hms-monitor"
            echo ""
        else
            print_message $RED "❌ Deployment failed. Check the logs above."
            exit 1
        fi
    else
        print_message $YELLOW "⏭️  Skipping deployment - SSH details not provided"
    fi
}

# Function to show resume instructions
show_resume_instructions() {
    print_message $PURPLE "🔄 IMPORTANT: Resume Capability"
    echo ""
    echo "If your credits run out and training stops:"
    echo ""
    echo "1. 🔋 Top up credits and restart Novita AI instance"
    echo "2. 🔗 SSH back into the instance"
    echo "3. 🚀 Quick restart:"
    echo "   cd /workspace"
    echo "   ./restart_training.sh"
    echo ""
    echo "Or use remote resume:"
    echo "   python3 deploy_novita_enhanced.py --resume"
    echo ""
    print_message $GREEN "✅ Training will resume exactly where it stopped!"
    echo ""
}

# Function to show monitoring commands
show_monitoring_commands() {
    print_message $CYAN "📈 Enhanced Monitoring Commands:"
    echo ""
    echo "Local (from your machine):"
    echo "  python3 deploy_novita_enhanced.py --status"
    echo "  python3 deploy_novita_enhanced.py --resume"
    echo "  python3 deploy_novita_enhanced.py --ssh"
    echo ""
    echo "Remote (on Novita AI instance):"
    echo "  hms-monitor          # Real-time dashboard"
    echo "  hms-status           # Quick status check"
    echo "  hms-resume           # Smart resume"
    echo "  backup-state         # Manual backup"
    echo "  ./restart_training.sh # Quick restart"
    echo ""
}

# Function to check current status
check_current_status() {
    if [[ -n "$SSH_HOST" ]] && [[ -n "$SSH_KEY_PATH" ]]; then
        print_message $BLUE "📊 Checking current training status..."
        python3 deploy_novita_enhanced.py --ssh-host "$SSH_HOST" --ssh-key "$SSH_KEY_PATH" --status
    else
        print_message $YELLOW "⚠️  Cannot check status - SSH details not provided"
    fi
}

# Main execution
main() {
    print_message $PURPLE "🧠 Enhanced HMS EEG Classification - Novita AI"
    print_message $PURPLE "Complete system with advanced features & resume capability"
    echo ""
    
    # Check prerequisites
    check_prerequisites
    echo ""
    
    # Show enhanced features
    show_enhanced_features
    
    # Show expected results
    show_expected_results
    
    # Deploy enhanced system
    deploy_enhanced_system
    echo ""
    
    # Show resume instructions
    show_resume_instructions
    
    # Show monitoring commands
    show_monitoring_commands
    
    # Check current status if possible
    if [[ "$SKIP_DEPLOY" == true ]]; then
        check_current_status
    fi
    
    echo ""
    print_message $GREEN "🎉 Enhanced HMS Novita AI setup complete!"
    print_message $CYAN "📚 For detailed instructions, see: README_ENHANCED_NOVITA.md"
    echo ""
}

# Run main function
main "$@" 