#!/bin/bash

# Fix Kaggle Download Script
# This script rebuilds the Docker container with the fixed Kaggle API version
# and attempts to download the dataset

set -e  # Exit on any error

echo "🔧 Fixing Kaggle API compatibility issue..."
echo "📦 Rebuilding Docker container with pinned Kaggle version (1.5.16)..."

# Rebuild the Docker image to include the fixed requirements
docker-compose build data-downloader

echo "✅ Docker container rebuilt with Kaggle API v1.5.16"
echo ""
echo "🔍 Checking Kaggle credentials..."

# Check if Kaggle credentials exist
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Kaggle credentials not found!"
    echo "Please ensure you have ~/.kaggle/kaggle.json with your API credentials."
    echo ""
    echo "To get your credentials:"
    echo "1. Go to https://www.kaggle.com/account"
    echo "2. Click 'Create New API Token'"
    echo "3. Save the kaggle.json file to ~/.kaggle/"
    echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

echo "✅ Kaggle credentials found"
echo ""
echo "📥 Starting dataset download with fixed Kaggle API..."

# Try the download
if make download; then
    echo ""
    echo "🎉 Dataset download completed successfully!"
    echo "✅ Kaggle API compatibility issue resolved"
else
    echo ""
    echo "⚠️  Download failed. Additional troubleshooting options:"
    echo ""
    echo "1. Check your Kaggle credentials:"
    echo "   cat ~/.kaggle/kaggle.json"
    echo ""
    echo "2. Verify you have access to the competition:"
    echo "   https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification"
    echo ""
    echo "3. Try downloading individual files:"
    echo "   docker-compose run --rm -v ~/.kaggle:/root/.kaggle:ro data-downloader --folders train_eegs"
    echo ""
    echo "4. Check container logs:"
    echo "   docker-compose logs data-downloader"
    echo ""
    exit 1
fi 