#!/bin/bash
# Fix file permissions for HMS EEG Classification System

echo "🔧 Fixing file permissions..."

# Make scripts executable
chmod +x *.py
chmod +x *.sh
chmod +x scripts/*.py

# Make data directories writable
mkdir -p data/raw data/processed data/models logs backups
chmod -R 755 data/
chmod -R 755 logs/
chmod -R 755 models/ 2>/dev/null || mkdir -p models && chmod -R 755 models/
chmod -R 755 backups/ 2>/dev/null || mkdir -p backups && chmod -R 755 backups/

# Make config readable
chmod -R 644 config/*

# Make source code readable
find src/ -type f -name "*.py" -exec chmod 644 {} \;

echo "✅ File permissions fixed!"
echo ""
echo "You can now run:"
echo "  make build    - Build containers"
echo "  make up       - Start services (CPU only)"
echo "  make up-gpu   - Start services with GPU (if available)"