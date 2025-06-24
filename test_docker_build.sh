#!/bin/bash
# Test script to verify Docker build works without frontend

echo "Testing Docker build without frontend..."

# Check if docker-compose.yml is valid
echo "1. Validating docker-compose.yml..."
docker-compose config > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ docker-compose.yml is valid"
else
    echo "✗ docker-compose.yml has errors"
    docker-compose config
fi

# Test building the base image
echo ""
echo "2. Testing Dockerfile build..."
docker build -t hms-test:latest --target base . > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Dockerfile builds successfully"
else
    echo "✗ Dockerfile build failed"
fi

# List services
echo ""
echo "3. Available services:"
docker-compose ps --services

echo ""
echo "Test complete!" 