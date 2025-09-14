#!/bin/bash

# Deploy pre-built image to Railway
# This avoids build timeouts by building locally and pushing to registry

echo "🚀 Building optimized image locally..."
docker build -f Dockerfile.slim-optimized -t mental-health-bot:latest .

echo "📦 Tagging for Railway registry..."
# Replace 'your-railway-registry' with your actual Railway registry
docker tag mental-health-bot:latest your-railway-registry/mental-health-bot:latest

echo "⬆️ Pushing to Railway registry..."
docker push your-railway-registry/mental-health-bot:latest

echo "✅ Image pushed! Now configure Railway to use the pre-built image."
