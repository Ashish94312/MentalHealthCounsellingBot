#!/bin/bash

# Deploy Mental Health Bot to Railway using pre-built image
echo "🚀 Building optimized image locally..."
docker build -f Dockerfile.slim-optimized -t mental-health-bot:latest .

echo "📦 Tagging for Docker Hub..."
docker tag mental-health-bot:latest ashish85297/mental-health-bot:latest

echo "⬆️ Pushing to Docker Hub..."
docker push ashish85297/mental-health-bot:latest

echo "✅ Image pushed to Docker Hub!"
echo "🔄 Railway will automatically pull the latest image on next deployment."
echo ""
echo "Your image is available at: https://hub.docker.com/r/ashish85297/mental-health-bot"
echo "Railway will use: ashish85297/mental-health-bot:latest"
