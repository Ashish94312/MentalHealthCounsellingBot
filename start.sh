#!/bin/bash

# Memory-efficient startup script for Railway
echo "🚀 Starting Mental Health Bot with memory optimizations..."

# Set memory-efficient environment variables
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run migrations
echo "📊 Running database migrations..."
python chatbot_project/manage.py migrate --noinput

# Start the application with memory optimizations
echo "🤖 Starting Gunicorn with memory optimizations..."
exec gunicorn chatbot_project.wsgi:application \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers 1 \
    --worker-class sync \
    --timeout 120 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --preload \
    --worker-connections 1000
