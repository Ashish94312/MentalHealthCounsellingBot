# Multi-stage build to reduce final image size
FROM python:3.11-slim AS builder

# Environment for builder stage
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-production.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r /tmp/requirements.txt

# Final stage - minimal runtime image
FROM python:3.11-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=chatbot_project.settings \
    PATH=/root/.local/bin:$PATH

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Set workdir
WORKDIR /app

# Copy only necessary application files
COPY chatbot_project/ /app/chatbot_project/

# Copy only the specific model directory needed for production
COPY tinylama-mental-health-mentalchat16k/ /app/tinylama-mental-health-mentalchat16k/

# Collect static files as root (before switching to non-root user)
RUN python chatbot_project/manage.py collectstatic --noinput

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /home/app && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/')" || exit 1

# Run migrations and start gunicorn
CMD ["sh", "-c", "python chatbot_project/manage.py migrate && gunicorn chatbot_project.wsgi:application --bind 0.0.0.0:${PORT:-8000} --workers 2 --timeout 120"]
