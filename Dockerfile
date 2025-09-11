# Use a slim Python base image
FROM python:3.11-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=chatbot_project.settings

# Workdir
WORKDIR /app

# System deps (build tools for any wheels)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# App source
COPY . /app

# Collect static files at build time
RUN python chatbot_project/manage.py collectstatic --noinput

# Expose the default port (Railway will set $PORT)
EXPOSE 8000

# Run migrations and start gunicorn
CMD ["sh", "-c", "python chatbot_project/manage.py migrate && gunicorn chatbot_project.wsgi:application --bind 0.0.0.0:${PORT:-8000}"]
