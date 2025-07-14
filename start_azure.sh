#!/bin/bash
# Azure App Service startup script for FastAPI LLM server
# This script handles the startup with proper error handling and logging

echo "=== Azure App Service FastAPI Startup ==="
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Port: ${PORT:-8000}"
echo "Azure SKU: ${WEBSITE_SKU:-Free}"

# Set Azure deployment flag
export AZURE_DEPLOYMENT=true

# Add Python path
export PYTHONPATH="/home/site/wwwroot:/home/site/wwwroot/app:$PYTHONPATH"

# Check if the main app exists
if [ -f "/home/site/wwwroot/app/main.py" ]; then
    echo "Found app/main.py - using app.main:app"
    APP_MODULE="app.main:app"
elif [ -f "/home/site/wwwroot/main.py" ]; then
    echo "Found main.py - using main:app"
    APP_MODULE="main:app"
else
    echo "ERROR: Could not find main.py or app/main.py"
    exit 1
fi

# Start with gunicorn + uvicorn worker for better Azure compatibility
echo "Starting FastAPI with gunicorn + uvicorn worker..."
exec gunicorn "$APP_MODULE" \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind "0.0.0.0:${PORT:-8000}" \
    --workers 1 \
    --timeout 600 \
    --keep-alive 30 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --access-logfile '-' \
    --error-logfile '-' \
    --log-level info 