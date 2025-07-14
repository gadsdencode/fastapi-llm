#!/bin/bash

# Azure App Service startup script for FastAPI LLM server
set -e

echo "Starting Azure App Service deployment..."
echo "PORT: ${PORT:-8000}"
echo "AZURE_DEPLOYMENT: ${AZURE_DEPLOYMENT:-true}"
echo "LOAD_MODEL_ON_STARTUP: ${LOAD_MODEL_ON_STARTUP:-false}"

# Ensure models directory exists
mkdir -p /app/models

# Set Python path
export PYTHONPATH=/app

# Start the application with Azure-optimized settings
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers 1 \
    --timeout-keep-alive 300 \
    --log-level info \
    --access-log \
    --loop uvloop \
    --http httptools 