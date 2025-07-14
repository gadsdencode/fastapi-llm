# Azure-optimized Dockerfile for FastAPI LLM server
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Azure App Service
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Azure-specific environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV AZURE_DEPLOYMENT=true
ENV LOAD_MODEL_ON_STARTUP=false
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
ENV MODEL_TYPE=gguf
ENV PORT=8000

# Azure App Service optimizations
ENV WEBSITE_TIME_ZONE="UTC"
ENV WEBSITE_ENABLE_SYNC_UPDATE_SITE=true

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory with proper permissions
RUN mkdir -p /app/models && \
    chmod 755 /app/models

# Make startup scripts executable
RUN chmod +x startup.py && \
    chmod +x start_azure.sh

# Create non-root user for security (Azure compatible)
RUN useradd --create-home --shell /bin/bash --uid 1000 app && \
    chown -R app:app /app && \
    usermod -aG root app
USER app

# Expose port (Azure App Service will use PORT env var)
EXPOSE $PORT

# Azure-optimized health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=2 \
    CMD curl -f http://localhost:$PORT/ping || exit 1

# Use Azure-optimized startup script
CMD ["python", "startup.py"] 