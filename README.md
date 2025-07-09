---
title: FastAPI
description: A FastAPI server
tags:
  - fastapi
  - hypercorn
  - python
---

# FastAPI LLM Inference Server

A production-ready FastAPI server for LLM inference with streaming support, containerized with Docker and deployable to Railway.

## Features

- 🚀 **Production-ready**: Built with FastAPI, proper error handling, and logging
- 🔄 **Streaming Support**: Real-time text generation with Server-Sent Events
- 🧠 **Multiple Model Types**: Extensible architecture supporting Hugging Face, GGUF, and Ollama
- 🔒 **Security**: Optional API key authentication and rate limiting
- 📊 **Monitoring**: Health checks, memory usage tracking, and uptime monitoring
- 🐳 **Containerized**: Docker support with multi-stage builds
- 🚂 **Railway Ready**: Configured for seamless Railway deployment

## Project Structure

```
fastapi-llm-server/
├── app/
│   ├── main.py                  # FastAPI application entrypoint
│   ├── models/llm_handler.py    # Model loader and inference handler
│   ├── schemas/models.py        # Pydantic models for API schemas
│   ├── api/endpoints.py         # REST API endpoints
│   └── __init__.py
├── models/                      # Model cache directory
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── railway.json                 # Railway deployment configuration
└── README.md                    # This file
```

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fastapi-llm-server
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables** (optional)
   ```bash
   export MODEL_NAME="microsoft/DialoGPT-small"
   export MODEL_TYPE="huggingface"
   export PORT=8000
   export TRANSFORMERS_CACHE="./models"
   export HF_HOME="./models"
   ```

4. **Run the server**
   ```bash
   uvicorn app.main:app --reload
   ```

5. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/ping
   - Model Info: http://localhost:8000/api/v1/model/info

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t fastapi-llm-server .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 \
     -e MODEL_NAME="microsoft/DialoGPT-small" \
     -e MODEL_TYPE="huggingface" \
     -v $(pwd)/models:/app/models \
     fastapi-llm-server
   ```

## API Endpoints

### Text Generation

#### POST `/api/v1/generate`
Generate text from a prompt (non-streaming).

**Request Body:**
```json
{
  "prompt": "Hello, how are you?",
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 256,
  "stop_sequences": ["Human:", "AI:"]
}
```

**Response:**
```json
{
  "generated_text": "I'm doing well, thank you for asking!",
  "prompt": "Hello, how are you?",
  "model_name": "microsoft/DialoGPT-small",
  "tokens_generated": 8,
  "generation_time": 0.45
}
```

#### POST `/api/v1/generate/stream`
Generate text with streaming response using Server-Sent Events.

**Request Body:** Same as `/generate`

**Response:** Stream of JSON chunks:
```json
{"text": "I'm", "is_final": false, "tokens_generated": 1}
{"text": " doing", "is_final": false, "tokens_generated": 2}
{"text": "", "is_final": true, "tokens_generated": 8}
```

### Model Management

#### GET `/api/v1/model/info`
Get information about the currently loaded model.

**Response:**
```json
{
  "name": "microsoft/DialoGPT-small",
  "type": "huggingface",
  "loaded": true,
  "memory_usage": {
    "system": {"total": 16000000000, "used": 8000000000},
    "gpu": {"allocated": 1000000000}
  },
  "parameters": {
    "total_params": 117000000,
    "trainable_params": 117000000
  },
  "load_time": 15.2
}
```

#### POST `/api/v1/model/load`
Load a new model on demand.

**Request Body:**
```json
{
  "model_name": "microsoft/DialoGPT-medium",
  "model_type": "huggingface",
  "force_reload": false
}
```

### Health Monitoring

#### GET `/api/v1/health`
Detailed health check with system information.

**Response:**
```json
{
  "status": "healthy",
  "uptime": 3600.5,
  "model_loaded": true,
  "memory_usage": {...},
  "timestamp": 1699123456.789
}
```

#### GET `/api/v1/ping`
Simple ping endpoint for basic health checks.

**Response:**
```json
{
  "status": "ok",
  "timestamp": 1699123456.789
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `microsoft/DialoGPT-small` | Hugging Face model name |
| `MODEL_TYPE` | `huggingface` | Model type (huggingface, gguf, ollama) |
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Server host |
| `TRANSFORMERS_CACHE` | `./models` | Model cache directory |
| `HF_HOME` | `./models` | Hugging Face home directory |
| `LOAD_MODEL_ON_STARTUP` | `true` | Load model on server startup |
| `API_KEY` | `None` | Optional API key for authentication |

### Authentication

To enable API key authentication, set the `API_KEY` environment variable:

```bash
export API_KEY="your-secret-api-key"
```

Then include the key in requests:

```bash
curl -H "x-api-key: your-secret-api-key" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello!"}' \
     http://localhost:8000/api/v1/generate
```

### Rate Limiting

Built-in rate limiting is configured per endpoint:
- `/generate`: 10 requests/minute
- `/generate/stream`: 5 requests/minute
- `/model/load`: 3 requests/minute
- `/health`: 60 requests/minute
- `/ping`: 120 requests/minute

## Railway Deployment

### Prerequisites

1. [Railway account](https://railway.app)
2. Railway CLI installed

### Deployment Steps

1. **Connect to Railway**
   ```bash
   railway login
   railway link
   ```

2. **Set environment variables**
   ```bash
   railway variables set MODEL_NAME=microsoft/DialoGPT-small
   railway variables set MODEL_TYPE=huggingface
   railway variables set TRANSFORMERS_CACHE=/app/models
   railway variables set HF_HOME=/app/models
   railway variables set LOAD_MODEL_ON_STARTUP=true
   ```

3. **Deploy**
   ```bash
   railway up
   ```

### Railway Environment Variables

Set these variables in your Railway dashboard:

```ini
PORT=8000
MODEL_NAME=microsoft/DialoGPT-small
MODEL_TYPE=huggingface
PYTHONPATH=/app
TRANSFORMERS_CACHE=/app/models
HF_HOME=/app/models
LOAD_MODEL_ON_STARTUP=true
```

## Extending Model Support

The server is designed to be extensible. To add support for new model types:

1. **Add model type to enum** in `app/schemas/models.py`:
   ```python
   class ModelType(str, Enum):
       HUGGINGFACE = "huggingface"
       GGUF = "gguf"
       OLLAMA = "ollama"
       YOUR_MODEL = "your_model"  # Add here
   ```

2. **Implement loader** in `app/models/llm_handler.py`:
   ```python
   async def load_model(self, model_name: str, model_type: ModelType = ModelType.HUGGINGFACE, force_reload: bool = False):
       # ... existing code ...
       elif model_type == ModelType.YOUR_MODEL:
           await self._load_your_model(model_name)
   ```

3. **Add dependencies** to `requirements.txt` as needed.

## Performance Optimization

### Memory Management

- Models are cached in the `models/` directory
- GPU memory is automatically managed by PyTorch
- System memory usage is monitored and reported

### Scaling

- Use multiple worker processes: `uvicorn app.main:app --workers 4`
- Deploy multiple instances behind a load balancer
- Consider GPU acceleration for larger models

## Troubleshooting

### Common Issues

1. **Model loading fails**
   - Check internet connection for model download
   - Verify model name and type
   - Ensure sufficient disk space in cache directory

2. **Out of memory errors**
   - Use smaller models or reduce batch sizes
   - Enable CPU offloading for large models
   - Monitor memory usage via `/api/v1/health`

3. **Slow inference**
   - Enable GPU acceleration if available
   - Reduce `max_tokens` in requests
   - Use model-specific optimizations

### Logs

Check application logs for detailed error information:

```bash
# Docker logs
docker logs <container-id>

# Railway logs
railway logs
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Support

For issues and questions:
- Check the [documentation](http://localhost:8000/docs)
- Review the logs for error details
- Open an issue on the repository
