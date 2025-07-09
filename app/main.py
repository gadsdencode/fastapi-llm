import os
import logging
import asyncio
from contextlib import asynccontextmanager

# Environment optimizations for CPU inference
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())
os.environ["VECLIB_MAXIMUM_THREADS"] = str(os.cpu_count())
os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())

# Enable Railway-specific optimizations
if os.getenv("RAILWAY_ENVIRONMENT"):
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "100000"
    os.environ["MALLOC_MMAP_THRESHOLD_"] = "131072"

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.api.endpoints import router, limiter
from app.models.llm_handler import llm_handler
from app.schemas.models import ModelType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "bartowski/Phi-3.5-mini-instruct_Uncensored-GGUF")
MODEL_TYPE = os.getenv("MODEL_TYPE", "huggingface") 
LOAD_MODEL_ON_STARTUP = os.getenv("LOAD_MODEL_ON_STARTUP", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("Starting FastAPI LLM Inference Server")
    logger.info(f"Model: {MODEL_NAME} (Type: {MODEL_TYPE})")
    logger.info(f"Load on startup: {LOAD_MODEL_ON_STARTUP}")
    
    # Load model on startup if configured
    if LOAD_MODEL_ON_STARTUP:
        try:
            logger.info("Loading model on startup...")
            model_type = ModelType(MODEL_TYPE)
            await llm_handler.load_model(MODEL_NAME, model_type)
            logger.info("Model loaded successfully on startup")
        except Exception as e:
            logger.error(f"Failed to load model on startup: {str(e)}")
            logger.info("Server will start without a loaded model")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI LLM Inference Server")


# Create FastAPI app
app = FastAPI(
    title="FastAPI LLM Inference Server",
    description="Production-ready FastAPI server for LLM inference with streaming support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic server information"""
    return {
        "message": "FastAPI LLM Inference Server",
        "version": "1.0.0",
        "model_loaded": llm_handler.is_loaded(),
        "current_model": llm_handler.model_name if llm_handler.is_loaded() else None,
        "endpoints": {
            "generate": "/api/v1/generate",
            "generate_stream": "/api/v1/generate/stream",
            "model_info": "/api/v1/model/info",
            "model_load": "/api/v1/model/load",
            "health": "/api/v1/health",
            "ping": "/api/v1/ping"
        }
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred"
        }
    )


# Health check endpoint (duplicate for Railway)
@app.get("/ping")
async def ping():
    """Simple ping endpoint for Railway health checks"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    ) 