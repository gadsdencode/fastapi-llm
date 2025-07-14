import os
import logging
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.api.endpoints import router, limiter
from app.models.llm_handler import llm_handler
from app.schemas.models import ModelType

# Environment optimizations for CPU inference
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())
os.environ["VECLIB_MAXIMUM_THREADS"] = str(os.cpu_count())
os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable Railway-specific optimizations
if os.getenv("RAILWAY_ENVIRONMENT"):
    os.environ["MALLOC_TRIM_THRESHOLD_"] = "100000"
    os.environ["MALLOC_MMAP_THRESHOLD_"] = "131072"

# Enable Azure-specific optimizations
if os.getenv("AZURE_DEPLOYMENT"):
    # Detect Azure App Service tier for optimal configuration
    azure_sku = os.getenv("WEBSITE_SKU", "Free")
    logger.info(f"Azure App Service tier detected: {azure_sku}")
    
    # Disable model loading on startup for faster Azure startup
    os.environ["LOAD_MODEL_ON_STARTUP"] = "false"
    
    if azure_sku in ["Free", "Shared"]:
        # Conservative settings for lower tiers
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["LLM_MAX_CONTEXT"] = "512"
        logger.info("Applied Free/Shared tier optimizations")
    elif azure_sku in ["Basic"]:
        # Basic tier optimizations
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"
        os.environ["LLM_MAX_CONTEXT"] = "1024"
        logger.info("Applied Basic tier optimizations")
    else:
        # Standard/Premium tier optimizations
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"
        os.environ["LLM_MAX_CONTEXT"] = "2048"
        logger.info("Applied Standard/Premium tier optimizations")

# Configuration from environment variables
MODEL_NAME = os.getenv(
    "MODEL_NAME", "TheBloke/Wizard-Vicuna-7B-Uncensored-GGUF"
)
MODEL_TYPE = os.getenv("MODEL_TYPE", "gguf")
LOAD_MODEL_ON_STARTUP = (
    os.getenv("LOAD_MODEL_ON_STARTUP", "true").lower() == "true"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("Starting FastAPI LLM Inference Server")
    logger.info(f"Azure deployment: {os.getenv('AZURE_DEPLOYMENT', 'false')}")
    logger.info(f"Port: {os.getenv('PORT', '8000')}")
    logger.info(f"Model: {MODEL_NAME} (Type: {MODEL_TYPE})")
    logger.info(f"Load on startup: {LOAD_MODEL_ON_STARTUP}")

    # Create a global HTTPX async client with OPTIMIZED connection pooling
    app.state.http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_keepalive_connections=50,    # Increased keepalive connections
            max_connections=200,             # Increased max connections
            keepalive_expiry=30.0           # Keep connections alive for 30s
        ),
        timeout=httpx.Timeout(
            connect=10.0,    # Connection timeout
            read=60.0,       # Read timeout for LLM responses
            write=10.0,      # Write timeout
            pool=5.0         # Pool timeout
        ),
        http2=True,          # Enable HTTP/2 for better performance
        follow_redirects=True
    )

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
    # Properly close the HTTPX client
    await app.state.http_client.aclose()


# Create FastAPI app
app = FastAPI(
    title="FastAPI LLM Inference Server",
    description="Production-ready FastAPI server for LLM inference "
                "with streaming support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware with optimized settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Limit to needed methods
    allow_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Add GZip compression middleware for faster responses
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Add performance headers middleware
class PerformanceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Add performance headers
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = (
            "no-cache, no-store, must-revalidate"
        )
        response.headers["Server"] = "FastAPI-LLM"
        return response


app.add_middleware(PerformanceMiddleware)

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
    workers = int(os.getenv("WORKERS", "1"))  # Default single worker, set via env

    # For Railway deployment with 48 cores, consider:
    # WORKERS=8-16 (conservative for shared environment)
    # For local development, keep workers=1

    if workers > 1:
        logger.info(f"Starting with {workers} workers for multi-core performance")
        # Use gunicorn for multi-worker deployment
        # Railway: set WORKERS=8 in environment variables
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            workers=workers,
            reload=False,
            log_level="info",
            access_log=False,      # Disable access logs for performance
            use_colors=False,      # Disable colors in production
            loop="uvloop",         # Use faster event loop if available
            http="httptools"       # Use faster HTTP parser if available
        )
    else:
        # Single worker mode (development)
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=False,
            log_level="info",
            loop="uvloop",         # Use faster event loop if available
            http="httptools"       # Use faster HTTP parser if available
        )
