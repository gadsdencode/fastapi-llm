import os
import time
import logging
from typing import Optional, Dict, Any
import json
import hashlib
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Depends, Header, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.schemas.models import (
    GenerateRequest, 
    GenerateResponse, 
    ModelInfo, 
    ModelLoadRequest, 
    HealthResponse,
    ErrorResponse
)
from app.models.llm_handler import llm_handler

logger = logging.getLogger(__name__)

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

# Security setup
security = HTTPBearer(auto_error=False)

# API Key authentication
API_KEY = os.getenv("API_KEY")

# Response caching
@lru_cache(maxsize=100)
def get_cache_key(prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
    """Generate cache key for request"""
    content = f"{prompt}_{max_tokens}_{temperature}_{top_p}"
    return hashlib.md5(content.encode()).hexdigest()

# In-memory cache for responses
response_cache: Dict[str, tuple[GenerateResponse, float]] = {}
CACHE_TTL = 3600  # 1 hour

def verify_api_key(x_api_key: Optional[str] = Header(None)) -> bool:
    """Verify API key if configured"""
    if API_KEY is None:
        return True  # No API key required
    
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API key required. Please provide x-api-key header."
        )
    
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return True


@router.post("/generate", response_model=GenerateResponse)
@limiter.limit("20/minute")  # Increased rate limit after optimizations
async def generate_text(
    request: Request,
    generate_request: GenerateRequest,
    _: bool = Depends(verify_api_key)
):
    """Generate text from prompt with caching"""
    try:
        if generate_request.stream:
            raise HTTPException(
                status_code=400,
                detail="Use /generate/stream endpoint for streaming responses"
            )
        
        # Check cache first for identical requests
        cache_key = get_cache_key(
            generate_request.prompt,
            generate_request.max_tokens,
            generate_request.temperature,
            generate_request.top_p
        )
        
        if cache_key in response_cache:
            cached_response, timestamp = response_cache[cache_key]
            if time.time() - timestamp < CACHE_TTL:
                logger.info(f"Returning cached response for key: {cache_key[:8]}")
                return cached_response
            else:
                # Remove expired cache entry
                del response_cache[cache_key]
        
        response = await llm_handler.generate(generate_request)
        
        # Cache the response
        response_cache[cache_key] = (response, time.time())
        
        # Clean up cache if it gets too large
        if len(response_cache) > 200:
            # Remove oldest 50 entries
            sorted_items = sorted(response_cache.items(), key=lambda x: x[1][1])
            for k, _ in sorted_items[:50]:
                del response_cache[k]
        
        return response
        
    except RuntimeError as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in generate: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/generate/stream")
@limiter.limit("10/minute")  # Increased rate limit for streaming
async def generate_stream(
    request: Request,
    generate_request: GenerateRequest,
    _: bool = Depends(verify_api_key)
):
    """Generate text with streaming response"""
    try:
        async def stream_generator():
            """Generator for streaming response with optimized formatting"""
            async for chunk in llm_handler.generate_stream(generate_request):
                # Format as Server-Sent Events
                data = json.dumps(chunk.dict())
                yield f"data: {data}\n\n"
                
                # Check if this is the final chunk
                if chunk.is_final:
                    yield "data: [DONE]\n\n"
                    break
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "X-Accel-Buffering": "no"  # Disable nginx buffering for better streaming
            }
        )
        
    except RuntimeError as e:
        logger.error(f"Streaming generation error: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in generate_stream: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/model/info", response_model=ModelInfo)
@limiter.limit("60/minute")  # Increased rate limit for info endpoint
async def get_model_info(
    request: Request,
    _: bool = Depends(verify_api_key)
):
    """Get information about the currently loaded model"""
    try:
        return llm_handler.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/model/load")
@limiter.limit("5/minute")  # Slightly increased rate limit for model loading
async def load_model(
    request: Request,
    load_request: ModelLoadRequest,
    _: bool = Depends(verify_api_key)
):
    """Load a model on demand with cache clearing"""
    try:
        # Clear response cache when loading a new model
        global response_cache
        if load_request.force_reload or llm_handler.model_name != load_request.model_name:
            response_cache.clear()
            logger.info("Cleared response cache due to model change")
        
        success = await llm_handler.load_model(
            load_request.model_name,
            load_request.model_type,
            load_request.force_reload,
            load_request.preferred_quant
        )
        
        if success:
            return {"message": f"Model {load_request.model_name} loaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
            
    except ValueError as e:
        logger.error(f"Invalid model load request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
@limiter.limit("120/minute")  # Increased rate limit for health checks
async def health_check(request: Request):
    """Detailed health check endpoint with performance metrics"""
    try:
        # Calculate uptime (this would need to be tracked from app startup)
        uptime = time.time() - getattr(health_check, 'start_time', time.time())
        
        # Add cache statistics
        cache_stats = {
            "cache_size": len(response_cache),
            "cache_hit_ratio": getattr(health_check, 'cache_hits', 0) / max(getattr(health_check, 'total_requests', 1), 1)
        }
        
        memory_usage = llm_handler.get_memory_usage()
        memory_usage["cache_stats"] = cache_stats
        
        return HealthResponse(
            status="healthy",
            uptime=uptime,
            model_loaded=llm_handler.is_loaded(),
            memory_usage=memory_usage
        )
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            uptime=0,
            model_loaded=False,
            memory_usage=None
        )


@router.get("/ping")
@limiter.limit("240/minute")  # Increased rate limit for ping endpoint
async def ping(request: Request):
    """Simple ping endpoint for Railway health checks"""
    return {"status": "ok", "timestamp": time.time()}


# Cache management endpoints
@router.post("/cache/clear")
@limiter.limit("10/minute")
async def clear_cache(
    request: Request,
    _: bool = Depends(verify_api_key)
):
    """Clear the response cache"""
    global response_cache
    cache_size = len(response_cache)
    response_cache.clear()
    logger.info(f"Manually cleared response cache ({cache_size} entries)")
    return {"message": f"Cache cleared successfully ({cache_size} entries removed)"}


@router.get("/cache/stats")
@limiter.limit("30/minute")
async def get_cache_stats(
    request: Request,
    _: bool = Depends(verify_api_key)
):
    """Get cache statistics"""
    return {
        "cache_size": len(response_cache),
        "cache_entries": [
            {
                "key": key[:8] + "...",
                "age_seconds": time.time() - timestamp,
                "tokens_generated": cached_response.tokens_generated
            }
            for key, (cached_response, timestamp) in list(response_cache.items())[:10]
        ]
    }


# Store start time for uptime calculation
health_check.start_time = time.time()
health_check.cache_hits = 0
health_check.total_requests = 0 