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
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

# Security setup
security = HTTPBearer(auto_error=False)

# API Key authentication
API_KEY = os.getenv("API_KEY")

# Redis connection setup with optimized pooling
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# OPTIMIZED Redis client with connection pooling
redis_client = redis.from_url(
    REDIS_URL, 
    decode_responses=True,
    max_connections=20,          # Connection pool size
    retry_on_timeout=True,       # Retry on timeout
    socket_connect_timeout=5,    # Connection timeout
    socket_timeout=5,            # Socket timeout
    health_check_interval=30     # Health check every 30s
)

# Comment out the in-memory cache for reference
# response_cache: Dict[str, tuple[GenerateResponse, float]] = {}
# CACHE_TTL = 3600  # 1 hour
CACHE_TTL = 3600  # 1 hour

async def get_cached_response(cache_key: str):
    try:
        cached = await redis_client.get(cache_key)
        if cached:
            # Use compressed JSON deserialization for better performance
            import json
            import gzip
            import base64
            
            try:
                # Try compressed format first
                compressed_data = base64.b64decode(cached)
                decompressed_data = gzip.decompress(compressed_data)
                data = json.loads(decompressed_data.decode('utf-8'))
                return GenerateResponse.model_validate(data)
            except:
                # Fallback to regular JSON
                return GenerateResponse.model_validate_json(cached)
    except Exception as e:
        logger.error(f"Redis get error: {e}")
    return None

async def cache_response(cache_key: str, response: GenerateResponse, ttl: int = CACHE_TTL):
    try:
        # Use compressed JSON serialization for better performance
        import json
        import gzip
        import base64
        
        # Serialize and compress
        json_data = response.model_dump_json()
        compressed_data = gzip.compress(json_data.encode('utf-8'))
        encoded_data = base64.b64encode(compressed_data).decode('utf-8')
        
        await redis_client.setex(cache_key, ttl, encoded_data)
    except Exception as e:
        logger.error(f"Redis set error: {e}")

async def clear_response_cache():
    try:
        # Use pipeline for better performance when clearing multiple keys
        async with redis_client.pipeline() as pipe:
            keys = await redis_client.keys("cache:*")
            if keys:
                for key in keys:
                    pipe.delete(key)
                await pipe.execute()
    except Exception as e:
        logger.error(f"Redis clear error: {e}")

async def get_cache_stats():
    try:
        # Use pipeline for multiple Redis operations
        async with redis_client.pipeline() as pipe:
            pipe.keys("cache:*")
            pipe.info("memory")
            results = await pipe.execute()
            
            keys = results[0] if results else []
            memory_info = results[1] if len(results) > 1 else {}
            
            return {
                "cache_keys": len(keys),
                "memory_usage": memory_info.get("used_memory_human", "unknown"),
                "redis_connections": memory_info.get("connected_clients", "unknown")
            }
    except Exception as e:
        logger.error(f"Redis stats error: {e}")
        return {"cache_keys": 0, "memory_usage": "unknown", "redis_connections": "unknown"}

# Response caching
# @lru_cache(maxsize=100)
def get_cache_key(prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
    """Generate cache key for request"""
    content = f"{prompt}_{max_tokens}_{temperature}_{top_p}"
    return hashlib.md5(content.encode()).hexdigest()

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


@router.post("/generate")
@limiter.limit("20/minute")  # Increased rate limit after optimizations
async def generate_text(
    request: Request,
    generate_request: GenerateRequest,
    _: bool = Depends(verify_api_key)
):
    """Generate text from prompt with caching and thread pool optimization"""
    try:
        if generate_request.stream:
            raise HTTPException(
                status_code=400,
                detail="Use /generate/stream endpoint for streaming responses"
            )
        
        # Check cache first for identical requests
        cache_key = f"cache:{get_cache_key(generate_request.prompt, generate_request.max_tokens, generate_request.temperature, generate_request.top_p)}"
        cached_response = await get_cached_response(cache_key)
        if cached_response:
            logger.info(f"Returning cached response for key: {cache_key[:16]}")
            return cached_response
        
        # OPTIMIZATION: Move LLM inference to thread pool to prevent event loop blocking
        import asyncio
        response = await asyncio.to_thread(llm_handler.generate_sync, generate_request)
        
        await cache_response(cache_key, response)
        
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
    """OPTIMIZED streaming response with proper async generator and thread pool"""
    try:
        # CRITICAL FIX: Proper async generator implementation
        async def optimized_stream_generator():
            """Optimized async generator that prevents Starlette streaming bug"""
            import asyncio
            import json
            
            # Buffer chunks for better performance
            chunk_buffer = []
            buffer_size = 3  # Send chunks in groups of 3
            
            try:
                # Move streaming to thread pool to prevent event loop blocking
                async for chunk in llm_handler.generate_stream_async(generate_request):
                    # Format as Server-Sent Events with optimized JSON serialization
                    data = json.dumps(chunk.dict(), separators=(',', ':'))  # Compact JSON
                    chunk_buffer.append(f"data: {data}\n\n")
                    
                    # Batch send for better network efficiency
                    if len(chunk_buffer) >= buffer_size or chunk.is_final:
                        # Send all buffered chunks at once
                        for buffered_chunk in chunk_buffer:
                            yield buffered_chunk
                        chunk_buffer = []
                        
                        # Small delay to prevent overwhelming client
                        if not chunk.is_final:
                            await asyncio.sleep(0.001)  # 1ms delay
                    
                    # Check if this is the final chunk
                    if chunk.is_final:
                        yield "data: [DONE]\n\n"
                        break
                        
            except Exception as e:
                # Send error in SSE format
                error_data = json.dumps({"error": str(e)})
                yield f"data: {error_data}\n\n"
                yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            optimized_stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Connection": "keep-alive", 
                "Content-Type": "text/event-stream; charset=utf-8",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Access-Control-Allow-Origin": "*",  # CORS for streaming
                "Access-Control-Expose-Headers": "*"
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
        await clear_response_cache()
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
            "cache_size": await get_cache_stats()["cache_keys"]
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
    await clear_response_cache()
    return {"message": "Cache cleared successfully"}


@router.get("/cache/stats")
@limiter.limit("30/minute")
async def get_cache_stats(
    request: Request,
    _: bool = Depends(verify_api_key)
):
    """Get cache statistics"""
    return await get_cache_stats()


# Store start time for uptime calculation
health_check.start_time = time.time()
health_check.cache_hits = 0
health_check.total_requests = 0 