import os
import time
import logging
from typing import Optional
import json

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
@limiter.limit("10/minute")
async def generate_text(
    request: Request,
    generate_request: GenerateRequest,
    _: bool = Depends(verify_api_key)
):
    """Generate text from prompt"""
    try:
        if generate_request.stream:
            raise HTTPException(
                status_code=400,
                detail="Use /generate/stream endpoint for streaming responses"
            )
        
        response = await llm_handler.generate(generate_request)
        return response
        
    except RuntimeError as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in generate: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/generate/stream")
@limiter.limit("5/minute")
async def generate_stream(
    request: Request,
    generate_request: GenerateRequest,
    _: bool = Depends(verify_api_key)
):
    """Generate text with streaming response"""
    try:
        async def stream_generator():
            """Generator for streaming response"""
            async for chunk in llm_handler.generate_stream(generate_request):
                # Format as Server-Sent Events
                data = json.dumps(chunk.dict())
                yield f"data: {data}\n\n"
                
                if chunk.is_final:
                    yield "data: [DONE]\n\n"
                    break
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except RuntimeError as e:
        logger.error(f"Streaming generation error: {str(e)}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in generate_stream: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/model/info", response_model=ModelInfo)
@limiter.limit("30/minute")
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
@limiter.limit("3/minute")
async def load_model(
    request: Request,
    load_request: ModelLoadRequest,
    _: bool = Depends(verify_api_key)
):
    """Load a model on demand"""
    try:
        success = await llm_handler.load_model(
            load_request.model_name,
            load_request.model_type,
            load_request.force_reload
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
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Detailed health check endpoint"""
    try:
        # Calculate uptime (this would need to be tracked from app startup)
        uptime = time.time() - getattr(health_check, 'start_time', time.time())
        
        return HealthResponse(
            status="healthy",
            uptime=uptime,
            model_loaded=llm_handler.is_loaded(),
            memory_usage=llm_handler.get_memory_usage()
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
@limiter.limit("120/minute")
async def ping(request: Request):
    """Simple ping endpoint for Railway health checks"""
    return {"status": "ok", "timestamp": time.time()}


# Store start time for uptime calculation
health_check.start_time = time.time()

# Rate limit error handler
# @router.exception_handler(RateLimitExceeded)
# async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
#     response = HTTPException(
#         status_code=429,
#         detail=f"Rate limit exceeded: {exc.detail}"
#     )
#     return response 