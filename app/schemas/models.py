from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List
from enum import Enum
import time


class ModelType(str, Enum):
    """Supported model types for extensibility"""
    HUGGINGFACE = "huggingface"
    GGUF = "gguf"
    OLLAMA = "ollama"


class GenerateRequest(BaseModel):
    """Request schema for text generation"""
    prompt: str = Field(..., description="Input text prompt", min_length=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    max_tokens: int = Field(512, ge=1, le=2048, description="Maximum tokens to generate")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Enable streaming response")


class GenerateResponse(BaseModel):
    """Response schema for text generation"""
    text: str = Field(..., description="Generated text output")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    generation_time: float = Field(..., description="Time taken for generation in seconds")
    tokens_per_second: float = Field(..., description="Generation speed in tokens per second")


class StreamChunk(BaseModel):
    """Schema for streaming response chunks"""
    delta: str = Field(..., description="Generated text chunk")
    tokens_generated: int = Field(0, description="Total tokens generated so far")
    generation_time: float = Field(0.0, description="Time elapsed since generation started")
    tokens_per_second: float = Field(0.0, description="Current generation speed")


class ModelInfo(BaseModel):
    """Model information response"""
    model_config = ConfigDict(protected_namespaces=())
    
    name: str = Field(..., description="Model name")
    type: ModelType = Field(..., description="Model type")
    loaded: bool = Field(..., description="Whether model is currently loaded")
    memory_usage: Optional[Dict[str, Any]] = Field(None, description="Memory usage statistics")
    load_time: Optional[float] = Field(None, description="Time taken to load model")


class ModelLoadRequest(BaseModel):
    """Request schema for loading a model"""
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str = Field(..., description="Name of the model to load")
    model_type: ModelType = Field(ModelType.HUGGINGFACE, description="Type of model to load")
    force_reload: bool = Field(False, description="Force reload even if already loaded")


class HealthResponse(BaseModel):
    """Health check response"""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str = Field(..., description="Service status")
    uptime: float = Field(..., description="Service uptime in seconds")
    model_loaded: bool = Field(..., description="Whether a model is currently loaded")
    memory_usage: Optional[Dict[str, Any]] = Field(None, description="System memory usage")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: float = Field(default_factory=time.time, description="Error timestamp") 