from pydantic import BaseModel, Field
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
    max_tokens: int = Field(256, ge=1, le=2048, description="Maximum tokens to generate")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Enable streaming response")


class GenerateResponse(BaseModel):
    """Response schema for text generation"""
    generated_text: str = Field(..., description="Generated text output")
    prompt: str = Field(..., description="Original input prompt")
    model_name: str = Field(..., description="Model used for generation")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    generation_time: float = Field(..., description="Time taken for generation in seconds")


class StreamChunk(BaseModel):
    """Schema for streaming response chunks"""
    text: str = Field(..., description="Generated text chunk")
    is_final: bool = Field(False, description="Whether this is the final chunk")
    tokens_generated: int = Field(0, description="Total tokens generated so far")


class ModelInfo(BaseModel):
    """Model information response"""
    name: str = Field(..., description="Model name")
    type: ModelType = Field(..., description="Model type")
    loaded: bool = Field(..., description="Whether model is currently loaded")
    memory_usage: Optional[Dict[str, Any]] = Field(None, description="Memory usage statistics")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Model parameters")
    load_time: Optional[float] = Field(None, description="Time taken to load model")


class ModelLoadRequest(BaseModel):
    """Request schema for loading a model"""
    model_name: str = Field(..., description="Name of the model to load")
    model_type: ModelType = Field(ModelType.HUGGINGFACE, description="Type of model to load")
    force_reload: bool = Field(False, description="Force reload even if already loaded")


class HealthResponse(BaseModel):
    """Health check response"""
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