from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing import Optional, Dict, Any, List
from enum import Enum
import time


class ModelType(str, Enum):
    """Supported model types for extensibility"""
    HUGGINGFACE = "huggingface"
    GGUF = "gguf"
    OLLAMA = "ollama"


class ChatMessage(BaseModel):
    """Individual message in a conversation"""
    role: str = Field(..., description="Message role: system, user, assistant")
    content: str = Field(..., description="Message content", min_length=1)
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        if v not in ['system', 'user', 'assistant']:
            raise ValueError('Role must be one of: system, user, assistant')
        return v


class GenerateRequest(BaseModel):
    """Enhanced request schema supporting both single prompts and multi-turn conversations"""
    # Legacy support - single prompt
    prompt: Optional[str] = Field(None, description="Single input text prompt (legacy mode)", min_length=1)
    
    # New multi-turn conversation support
    messages: Optional[List[ChatMessage]] = Field(None, description="Multi-turn conversation messages")
    
    # Existing generation parameters
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    max_tokens: int = Field(512, ge=1, le=2048, description="Maximum tokens to generate")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Enable streaming response")
    
    @model_validator(mode='after')
    def validate_prompt_or_messages(self):
        """Ensure either prompt or messages is provided, but not both"""
        prompt = self.prompt
        messages = self.messages
        
        if not prompt and not messages:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        if prompt and messages:
            raise ValueError("Provide either 'prompt' OR 'messages', not both")
        
        # Validate messages if provided
        if messages and len(messages) == 0:
            raise ValueError("Messages list cannot be empty")
            
        return self


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
    is_final: bool = Field(False, description="Whether this is the final chunk")


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
    preferred_quant: Optional[str] = Field(None, description="Preferred quantization (e.g., 'Q4_K_M', 'Q8_0'). If not found, falls back to defaults.")


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