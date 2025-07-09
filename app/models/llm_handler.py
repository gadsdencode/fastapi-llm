import os
import time
import logging
import psutil
from typing import Optional, Dict, Any, Iterator, AsyncIterator
from threading import Thread
from queue import Queue, Empty
import asyncio

from llama_cpp import Llama

from ..schemas.models import ModelType, GenerateRequest, GenerateResponse, StreamChunk, ModelInfo

logger = logging.getLogger(__name__)


class LLMHandler:
    """Handles LLM model loading, inference, and streaming"""
    
    def __init__(self):
        self.model = None
        self.model_name: Optional[str] = None
        self.model_type: Optional[ModelType] = None
        self.load_time: Optional[float] = None
        logger.info("LLM Handler initialized for GGUF models")
    
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded"""
        return self.model is not None
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        memory = psutil.virtual_memory()
        
        return {
            "system": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used
            }
        }
    
    async def load_model(self, model_name: str, model_type: ModelType = ModelType.HUGGINGFACE, force_reload: bool = False) -> bool:
        """Load a model based on type"""
        if self.is_loaded() and self.model_name == model_name and not force_reload:
            logger.info(f"Model {model_name} already loaded")
            return True
        
        logger.info(f"Loading model: {model_name} (type: {model_type})")
        start_time = time.time()
        
        try:
            # For GGUF models from Hugging Face
            if model_type == ModelType.GGUF or "/" in model_name:
                # Download from Hugging Face repo
                self.model = Llama.from_pretrained(
                    repo_id=model_name,
                    filename="*Q8_0.gguf",  # Default to Q8_0 quantization
                    n_ctx=2048,  # Context window size
                    n_threads=None,  # Use all available threads
                    verbose=False
                )
            else:
                # Local file path
                self.model = Llama(
                    model_path=model_name,
                    n_ctx=2048,  # Context window size
                    n_threads=None,  # Use all available threads
                    verbose=False
                )
            
            self.model_name = model_name
            self.model_type = model_type
            self.load_time = time.time() - start_time
            
            logger.info(f"Model loaded successfully in {self.load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            self.model = None
            raise
    
    def get_model_info(self) -> ModelInfo:
        """Get information about the currently loaded model"""
        if not self.is_loaded():
            return ModelInfo(
                name="None",
                type=ModelType.HUGGINGFACE,
                loaded=False
            )
        
        return ModelInfo(
            name=self.model_name,
            type=self.model_type,
            loaded=True,
            memory_usage=self.get_memory_usage(),
            load_time=self.load_time
        )
    
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate text from prompt"""
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Please load a model first.")
        
        start_time = time.time()
        
        try:
            # Generate text using llama-cpp-python chat completion
            output = self.model.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": request.prompt
                    }
                ],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop_sequences or []
            )
            
            generated_text = output['choices'][0]['message']['content']
            tokens_generated = output['usage']['completion_tokens']
            generation_time = time.time() - start_time
            
            return GenerateResponse(
                text=generated_text,
                tokens_generated=tokens_generated,
                generation_time=generation_time,
                tokens_per_second=tokens_generated / generation_time if generation_time > 0 else 0
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise RuntimeError(f"Text generation failed: {str(e)}")
    
    async def generate_stream(self, request: GenerateRequest) -> AsyncIterator[StreamChunk]:
        """Generate text with streaming response"""
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Please load a model first.")
        
        start_time = time.time()
        token_count = 0
        
        try:
            # Create streaming generator
            stream = self.model.create_chat_completion(
                messages=[
                    {
                        "role": "user", 
                        "content": request.prompt
                    }
                ],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop_sequences or [],
                stream=True
            )
            
            for output in stream:
                token_count += 1
                delta = output['choices'][0]['delta'].get('content', '')
                
                # Calculate timing info
                current_time = time.time()
                elapsed_time = current_time - start_time
                tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0
                
                yield StreamChunk(
                    delta=delta,
                    tokens_generated=token_count,
                    generation_time=elapsed_time,
                    tokens_per_second=tokens_per_second
                )
                
                # Allow other coroutines to run
                await asyncio.sleep(0)
                
        except Exception as e:
            logger.error(f"Streaming generation failed: {str(e)}")
            raise RuntimeError(f"Streaming generation failed: {str(e)}")
    
    def unload_model(self):
        """Unload the current model to free memory"""
        if self.model:
            del self.model
            self.model = None
            self.model_name = None
            self.model_type = None
            self.load_time = None
            logger.info("Model unloaded successfully")


# Global instance
llm_handler = LLMHandler() 