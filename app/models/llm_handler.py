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
            # For Railway deployment, use a smaller model that fits in memory constraints
            if model_type == ModelType.GGUF or "/" in model_name:
                # Use huggingface_hub to download GGUF files
                from huggingface_hub import hf_hub_download
                import tempfile
                import os
                
                # Download a smaller quantized model suitable for Railway's 8GB memory limit
                # Using Q4_K_M quantization for balance of quality and size
                logger.info("Downloading GGUF model from Hugging Face...")
                
                # First, list all files in the repository to find GGUF files
                from huggingface_hub import list_repo_files
                try:
                    repo_files = list_repo_files(repo_id=model_name, repo_type="model")
                    gguf_files = [f for f in repo_files if f.endswith('.gguf')]
                    
                    if not gguf_files:
                        raise ValueError(f"No GGUF files found in {model_name}")
                    
                    logger.info(f"Found GGUF files: {gguf_files}")
                    
                    # Try different quantization patterns in order of preference (smaller first for Railway)
                    preferred_patterns = ["Q2_K", "Q3_K_S", "Q4_0", "Q4_K_S", "Q5_0", "Q5_K_S", "Q6_K", "Q8_0"]
                    model_path = None
                    selected_file = None
                    
                    # Find the best matching file based on quantization preference
                    for pattern in preferred_patterns:
                        matching_files = [f for f in gguf_files if pattern in f]
                        if matching_files:
                            selected_file = matching_files[0]  # Take first match
                            break
                    
                    # If no preferred quantization found, take any GGUF file
                    if not selected_file:
                        selected_file = gguf_files[0]
                    
                    logger.info(f"Attempting to download {selected_file}")
                    model_path = hf_hub_download(
                        repo_id=model_name,
                        filename=selected_file,
                        cache_dir="/tmp/models"
                    )
                    logger.info(f"Successfully downloaded {selected_file}")
                    
                except Exception as e:
                    logger.error(f"Failed to list or download files from {model_name}: {e}")
                    raise ValueError(f"Could not download any GGUF file from {model_name}")
                
                if not model_path:
                    raise RuntimeError(f"Could not download any GGUF file from {model_name}")
                
                # Load the model with Railway-optimized settings
                logger.info(f"Loading model from path: {model_path}")
                self.model = Llama(
                    model_path=model_path,
                    n_ctx=1024,  # Reduced context for memory efficiency
                    n_threads=None,  # Let llama.cpp decide thread count
                    n_gpu_layers=0,  # CPU-only for Railway
                    use_mmap=True,  # Enable memory mapping
                    use_mlock=False,  # Disable memory locking for Railway
                    verbose=True,  # Enable verbose for debugging
                    n_batch=128,  # Smaller batch size for Railway
                    rope_scaling_type=None,  # Default rope scaling
                    rope_freq_base=0.0,  # Use model defaults
                    rope_freq_scale=0.0  # Use model defaults
                )
            else:
                # Local file path
                logger.info(f"Loading local model from path: {model_name}")
                self.model = Llama(
                    model_path=model_name,
                    n_ctx=1024,  # Reduced context window
                    n_threads=None,  # Let llama.cpp decide thread count
                    n_gpu_layers=0,  # CPU-only
                    use_mmap=True,
                    use_mlock=False,
                    verbose=True,  # Enable verbose for debugging
                    n_batch=128,  # Smaller batch size for Railway
                    rope_scaling_type=None,  # Default rope scaling
                    rope_freq_base=0.0,  # Use model defaults
                    rope_freq_scale=0.0  # Use model defaults
                )
            
            self.model_name = model_name
            self.model_type = model_type
            self.load_time = time.time() - start_time
            
            logger.info(f"Model loaded successfully in {self.load_time:.2f}s")
            return True
            
        except Exception as e:
            error_msg = f"Failed to load model {model_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception args: {e.args}")
            self.model = None
            raise RuntimeError(error_msg) from e
    
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