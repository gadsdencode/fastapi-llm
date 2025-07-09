import os
import time
import logging
import psutil
from typing import Optional, Dict, Any, Iterator, AsyncIterator
from threading import Thread
from queue import Queue, Empty
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from llama_cpp import Llama

from ..schemas.models import ModelType, GenerateRequest, GenerateResponse, StreamChunk, ModelInfo

logger = logging.getLogger(__name__)


class LLMHandler:
    """Handles LLM model loading, inference, and streaming with optimizations"""
    
    def __init__(self):
        self.model = None
        self.model_name: Optional[str] = None
        self.model_type: Optional[ModelType] = None
        self.load_time: Optional[float] = None
        # Thread pool for async inference
        self.executor = ThreadPoolExecutor(max_workers=2)
        logger.info("LLM Handler initialized for GGUF models with threading support")
    
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
        """Load a model based on type with optimized parameters"""
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
                # Optimized quantization selection for speed/quality balance
                logger.info("Downloading GGUF model from Hugging Face...")
                
                # First, list all files in the repository to find GGUF files
                from huggingface_hub import list_repo_files
                try:
                    repo_files = list_repo_files(repo_id=model_name, repo_type="model")
                    gguf_files = [f for f in repo_files if f.endswith('.gguf')]
                    
                    if not gguf_files:
                        raise ValueError(f"No GGUF files found in {model_name}")
                    
                    logger.info(f"Found GGUF files: {gguf_files}")
                    
                    # Optimized quantization patterns for CPU inference speed
                    preferred_patterns = [
                        "Q4_K_M",    # Best speed/quality balance for CPU inference
                        "Q4_K_S",    # Faster than Q4_K_M, slightly lower quality
                        "Q4_0",      # Fast quantization
                        "Q5_K_M",    # Higher quality but slower
                        "Q6_K",      # High quality, slower
                        "Q3_K_M",    # Fastest, lower quality
                        "Q3_K_S"
                    ]
                    model_path = None
                    selected_file = None
                    
                    # First, try to find the exact Q4_K_M file for Phi-3.5-mini-instruct_Uncensored
                    phi35_q4km_file = "Phi-3.5-mini-instruct_Uncensored-Q4_K_M.gguf"
                    if phi35_q4km_file in gguf_files:
                        selected_file = phi35_q4km_file
                        logger.info(f"Found optimized target file: {selected_file}")
                    else:
                        # Find the best matching file based on quantization preference
                        for pattern in preferred_patterns:
                            matching_files = [f for f in gguf_files if pattern in f]
                            if matching_files:
                                selected_file = matching_files[0]  # Take first match
                                logger.info(f"Selected optimized quantization: {pattern} from file: {selected_file}")
                                break
                        
                        # If no preferred quantization found, take any GGUF file
                        if not selected_file:
                            selected_file = gguf_files[0]
                            logger.warning(f"No preferred quantization found, using: {selected_file}")
                    
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
                
                # Load the model with optimized settings for CPU inference performance
                logger.info(f"Loading model from path: {model_path}")
                self.model = Llama(
                    model_path=model_path,
                    n_ctx=1024,  # Reduced context window for faster inference
                    n_threads=os.cpu_count(),  # Use all available CPU cores
                    n_gpu_layers=0,  # CPU-only for Railway
                    use_mmap=True,  # Enable memory mapping
                    use_mlock=False,  # Disable memory locking for Railway compatibility
                    verbose=False,  # Disable verbose to reduce overhead
                    n_batch=512,  # Increased batch size for better throughput
                    # Performance optimizations:
                    logits_all=False,  # Only compute logits for generation
                    embedding=False,  # Disable embeddings if not needed
                    rope_freq_base=10000.0,  # Optimize for the model
                    rope_freq_scale=1.0,
                    seed=-1,  # Random seed
                    f16_kv=True,  # Use FP16 for key-value cache
                    low_vram=True,  # Enable low VRAM mode for better memory efficiency
                    rope_scaling_type=None,  # Default rope scaling
                    numa=False  # Disable NUMA for Railway
                )
            else:
                # Local file path
                logger.info(f"Loading local model from path: {model_name}")
                self.model = Llama(
                    model_path=model_name,
                    n_ctx=1024,  # Reduced context window for faster inference
                    n_threads=os.cpu_count(),  # Use all available CPU cores
                    n_gpu_layers=0,  # CPU-only
                    use_mmap=True,
                    use_mlock=False,
                    verbose=False,  # Disable verbose to reduce overhead
                    n_batch=512,  # Increased batch size for better performance
                    # Performance optimizations:
                    logits_all=False,
                    embedding=False,
                    rope_freq_base=10000.0,
                    rope_freq_scale=1.0,
                    seed=-1,
                    f16_kv=True,
                    low_vram=True,
                    rope_scaling_type=None,
                    numa=False
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
        """Generate text from prompt using thread pool to avoid blocking"""
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Please load a model first.")
        
        # Run inference in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            partial(self._sync_generate, request)
        )
        return result
    
    def _sync_generate(self, request: GenerateRequest) -> GenerateResponse:
        """Synchronous generation method with optimized parameters"""
        start_time = time.time()
        
        try:
            # Use create_completion for better performance than chat completion
            output = self.model.create_completion(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop_sequences or [],
                # Performance optimizations:
                repeat_penalty=1.1,  # Prevent repetition
                tfs_z=1.0,  # Tail free sampling
                typical_p=1.0,  # Typical sampling
                mirostat_mode=0,  # Disable mirostat for speed
                stream=False,
                echo=False  # Don't echo the prompt
            )
            
            generated_text = output['choices'][0]['text']
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
        """Generate text with optimized streaming response"""
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Please load a model first.")
        
        start_time = time.time()
        token_count = 0
        
        try:
            # Use thread pool for streaming to avoid blocking
            loop = asyncio.get_event_loop()
            
            def stream_generator():
                return self.model.create_completion(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stop=request.stop_sequences or [],
                    stream=True,
                    # Optimizations for streaming:
                    repeat_penalty=1.1,
                    tfs_z=1.0,
                    typical_p=1.0,
                    mirostat_mode=0,
                    echo=False
                )
            
            # Get streaming iterator in thread
            stream_iter = await loop.run_in_executor(self.executor, stream_generator)
            
            for output in stream_iter:
                if 'choices' in output and len(output['choices']) > 0:
                    choice = output['choices'][0]
                    delta = choice.get('text', '')
                    finish_reason = choice.get('finish_reason')
                    
                    if delta:
                        token_count += 1
                        
                        current_time = time.time()
                        elapsed_time = current_time - start_time
                        tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0
                        
                        # Check if this is the final chunk
                        is_final = (
                            finish_reason is not None or
                            token_count >= request.max_tokens or
                            (request.stop_sequences and any(stop in delta for stop in request.stop_sequences))
                        )
                        
                        yield StreamChunk(
                            delta=delta,
                            tokens_generated=token_count,
                            generation_time=elapsed_time,
                            tokens_per_second=tokens_per_second,
                            is_final=is_final
                        )
                        
                        # Break if this was the final chunk
                        if is_final:
                            break
                            
                        # Smaller sleep for faster streaming
                        await asyncio.sleep(0.001)
                        
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
    
    def __del__(self):
        """Cleanup thread pool on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Global instance
llm_handler = LLMHandler() 