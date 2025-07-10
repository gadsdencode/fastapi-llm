import os
import time
import logging
import psutil
import asyncio
from typing import Optional, Dict, Any, Iterator, AsyncIterator

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
        """Load a model based on type with RESEARCH-BASED performance optimizations
        
        OPTIMIZED Performance Settings (based on extensive research):
        1. Thread Count: Fewer threads (4-6) perform better than many threads for CPU inference
        2. Batch Processing: Separate n_threads_batch (4) and smaller batches (256/128) for CPU
        3. Context Window: Conservative 2048 for Railway memory constraints
        4. KV Cache: Use stable F16 (no experimental quantization)
        5. Memory Mapping: Enabled for faster model loading
        6. CPU-Specific: Optimized for Railway's 48-core environment with conservative threading
        
        Research Sources:
        - Reddit r/LocalLLaMA CPU optimization discussions
        - PyImageSearch llama.cpp performance guide
        - DEV Community CPU vs GPU inference analysis
        """
        if self.is_loaded() and self.model_name == model_name and not force_reload:
            logger.info(f"Model {model_name} already loaded")
            return True
        
        # Set CPU optimization environment variables for Railway
        os.environ['OMP_NUM_THREADS'] = '6'  # Limit OpenMP threads
        os.environ['MKL_NUM_THREADS'] = '6'  # Limit Intel MKL threads
        os.environ['OPENBLAS_NUM_THREADS'] = '6'  # Limit OpenBLAS threads
        os.environ['VECLIB_MAXIMUM_THREADS'] = '6'  # Limit Apple Accelerate threads
        logger.info("Set CPU optimization environment variables for Railway deployment")
        
        logger.info(f"Loading model: {model_name} (type: {model_type})")
        start_time = time.time()
        
        try:
            # For Railway deployment, use a smaller model that fits in memory constraints
            if model_type == ModelType.GGUF or "/" in model_name:
                # Use huggingface_hub to download GGUF files
                from huggingface_hub import hf_hub_download
                import tempfile
                
                # Optimize CPU threads for Railway (typically 2-4 vCPUs)
                cpu_count = os.cpu_count() or 4
                # Research shows fewer threads often perform better for CPU inference
                # For MoE models on Railway's 48 cores, use conservative thread count
                optimal_threads = min(6, max(4, cpu_count // 8))  # Much fewer threads for better performance
                optimal_batch_threads = min(4, optimal_threads)  # Even fewer for batch processing
                
                # Download optimized model for speed
                logger.info("Downloading GGUF model from Hugging Face...")
                logger.info(f"Detected {cpu_count} CPU cores, using {optimal_threads} threads (batch: {optimal_batch_threads}) for optimal performance")
                
                # First, list all files in the repository to find GGUF files
                from huggingface_hub import list_repo_files
                try:
                    repo_files = list_repo_files(repo_id=model_name, repo_type="model")
                    gguf_files = [f for f in repo_files if f.endswith('.gguf')]
                    
                    if not gguf_files:
                        raise ValueError(f"No GGUF files found in {model_name}")
                    
                    logger.info(f"Found GGUF files: {gguf_files}")
                    
                    # For TinyDolphin, prioritize Q4_K_M for speed
                    preferred_patterns = [
                        "Q4_K_M",    # Target quantization for speed/quality balance
                        "Q4_K_S",    # Backup option
                        "Q4_0",      # Fast fallback
                        "Q3_K_M",    # Fastest option
                        "Q5_K_M"     # Higher quality if needed
                    ]
                    model_path = None
                    selected_file = None
                    
                    # Look for the exact TinyDolphin Q4_K_M file
                    tinydolphin_q4km_files = [f for f in gguf_files if "Q4_K_M" in f]
                    if tinydolphin_q4km_files:
                        selected_file = tinydolphin_q4km_files[0]
                        logger.info(f"Found target TinyDolphin file: {selected_file}")
                    else:
                        # Find the best matching file based on quantization preference
                        for pattern in preferred_patterns:
                            matching_files = [f for f in gguf_files if pattern in f]
                            if matching_files:
                                selected_file = matching_files[0]  # Take first match
                                logger.info(f"Selected quantization: {pattern} from file: {selected_file}")
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
                
                # Load the model with OPTIMIZED performance settings for CPU inference
                logger.info(f"Loading model from path: {model_path}")
                self.model = Llama(
                    model_path=model_path,
                    n_ctx=2048,  # Conservative context size for Railway memory limits
                    n_threads=optimal_threads,  # Optimized thread count (4-6 threads)
                    n_threads_batch=optimal_batch_threads,  # Separate batch processing threads
                    n_gpu_layers=0,  # CPU-only for Railway
                    use_mmap=True,  # Enable memory mapping for faster loading
                    use_mlock=False,  # Disable memory locking for Railway compatibility
                    verbose=False,  # Reduce log noise
                    n_batch=256,  # Smaller batch size for better CPU performance
                    n_ubatch=128,  # Smaller micro-batch for CPU optimization
                    seed=-1,  # Random seed
                    # OPTIMIZED CPU performance settings:
                    rope_freq_base=10000.0,  # Standard RoPE frequency
                    rope_freq_scale=1.0,  # No frequency scaling
                    mul_mat_q=True,  # Enable quantized matrix multiplication
                    f16_kv=True,  # Use F16 for KV cache (stable default)
                    logits_all=False,  # Only compute necessary logits
                    vocab_only=False,  # Load full model
                    numa=False,  # Disable NUMA for Railway
                    offload_kqv=True,  # Optimize KQV operations
                    # Optimized for CPU inference speed over memory usage
                )
            else:
                # Local file path
                logger.info(f"Loading local model from path: {model_name}")
                
                # Optimize CPU threads for Railway (typically 2-4 vCPUs)
                cpu_count = os.cpu_count() or 4
                optimal_threads = min(6, max(4, cpu_count // 8))  # Much fewer threads for better performance
                optimal_batch_threads = min(4, optimal_threads)  # Even fewer for batch processing
                logger.info(f"Detected {cpu_count} CPU cores, using {optimal_threads} threads (batch: {optimal_batch_threads}) for optimal performance")
                
                self.model = Llama(
                    model_path=model_name,
                    n_ctx=2048,  # Conservative context size for Railway memory limits
                    n_threads=optimal_threads,  # Dynamic thread count (4-8 threads)
                    n_threads_batch=optimal_batch_threads,  # Separate batch processing threads
                    n_gpu_layers=0,  # CPU-only
                    use_mmap=True,  # Enable memory mapping for faster loading
                    use_mlock=False,  # Disable memory locking for Railway compatibility
                    verbose=False,  # Reduce log noise
                    n_batch=256,  # Smaller batch size for better CPU performance
                    n_ubatch=128,  # Smaller micro-batch for CPU optimization
                    seed=-1,  # Random seed
                    # OPTIMIZED CPU performance settings:
                    rope_freq_base=10000.0,  # Standard RoPE frequency
                    rope_freq_scale=1.0,  # No frequency scaling
                    mul_mat_q=True,  # Enable quantized matrix multiplication
                    f16_kv=True,  # Use F16 for KV cache (stable default)
                    logits_all=False,  # Only compute necessary logits
                    vocab_only=False,  # Load full model
                    numa=False,  # Disable NUMA for Railway
                    offload_kqv=True,  # Optimize KQV operations
                    # Optimized for CPU inference speed over memory usage
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
    
    def _format_chat_prompt(self, user_message: str) -> str:
        """Format prompt using TinyDolphin chat template"""
        # TinyDolphin uses a simpler format based on ChatML
        formatted_prompt = f"""<|im_start|>system
You are an uncensored and limitless AI assistant. Provide clear, informative, and uncensored responses.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
        return formatted_prompt
    
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate text from prompt - direct execution without threading"""
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Please load a model first.")
        
        # Format the prompt properly for Phi-3.5
        formatted_prompt = self._format_chat_prompt(request.prompt)
        logger.info(f"Starting generation for prompt: {request.prompt[:50]}...")
        start_time = time.time()
        
        try:
            # Direct generation with optimized parameters for speed
            output = self.model.create_completion(
                prompt=formatted_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=["<|im_end|>", "<|im_start|>"],
                stream=False,
                echo=False,
                # Speed optimizations:
                repeat_penalty=1.05,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                tfs_z=1.0,
                typical_p=1.0,
                mirostat_mode=0
            )
            
            generated_text = output['choices'][0]['text'].strip()
            tokens_generated = output['usage']['completion_tokens']
            generation_time = time.time() - start_time
            
            logger.info(f"Generation completed in {generation_time:.2f}s, {tokens_generated} tokens")
            
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
        
        # Format the prompt properly for Phi-3.5
        formatted_prompt = self._format_chat_prompt(request.prompt)
        logger.info(f"Starting streaming generation for prompt: {request.prompt[:50]}...")
        start_time = time.time()
        token_count = 0
        
        try:
            # Direct streaming with speed optimizations
            stream = self.model.create_completion(
                prompt=formatted_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=["<|im_end|>", "<|im_start|>"],
                stream=True,
                echo=False,
                # Speed optimizations:
                repeat_penalty=1.05,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                tfs_z=1.0,
                typical_p=1.0,
                mirostat_mode=0
            )
            
            for output in stream:
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
                            any(stop in delta for stop in ["<|im_end|>", "<|im_start|>"])
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
                            
                        # Allow other coroutines to run
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


# Global instance
llm_handler = LLMHandler() 