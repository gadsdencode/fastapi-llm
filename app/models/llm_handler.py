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
        
        # Set CPU optimization environment variables for Railway's 48-core deployment
        # Research shows optimal performance at 50-75% of available cores for high-core systems
        optimal_env_threads = '32'  # 67% of 48 cores for maximum throughput
        os.environ['OMP_NUM_THREADS'] = optimal_env_threads
        os.environ['MKL_NUM_THREADS'] = optimal_env_threads  
        os.environ['OPENBLAS_NUM_THREADS'] = optimal_env_threads
        os.environ['VECLIB_MAXIMUM_THREADS'] = optimal_env_threads
        logger.info(f"Set CPU optimization environment variables for Railway's 48-core deployment: {optimal_env_threads} threads")
        
        logger.info(f"Loading model: {model_name} (type: {model_type})")
        start_time = time.time()
        
        try:
            # For Railway deployment, use a smaller model that fits in memory constraints
            if model_type == ModelType.GGUF or "/" in model_name:
                # Use huggingface_hub to download GGUF files
                from huggingface_hub import hf_hub_download
                import tempfile
                
                # Optimize CPU threads for Railway's 48-core deployment
                cpu_count = os.cpu_count() or 48
                # Research-based optimization: Use 50-75% of cores for high-core systems
                # MoE models can handle higher thread counts due to parallel expert architecture
                if cpu_count >= 32:  # High-core system (Railway 48-core)
                    optimal_threads = int(cpu_count * 0.67)  # 67% of cores (32 threads for 48 cores)
                    optimal_batch_threads = int(cpu_count * 0.42)  # 42% for batch (20 threads for 48 cores)
                else:  # Fallback for smaller systems
                    optimal_threads = max(4, cpu_count // 2)
                    optimal_batch_threads = max(2, optimal_threads // 2)
                
                # Download optimized model for speed
                logger.info("Downloading GGUF model from Hugging Face...")
                logger.info(f"Detected {cpu_count} CPU cores, using {optimal_threads} threads (batch: {optimal_batch_threads}) for high-performance inference")
                
                # First, list all files in the repository to find GGUF files
                from huggingface_hub import list_repo_files
                try:
                    repo_files = list_repo_files(repo_id=model_name, repo_type="model")
                    gguf_files = [f for f in repo_files if f.endswith('.gguf')]
                    
                    if not gguf_files:
                        raise ValueError(f"No GGUF files found in {model_name}")
                    
                    logger.info(f"Found GGUF files: {gguf_files}")
                    
                    # Prioritize quantizations based on model type
                    if "llama-3.2" in model_name.lower():
                        # For Llama 3.2, prioritize Q8_0 for accuracy, then Q4_K_M for speed
                        preferred_patterns = [
                            "Q8_0",      # Target quantization for Llama 3.2 (high accuracy)
                            "Q4_K_M",    # Good speed/quality balance
                            "Q5_K_M",    # Higher quality
                            "Q4_K_S",    # Backup option
                            "Q4_0"       # Fast fallback
                        ]
                    else:
                        # For TinyDolphin and other models, prioritize Q4_K_M for speed
                        preferred_patterns = [
                            "Q4_K_M",    # Target quantization for speed/quality balance
                            "Q4_K_S",    # Backup option
                            "Q4_0",      # Fast fallback
                            "Q3_K_M",    # Fastest option
                            "Q5_K_M"     # Higher quality if needed
                        ]
                    model_path = None
                    selected_file = None
                    
                    # Look for target quantization files based on model type
                    if "llama-3.2" in model_name.lower():
                        # Look for Q8_0 for Llama 3.2 models
                        target_files = [f for f in gguf_files if "q8_0" in f.lower()]
                        if target_files:
                            selected_file = target_files[0]
                            logger.info(f"Found target Llama 3.2 Q8_0 file: {selected_file}")
                        else:
                            target_files = [f for f in gguf_files if "q4_k_m" in f.lower()]
                            if target_files:
                                selected_file = target_files[0]
                                logger.info(f"Found fallback Llama 3.2 Q4_K_M file: {selected_file}")
                    else:
                        # Look for Q4_K_M for other models (TinyDolphin)
                        target_files = [f for f in gguf_files if "q4_k_m" in f.lower()]
                        if target_files:
                            selected_file = target_files[0]
                            logger.info(f"Found target Q4_K_M file: {selected_file}")
                    
                    if not selected_file:
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
                    n_threads=optimal_threads,  # Optimized thread count (32 threads for 48 cores)
                    n_threads_batch=optimal_batch_threads,  # Separate batch processing threads (20 threads)
                    n_gpu_layers=0,  # CPU-only for Railway
                    use_mmap=True,  # Enable memory mapping for faster loading
                    use_mlock=False,  # Disable memory locking for Railway compatibility
                    verbose=False,  # Reduce log noise
                    n_batch=512,  # Larger batch size for high-core systems  
                    n_ubatch=256,  # Larger micro-batch for 48-core optimization
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
                
                # Optimize CPU threads for Railway's 48-core deployment  
                cpu_count = os.cpu_count() or 48
                # Research-based optimization: Use 50-75% of cores for high-core systems
                # MoE models can handle higher thread counts due to parallel expert architecture
                if cpu_count >= 32:  # High-core system (Railway 48-core)
                    optimal_threads = int(cpu_count * 0.67)  # 67% of cores (32 threads for 48 cores)
                    optimal_batch_threads = int(cpu_count * 0.42)  # 42% for batch (20 threads for 48 cores)
                else:  # Fallback for smaller systems
                    optimal_threads = max(4, cpu_count // 2)
                    optimal_batch_threads = max(2, optimal_threads // 2)
                logger.info(f"Detected {cpu_count} CPU cores, using {optimal_threads} threads (batch: {optimal_batch_threads}) for high-performance inference")
                
                self.model = Llama(
                    model_path=model_name,
                    n_ctx=2048,  # Conservative context size for Railway memory limits
                    n_threads=optimal_threads,  # Optimized thread count (32 threads for 48 cores)
                    n_threads_batch=optimal_batch_threads,  # Separate batch processing threads (20 threads)
                    n_gpu_layers=0,  # CPU-only
                    use_mmap=True,  # Enable memory mapping for faster loading
                    use_mlock=False,  # Disable memory locking for Railway compatibility
                    verbose=False,  # Reduce log noise
                    n_batch=512,  # Larger batch size for high-core systems
                    n_ubatch=256,  # Larger micro-batch for 48-core optimization
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
        """Format prompt using appropriate chat template based on model"""
        # Determine format based on model name
        if self.model_name and ("llama-3.2" in self.model_name.lower() or "llama-3" in self.model_name.lower()):
            # Llama 3.2/3.x Instruct format with proper tokens
            formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            # TinyDolphin/ChatML format for other models
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
        
        # Format the prompt properly for the loaded model
        formatted_prompt = self._format_chat_prompt(request.prompt)
        logger.info(f"Starting generation for prompt: {request.prompt[:50]}...")
        start_time = time.time()
        
        try:
            # Direct generation with optimized parameters for speed
            # Use appropriate stop tokens based on model type
            stop_tokens = ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"] if self.model_name and ("llama-3.2" in self.model_name.lower() or "llama-3" in self.model_name.lower()) else ["<|im_end|>", "<|im_start|>"]
            
            output = self.model.create_completion(
                prompt=formatted_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop_tokens,
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
        
        # Format the prompt properly for the loaded model
        formatted_prompt = self._format_chat_prompt(request.prompt)
        logger.info(f"Starting streaming generation for prompt: {request.prompt[:50]}...")
        start_time = time.time()
        token_count = 0
        
        try:
            # Direct streaming with speed optimizations
            # Use appropriate stop tokens based on model type
            stop_tokens = ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"] if self.model_name and ("llama-3.2" in self.model_name.lower() or "llama-3" in self.model_name.lower()) else ["<|im_end|>", "<|im_start|>"]
            
            stream = self.model.create_completion(
                prompt=formatted_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop_tokens,
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
                            any(stop in delta for stop in stop_tokens)
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