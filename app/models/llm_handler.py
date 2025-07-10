import os
import time
import logging
import psutil
import asyncio
import re
from typing import Optional, Dict, Any, Iterator, AsyncIterator, Tuple, List
from dataclasses import dataclass

from llama_cpp import Llama

logger = logging.getLogger(__name__)

# Add transformers import for tokenizer integration
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available, Hugging Face chat templates will not be used")

from ..schemas.models import ModelType, GenerateRequest, GenerateResponse, StreamChunk, ModelInfo, ChatMessage
from ..config import get_template_manager


@dataclass
class GGUFFileGroup:
    """Represents a group of GGUF files (single or multi-part)"""
    base_name: str
    files: List[str]
    quantization: str
    is_split: bool
    total_parts: Optional[int] = None
    primary_file: Optional[str] = None

    def __post_init__(self):
        if not self.primary_file:
            # For split files, primary is usually the first part
            # For single files, primary is the only file
            self.primary_file = self.files[0] if self.files else None


class QuantizationMatcher:
    """Robust quantization detection using regex patterns"""
    
    # Define regex patterns for common quantization formats
    QUANTIZATION_PATTERNS = {
        'Q2_K': re.compile(r'q2[-_]?k(?![\w])', re.IGNORECASE),
        'Q3_K_S': re.compile(r'q3[-_]?k[-_]?s', re.IGNORECASE),
        'Q3_K_M': re.compile(r'q3[-_]?k[-_]?m', re.IGNORECASE),
        'Q3_K_L': re.compile(r'q3[-_]?k[-_]?l', re.IGNORECASE),
        'Q4_0': re.compile(r'q4[-_]?0', re.IGNORECASE),
        'Q4_1': re.compile(r'q4[-_]?1', re.IGNORECASE),
        'Q4_K_S': re.compile(r'q4[-_]?k[-_]?s', re.IGNORECASE),
        'Q4_K_M': re.compile(r'q4[-_]?k[-_]?m', re.IGNORECASE),
        'Q5_0': re.compile(r'q5[-_]?0', re.IGNORECASE),
        'Q5_1': re.compile(r'q5[-_]?1', re.IGNORECASE),
        'Q5_K_S': re.compile(r'q5[-_]?k[-_]?s', re.IGNORECASE),
        'Q5_K_M': re.compile(r'q5[-_]?k[-_]?m', re.IGNORECASE),
        'Q6_K': re.compile(r'q6[-_]?k(?![\w])', re.IGNORECASE),
        'Q8_0': re.compile(r'q8[-_]?0', re.IGNORECASE),
        'F16': re.compile(r'f16(?![\w])', re.IGNORECASE),
        'F32': re.compile(r'f32(?![\w])', re.IGNORECASE),
    }

    @classmethod
    def detect_quantization(cls, filename: str) -> Optional[str]:
        """
        Detect quantization type from filename using regex patterns
        
        Args:
            filename: GGUF filename to analyze
            
        Returns:
            Detected quantization type or None if not found
        """
        for quant_type, pattern in cls.QUANTIZATION_PATTERNS.items():
            if pattern.search(filename):
                return quant_type
        return None

    @classmethod
    def normalize_quantization(cls, quant_input: str) -> Optional[str]:
        """
        Normalize user input quantization to standard format
        
        Args:
            quant_input: User-provided quantization string
            
        Returns:
            Normalized quantization type or None if invalid
        """
        # Try to match against our known patterns
        for quant_type, pattern in cls.QUANTIZATION_PATTERNS.items():
            if pattern.search(quant_input):
                return quant_type
        return None


class GGUFFileHandler:
    """Handles GGUF file detection, grouping, and multi-file support"""
    
    # Pattern for detecting split GGUF files (e.g., model-00001-of-00003.gguf)
    SPLIT_FILE_PATTERN = re.compile(r'^(.+)-(\d{5})-of-(\d{5})\.gguf$', re.IGNORECASE)
    
    @classmethod
    def group_gguf_files(cls, files: List[str]) -> List[GGUFFileGroup]:
        """
        Group GGUF files by base name and detect multi-file models
        
        Args:
            files: List of GGUF filenames
            
        Returns:
            List of GGUFFileGroup objects representing file groups
        """
        single_files = []
        split_groups = {}
        
        for filename in files:
            # Check if this is a split file
            match = cls.SPLIT_FILE_PATTERN.match(filename)
            if match:
                base_name = match.group(1)
                part_num = int(match.group(2))
                total_parts = int(match.group(3))
                
                if base_name not in split_groups:
                    split_groups[base_name] = {
                        'files': [],
                        'total_parts': total_parts,
                        'quantization': None
                    }
                
                split_groups[base_name]['files'].append((part_num, filename))
                
                # Detect quantization from the base name
                if not split_groups[base_name]['quantization']:
                    split_groups[base_name]['quantization'] = QuantizationMatcher.detect_quantization(base_name)
            else:
                # Single file
                single_files.append(filename)
        
        # Create file groups
        groups = []
        
        # Add single files
        for filename in single_files:
            quantization = QuantizationMatcher.detect_quantization(filename) or 'UNKNOWN'
            groups.append(GGUFFileGroup(
                base_name=filename.replace('.gguf', ''),
                files=[filename],
                quantization=quantization,
                is_split=False,
                primary_file=filename
            ))
        
        # Add split file groups
        for base_name, group_data in split_groups.items():
            # Sort files by part number
            sorted_files = sorted(group_data['files'], key=lambda x: x[0])
            file_list = [f[1] for f in sorted_files]
            
            # Verify we have all parts
            expected_parts = group_data['total_parts']
            if len(file_list) != expected_parts:
                logger.warning(f"Incomplete split model {base_name}: found {len(file_list)}/{expected_parts} parts")
                continue
            
            quantization = group_data['quantization'] or 'UNKNOWN'
            groups.append(GGUFFileGroup(
                base_name=base_name,
                files=file_list,
                quantization=quantization,
                is_split=True,
                total_parts=expected_parts,
                primary_file=file_list[0]  # First part as primary
            ))
        
        return groups

    @classmethod
    def select_best_group(cls, groups: List[GGUFFileGroup], preferred_quant: Optional[str] = None, 
                         model_preferences: Optional[List[str]] = None) -> Optional[GGUFFileGroup]:
        """
        Select the best GGUF file group based on preferences
        
        Args:
            groups: List of available file groups
            preferred_quant: User's preferred quantization (takes priority)
            model_preferences: Default quantization preferences for model type
            
        Returns:
            Best matching GGUFFileGroup or None if no suitable group found
        """
        if not groups:
            return None
        
        # First, try to match user's preferred quantization
        if preferred_quant:
            normalized_pref = QuantizationMatcher.normalize_quantization(preferred_quant)
            if normalized_pref:
                for group in groups:
                    if group.quantization == normalized_pref:
                        logger.info(f"Selected preferred quantization {normalized_pref}: {group.primary_file}")
                        return group
                logger.warning(f"Preferred quantization {preferred_quant} not found, falling back to defaults")
        
        # Fall back to model-specific preferences
        if model_preferences:
            for pref_quant in model_preferences:
                for group in groups:
                    if group.quantization == pref_quant:
                        logger.info(f"Selected fallback quantization {pref_quant}: {group.primary_file}")
                        return group
        
        # Last resort: take the first group
        selected = groups[0]
        logger.info(f"No preferred quantization found, using: {selected.primary_file} ({selected.quantization})")
        return selected


class LLMHandler:
    """Handles LLM model loading, inference, and streaming with optimizations"""
    
    def __init__(self):
        self.model = None
        self.model_name: Optional[str] = None
        self.model_type: Optional[ModelType] = None
        self.load_time: Optional[float] = None
        self.template_manager = get_template_manager()
        # Add tokenizer for Hugging Face chat template integration
        self.tokenizer: Optional['AutoTokenizer'] = None
        logger.info("Enhanced LLM Handler initialized with flexible template support")
    
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
    
    def _get_optimal_threads(self, cpu_count: int) -> Tuple[int, int]:
        """Calculate optimal thread counts for CPU inference
        
        Research-based optimization: Use 50-75% of cores for high-core systems
        MoE models can handle higher thread counts due to parallel expert architecture
        
        Args:
            cpu_count: Number of available CPU cores
            
        Returns:
            Tuple of (optimal_threads, optimal_batch_threads)
        """
        if cpu_count >= 32:  # High-core system (Railway 48-core)
            optimal_threads = int(cpu_count * 0.67)  # 67% of cores (32 threads for 48 cores)
            optimal_batch_threads = int(cpu_count * 0.42)  # 42% for batch (20 threads for 48 cores)
        else:  # Fallback for smaller systems
            optimal_threads = max(4, cpu_count // 2)
            optimal_batch_threads = max(2, optimal_threads // 2)
        
        logger.info(f"Detected {cpu_count} CPU cores, using {optimal_threads} threads (batch: {optimal_batch_threads}) for high-performance inference")
        return optimal_threads, optimal_batch_threads
    
    def _setup_cpu_optimization_env(self, optimal_threads: int) -> None:
        """Set CPU optimization environment variables for Railway's 48-core deployment"""
        env_threads = str(optimal_threads)
        os.environ['OMP_NUM_THREADS'] = env_threads
        os.environ['MKL_NUM_THREADS'] = env_threads  
        os.environ['OPENBLAS_NUM_THREADS'] = env_threads
        os.environ['VECLIB_MAXIMUM_THREADS'] = env_threads
        logger.info(f"Set CPU optimization environment variables for Railway's 48-core deployment: {env_threads} threads")
    
    def _get_base_llama_config(self, optimal_threads: int, optimal_batch_threads: int) -> Dict[str, Any]:
        """Get base Llama configuration with optimized performance settings
        
        OPTIMIZED Performance Settings (based on extensive research):
        1. Thread Count: Fewer threads (4-6) perform better than many threads for CPU inference
        2. Batch Processing: Separate n_threads_batch (4) and smaller batches (256/128) for CPU
        3. Context Window: Conservative 2048 for Railway memory constraints
        4. KV Cache: Use stable F16 (no experimental quantization)
        5. Memory Mapping: Enabled for faster model loading
        6. CPU-Specific: Optimized for Railway's 48-core environment with conservative threading
        
        Args:
            optimal_threads: Optimized thread count for inference
            optimal_batch_threads: Optimized thread count for batch processing
            
        Returns:
            Dictionary of base Llama configuration parameters
        """
        return {
            "n_ctx": 2048,  # Conservative context size for Railway memory limits
            "n_threads": optimal_threads,  # Optimized thread count
            "n_threads_batch": optimal_batch_threads,  # Separate batch processing threads
            "n_gpu_layers": 0,  # CPU-only for Railway
            "use_mmap": True,  # Enable memory mapping for faster loading
            "use_mlock": False,  # Disable memory locking for Railway compatibility
            "verbose": False,  # Reduce log noise
            "n_batch": 512,  # Larger batch size for high-core systems  
            "n_ubatch": 256,  # Larger micro-batch for 48-core optimization
            "seed": -1,  # Random seed
            # OPTIMIZED CPU performance settings:
            "rope_freq_base": 10000.0,  # Standard RoPE frequency
            "rope_freq_scale": 1.0,  # No frequency scaling
            "mul_mat_q": True,  # Enable quantized matrix multiplication
            "f16_kv": True,  # Use F16 for KV cache (stable default)
            "logits_all": False,  # Only compute necessary logits
            "vocab_only": False,  # Load full model
            "numa": False,  # Disable NUMA for Railway
            "offload_kqv": True,  # Optimize KQV operations
        }

    async def load_model(self, model_name: str, model_type: ModelType = ModelType.HUGGINGFACE, force_reload: bool = False, preferred_quant: Optional[str] = None) -> bool:
        """Load a model based on type with RESEARCH-BASED performance optimizations"""
        if self.is_loaded() and self.model_name == model_name and not force_reload:
            logger.info(f"Model {model_name} already loaded")
            return True
        
        # Optimize CPU threads for Railway's 48-core deployment
        cpu_count = os.cpu_count() or 48
        optimal_threads, optimal_batch_threads = self._get_optimal_threads(cpu_count)
        self._setup_cpu_optimization_env(optimal_threads)
        
        logger.info(f"Loading model: {model_name} (type: {model_type})")
        start_time = time.time()
        
        try:
            # Get base configuration for Llama initialization
            base_config = self._get_base_llama_config(optimal_threads, optimal_batch_threads)
            
            # Determine model path based on type
            if model_type == ModelType.GGUF or "/" in model_name:
                model_path = await self._download_gguf_model(model_name, preferred_quant)
            else:
                model_path = model_name
                logger.info(f"Loading local model from path: {model_name}")
            
            # Load the model with optimized configuration
            logger.info(f"Loading model from path: {model_path}")
            self.model = Llama(
                model_path=model_path,
                **base_config
            )
            
            self.model_name = model_name
            self.model_type = model_type
            self.load_time = time.time() - start_time
            
            # Try to load tokenizer for Hugging Face chat template integration
            await self._load_tokenizer(model_name)
            
            logger.info(f"Model loaded successfully in {self.load_time:.2f}s")
            return True
            
        except Exception as e:
            error_msg = f"Failed to load model {model_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception args: {e.args}")
            self.model = None
            raise RuntimeError(error_msg) from e
    
    async def _download_gguf_model(self, model_name: str, preferred_quant: Optional[str] = None) -> str:
        """Download GGUF model from Hugging Face Hub with robust file selection
        
        Args:
            model_name: Model repository name on Hugging Face
            preferred_quant: User's preferred quantization (e.g., 'Q4_K_M', 'Q8_0')
            
        Returns:
            Path to downloaded primary model file
        """
        # Use huggingface_hub to download GGUF files
        from huggingface_hub import hf_hub_download, list_repo_files
        
        logger.info(f"Downloading GGUF model from Hugging Face: {model_name}")
        if preferred_quant:
            logger.info(f"User preferred quantization: {preferred_quant}")
        
        try:
            # List all files in the repository to find GGUF files
            repo_files = list_repo_files(repo_id=model_name, repo_type="model")
            gguf_files = [f for f in repo_files if f.endswith('.gguf')]
            
            if not gguf_files:
                raise ValueError(f"No GGUF files found in {model_name}")
            
            logger.info(f"Found {len(gguf_files)} GGUF files: {gguf_files}")
            
            # Group GGUF files and detect multi-file models
            file_groups = GGUFFileHandler.group_gguf_files(gguf_files)
            
            if not file_groups:
                raise ValueError(f"No valid GGUF file groups found in {model_name}")
            
            logger.info(f"Detected {len(file_groups)} file groups:")
            for group in file_groups:
                logger.info(f"  - {group.base_name}: {group.quantization} ({'split' if group.is_split else 'single'}, {len(group.files)} files)")
            
            # Define model-specific quantization preferences
            if "llama-3.2" in model_name.lower() or "llama-3" in model_name.lower():
                # For Llama 3.2/3.x, prioritize Q8_0 for accuracy, then Q4_K_M for speed
                model_preferences = ["Q8_0", "Q4_K_M", "Q5_K_M", "Q4_K_S", "Q4_0"]
            else:
                # For other models, prioritize Q4_K_M for speed/quality balance
                model_preferences = ["Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q5_K_M", "Q8_0"]
            
            # Select the best file group based on preferences
            selected_group = GGUFFileHandler.select_best_group(
                file_groups, 
                preferred_quant=preferred_quant,
                model_preferences=model_preferences
            )
            
            if not selected_group:
                raise ValueError(f"No suitable GGUF file group found in {model_name}")
            
            # Download all files in the selected group
            downloaded_files = []
            for filename in selected_group.files:
                logger.info(f"Downloading {filename}...")
                file_path = hf_hub_download(
                    repo_id=model_name,
                    filename=filename,
                    cache_dir="/tmp/models"
                )
                downloaded_files.append(file_path)
                logger.info(f"Successfully downloaded {filename}")
            
            # Return the primary file path
            primary_path = None
            for i, filename in enumerate(selected_group.files):
                if filename == selected_group.primary_file:
                    primary_path = downloaded_files[i]
                    break
            
            if not primary_path:
                primary_path = downloaded_files[0]  # Fallback to first file
            
            if selected_group.is_split:
                logger.info(f"Downloaded multi-file model: {len(downloaded_files)} parts, primary: {selected_group.primary_file}")
            else:
                logger.info(f"Downloaded single-file model: {selected_group.primary_file}")
            
            logger.info(f"Model quantization: {selected_group.quantization}")
            return primary_path
            
        except Exception as e:
            logger.error(f"Failed to download GGUF model from {model_name}: {e}")
            raise ValueError(f"Could not download GGUF model from {model_name}: {str(e)}")
    
    async def _load_tokenizer(self, model_name: str) -> None:
        """Load Hugging Face tokenizer for chat template integration"""
        if not TRANSFORMERS_AVAILABLE:
            logger.debug("Transformers not available, skipping tokenizer loading")
            return
            
        try:
            # For GGUF models, extract the base repo name
            if "/" in model_name and not model_name.startswith("./"):
                repo_name = model_name
                logger.info(f"Attempting to load tokenizer for: {repo_name}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    repo_name,
                    trust_remote_code=True,
                    use_fast=False  # Use slow tokenizer for better compatibility
                )
                logger.info(f"Successfully loaded tokenizer for {repo_name}")
            else:
                logger.debug(f"Local model path detected, skipping tokenizer loading: {model_name}")
                
        except Exception as e:
            logger.debug(f"Could not load tokenizer for {model_name}: {e}")
            self.tokenizer = None
    
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
    
    def _try_builtin_chat_template(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Try to use Hugging Face tokenizer's chat template if available"""
        try:
            # First priority: Use Hugging Face tokenizer's apply_chat_template
            if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template'):
                # Check if the tokenizer has a chat template
                if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    logger.info("Using Hugging Face tokenizer chat template")
                    return formatted_prompt
                    
            # Secondary: Check if llama_cpp model has apply_chat_template method
            if hasattr(self.model, 'apply_chat_template'):
                formatted_prompt = self.model.apply_chat_template(messages, tokenize=False)
                logger.info("Using llama_cpp built-in chat template")
                return formatted_prompt
                
        except Exception as e:
            logger.debug(f"Built-in chat template not available or failed: {e}")
            
        return None
    
    def _format_multi_turn_prompt(self, messages: List[ChatMessage]) -> str:
        """Format multi-turn conversation using flexible template system"""
        
        # Convert to dict format for compatibility
        message_dicts = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        # Try built-in template first (future enhancement)
        builtin_prompt = self._try_builtin_chat_template(message_dicts)
        if builtin_prompt:
            logger.info("Using llama_cpp built-in chat template")
            return builtin_prompt
        
        # Use config-based template
        logger.info("Using config-based chat template")
        template_config = self.template_manager.get_template_for_model(self.model_name or "")
        
        if not template_config:
            raise ValueError("No suitable chat template found")
            
        return self._apply_template_config(messages, template_config)
    
    def _apply_template_config(self, messages: List[ChatMessage], template_config: Dict[str, Any]) -> str:
        """Apply template configuration to format messages"""
        format_config = template_config['format']
        formatted_parts = []
        
        # Add conversation start if present
        if 'conversation_start' in format_config:
            formatted_parts.append(format_config['conversation_start'])
        
        for message in messages:
            role = message.role
            content = message.content
            
            # Add role-specific formatting
            if f"{role}_start" in format_config:
                formatted_parts.append(format_config[f"{role}_start"])
            
            formatted_parts.append(content)
            
            if f"{role}_end" in format_config:
                formatted_parts.append(format_config[f"{role}_end"])
        
        # Add assistant start for generation
        if 'assistant_start' in format_config:
            formatted_parts.append(format_config['assistant_start'])
            
        formatted_prompt = ''.join(formatted_parts)
        
        # Defensive check: Remove duplicate leading <|begin_of_text|> tokens
        # This prevents issues if templates accidentally include them or future llama_cpp changes
        while formatted_prompt.startswith("<|begin_of_text|><|begin_of_text|>"):
            formatted_prompt = formatted_prompt[len("<|begin_of_text|>"):]
            logger.warning("Removed duplicate leading <|begin_of_text|> token from prompt")
            
        return formatted_prompt
    
    def _convert_prompt_to_messages(self, prompt: str) -> List[ChatMessage]:
        """Convert legacy single prompt to message format"""
        system_message = self.template_manager.get_default_system_message(self.model_name or "")
        
        return [
            ChatMessage(role="system", content=system_message),
            ChatMessage(role="user", content=prompt)
        ]
    
    def _get_stop_tokens_for_model(self) -> List[str]:
        """Get appropriate stop tokens for the current model including EOS token"""
        # Start with template-based stop tokens
        stop_tokens = self.template_manager.get_stop_tokens_for_model(self.model_name or "")
        
        # Add model-specific EOS token if tokenizer is available
        if self.tokenizer:
            try:
                # Add EOS token if available
                if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                    if self.tokenizer.eos_token not in stop_tokens:
                        stop_tokens.append(self.tokenizer.eos_token)
                        logger.debug(f"Added EOS token to stop tokens: {self.tokenizer.eos_token}")
                
                # Add additional special tokens that indicate end of generation
                special_tokens_to_check = ['</s>', '<|end|>', '<|endoftext|>', '<|im_end|>', '<|eot_id|>']
                for token in special_tokens_to_check:
                    if (hasattr(self.tokenizer, 'special_tokens_map') and 
                        token in self.tokenizer.special_tokens_map.values() and 
                        token not in stop_tokens):
                        stop_tokens.append(token)
                        logger.debug(f"Added special stop token: {token}")
                        
            except Exception as e:
                logger.debug(f"Could not extract stop tokens from tokenizer: {e}")
        
        # Add llama_cpp model-specific EOS token if available
        if self.model and hasattr(self.model, 'token_eos'):
            try:
                eos_token_id = self.model.token_eos()
                # Convert token ID to text if possible
                if hasattr(self.model, 'detokenize'):
                    eos_text = self.model.detokenize([eos_token_id]).decode('utf-8', errors='ignore')
                    if eos_text and eos_text not in stop_tokens:
                        stop_tokens.append(eos_text)
                        logger.debug(f"Added model EOS token: {eos_text}")
            except Exception as e:
                logger.debug(f"Could not extract EOS token from model: {e}")
        
        return stop_tokens
    
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Enhanced generation supporting both single prompts and multi-turn conversations"""
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Please load a model first.")
        
        # Handle backward compatibility and multi-turn support
        if request.prompt:
            messages = self._convert_prompt_to_messages(request.prompt)
            logger.info(f"Converted single prompt to messages format: {request.prompt[:50]}...")
        else:
            messages = request.messages
            logger.info(f"Processing {len(messages)} messages in conversation")
        
        # Format using enhanced template system
        formatted_prompt = self._format_multi_turn_prompt(messages)
        
        # Get appropriate stop tokens
        stop_tokens = self._get_stop_tokens_for_model()
        
        start_time = time.time()
        
        try:
            
            output = self.model.create_completion(
                prompt=formatted_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop_tokens,
                stream=False,
                echo=False,
                # User-configurable generation parameters
                repeat_penalty=request.repeat_penalty,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                tfs_z=request.tfs_z,
                typical_p=request.typical_p,
                mirostat_mode=request.mirostat_mode,
                mirostat_tau=request.mirostat_tau,
                mirostat_eta=request.mirostat_eta
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
        """Enhanced streaming generation with improved stop token handling and error logging"""
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Please load a model first.")
        
        # Handle backward compatibility and multi-turn support
        if request.prompt:
            messages = self._convert_prompt_to_messages(request.prompt)
            logger.info(f"Starting streaming for single prompt: {request.prompt[:50]}...")
        else:
            messages = request.messages
            logger.info(f"Starting streaming for {len(messages)} messages in conversation")
        
        # Format using enhanced template system
        formatted_prompt = self._format_multi_turn_prompt(messages)
        
        # Get appropriate stop tokens
        stop_tokens = self._get_stop_tokens_for_model()
        logger.debug(f"Using stop tokens: {stop_tokens}")
        
        start_time = time.time()
        token_count = 0
        
        try:
            stream = self.model.create_completion(
                prompt=formatted_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=stop_tokens,
                stream=True,
                echo=False,
                # User-configurable generation parameters
                repeat_penalty=request.repeat_penalty,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
                tfs_z=request.tfs_z,
                typical_p=request.typical_p,
                mirostat_mode=request.mirostat_mode,
                mirostat_tau=request.mirostat_tau,
                mirostat_eta=request.mirostat_eta
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
                        
                        # Rely primarily on llama_cpp's finish_reason for completion detection
                        is_final = (
                            finish_reason is not None or  # Primary: trust llama_cpp's finish_reason
                            token_count >= request.max_tokens  # Safety net: max tokens reached
                        )
                        
                        # Log completion reason for debugging
                        if is_final:
                            if finish_reason:
                                completion_reason = f"finish_reason={finish_reason}"
                            elif token_count >= request.max_tokens:
                                completion_reason = "max_tokens_reached"
                            else:
                                completion_reason = "unknown"
                            
                            logger.info(f"Stream completion: {completion_reason}, tokens={token_count}, "
                                      f"elapsed={elapsed_time:.2f}s")
                        
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
            # Log error information for debugging when generation fails mid-stream
            elapsed_time = time.time() - start_time
            logger.error(f"Streaming generation failed after {token_count} tokens and {elapsed_time:.2f}s. "
                        f"Error: {str(e)}")
            raise RuntimeError(f"Streaming generation failed: {str(e)}")
    
    def unload_model(self):
        """Unload the current model to free memory"""
        if self.model:
            del self.model
            self.model = None
            self.model_name = None
            self.model_type = None
            self.load_time = None
            
        # Also clear tokenizer
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
            
        logger.info("Model and tokenizer unloaded successfully")


# Global instance
llm_handler = LLMHandler() 