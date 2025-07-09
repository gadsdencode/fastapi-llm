import os
import time
import logging
import psutil
from typing import Optional, Dict, Any, Iterator, AsyncIterator
from threading import Thread
from queue import Queue, Empty
import asyncio

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline, 
    TextIteratorStreamer,
    StoppingCriteria,
    StoppingCriteriaList
)
import torch

from ..schemas.models import ModelType, GenerateRequest, GenerateResponse, StreamChunk, ModelInfo

logger = logging.getLogger(__name__)


class StopSequenceCriteria(StoppingCriteria):
    """Custom stopping criteria for stop sequences"""
    
    def __init__(self, stop_sequences: list, tokenizer):
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Decode the last few tokens to check for stop sequences
        last_tokens = self.tokenizer.decode(input_ids[0][-20:], skip_special_tokens=True)
        return any(stop_seq in last_tokens for stop_seq in self.stop_sequences)


class LLMHandler:
    """Handles LLM model loading, inference, and streaming"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_name: Optional[str] = None
        self.model_type: Optional[ModelType] = None
        self.load_time: Optional[float] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
    
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded"""
        return self.model is not None and self.tokenizer is not None
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        memory = psutil.virtual_memory()
        gpu_memory = {}
        
        if torch.cuda.is_available():
            gpu_memory = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "max_allocated": torch.cuda.max_memory_allocated(),
            }
        
        return {
            "system": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used
            },
            "gpu": gpu_memory
        }
    
    async def load_model(self, model_name: str, model_type: ModelType = ModelType.HUGGINGFACE, force_reload: bool = False) -> bool:
        """Load a model based on type"""
        if self.is_loaded() and self.model_name == model_name and not force_reload:
            logger.info(f"Model {model_name} already loaded")
            return True
        
        logger.info(f"Loading model: {model_name} (type: {model_type})")
        start_time = time.time()
        
        try:
            if model_type == ModelType.HUGGINGFACE:
                await self._load_huggingface_model(model_name)
            elif model_type == ModelType.GGUF:
                raise NotImplementedError("GGUF model loading not yet implemented")
            elif model_type == ModelType.OLLAMA:
                raise NotImplementedError("Ollama model loading not yet implemented")
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            self.model_name = model_name
            self.model_type = model_type
            self.load_time = time.time() - start_time
            
            logger.info(f"Model loaded successfully in {self.load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            self.model = None
            self.tokenizer = None
            self.pipeline = None
            raise
    
    async def _load_huggingface_model(self, model_name: str):
        """Load a Hugging Face model"""
        # Set cache directory from environment
        cache_dir = os.getenv("TRANSFORMERS_CACHE", "./models")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Create pipeline for easier inference
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
    
    def get_model_info(self) -> ModelInfo:
        """Get information about the currently loaded model"""
        if not self.is_loaded():
            return ModelInfo(
                name="None",
                type=ModelType.HUGGINGFACE,
                loaded=False
            )
        
        parameters = None
        if self.model:
            parameters = {
                "total_params": sum(p.numel() for p in self.model.parameters()),
                "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "device": str(self.model.device) if hasattr(self.model, 'device') else self.device
            }
        
        return ModelInfo(
            name=self.model_name,
            type=self.model_type,
            loaded=True,
            memory_usage=self.get_memory_usage(),
            parameters=parameters,
            load_time=self.load_time
        )
    
    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate text from prompt"""
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Please load a model first.")
        
        start_time = time.time()
        
        # Prepare stopping criteria
        stopping_criteria = []
        if request.stop_sequences:
            stopping_criteria.append(
                StopSequenceCriteria(request.stop_sequences, self.tokenizer)
            )
        
        # Generate text
        try:
            outputs = self.pipeline(
                request.prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList(stopping_criteria) if stopping_criteria else None,
                return_full_text=False
            )
            
            generated_text = outputs[0]['generated_text']
            
            # Count tokens
            tokens_generated = len(self.tokenizer.encode(generated_text))
            generation_time = time.time() - start_time
            
            return GenerateResponse(
                generated_text=generated_text,
                prompt=request.prompt,
                model_name=self.model_name,
                tokens_generated=tokens_generated,
                generation_time=generation_time
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    async def generate_stream(self, request: GenerateRequest) -> AsyncIterator[StreamChunk]:
        """Generate text with streaming response"""
        if not self.is_loaded():
            raise RuntimeError("No model loaded. Please load a model first.")
        
        # Create streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            timeout=30.0,
            skip_prompt=True,
            skip_special_tokens=True
        )
        
        # Prepare generation kwargs
        generation_kwargs = {
            "inputs": request.prompt,
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
            "return_full_text": False
        }
        
        # Add stopping criteria if provided
        if request.stop_sequences:
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([
                StopSequenceCriteria(request.stop_sequences, self.tokenizer)
            ])
        
        # Start generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs={
            **generation_kwargs,
            "inputs": self.tokenizer.encode(request.prompt, return_tensors="pt").to(self.device)
        })
        thread.start()
        
        # Stream the results
        tokens_generated = 0
        try:
            for text in streamer:
                if text:
                    tokens_generated += len(self.tokenizer.encode(text))
                    yield StreamChunk(
                        text=text,
                        is_final=False,
                        tokens_generated=tokens_generated
                    )
            
            # Send final chunk
            yield StreamChunk(
                text="",
                is_final=True,
                tokens_generated=tokens_generated
            )
            
        except Exception as e:
            logger.error(f"Streaming generation failed: {str(e)}")
            yield StreamChunk(
                text=f"Error: {str(e)}",
                is_final=True,
                tokens_generated=tokens_generated
            )
        finally:
            thread.join(timeout=1.0)


# Global instance
llm_handler = LLMHandler() 