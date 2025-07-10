# Performance Optimizations for Railway 48-Core Deployment

## Overview
Optimized FastAPI LLM server for Railway's 48-core CPU deployment using research-based threading and batch configurations.

## Key Optimizations Applied

### 1. High-Core CPU Threading
**Before**: 6 threads (12.5% of 48 cores)
**After**: 32 threads (67% of 48 cores)

**Research Basis**:
- Studies show optimal performance at 50-75% of available cores for high-core systems
- Beyond 75%, threads compete for memory bandwidth rather than compute
- MoE models can handle higher thread counts due to parallel expert architecture

### 2. Optimized Thread Configuration
- **Main Threads**: 32 (67% of 48 cores)
- **Batch Threads**: 20 (42% of 48 cores) 
- **Environment Variables**: Set to 32 threads for BLAS libraries

### 3. Batch Size Optimization
**Before**: n_batch=256, n_ubatch=128
**After**: n_batch=512, n_ubatch=256

**Rationale**: Higher batch sizes better utilize the increased thread count and memory bandwidth.

### 4. Model Selection
- **Target**: TinyDolphin-3x-MoE-Q4_K_M-GGUF
- **Quantization**: Q4_K_M for optimal speed/quality balance
- **Architecture**: Mixture of Experts for efficient parallel processing

### 5. Memory and KV Cache
- **KV Cache**: F16 (stable, no experimental quantization)
- **Context Size**: 2048 (conservative for Railway memory limits)
- **Memory Mapping**: Enabled for faster model loading

## Expected Performance Improvements

### Speed Targets
- **Current**: ~5 tokens/second
- **Target**: 40-80 tokens/second (8-16x improvement)
- **Response Time**: 8-15 seconds for 512 tokens

### Utilization Improvements
- **CPU Usage**: From 12.5% to 67% of available cores
- **Thread Efficiency**: Optimized for MoE architecture
- **Memory Bandwidth**: Better utilization without saturation

## Research Sources
1. Reddit r/LocalLLaMA CPU optimization discussions
2. GitHub llama.cpp performance analysis
3. High-core system benchmarking studies
4. MoE model threading optimization research

## Monitoring
- Watch for CPU utilization increases from ~12% to ~67%
- Monitor memory usage to ensure no saturation
- Track tokens/second improvements in generation logs
- Verify stable model loading without context failures

## Notes
- Configuration automatically scales down for smaller systems (<32 cores)
- Environment variables prevent BLAS library oversubscription
- Batch processing optimized for Railway's memory constraints 