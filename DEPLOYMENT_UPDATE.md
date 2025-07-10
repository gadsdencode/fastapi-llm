# Deployment Update: Llama 3.2 3B Instruct Model

## Changes Made

### 1. Updated Default Model Configuration
**File**: `app/main.py`
- Changed default MODEL_NAME to `v8karlo/Llama-3.2-3B-Instruct-uncensored-Q8_0-GGUF`
- Changed default MODEL_TYPE to `gguf`
- Enabled LOAD_MODEL_ON_STARTUP by default

### 2. Updated Environment Configuration
**File**: `env`
- Updated MODEL_NAME to Llama 3.2 3B Instruct model
- Changed MODEL_TYPE from `huggingface` to `gguf`
- Maintained LOAD_MODEL_ON_STARTUP=true

### 3. Implementation Features
- **Automatic Model Detection**: System detects Llama 3.2 models and applies correct chat template
- **Optimized Threading**: 32 threads (67% of 48 cores) for maximum performance
- **Q8_0 Quantization**: Higher accuracy than previous Q4_K_M quantization
- **Proper Stop Tokens**: Uses Llama 3.2 specific tokens (`<|eot_id|>`, etc.)

## Railway Deployment

### Environment Variables to Set (Optional Override)
If you need to override the defaults on Railway, set these environment variables:

```bash
MODEL_NAME=v8karlo/Llama-3.2-3B-Instruct-uncensored-Q8_0-GGUF
MODEL_TYPE=gguf
LOAD_MODEL_ON_STARTUP=true
```

### Expected Startup Logs
After deployment, you should see logs like:
```
INFO - Loading model: v8karlo/Llama-3.2-3B-Instruct-uncensored-Q8_0-GGUF (type: ModelType.GGUF)
INFO - Found target Llama 3.2 Q8_0 file: llama-3.2-3b-instruct-uncensored-q8_0.gguf
INFO - Detected 48 CPU cores, using 32 threads (batch: 20) for high-performance inference
INFO - Model loaded successfully in X.XXs
```

### Performance Expectations
- **Model Size**: ~3.6GB (Q8_0 quantization)
- **Load Time**: 8-15 seconds
- **Generation Speed**: 40-80 tokens/second (8-16x improvement)
- **CPU Utilization**: ~67% of 48 cores
- **Memory Usage**: More efficient with better quantization

## Verification

### Test the New Model
1. **Basic Generation Test**:
   ```bash
   curl -X POST "https://your-railway-app.railway.app/api/v1/generate" \
   -H "Content-Type: application/json" \
   -d '{
     "prompt": "What is the capital of France?",
     "max_tokens": 100,
     "temperature": 0.7
   }'
   ```

2. **Model Info Check**:
   ```bash
   curl "https://your-railway-app.railway.app/api/v1/model/info"
   ```

### Expected Response Format
The model should respond with proper Llama 3.2 formatting and improved quality due to Q8_0 quantization.

## Rollback Plan
If issues occur, you can quickly rollback by setting these Railway environment variables:
```bash
MODEL_NAME=v8karlo/UNCENSORED-TinyDolphin-3x-MoE-Q4_K_M-GGUF
MODEL_TYPE=huggingface
```

## Next Steps
1. Deploy the updated code to Railway
2. Monitor startup logs for successful model loading
3. Test generation performance and quality
4. Verify CPU utilization improvements (~67%)
5. Benchmark token generation speed (target: 40-80 tok/s) 