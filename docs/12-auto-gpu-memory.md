# Automatic GPU Memory Utilization

## Overview

Model Garden now features **automatic GPU memory utilization calculation** to optimize vLLM inference performance. When `gpu_memory_utilization` is set to `0` (now the default), the system intelligently calculates the optimal memory utilization based on:

- **Available GPU VRAM**: Detected automatically from the GPU
- **Model size**: Estimated from model files or inferred from model name
- **Sequence length**: Based on `max_model_len` parameter
- **Tensor parallelism**: Accounts for model sharding across multiple GPUs

## How It Works

### 1. GPU Memory Detection
```python
def get_gpu_memory_gb() -> float
```
Detects total GPU memory using PyTorch CUDA APIs. Returns 0.0 if no GPU is available.

### 2. Model Size Estimation
```python
def estimate_model_size_gb(model_path: str) -> float
```
Estimates model size in GB by:
- **For local models**: Scanning weight files (`.safetensors`, `.bin`)
- **For HuggingFace models**: Extracting parameter count from model name (e.g., "7B", "3B")
- **From config**: Calculating from architecture parameters in `config.json`
- **Default fallback**: Assumes 7GB if no size information is available

The regex pattern `(\d+(?:\.\d+)?)[Bb](?!it)` matches:
- ✅ `7B`, `13B`, `3B`, `1.1B` (parameter counts)
- ❌ `4bit`, `8bit` (quantization indicators)

### 3. KV Cache Estimation
The algorithm estimates the KV cache size based on:
- Model size (scales with number of parameters)
- Maximum sequence length (`max_model_len`)
- vLLM's paged attention efficiency
- Rule of thumb: ~0.4 GB per 1K tokens for 7B models (vLLM is memory efficient)

Formula:
```python
kv_cache_gb = (model_size_gb / 7.0) * (max_model_len / 1000) * 0.4
```

### 4. Utilization Calculation
```python
def calculate_gpu_memory_utilization(
    model_path: str,
    max_model_len: Optional[int] = None,
    tensor_parallel_size: int = 1,
) -> float
```

The final utilization is calculated with three strategies:

#### Conservative (Small Models)
- **Condition**: Model + KV cache < 70% of GPU memory
- **Utilization**: 0.50-0.75 (leaves plenty of room for batching)
- **Example**: 3B model on 24GB GPU with 4K context → 0.58
- **Example**: 1B model on 24GB GPU with 2K context → 0.50

#### Standard (Medium Models)
- **Condition**: Model + KV cache between 70-100% of GPU memory
- **Utilization**: 0.88
- **Example**: 7B model on 24GB GPU with 4K context → 0.88

#### Aggressive (Large Models)
- **Condition**: Model + KV cache ≥ 100% of GPU memory
- **Utilization**: 0.95 (maximum safe utilization)
- **Example**: 7B model on 24GB GPU with 8K context → 0.95

All calculations include a **20% safety margin** for temporary buffers, CUDA graphs, and overhead.

## Usage

### CLI (Default Auto Mode)
```bash
# Auto mode (recommended)
uv run model-garden serve-model --model-path ./models/my-model

# Manual override
uv run model-garden serve-model \
    --model-path ./models/my-model \
    --gpu-memory-utilization 0.85
```

### API (Default Auto Mode)
```python
# Auto mode
service = InferenceService(
    model_path="Qwen/Qwen2.5-VL-3B-Instruct",
    gpu_memory_utilization=0.0,  # 0 = auto
    max_model_len=4096,
)

# Manual override
service = InferenceService(
    model_path="Qwen/Qwen2.5-VL-3B-Instruct",
    gpu_memory_utilization=0.85,
    max_model_len=4096,
)
```

### REST API
```bash
curl -X POST http://localhost:8000/api/v1/inference/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "Qwen/Qwen2.5-VL-3B-Instruct",
    "gpu_memory_utilization": 0.0,
    "max_model_len": 4096
  }'
```

## Output During Model Loading

When auto mode is enabled, you'll see detailed information during model loading:

```
🔧 Auto mode enabled for GPU memory utilization
💾 Detected GPU memory: 23.5 GB
📊 Estimated model size: 6.0 GB
🗄️  Estimated KV cache: 1.4 GB (for max_model_len=4096)
✓ Model requires ~38% of GPU memory
  Using conservative utilization: 0.58 (leaves room for batching)
✓ Calculated GPU memory utilization: 0.58
```

## Benefits

1. **Optimal Performance**: Maximizes throughput without risking OOM errors
2. **Prevents OOM**: Conservative for small models, aggressive only when needed
3. **Better Batching**: Leaves headroom for concurrent requests on smaller models
4. **Zero Configuration**: Works out of the box for most use cases
5. **Flexible**: Can still manually override when needed

## Testing

Run the comprehensive test suite:
```bash
uv run python test_gpu_memory_auto.py
```

Tests cover:
- GPU memory detection
- Model size estimation (local files and HuggingFace models)
- Utilization calculation for various model sizes and context lengths
- InferenceService integration with auto mode

## Implementation Details

### File Changes
1. **`model_garden/inference.py`**:
   - Added `get_gpu_memory_gb()`
   - Added `estimate_model_size_gb()`
   - Added `calculate_gpu_memory_utilization()`
   - Updated `InferenceService.__init__()` default to `0.0`
   - Updated `InferenceService.load_model()` to calculate when `0.0`

2. **`model_garden/cli.py`**:
   - Updated `--gpu-memory-utilization` default from `0.9` to `0.0`
   - Updated help text to indicate `0 = auto`

3. **`model_garden/api.py`**:
   - Updated `LoadModelRequest.gpu_memory_utilization` default to `0.0`
   - Updated environment variable default from `"0.9"` to `"0.0"`

### Backward Compatibility
- ✅ Existing scripts with manual `gpu_memory_utilization` values continue to work
- ✅ Default behavior is now smarter (auto instead of fixed 0.9)
- ✅ Zero-downtime migration (defaults changed, but functionality preserved)

## Future Enhancements

Potential improvements for future versions:
- Support for dynamic batch size estimation
- Integration with vLLM's memory profiler for more accurate estimates
- Per-request memory tracking for multi-tenant scenarios
- Support for multi-GPU configurations with different VRAM sizes
- Model-specific tuning profiles (vision models, mixture-of-experts, etc.)
