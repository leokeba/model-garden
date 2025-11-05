# Vision LoRA Automatic Merging for Inference

## Overview

This document describes the automatic vision LoRA merging feature implemented to work around vLLM's limitation: **vLLM cannot load LoRA adapters on vision-language models**.

## Problem

vLLM's LoRA support only works with text-only models. When trying to load a vision LoRA adapter (e.g., fine-tuned Qwen2.5-VL), vLLM would fail because:
1. Vision models have different architectures (vision encoders + language models)
2. vLLM's LoRA implementation doesn't support the vision encoder components
3. This prevents users from serving fine-tuned vision models via the inference API

## Solution

We implemented automatic LoRA merging for vision models:

1. **Detection**: Automatically detect when a model is:
   - A LoRA adapter (`adapter_config.json` present)
   - A vision model (name contains "VL", has `processor_config.json`, etc.)

2. **Automatic Merge**: Before loading into vLLM:
   - Load the base vision model using Unsloth's `FastVisionModel`
   - Load the LoRA adapter on top using PEFT
   - Merge the adapter weights into the base model
   - Save to a temporary directory
   - Load the merged model into vLLM (no LoRA needed)

3. **Cleanup**: Automatically delete temporary merged models when:
   - The model is unloaded
   - The service is shut down

## Implementation Details

### New Functions

#### `merge_vision_lora_adapter()` (vision_training.py)

```python
def merge_vision_lora_adapter(
    adapter_path: str,
    output_dir: str,
    base_model: Optional[str] = None,
    max_seq_length: int = 16384,
    load_in_4bit: bool = True,
    save_method: str = "merged_16bit",
    maximum_memory_usage: float = 0.75,
    max_shard_size: str = "5GB",
) -> str
```

Merges a vision LoRA adapter with its base model for inference.

**Features:**
- Auto-detects base model from `adapter_config.json`
- Supports both local and HuggingFace Hub adapters
- Memory-efficient merging (4-bit loading, 16-bit output)
- Automatic cleanup after merge

#### `is_vision_model()` (inference.py)

```python
def is_vision_model(model_path: str) -> bool
```

Detects vision-language models by checking:
- Model name (contains "VL", "vision", etc.)
- Presence of `processor_config.json`
- Vision-specific config fields (`vision_config`, `visual_config`)
- Architecture names in `config.json`

### Modified Components

#### `InferenceService` Class

**New Attributes:**
- `is_vision_lora_adapter`: Flag indicating vision LoRA adapter
- `merged_vision_model_path`: Path to temporary merged model

**Updated Methods:**

1. **`load_model()`**:
   - Detects vision LoRA adapters
   - Calls `merge_vision_lora_adapter()` to create merged model
   - Loads merged model instead of base + adapter
   - Disables LoRA support for merged models

2. **`unload_model()`**:
   - Cleans up temporary merged model directory
   - Uses `shutil.rmtree()` to delete all files

3. **`_generate_complete()` / `_generate_streaming()`**:
   - Skip LoRA request creation for vision models (check `not self.is_vision_lora_adapter`)
   - Prevents vLLM errors when generating with merged models

4. **`get_model_info()`**:
   - Reports vision adapter status
   - Includes merge information in model info

## Usage

### Automatic (Recommended)

Simply serve a vision LoRA adapter like any other model:

```bash
# Serve a vision LoRA adapter - automatic merge happens behind the scenes
uv run model-garden serve-model --model-path ./models/my-qwen-vl-adapter

# Or via API
curl -X POST http://localhost:8000/api/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_path": "./models/my-qwen-vl-adapter"}'
```

**What happens:**
1. System detects it's a vision LoRA adapter
2. Automatically merges with base model (Qwen2.5-VL-3B-Instruct)
3. Saves merged model to temp directory (`$HF_HOME/temp_merges/model-garden-merged-<timestamp>`)
4. Loads merged model into vLLM
5. Cleans up temp directory when model is unloaded

### Manual Merge (Advanced)

You can also manually merge adapters for reuse:

```python
from model_garden.vision_training import merge_vision_lora_adapter

# Merge adapter and save permanently
merged_path = merge_vision_lora_adapter(
    adapter_path="./models/my-qwen-vl-adapter",
    output_dir="./models/my-qwen-vl-merged",
    save_method="merged_16bit"
)

# Now serve the merged model (faster loading, no merge needed)
# uv run model-garden serve-model --model-path ./models/my-qwen-vl-merged
```

## Benefits

1. **Seamless Experience**: Users don't need to know about vLLM's LoRA limitation
2. **Memory Efficient**: Merging uses 4-bit loading, final model is 16-bit for quality
3. **Automatic Cleanup**: No manual cleanup needed, temp files are removed automatically
4. **Works with Hub**: Supports both local adapters and HuggingFace Hub models
5. **Backward Compatible**: Text model adapters still use vLLM's native LoRA support

## Performance Considerations

### Memory Usage

- **Merge Process**: Requires ~12GB RAM + ~8GB VRAM for Qwen2.5-VL-3B (4-bit loading)
- **Final Model**: ~7GB VRAM for merged 16-bit model
- **Temporary Storage**: ~6GB disk space during merge (cleaned up after)

### Loading Time

- **First Load**: Slower due to merge (~2-3 minutes for 3B model)
- **Subsequent Loads**: If you keep the merged model, no merge needed
- **Text Models**: No impact, still use fast LoRA loading

### Optimization Tips

1. **Manual Pre-merge**: For production, merge adapters once and save permanently
2. **Reduce Memory**: Use `maximum_memory_usage=0.5` if running out of RAM
3. **Smaller Shards**: Use `max_shard_size="2GB"` to reduce peak memory
4. **Reuse Merged Models**: Save merged models to avoid repeated merging

## Testing

Run the test suite:

```bash
python test_vision_lora_merge.py
```

Tests verify:
- Vision model detection
- LoRA adapter detection
- Merge function availability
- InferenceService attributes

## Example Workflow

### Training a Vision LoRA

```bash
uv run model-garden train-vision \
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset ./data/my_vision_dataset.jsonl \
  --output ./models/my-qwen-vl-adapter \
  --save-method lora
```

### Serving the Adapter (Automatic Merge)

```bash
uv run model-garden serve-model \
  --model-path ./models/my-qwen-vl-adapter
```

Console output:
```
Loading model: ./models/my-qwen-vl-adapter
📦 Detected LoRA adapter repository: ./models/my-qwen-vl-adapter
🔍 Found base model in adapter config: Qwen/Qwen2.5-VL-3B-Instruct
⚠️  Detected vision-language model adapter
   vLLM doesn't support LoRA on vision models - merging adapter with base model first
🔧 Merging vision LoRA adapter...
   Adapter: ./models/my-qwen-vl-adapter
   Base model: Qwen/Qwen2.5-VL-3B-Instruct
   Output: /tmp/model-garden-merged-1729699200
Loading base model...
✓ Base model loaded
Loading LoRA adapter from ./models/my-qwen-vl-adapter...
✓ LoRA adapter loaded
Merging adapter and saving (merged_16bit)...
✓ Model merged and saved in 16-bit precision
✓ Processor saved
🧹 Cleaning up memory...
✓ Memory cleaned up
✨ Vision LoRA adapter merged successfully!
📦 Loading merged vision model into vLLM...
✓ Base model loaded successfully
```

### Making Requests

```python
import requests

response = requests.post(
    "http://localhost:8000/api/chat/completions",
    json={
        "messages": [
            {
                "role": "user",
                "content": "What do you see in this image?"
            }
        ],
        "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
        "max_tokens": 512
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

## Troubleshooting

### Out of Memory During Merge

**Problem**: Merge fails with OOM error

**Solution**: Reduce memory usage parameters
```python
merge_vision_lora_adapter(
    ...,
    maximum_memory_usage=0.5,  # Lower from 0.75
    max_shard_size="2GB"       # Lower from 5GB
)
```

### Merge Takes Too Long

**Problem**: Merge process is slow

**Solution**: Pre-merge adapters and save permanently
```bash
# Merge once and save
uv run python -c "
from model_garden.vision_training import merge_vision_lora_adapter
merge_vision_lora_adapter(
    './models/adapter',
    './models/merged',
    save_method='merged_16bit'
)
"

# Serve merged model (no merge needed)
uv run model-garden serve-model --model-path ./models/merged
```

### Temp Directory Not Cleaned Up

**Problem**: `$HF_HOME/temp_merges/model-garden-merged-*` directories remain

**Solution**: Manual cleanup
```bash
rm -rf $HF_HOME/temp_merges/model-garden-merged-*
```

Or restart the service to trigger cleanup on shutdown.

**Note**: Temporary merged models are stored in `$HF_HOME/temp_merges/` (not `/tmp/`) to avoid filling up the main system drive.

## Future Improvements

1. **Caching**: Cache merged models to avoid re-merging
2. **Parallel Loading**: Merge in background while loading base model
3. **Partial Merge**: Only merge necessary layers for inference
4. **GGUF Support**: Add support for GGUF format merging
5. **Multi-Adapter**: Support multiple vision adapters simultaneously

## Related Documentation

- [Vision Training Guide](VISION_TRAINING.md) - How to train vision models
- [Inference API](API_COMPARISON.md) - Using the inference API
- [vLLM Documentation](https://docs.vllm.ai/) - vLLM's official docs
- [Unsloth Documentation](https://github.com/unslothai/unsloth) - Unsloth features

## Summary

This feature enables seamless serving of fine-tuned vision-language models by automatically merging LoRA adapters before loading into vLLM. The implementation is transparent to users, handles cleanup automatically, and maintains backward compatibility with text models.
