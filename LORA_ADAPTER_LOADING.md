# LoRA Adapter Loading Guide

Model Garden now supports loading LoRA adapters directly from HuggingFace Hub or local paths, without requiring you to merge them first. This provides several benefits:

- **Faster deployment**: No need to merge weights before inference
- **Memory efficiency**: Base model + adapters use less memory than merged models
- **Flexibility**: Switch between different adapters on the same base model
- **Hub integration**: Load adapters directly from HuggingFace Hub

## Quick Start

### Loading a LoRA Adapter from HuggingFace Hub

```bash
# The adapter's base model is automatically detected from adapter_config.json
uv run model-garden serve-model \
    --model-path Barth371/Qwen2.5-VL-72B-Instruct-bnb-4bit-2025-10-21_16-26_batch_size_4_cmr-block-2_adapters_4bit \
    --port 8000
```

### Loading a Local LoRA Adapter

```bash
# For local adapters, base model is also auto-detected
uv run model-garden serve-model \
    --model-path ./models/my-adapter \
    --port 8000
```

### Explicit Base Model Override

If your adapter doesn't have a valid `adapter_config.json` or you want to use a different base model:

```bash
uv run model-garden serve-model \
    --model-path ./models/my-adapter \
    --base-model Qwen/Qwen2.5-VL-72B-Instruct-bnb-4bit \
    --port 8000
```

## How It Works

### Automatic Detection

When you provide a model path, Model Garden:

1. **Checks for adapter_config.json**: If found, recognizes it as a LoRA adapter
2. **Extracts base model**: Reads `base_model_name_or_path` from the config
3. **Loads base model**: Uses vLLM to load the base model with LoRA support enabled
4. **Applies adapter**: Automatically applies the LoRA adapter to all requests

### LoRA Request Flow

```
User Request → vLLM Engine (Base Model) → LoRA Adapter → Response
```

Every inference request automatically includes the LoRA adapter, so you get fine-tuned model behavior without merging.

## CLI Commands

### Serve Model (Persistent Server)

```bash
# Basic usage with LoRA adapter
uv run model-garden serve-model \
    --model-path <adapter-path-or-hub-id>

# With custom LoRA settings
uv run model-garden serve-model \
    --model-path <adapter-path-or-hub-id> \
    --max-loras 4 \
    --max-lora-rank 128

# Disable LoRA (force merged model loading)
uv run model-garden serve-model \
    --model-path <model-path> \
    --no-enable-lora
```

### One-off Generation

```bash
# Generate with LoRA adapter
uv run model-garden inference-generate \
    --model-path Barth371/Qwen2.5-VL-72B-Instruct-bnb-4bit-2025-10-21_16-26_batch_size_4_cmr-block-2_adapters_4bit \
    --prompt "Extract the client name from this document"

# With streaming
uv run model-garden inference-generate \
    --model-path <adapter-path> \
    --prompt "Your prompt here" \
    --stream
```

## Configuration Options

### LoRA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--enable-lora` | `true` | Enable LoRA adapter support |
| `--max-loras` | `1` | Maximum concurrent LoRA adapters |
| `--max-lora-rank` | `64` | Maximum LoRA rank to support |

### Model Parameters

All standard model parameters work with adapters:

```bash
uv run model-garden serve-model \
    --model-path <adapter-path> \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192 \
    --dtype float16
```

## HuggingFace Hub Integration

### Authentication

For private adapters or base models, set your HuggingFace token:

```bash
export HF_TOKEN=your_token_here
```

### Repository Structure

Your adapter repository should contain:

```
your-adapter-repo/
├── adapter_config.json     # Required: contains base_model_name_or_path
├── adapter_model.safetensors  # or adapter_model.bin
├── README.md              # Optional
└── tokenizer files...     # Optional: if different from base model
```

### Example adapter_config.json

```json
{
  "base_model_name_or_path": "Qwen/Qwen2.5-VL-72B-Instruct-bnb-4bit",
  "peft_type": "LORA",
  "r": 16,
  "lora_alpha": 16,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
  "lora_dropout": 0.0,
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

## API Usage

Once your adapter is loaded via `serve-model`, use the standard OpenAI-compatible API:

```python
import requests

# Chat completion
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "your-adapter",  # Any string works
        "messages": [
            {"role": "user", "content": "Your prompt here"}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

## Vision Models

LoRA adapters work seamlessly with vision-language models:

```bash
# Load a vision model adapter
uv run model-garden serve-model \
    --model-path Barth371/Qwen2.5-VL-72B-Instruct-bnb-4bit-2025-10-21_16-26_batch_size_4_cmr-block-2_adapters_4bit

# Use with vision API
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vision-adapter",
    "messages": [{
      "role": "user",
      "content": "Describe this image"
    }],
    "image": "data:image/jpeg;base64,..."
  }'
```

## Troubleshooting

### "Could not determine base model"

**Problem**: `adapter_config.json` is missing or doesn't contain `base_model_name_or_path`.

**Solution**: Specify base model explicitly:
```bash
uv run model-garden serve-model \
    --model-path <adapter-path> \
    --base-model <base-model-name>
```

### "LoRA support not available"

**Problem**: Your vLLM version doesn't support LoRA.

**Solution**: Update vLLM:
```bash
uv pip install --upgrade vllm
```

### Memory Issues

**Problem**: Out of memory when loading large models.

**Solution**: Adjust GPU memory utilization or use tensor parallelism:
```bash
uv run model-garden serve-model \
    --model-path <adapter-path> \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 2
```

### Adapter Not Applied

**Problem**: Model generates responses as if adapter isn't loaded.

**Solution**: Check that:
1. `adapter_config.json` exists in the adapter path
2. LoRA is enabled (`--enable-lora` is default, but check you didn't disable it)
3. Base model matches the one used during training

## Performance Considerations

### Memory Usage

- **Base model + adapter**: ~10-20% more memory than base model alone
- **Multiple adapters**: Each additional adapter uses minimal extra memory
- **vs. Merged model**: Adapters use similar or slightly less memory

### Latency

- **First request**: Slightly slower (~100-200ms) as adapter is loaded
- **Subsequent requests**: Identical to merged model performance
- **Switching adapters**: Fast (if using multi-adapter setup)

### Throughput

- **Single adapter**: Same throughput as merged model
- **Multiple adapters**: vLLM batches requests efficiently across adapters

## Best Practices

1. **Use descriptive adapter names**: Name adapters after their fine-tuning task
2. **Version control**: Use Git tags or version suffixes in Hub repo names
3. **Document base models**: Always specify base model in adapter_config.json
4. **Test locally first**: Verify adapter loading before deploying to production
5. **Monitor memory**: Start with conservative GPU memory settings

## Examples

### Document Extraction (Vision Model)

```bash
# Serve CMR extraction adapter
uv run model-garden serve-model \
    --model-path Barth371/Qwen2.5-VL-72B-Instruct-bnb-4bit-2025-10-21_16-26_batch_size_4_cmr-block-2_adapters_4bit \
    --port 8000 \
    --max-model-len 8192

# Use for extraction
python -c "
import requests
import base64

with open('document.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    'http://localhost:8000/v1/chat/completions',
    json={
        'model': 'cmr-extractor',
        'messages': [{'role': 'user', 'content': 'Extract CMR data'}],
        'image': img_b64,
        'max_tokens': 4096
    }
)

print(response.json()['choices'][0]['message']['content'])
"
```

### Text Generation (Language Model)

```bash
# Serve text generation adapter
uv run model-garden serve-model \
    --model-path username/my-custom-adapter \
    --port 8000

# Generate text
uv run model-garden inference-generate \
    --model-path username/my-custom-adapter \
    --prompt "Write a creative story about AI" \
    --max-tokens 1024 \
    --stream
```

## Migration from Merged Models

If you currently use merged models, switching to adapters is straightforward:

### Before (Merged Model)
```bash
uv run model-garden serve-model \
    --model-path ./models/merged-model
```

### After (LoRA Adapter)
```bash
uv run model-garden serve-model \
    --model-path ./models/adapter
    # Base model is auto-detected from adapter_config.json
```

The API remains identical - no client code changes needed!

## Additional Resources

- [vLLM LoRA Documentation](https://docs.vllm.ai/en/latest/models/lora.html)
- [PEFT Library](https://github.com/huggingface/peft)
- [Model Garden Training Guide](./VISION_SUPPORT.md)

## Support

For issues or questions:
1. Check this guide first
2. Review error messages for specific problems
3. Open an issue on GitHub with:
   - Model/adapter paths
   - Command used
   - Full error output
   - vLLM version (`pip show vllm`)
