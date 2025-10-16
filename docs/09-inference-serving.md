# Inference Serving with vLLM

## Overview

Model Garden now includes a high-performance inference engine powered by vLLM. This provides:

- **High-throughput text generation** with PagedAttention and continuous batching
- **Vision-language model support** (Qwen2.5-VL and compatible models)
- **OpenAI-compatible API** for easy integration
- **Streaming support** for real-time responses
- **Multi-GPU support** via tensor parallelism
- **Quantization support** (AWQ, GPTQ, SqueezeLLM, FP8)
- **CLI and API interfaces** for flexible deployment
- **Multimodal input support** (text, images via URL, base64, or file path)

## Architecture

The inference system has three layers:

1. **Service Layer** (`model_garden/inference.py`)
   - `InferenceService` class manages vLLM AsyncLLMEngine
   - Handles model loading/unloading, generation, chat completions
   - Supports streaming and non-streaming modes

2. **API Layer** (`model_garden/api.py`)
   - FastAPI endpoints for model management and inference
   - Server-Sent Events (SSE) for streaming responses
   - OpenAI-compatible chat completion format

3. **CLI Layer** (`model_garden/cli.py`)
   - Commands for serving models and one-off generation
   - Interactive chat interface

## API Endpoints

### Model Management

#### Load Model
```http
POST /api/v1/inference/load
Content-Type: application/json

{
  "model_path": "./models/my-model",
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": 0.9,
  "quantization": "awq",
  "max_model_len": 4096
}
```

#### Unload Model
```http
POST /api/v1/inference/unload
```

#### Check Status
```http
GET /api/v1/inference/status
```

Response:
```json
{
  "is_loaded": true,
  "model_path": "./models/my-model",
  "timestamp": "2024-01-01T12:00:00"
}
```

### Text Generation

#### Generate Text
```http
POST /api/v1/inference/generate
Content-Type: application/json

{
  "prompt": "Once upon a time",
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "stream": false
}
```

Response (non-streaming):
```json
{
  "text": "Once upon a time, in a land far away...",
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 50,
    "total_tokens": 55
  }
}
```

Response (streaming with `stream: true`):
```
data: {"text": "Once", "finish_reason": null}
data: {"text": " upon", "finish_reason": null}
data: {"text": " a", "finish_reason": null}
...
data: {"text": "", "finish_reason": "stop"}
```

### Chat Completions

#### OpenAI-Compatible Chat
```http
POST /api/v1/chat/completions
Content-Type: application/json

{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "What is quantum computing?"}
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "stream": false
}
```

#### Chat with Vision (Multimodal)

For vision-language models like Qwen2.5-VL, you can include images in messages:

```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "Qwen/Qwen2.5-VL-3B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What do you see in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }
  ],
  "max_tokens": 200
}
```

**Supported image formats:**
- **HTTP/HTTPS URLs**: `{"url": "https://example.com/image.jpg"}`
- **Base64 data URLs**: `{"url": "data:image/png;base64,iVBORw0KGgo..."}`
- **Local file paths**: Supported via direct API (not recommended for production)

**Example with base64 image:**
```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "Qwen/Qwen2.5-VL-3B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
          }
        }
      ]
    }
  ],
  "max_tokens": 100
}
```

Response (non-streaming):
```json
{
  "id": "chatcmpl-140038686648448",
  "object": "chat.completion",
  "created": 1704110400,
  "model": "Qwen/Qwen2.5-VL-3B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "In the image, I can see a vintage car parked on a street..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 424,
    "completion_tokens": 76,
    "total_tokens": 500
  }
}
```

Response (non-streaming):
```json
{
  "id": "chat-123",
  "object": "chat.completion",
  "created": 1704110400,
  "model": "./models/my-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Quantum computing is..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 100,
    "total_tokens": 115
  }
}
```

Response (streaming with `stream: true`):
```
data: {"id": "chat-123", "object": "chat.completion.chunk", "created": 1704110400, "model": "./models/my-model", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Quantum"}, "finish_reason": null}]}
data: {"id": "chat-123", "object": "chat.completion.chunk", "created": 1704110400, "model": "./models/my-model", "choices": [{"index": 0, "delta": {"content": " computing"}, "finish_reason": null}]}
...
data: [DONE]
```

## CLI Commands

### 1. Serve Model (Persistent Server)

Start an inference server that keeps running:

```bash
# Basic usage
uv run model-garden serve-model --model-path ./models/my-model

# With custom port
uv run model-garden serve-model \
  --model-path ./models/my-model \
  --port 8080

# Multi-GPU with tensor parallelism
uv run model-garden serve-model \
  --model-path ./models/my-model \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8

# With quantization
uv run model-garden serve-model \
  --model-path ./models/my-model \
  --quantization awq
```

Options:
- `--model-path`: Path to model (required)
- `--port`: Server port (default: 8000)
- `--host`: Bind address (default: 0.0.0.0)
- `--tensor-parallel-size`: Number of GPUs (default: 1)
- `--gpu-memory-utilization`: GPU memory fraction (default: 0.9)
- `--quantization`: Quantization method (awq/gptq/squeezellm/fp8)
- `--max-model-len`: Maximum context length

Once running, access:
- API: `http://localhost:8000`
- Documentation: `http://localhost:8000/docs`
- Frontend: `http://localhost:8000` (if configured)

### 2. One-off Generation

Generate text without starting a server:

```bash
# Basic generation
uv run model-garden inference-generate \
  --model-path ./models/my-model \
  --prompt "Once upon a time"

# With streaming output
uv run model-garden inference-generate \
  --model-path ./models/my-model \
  --prompt "Explain quantum computing" \
  --max-tokens 512 \
  --temperature 0.8 \
  --stream

# With quantization
uv run model-garden inference-generate \
  --model-path ./models/my-model \
  --prompt "Write a poem" \
  --quantization awq
```

Options:
- `--model-path`: Path to model (required)
- `--prompt`: Text prompt (required)
- `--max-tokens`: Max output tokens (default: 256)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Nucleus sampling (default: 0.9)
- `--stream/--no-stream`: Enable streaming (default: false)
- `--tensor-parallel-size`: Number of GPUs (default: 1)
- `--quantization`: Quantization method

### 3. Interactive Chat

Start a chat session:

```bash
# Basic chat
uv run model-garden inference-chat --model-path ./models/my-model

# With system prompt
uv run model-garden inference-chat \
  --model-path ./models/my-model \
  --system-prompt "You are a helpful AI assistant"

# With custom parameters
uv run model-garden inference-chat \
  --model-path ./models/my-model \
  --temperature 0.8 \
  --max-tokens 1024
```

Options:
- `--model-path`: Path to model (required)
- `--system-prompt`: System prompt for context
- `--max-tokens`: Max tokens per response (default: 512)
- `--temperature`: Sampling temperature (default: 0.7)
- `--tensor-parallel-size`: Number of GPUs (default: 1)
- `--quantization`: Quantization method

Usage:
- Type your message and press Enter
- Type 'exit', 'quit', or press Ctrl+D to end
- Chat history is maintained during the session

## Configuration Options

### Tensor Parallelism

Split model across multiple GPUs:

```python
service = InferenceService(tensor_parallel_size=4)  # Use 4 GPUs
```

Benefits:
- Handle larger models
- Increased throughput
- Lower per-GPU memory usage

### GPU Memory Utilization

Control how much GPU memory vLLM uses:

```python
service = InferenceService(gpu_memory_utilization=0.8)  # Use 80% of GPU memory
```

Recommendations:
- 0.9 (default): Maximum performance
- 0.7-0.8: Leave room for other processes
- 0.5-0.6: Conservative, avoid OOM errors

### Quantization

Reduce memory usage with quantization:

```python
service = InferenceService(quantization="awq")
```

Supported methods:
- **AWQ**: Activation-aware Weight Quantization (recommended)
- **GPTQ**: General Post-Training Quantization
- **SqueezeLLM**: Efficient quantization
- **FP8**: 8-bit floating point

Note: Model must be pre-quantized in the same format.

### Maximum Model Length

Override default context length:

```python
service = InferenceService(max_model_len=8192)
```

Useful for:
- Models with sliding window attention
- Controlling memory usage
- Handling long contexts

## Python API Usage

### Direct Service Usage

```python
from model_garden.inference import InferenceService
import asyncio

async def main():
    # Create service
    service = InferenceService(
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9
    )
    
    # Load model
    await service.load_model("./models/my-model")
    
    # Generate text (non-streaming)
    result = await service.generate(
        prompt="Once upon a time",
        max_tokens=256,
        temperature=0.7
    )
    print(result["text"])
    
    # Generate with images (vision models)
    result = await service.generate(
        prompt="Describe this image",
        images=["https://example.com/image.jpg"],
        max_tokens=200
    )
    print(result["text"])
    
    # Generate text (streaming)
    async for chunk in service.generate(
        prompt="Write a story",
        max_tokens=512,
        stream=True
    ):
        print(chunk["text"], end="")
    
    # Chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is AI?"}
    ]
    result = await service.chat_completion(
        messages=messages,
        max_tokens=512
    )
    print(result["choices"][0]["message"]["content"])
    
    # Cleanup
    await service.unload_model()

asyncio.run(main())
```

### Using with FastAPI (already integrated)

The inference service is already integrated into the FastAPI app. Just start the server:

```bash
uv run model-garden serve-model --model-path ./models/my-model
```

Then make HTTP requests to the endpoints listed above.

## Performance Tips

1. **Use tensor parallelism** for large models (>30B parameters)
2. **Enable quantization** if memory is limited
3. **Adjust gpu_memory_utilization** based on available VRAM
4. **Use streaming** for better user experience with long responses
5. **Batch requests** when possible (vLLM handles this automatically)
6. **Monitor GPU usage** with `nvidia-smi` to optimize settings

## Troubleshooting

### Out of Memory (OOM) Errors

1. Reduce `gpu_memory_utilization` (try 0.7 or 0.8)
2. Use quantization (`--quantization awq`)
3. Reduce `max_model_len`
4. Increase `tensor_parallel_size` to use more GPUs

### Slow Generation

1. Check GPU utilization with `nvidia-smi`
2. Increase `gpu_memory_utilization` if memory is available
3. Use tensor parallelism with more GPUs
4. Ensure no other processes are using the GPU

### Model Loading Fails

1. Verify model path is correct
2. Check model format is supported by vLLM
3. Ensure sufficient GPU memory
4. Check quantization format matches model

### Streaming Not Working

1. Verify `stream: true` in request
2. Check client supports Server-Sent Events (SSE)
3. Ensure no proxy buffering responses
4. Test with `curl --no-buffer`

### Vision Model Issues

#### EngineCore Crashes with Images

**Symptom**: Error "EngineCore encountered an issue" when sending images

**Solution**: This was fixed in recent updates. Ensure you have the latest code with these fixes:
- Images are always passed as lists to vLLM (not single strings)
- Base64 images are converted to temporary PNG files
- Token counting properly extracts text from result dicts

#### Base64 Images Not Working

**Symptom**: Images encoded as base64 data URLs fail to process

**Solution**: Use the format `data:image/png;base64,<BASE64_DATA>` and ensure:
- The base64 string is properly encoded
- The image format (PNG, JPEG, etc.) matches the MIME type
- The image size is reasonable (< 10MB recommended)

#### Image URLs Timeout

**Symptom**: Requests with image URLs take too long or timeout

**Solution**:
- Ensure the image URL is accessible from the server
- Check firewall/network settings
- Consider downloading and using local images or base64 encoding
- Increase request timeout on client side

## Supported Models

### Text Models (Tested)
- ✅ **TinyLlama-1.1B**: Fast, lightweight model for testing
- ✅ Any Llama-compatible model
- ✅ Mistral, Mixtral models
- ✅ Qwen2 text models

### Vision-Language Models (Tested)
- ✅ **Qwen/Qwen2.5-VL-3B-Instruct**: 3B parameter vision-language model
  - Supports images via URL, base64, or file path
  - Excellent for document understanding and image analysis
  - ~7.16 GiB GPU memory required
- ✅ Other Qwen2-VL variants (should work similarly)

### Compatibility Notes
- vLLM V1 engine is automatically enabled for vision models (with chunked prefill)
- Vision models require special prompt formatting (handled automatically)
- Multi-image support available (pass multiple images in content array)

## Examples

### Example 1: Question Answering

```bash
uv run model-garden inference-generate \
  --model-path ./models/my-model \
  --prompt "What is the capital of France?" \
  --max-tokens 50
```

### Example 2: Code Generation

```bash
uv run model-garden inference-chat \
  --model-path ./models/my-model \
  --system-prompt "You are an expert programmer"
```

Then chat:
```
You: Write a Python function to calculate factorial
Assistant: Here's a Python function to calculate factorial:

def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)
```

### Example 3: Vision Model - Image Analysis

```bash
# Start server with vision model
uv run model-garden serve --host 0.0.0.0 --port 8000

# Load the model
curl -X POST http://localhost:8000/api/v1/inference/load \
  -H "Content-Type: application/json" \
  -d '{"model_path": "Qwen/Qwen2.5-VL-3B-Instruct"}'

# Analyze an image from URL
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this image in detail"},
          {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
        ]
      }
    ],
    "max_tokens": 200
  }'
```

### Example 4: Vision Model - Document Analysis with Base64

```python
import base64
import requests

# Read and encode image
with open("document.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Send to API
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this document"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                    }
                ]
            }
        ],
        "max_tokens": 500
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### Example 5: High-Throughput Server

```bash
# Start server with optimized settings
uv run model-garden serve-model \
  --model-path ./models/my-model \
  --port 8000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9

# Make requests with curl
curl -X POST http://localhost:8000/api/v1/inference/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 100}'
```

### Example 4: OpenAI Python Client

Since the API is OpenAI-compatible, you can use the official OpenAI Python client:

```python
from openai import OpenAI

# Point to your local server
client = OpenAI(
    base_url="http://localhost:8000/api/v1",
    api_key="dummy"  # Not used but required by client
)

# Chat completion
response = client.chat.completions.create(
    model="my-model",  # Ignored by vLLM
    messages=[
        {"role": "user", "content": "What is machine learning?"}
    ],
    max_tokens=512
)

print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="my-model",
    messages=[{"role": "user", "content": "Tell me a story"}],
    max_tokens=1024,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Next Steps

- Implement Web UI for inference with image upload support
- Add model benchmarking tools for vision models
- Implement request queuing and prioritization
- Add metrics and monitoring for multimodal requests
- Expand vision model support to other architectures
- Add video input support for video-language models
