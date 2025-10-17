# Model Garden - AI Coding Instructions

## Project Overview
Model Garden is a comprehensive platform for fine-tuning and serving LLMs/VLMs with carbon tracking. The architecture spans three layers: **CLI/Frontend** → **FastAPI Backend** → **Compute Layer** (Unsloth training + vLLM inference).

## Architecture Boundaries

### Core Modules (`model_garden/`)
- **`training.py`**: Unsloth-based fine-tuning (text-only models)
- **`vision_training.py`**: Vision-language model fine-tuning (Qwen2.5-VL)  
- **`inference.py`**: vLLM inference engine with auto-quantization detection
- **`api.py`**: FastAPI backend with WebSocket job monitoring
- **`cli.py`**: Click-based CLI interface

### Key Integration Points
- **Environment Setup**: Always configure `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE` before importing HF libraries
- **Quantization Logic**: Auto-detect merged vs quantized models in `inference.py:detect_quantization_method()`
- **Job Management**: Training jobs stored as JSON in `storage/training_jobs.json`

## Critical Workflows

### Development Setup
```bash
# Required: Use uv for dependency management
uv sync  # Install all dependencies
uv run model-garden --help  # CLI entry point

# GPU cleanup (important for vLLM memory issues)
./cleanup_gpu.sh  # Kill vLLM processes and clear GPU memory
```

### Training Workflows
```bash
# Text-only fine-tuning (uses training.py)
uv run model-garden train --base-model unsloth/tinyllama-bnb-4bit --dataset ./data/sample.jsonl

# Vision-language fine-tuning (uses vision_training.py) 
uv run model-garden train-vision --base-model Qwen/Qwen2.5-VL-3B-Instruct --dataset ./data/vision.jsonl
```

### Inference Patterns
```bash
# Auto-detects quantization and serves via vLLM
uv run model-garden serve-model --model-path ./models/my-model

# Full API server with UI
uv run model-garden serve  # Serves FastAPI + static frontend
```

## Project-Specific Conventions

### Dataset Formats
- **Text**: `{"instruction": "...", "input": "...", "output": "..."}`
- **Vision**: `{"text": "...", "image": "/path/to/img.jpg", "response": "..."}`

### Model Save Strategies
- **LoRA adapters**: `save_method="lora"` (adapter_config.json present)
- **Merged models**: `save_method="merged_16bit"` (standard .safetensors files)

### Error Handling Patterns
- Suppress known warnings with `os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"`
- Always import `unsloth` first for optimal performance
- Use `console.print()` from Rich for consistent CLI output

### Testing Approach
- Integration tests in root directory: `test_*.py`
- API tests use live server endpoints (`API_BASE = "http://localhost:8000"`)
- Vision tests use sample images from `data/test_images/`

## Frontend Integration (Svelte)
- Built with SvelteKit + TailwindCSS in `frontend/`
- Static files served by FastAPI at `/` (see `api.py`)
- Use `npm run build` to generate production build consumed by backend

## Common Pitfalls
- **Memory**: vLLM processes persist after crashes - always use `cleanup_gpu.sh`
- **Quantization**: Merged fine-tuned models look quantized but aren't - check file patterns, not just config
- **Environment**: HuggingFace cache paths must be set before any HF imports
- **Dependencies**: Always use `uv` - pip installations may miss optimizations