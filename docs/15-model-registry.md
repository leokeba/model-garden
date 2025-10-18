# Model Registry - Centralized Model Management

## Overview

The Model Registry is a centralized system for managing all supported models in Model Garden. It provides a single source of truth for model configurations, requirements, capabilities, and default settings.

## Purpose

**Problems Solved:**
- ✅ Eliminates scattered model lists across frontend and backend
- ✅ Provides consistent default configurations
- ✅ Enables validation before training/inference
- ✅ Simplifies adding new models
- ✅ Documents model requirements and capabilities

## Architecture

### Components

```
storage/
├── models_registry.json        # Model metadata and configurations
└── models_registry.schema.json # JSON schema for validation

model_garden/
└── model_registry.py           # Python API for registry access

API Endpoints:
├── GET  /api/v1/registry/models          # List all supported models
├── GET  /api/v1/registry/models/{id}     # Get model details
├── GET  /api/v1/registry/categories      # Get categories
├── POST /api/v1/registry/validate/training   # Validate for training
└── POST /api/v1/registry/validate/inference  # Validate for inference
```

## Registry Schema

### Model Entry Structure

```json
{
  "model_id": {
    "id": "unsloth/tinyllama-bnb-4bit",
    "name": "TinyLlama 1.1B (4-bit)",
    "category": "text-llm",
    "provider": "unsloth",
    "base_architecture": "llama",
    "parameters": "1.1B",
    "description": "Compact model perfect for testing",
    "tags": ["small", "fast", "testing", "quantized"],
    "status": "stable",
    "quantization": {
      "method": "4bit",
      "type": "bitsandbytes"
    },
    "requirements": {
      "min_vram_gb": 4,
      "recommended_vram_gb": 6,
      "min_ram_gb": 8,
      "cuda_compute_capability": "7.0"
    },
    "capabilities": {
      "training": true,
      "inference": true,
      "vision": false,
      "structured_outputs": true,
      "streaming": true,
      "function_calling": false
    },
    "training_defaults": {
      "hyperparameters": {...},
      "lora_config": {...},
      "selective_loss": {...}
    },
    "inference_defaults": {
      "max_model_len": 2048,
      "dtype": "auto",
      "gpu_memory_utilization": 0.85,
      "quantization": null
    },
    "urls": {
      "huggingface": "https://...",
      "docs": "https://..."
    }
  }
}
```

### Field Descriptions

#### Core Fields

- **`id`**: Unique identifier (usually HuggingFace model ID)
- **`name`**: Human-readable display name
- **`category`**: Category key (`text-llm` or `vision-vlm`)
- **`provider`**: Model creator (`unsloth`, `qwen`, `meta`, etc.)
- **`base_architecture`**: Base model type (`llama`, `mistral`, `qwen-vl`, etc.)
- **`parameters`**: Model size (e.g., `"7B"`, `"1.1B"`)
- **`description`**: Brief description for UI display
- **`tags`**: Searchable tags (e.g., `["quantized", "recommended"]`)
- **`status`**: Support level (`stable`, `experimental`, `deprecated`)

#### Quantization

```json
"quantization": {
  "method": "4bit",        // null, "4bit", "8bit", "awq", "gptq", "fp8"
  "type": "bitsandbytes"   // null, "bitsandbytes", "awq", "gptq"
}
```

#### Requirements

```json
"requirements": {
  "min_vram_gb": 4,               // Minimum GPU VRAM
  "recommended_vram_gb": 6,        // Recommended VRAM
  "min_ram_gb": 8,                 // System RAM
  "cuda_compute_capability": "7.0", // Min CUDA version
  "min_gpus": 1                    // GPUs required (optional)
}
```

#### Capabilities

```json
"capabilities": {
  "training": true,           // Can be fine-tuned
  "inference": true,          // Can be served via vLLM
  "vision": false,            // Multimodal vision support
  "structured_outputs": true, // JSON schema generation
  "streaming": true,          // Streaming responses
  "function_calling": false   // Tool/function calling
}
```

#### Training Defaults

```json
"training_defaults": {
  "hyperparameters": {
    "learning_rate": 0.0002,
    "num_epochs": 3,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 10,
    "max_seq_length": 2048,
    "optim": "adamw_8bit",
    "lr_scheduler_type": "linear"
  },
  "lora_config": {
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "use_rslora": false
  },
  "selective_loss": {
    "supported": true,          // Only for vision models
    "default_level": "conservative",
    "recommended": true
  }
}
```

#### Inference Defaults

```json
"inference_defaults": {
  "max_model_len": 2048,
  "dtype": "auto",
  "gpu_memory_utilization": 0.85,
  "quantization": null,
  "tensor_parallel_size": 1
}
```

## Python API

### Basic Usage

```python
from model_garden.model_registry import get_registry, get_model

# Get registry instance
registry = get_registry()

# Get all models
all_models = registry.get_all_models()

# Get specific model
model = get_model("unsloth/tinyllama-bnb-4bit")

print(f"Model: {model.name}")
print(f"Min VRAM: {model.requirements.min_vram_gb} GB")
print(f"Supports vision: {model.is_vision_model}")
```

### Filter by Category

```python
# Get text-only models
text_models = registry.get_text_models()

# Get vision-language models
vision_models = registry.get_vision_models()

# Get by category string
llm_models = registry.get_models_by_category("text-llm")
```

### Filter by Tags

```python
# Get all recommended models
recommended = registry.get_recommended_models()

# Get quantized models
quantized = registry.get_models_by_tag("quantized")

# Get stable (production-ready) models
stable = registry.get_stable_models()
```

### Get Default Configurations

```python
model = get_model("Qwen/Qwen2.5-VL-7B-Instruct")

# Get training hyperparameters
hyperparams = model.get_training_hyperparameters()
print(f"Learning rate: {hyperparams['learning_rate']}")
print(f"Batch size: {hyperparams['batch_size']}")

# Get LoRA configuration
lora_config = model.get_lora_config()
print(f"LoRA rank: {lora_config['r']}")

# Get inference settings
inference_config = model.get_inference_config()
print(f"Max model len: {inference_config['max_model_len']}")
```

### Validation

```python
from model_garden.model_registry import validate_model_for_training

# Validate before starting training
is_valid, error = validate_model_for_training("unsloth/tinyllama-bnb-4bit")

if not is_valid:
    print(f"Cannot train: {error}")
else:
    # Proceed with training
    pass
```

### Check Capabilities

```python
model = get_model("Qwen/Qwen2.5-VL-3B-Instruct")

# Check specific capabilities
if model.is_vision_model:
    print("This is a vision-language model")

if model.supports_selective_loss:
    print("Supports selective loss for structured outputs")

if model.capabilities.streaming:
    print("Supports streaming inference")
```

## REST API

### Get All Models

```bash
# Get all models
curl http://localhost:8000/api/v1/registry/models

# Filter by category
curl http://localhost:8000/api/v1/registry/models?category=text-llm
curl http://localhost:8000/api/v1/registry/models?category=vision-vlm
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "unsloth/tinyllama-bnb-4bit",
      "name": "TinyLlama 1.1B (4-bit)",
      "category": "text-llm",
      "parameters": "1.1B",
      "description": "Compact model perfect for testing",
      "tags": ["small", "fast", "testing", "quantized"],
      "status": "stable",
      "is_vision": false,
      "is_quantized": true,
      "min_vram_gb": 4,
      "recommended_vram_gb": 6
    }
  ],
  "total": 1
}
```

### Get Model Details

```bash
curl http://localhost:8000/api/v1/registry/models/unsloth/tinyllama-bnb-4bit
```

**Response includes:**
- Complete model information
- Training defaults (hyperparameters + LoRA config)
- Inference defaults
- Requirements and capabilities
- URLs to documentation

### Validate for Training

```bash
curl -X POST http://localhost:8000/api/v1/registry/validate/training \
  -H "Content-Type: application/json" \
  -d '{"model_id": "unsloth/tinyllama-bnb-4bit"}'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "is_valid": true,
    "error_message": null
  }
}
```

### Get Categories

```bash
curl http://localhost:8000/api/v1/registry/categories
```

**Response:**
```json
{
  "success": true,
  "data": {
    "text-llm": {
      "name": "Text-Only Language Models",
      "description": "Models for text generation and completion tasks",
      "icon": "✍️"
    },
    "vision-vlm": {
      "name": "Vision-Language Models",
      "description": "Multimodal models for image + text understanding",
      "icon": "🎨"
    }
  }
}
```

## Frontend Integration

### Fetch Models for UI

```typescript
// In your Svelte/React component
async function loadModels() {
  const response = await fetch('/api/v1/registry/models?category=text-llm');
  const { data } = await response.json();
  
  return data.map(model => ({
    value: model.id,
    label: `${model.name} - ${model.description}`,
    vram: model.recommended_vram_gb,
    tags: model.tags
  }));
}
```

### Populate Form Defaults

```typescript
async function getModelDefaults(modelId: string) {
  const response = await fetch(`/api/v1/registry/models/${modelId}`);
  const { data } = await response.json();
  
  // Use training defaults to populate form
  return {
    hyperparameters: data.training_defaults.hyperparameters,
    loraConfig: data.training_defaults.lora_config,
    // Adjust based on vision vs text
    isVision: data.capabilities.vision
  };
}
```

## Adding New Models

### Step 1: Add to Registry

Edit `storage/models_registry.json`:

```json
{
  "models": {
    "your-org/your-model": {
      "id": "your-org/your-model",
      "name": "Your Model Name",
      "category": "text-llm",
      "provider": "your-org",
      "base_architecture": "llama",
      "parameters": "7B",
      "description": "Your model description",
      "tags": ["custom", "experimental"],
      "status": "experimental",
      // ... fill in all required fields
    }
  }
}
```

### Step 2: Validate Schema

```bash
# Install JSON schema validator
pip install jsonschema

# Validate registry
python -c "
import json
import jsonschema

with open('storage/models_registry.schema.json') as f:
    schema = json.load(f)
    
with open('storage/models_registry.json') as f:
    data = json.load(f)
    
jsonschema.validate(data, schema)
print('✓ Registry is valid')
"
```

### Step 3: Test in Python

```python
from model_garden.model_registry import get_model

model = get_model("your-org/your-model")
assert model is not None
print(f"Successfully added: {model.name}")
```

### Step 4: Update Frontend

The model will automatically appear in the UI! No frontend changes needed.

## Migration Guide

### From Hardcoded Lists

**Before (in Svelte component):**
```typescript
const textModels = [
  "unsloth/tinyllama-bnb-4bit",
  "unsloth/phi-2-bnb-4bit",
  // ... hardcoded list
];
```

**After (use registry API):**
```typescript
const { data: textModels } = await fetch('/api/v1/registry/models?category=text-llm')
  .then(r => r.json());
```

### From Hardcoded Defaults

**Before:**
```python
# Hardcoded in training function
hyperparameters = {
    "learning_rate": 0.0002,  # Different per model!
    "batch_size": 2,
    # ...
}
```

**After:**
```python
from model_garden.model_registry import get_model

model = get_model(base_model)
hyperparameters = model.get_training_hyperparameters()
```

## Best Practices

### 1. Use Registry for Validation

```python
# Always validate before training
is_valid, error = validate_model_for_training(model_id)
if not is_valid:
    raise ValueError(f"Invalid model: {error}")
```

### 2. Respect Default Configurations

```python
# Get defaults from registry
model = get_model(model_id)
defaults = model.get_training_hyperparameters()

# Merge with user overrides
final_config = {**defaults, **user_overrides}
```

### 3. Use Tags for Filtering

```json
"tags": ["quantized", "recommended", "vision", "small"]
```

Tags should be:
- **Descriptive**: "small", "large", "vision", "quantized"
- **Functional**: "fast", "high-quality", "experimental"
- **Use-case**: "testing", "production", "research"

### 4. Document Model Status

- **`stable`**: Production-ready, well-tested
- **`experimental`**: New, may have issues
- **`deprecated`**: Replaced by newer models

### 5. Keep Requirements Accurate

Test models and update VRAM requirements:
```json
"requirements": {
  "min_vram_gb": 4,      // Actual minimum
  "recommended_vram_gb": 6  // Comfortable operation
}
```

## Troubleshooting

### Registry Not Found

```python
FileNotFoundError: Registry not found: storage/models_registry.json
```

**Solution:** Ensure the registry file exists at the correct path.

### Schema Validation Errors

Install validator:
```bash
npm install -g ajv-cli
ajv validate -s storage/models_registry.schema.json -d storage/models_registry.json
```

### Model Not Appearing in UI

1. Check model is in `storage/models_registry.json`
2. Verify `status` is not `deprecated`
3. Check API response: `curl http://localhost:8000/api/v1/registry/models`
4. Clear browser cache

## Future Enhancements

### Planned Features

1. **Auto-detection**: Scan HuggingFace for new models
2. **Versioning**: Track registry schema versions
3. **User models**: Allow users to register custom models
4. **Benchmarks**: Include performance metrics
5. **Cost tracking**: Estimate training/inference costs
6. **Compatibility matrix**: Show which features work together

### Community Contributions

To add a new model to the registry:

1. Fork the repository
2. Add model to `storage/models_registry.json`
3. Validate schema
4. Test locally
5. Submit pull request with:
   - Model configuration
   - VRAM requirements (tested)
   - Sample training results

## Summary

The Model Registry provides:

- ✅ **Single source of truth** for all supported models
- ✅ **Centralized configuration** management
- ✅ **Validation** before training/inference
- ✅ **Easy model addition** without code changes
- ✅ **Auto-populated UI** forms
- ✅ **Consistent defaults** across the platform

**Key Files:**
- `storage/models_registry.json` - Model metadata
- `storage/models_registry.schema.json` - Validation schema
- `model_garden/model_registry.py` - Python API
- `model_garden/api.py` - REST endpoints

**Key Endpoints:**
- `GET /api/v1/registry/models` - List models
- `GET /api/v1/registry/models/{id}` - Model details
- `POST /api/v1/registry/validate/training` - Validate training
- `POST /api/v1/registry/validate/inference` - Validate inference
