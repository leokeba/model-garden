# HuggingFace Hub Model Upload

This document describes how to upload trained models from Model Garden to HuggingFace Hub, enabling easy sharing and deployment of your fine-tuned models.

## Overview

Model Garden now supports uploading trained models directly to HuggingFace Hub from both the API and the web interface. This allows you to:

- **Share models** with the community or your team
- **Version control** your trained models
- **Deploy models** using HuggingFace's infrastructure
- **Document models** with automatic README generation

## Prerequisites

### 1. HuggingFace Account
Create a free account at [huggingface.co](https://huggingface.co)

### 2. HuggingFace Token
Generate an access token with write permissions:

1. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Select "Write" permissions
4. Copy the token

### 3. Configure Token in Model Garden

Add your token to the `.env` file:

```bash
HF_TOKEN=hf_your_token_here
```

Or export it as an environment variable:

```bash
export HF_TOKEN=hf_your_token_here
```

## Usage

### Web Interface

1. **Navigate to Models page**: Open the Model Garden UI and go to the Models section
2. **Select a model**: Find the model you want to upload
3. **Click "Upload to Hub"**: Click the 🤗 button on the model card
4. **Fill in the form**:
   - **Repository ID**: Format: `your-username/model-name` (e.g., `john/my-finetuned-llama`)
   - **Visibility**: Check "Make repository private" if you want a private repository
   - **Commit Message**: Describe this version (default: "Upload model from Model Garden")
   - **Description**: Describe your model's purpose and training details
5. **Upload**: Click "🚀 Upload to Hub"
6. **Wait**: The upload may take several minutes depending on model size
7. **Success**: You'll see a success message with a link to your HuggingFace repository

### API

Upload a model programmatically using the REST API:

```bash
curl -X POST "http://localhost:8000/api/v1/models/{model_id}/upload-to-hub" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_id": "your-username/model-name",
    "private": false,
    "commit_message": "Upload model from Model Garden",
    "repo_description": "My fine-tuned model"
  }'
```

**Python example:**

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/models/my-model-id/upload-to-hub",
    json={
        "repo_id": "your-username/my-finetuned-model",
        "private": False,
        "commit_message": "Upload fine-tuned Llama 3B",
        "repo_description": "Fine-tuned on custom dataset for task X"
    }
)

result = response.json()
print(f"Model uploaded to: {result['url']}")
```

**Response:**

```json
{
  "success": true,
  "message": "Model uploaded successfully to HuggingFace Hub",
  "repo_id": "your-username/my-finetuned-model",
  "url": "https://huggingface.co/your-username/my-finetuned-model",
  "commit_url": "https://huggingface.co/your-username/my-finetuned-model/commit/abc123..."
}
```

## What Gets Uploaded

When you upload a model, Model Garden uploads:

1. **Model files**: All `.safetensors`, `.bin`, and `.pth` files
2. **Configuration files**: 
   - `config.json` (model configuration)
   - `adapter_config.json` (for LoRA models)
   - `tokenizer_config.json`, `tokenizer.json`
   - `special_tokens_map.json`
3. **Tokenizer files**: `vocab.json`, `merges.txt`, etc.
4. **README.md**: Auto-generated model card with:
   - Base model information
   - Training configuration
   - Usage examples (transformers and Model Garden)
   - Carbon footprint data

## Model Card

The auto-generated README includes:

```markdown
---
license: apache-2.0
base_model: unsloth/Llama-3.2-3B-Instruct
tags:
  - model-garden
  - fine-tuned
  - text-generation
---

# My Fine-tuned Model

Model fine-tuned with Model Garden. Base model: unsloth/Llama-3.2-3B-Instruct

## Model Details

- **Base Model**: unsloth/Llama-3.2-3B-Instruct
- **Fine-tuned with**: Model Garden
- **Training Date**: 2024-01-15
- **Model Type**: text-generation

## Usage

### With Model Garden

```bash
# Serve the model
uv run model-garden serve-model --model-path your-username/my-finetuned-model

# Generate text
uv run model-garden inference-generate \
    --model-path your-username/my-finetuned-model \
    --prompt "Your prompt here"
```

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-username/my-finetuned-model")
tokenizer = AutoTokenizer.from_pretrained("your-username/my-finetuned-model")
```

## Training Details

This model was fine-tuned using Model Garden with:
- **Dataset**: custom
- **Training Steps**: 1000
- **LoRA Rank**: 16

## Carbon Footprint

Training emissions: 12.5 gCO2eq
```

## Using Uploaded Models

Once uploaded, you can use your models directly from HuggingFace Hub:

### In Model Garden

```bash
# Load from Hub
uv run model-garden serve-model --model-path your-username/my-finetuned-model

# Generate
uv run model-garden inference-generate \
    --model-path your-username/my-finetuned-model \
    --prompt "Hello, how are you?"
```

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load directly from Hub
model = AutoModelForCausalLM.from_pretrained("your-username/my-finetuned-model")
tokenizer = AutoTokenizer.from_pretrained("your-username/my-finetuned-model")

# Generate
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### With vLLM

```bash
# Serve with vLLM
vllm serve your-username/my-finetuned-model

# Query
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "your-username/my-finetuned-model",
        "prompt": "Hello, how are you?",
        "max_tokens": 100
    }'
```

## Repository Management

### Visibility

- **Public repositories**: Anyone can view and use your model
- **Private repositories**: Only you (and collaborators) can access the model
- You can change visibility later on HuggingFace Hub

### Versioning

Each upload creates a new commit in the repository. The commit history shows:
- Upload timestamp
- Commit message
- Changes to files

### Updating Models

To upload a new version:
1. Re-train your model with new data
2. Upload again to the same repository ID
3. The new version will replace the old one (previous versions remain in Git history)

## Troubleshooting

### Authentication Error

```
Failed to upload model: 401 Unauthorized
```

**Solution**: Ensure your `HF_TOKEN` is correctly set and has write permissions.

```bash
# Check token is set
echo $HF_TOKEN

# Re-export if needed
export HF_TOKEN=hf_your_token_here
```

### Repository Already Exists

If the repository exists and you're not the owner, you'll get a permission error.

**Solutions**:
- Use a different repository name
- Delete the existing repository on HuggingFace Hub
- Contact the repository owner for access

### Upload Timeout

Large models (>10GB) may timeout on slow connections.

**Solutions**:
- Use a faster internet connection
- Upload from a server with better bandwidth
- Consider uploading compressed versions

### Disk Space

Ensure you have enough disk space for the upload process:

```bash
# Check disk space
df -h

# Clean up old models if needed
uv run model-garden clean-cache
```

## Best Practices

1. **Descriptive Names**: Use clear, descriptive repository names
   - ✅ `llama-3b-medical-qa`
   - ❌ `model1`

2. **Meaningful Commits**: Write clear commit messages
   - ✅ "Fine-tuned on 10k medical Q&A pairs, improved accuracy by 15%"
   - ❌ "Upload"

3. **Documentation**: Include training details in the description
   - Dataset source and size
   - Training hyperparameters
   - Performance metrics
   - Intended use cases

4. **Licensing**: Ensure you comply with the base model's license

5. **Privacy**: Don't upload models trained on sensitive data to public repositories

6. **Testing**: Test models locally before uploading

## API Reference

### POST /api/v1/models/{model_id}/upload-to-hub

Upload a model to HuggingFace Hub.

**Path Parameters:**
- `model_id` (string): ID of the model to upload

**Request Body:**
```json
{
  "repo_id": "username/repo-name",
  "private": false,
  "commit_message": "Upload model from Model Garden",
  "repo_description": "Model description"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Model uploaded successfully to HuggingFace Hub",
  "repo_id": "username/repo-name",
  "url": "https://huggingface.co/username/repo-name",
  "commit_url": "https://huggingface.co/username/repo-name/commit/abc123"
}
```

**Errors:**
- `400`: Invalid repository ID format
- `401`: HF_TOKEN not configured or invalid
- `404`: Model not found
- `500`: Upload failed

## Examples

### Example 1: Upload LoRA Adapter

```bash
# Train a LoRA adapter
uv run model-garden train \
    --base-model unsloth/Llama-3.2-3B-Instruct \
    --dataset ./data/my-dataset.jsonl \
    --output-dir ./models/my-lora

# Upload to Hub
curl -X POST "http://localhost:8000/api/v1/models/my-lora/upload-to-hub" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_id": "myusername/llama-3b-medical-lora",
    "private": false,
    "commit_message": "LoRA adapter for medical Q&A",
    "repo_description": "Fine-tuned Llama 3B with LoRA for medical question answering"
  }'
```

### Example 2: Upload Merged Model

```bash
# Train and merge model
uv run model-garden train \
    --base-model unsloth/Llama-3.2-3B-Instruct \
    --dataset ./data/my-dataset.jsonl \
    --output-dir ./models/my-merged \
    --save-method merged_16bit

# Upload to Hub
curl -X POST "http://localhost:8000/api/v1/models/my-merged/upload-to-hub" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_id": "myusername/llama-3b-medical-merged",
    "private": true,
    "commit_message": "Merged model for internal use",
    "repo_description": "Production-ready merged model"
  }'
```

### Example 3: Vision Model Upload

```bash
# Train vision model
uv run model-garden train-vision \
    --base-model Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset ./data/vision-data.jsonl \
    --output-dir ./models/qwen-vision

# Upload to Hub
curl -X POST "http://localhost:8000/api/v1/models/qwen-vision/upload-to-hub" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_id": "myusername/qwen-vl-food-recognition",
    "private": false,
    "commit_message": "Vision model for food recognition",
    "repo_description": "Qwen2.5-VL fine-tuned on food images"
  }'
```

## Integration with CI/CD

Automate model uploads in your training pipeline:

```python
#!/usr/bin/env python3
"""Automated training and upload pipeline"""

import os
import requests
from datetime import datetime

def train_and_upload_model():
    """Train a model and upload to HuggingFace Hub"""
    
    # 1. Train model
    os.system("""
        uv run model-garden train \
            --base-model unsloth/Llama-3.2-3B-Instruct \
            --dataset ./data/latest.jsonl \
            --output-dir ./models/production
    """)
    
    # 2. Upload to Hub
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    response = requests.post(
        "http://localhost:8000/api/v1/models/production/upload-to-hub",
        json={
            "repo_id": f"myorg/llama-3b-production-{timestamp}",
            "private": True,
            "commit_message": f"Automated training run {timestamp}",
            "repo_description": "Production model - automated daily update"
        }
    )
    
    if response.ok:
        print(f"✅ Model uploaded: {response.json()['url']}")
    else:
        print(f"❌ Upload failed: {response.text}")
        raise Exception("Upload failed")

if __name__ == "__main__":
    train_and_upload_model()
```

## Related Documentation

- [LoRA Adapter Loading](./LORA_ADAPTER_LOADING.md) - Load adapters from Hub
- [Training Guide](./docs/training.md) - Fine-tuning models
- [Inference Guide](./docs/inference.md) - Serving models
- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/index)
