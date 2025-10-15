# Quick Start - Fine-tuning

This guide will help you quickly get started with fine-tuning a small model.

## Installation

```bash
# Install dependencies with uv (recommended)
uv sync

# Or install manually with uv pip
uv pip install -e .

# Or with pip if you have a traditional venv
pip install -e .
```

## Quick Test

### 1. Create a sample dataset

```bash
uv run model-garden create-dataset --output ./data/sample.jsonl --num-examples 100
```

### 2. Fine-tune a small model

We'll use TinyLlama (1.1B parameters) which is small enough to train on most GPUs:

```bash
uv run model-garden train \
  --base-model unsloth/tinyllama-bnb-4bit \
  --dataset ./data/sample.jsonl \
  --output-dir ./models/my-first-model \
  --epochs 1 \
  --batch-size 2 \
  --max-steps 50
```

This will:
- Load TinyLlama in 4-bit quantization (~700MB VRAM)
- Apply LoRA adapters for efficient fine-tuning
- Train for 50 steps (should take 5-10 minutes on a modern GPU)
- Save the fine-tuned model

### 3. Generate text

```bash
uv run model-garden generate ./models/my-first-model \
  --prompt "Explain machine learning in simple terms" \
  --max-tokens 128
```

## Using Real Datasets

### From Local File

Your dataset should be in JSONL format with these fields:
- `instruction`: The task description
- `input`: Optional context or input
- `output`: The expected response

Example `dataset.jsonl`:
```json
{"instruction": "Write a poem about AI", "input": "", "output": "Silicon minds awakening..."}
{"instruction": "Explain recursion", "input": "", "output": "A function that calls itself..."}
```

Then train:
```bash
uv run model-garden train \
  --base-model unsloth/tinyllama-bnb-4bit \
  --dataset ./data/dataset.jsonl \
  --output-dir ./models/my-model \
  --epochs 3
```

### From HuggingFace Hub

```bash
uv run model-garden train \
  --base-model unsloth/tinyllama-bnb-4bit \
  --dataset yahma/alpaca-cleaned \
  --output-dir ./models/alpaca-model \
  --from-hub \
  --epochs 3
```

## Training Tips

### For Small GPUs (8-12GB)
```bash
uv run model-garden train \
  --base-model unsloth/tinyllama-bnb-4bit \
  --dataset ./data/dataset.jsonl \
  --output-dir ./models/my-model \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-seq-length 512
```

### For Larger GPUs (16GB+)
```bash
uv run model-garden train \
  --base-model unsloth/phi-2-bnb-4bit \
  --dataset ./data/dataset.jsonl \
  --output-dir ./models/my-model \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --max-seq-length 2048
```

### For Fast Testing
```bash
uv run model-garden train \
  --base-model unsloth/tinyllama-bnb-4bit \
  --dataset ./data/dataset.jsonl \
  --output-dir ./models/test-model \
  --max-steps 50 \
  --save-steps 25
```

## Available Models

Unsloth provides optimized 4-bit versions of popular models:

- `unsloth/tinyllama-bnb-4bit` (1.1B) - Great for testing
- `unsloth/phi-2-bnb-4bit` (2.7B) - Excellent quality/size ratio
- `unsloth/mistral-7b-bnb-4bit` (7B) - High quality
- `unsloth/llama-2-7b-bnb-4bit` (7B) - Popular choice
- `unsloth/llama-3-8b-bnb-4bit` (8B) - Latest Llama

## CLI Options

```bash
uv run model-garden train --help
```

Key options:
- `--base-model`: Model to fine-tune
- `--dataset`: Dataset path or HuggingFace ID
- `--output-dir`: Where to save the model
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size (lower for less VRAM)
- `--learning-rate`: Learning rate (default: 2e-4)
- `--max-seq-length`: Max sequence length (lower for less VRAM)
- `--lora-r`: LoRA rank (higher = more parameters)
- `--save-method`: How to save (`lora`, `merged_16bit`, `merged_4bit`)

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` to 1
- Reduce `--max-seq-length` to 512 or 1024
- Increase `--gradient-accumulation-steps`

### Slow Training
- Increase `--batch-size` if you have VRAM
- Use a smaller model for testing

### Model Not Loading
- Make sure you have CUDA installed
- Try a different base model
- Check your GPU VRAM (needs at least 6GB)

## Next Steps

- Try different base models
- Experiment with hyperparameters
- Use your own datasets
- Share your fine-tuned models on HuggingFace Hub
