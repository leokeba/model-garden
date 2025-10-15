# Installation & Testing Instructions

## Prerequisites

Before installing, make sure you have:

1. **Python 3.11 or higher**
   ```bash
   python --version
   ```

2. **CUDA-capable GPU** (NVIDIA)
   - Minimum 6GB VRAM (for TinyLlama)
   - 12GB+ recommended for larger models
   - Check with: `nvidia-smi`

3. **CUDA Toolkit** (11.8 or 12.1)
   - Required for PyTorch with CUDA support
   - Download from: https://developer.nvidia.com/cuda-downloads

4. **uv package manager** (optional but recommended)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

## Installation Steps

### Option 1: Using uv (Recommended)

```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Model Garden
uv pip install -e .
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install Model Garden
pip install -e .
```

**Note**: Initial installation will take 5-10 minutes as it downloads PyTorch, Transformers, and other large dependencies.

## Verify Installation

```bash
# Check if CLI is available
model-garden --version

# Should output: model-garden, version 0.1.0
```

## Quick Test

### 1. Create Test Dataset

```bash
model-garden create-dataset --output ./data/test.jsonl --num-examples 50
```

This creates a simple dataset with 50 examples for testing.

### 2. Run a Quick Training Test

```bash
model-garden train \
  --base-model unsloth/tinyllama-bnb-4bit \
  --dataset ./data/test.jsonl \
  --output-dir ./models/test-model \
  --epochs 1 \
  --batch-size 2 \
  --max-steps 10 \
  --logging-steps 2
```

**What to expect:**
- First run will download TinyLlama model (~700MB)
- Training should take 2-5 minutes for 10 steps
- You'll see progress logs every 2 steps
- Final model will be saved to `./models/test-model`

### 3. Test Generation

```bash
model-garden generate ./models/test-model \
  --prompt "Explain what artificial intelligence is" \
  --max-tokens 100
```

## Full Training Example

Once the quick test works, try a real training run:

```bash
model-garden train \
  --base-model unsloth/tinyllama-bnb-4bit \
  --dataset ./data/example_dataset.jsonl \
  --output-dir ./models/ml-explainer \
  --epochs 3 \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-4 \
  --max-seq-length 512 \
  --logging-steps 5
```

This will:
- Train on the example ML Q&A dataset
- Take 10-20 minutes depending on your GPU
- Create a model that can explain ML concepts

Then test it:

```bash
model-garden generate ./models/ml-explainer \
  --prompt "What is deep learning?" \
  --max-tokens 150
```

## Troubleshooting

### CUDA Out of Memory

If you get CUDA OOM errors:

```bash
model-garden train \
  --base-model unsloth/tinyllama-bnb-4bit \
  --dataset ./data/test.jsonl \
  --output-dir ./models/test-model \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-seq-length 512 \
  --max-steps 10
```

### No CUDA Device Found

Make sure:
1. You have an NVIDIA GPU
2. CUDA is installed: `nvcc --version`
3. PyTorch can see CUDA:
   ```python
   python -c "import torch; print(torch.cuda.is_available())"
   ```

If false, you may need to reinstall PyTorch with CUDA support:
```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
```

### Slow Download Speeds

First model download can be slow. The model is cached in `~/.cache/huggingface/hub/`, so subsequent runs will be fast.

### Import Errors

If you see import errors, try:
```bash
pip install --upgrade transformers datasets accelerate peft trl
```

## What's Next?

Once everything works:

1. **Use Real Data**: Replace the sample dataset with your own
2. **Try Larger Models**: Use `unsloth/phi-2-bnb-4bit` or `unsloth/mistral-7b-bnb-4bit`
3. **Tune Hyperparameters**: Adjust learning rate, batch size, etc.
4. **Share Models**: Push to HuggingFace Hub
5. **Add More Features**: Help implement API server, web UI, carbon tracking

## Performance Benchmarks

Expected training speed on different GPUs (TinyLlama, batch_size=2):

- **RTX 4090** (24GB): ~150 tokens/sec
- **RTX 3090** (24GB): ~120 tokens/sec
- **RTX 3080** (10GB): ~80 tokens/sec
- **RTX 3060** (12GB): ~60 tokens/sec

## Need Help?

- Check the [Quick Start Guide](./QUICKSTART.md)
- Read the [full documentation](./docs/)
- Open an issue on GitHub
- Check HuggingFace Transformers docs

## Success Criteria

✅ Installation complete if:
- `model-garden --version` works
- Can create sample dataset
- Can run training for 10 steps
- Can generate text from trained model

Happy fine-tuning! 🌱
