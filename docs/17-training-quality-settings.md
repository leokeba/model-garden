# Training Quality Settings Guide

## Overview

Model Garden supports both **memory-optimized** (default) and **quality-optimized** training modes. This guide helps you choose the right settings for your use case.

## Quick Start

### Default Mode (Memory-Optimized)

Best for: Consumer GPUs (< 16GB VRAM), quick experiments

```bash
uv run model-garden train \
  --base-model unsloth/llama-3.1-8b-bnb-4bit \
  --dataset ./data/train.jsonl \
  --output-dir ./models/my-model \
  --epochs 3
```

**Memory**: ~6-8GB VRAM for 8B models  
**Quality**: Good for most use cases

---

### Quality Mode (Accuracy-Optimized)

Best for: High-end GPUs (24GB+ VRAM), production models, research

```bash
uv run model-garden train \
  --base-model unsloth/llama-3.1-8b \
  --dataset ./data/train.jsonl \
  --output-dir ./models/my-model \
  --quality-mode \
  --lora-r 64 \
  --epochs 3
```

**Memory**: ~24-32GB VRAM for 8B models  
**Quality**: Maximum accuracy

Quality mode automatically enables:
- 16-bit precision (`--load-in-16bit`)
- Standard gradient checkpointing (`--use-gradient-checkpointing true`)
- Better optimizer (`--optim adamw_torch`)
- RSLoRA for high ranks (if `--lora-r >= 32`)

---

## Detailed Settings

### 1. Precision (Quantization)

Controls model weight precision:

```bash
# Default: 4-bit (lowest memory)
uv run model-garden train --base-model unsloth/llama-3.1-8b-bnb-4bit ...

# 8-bit (balanced)
uv run model-garden train --base-model unsloth/llama-3.1-8b --load-in-8bit ...

# 16-bit (best quality)
uv run model-garden train --base-model unsloth/llama-3.1-8b --load-in-16bit ...
```

**Impact**:
- 4-bit: 1x memory, ~95% quality
- 8-bit: 2x memory, ~98% quality
- 16-bit: 4x memory, 100% quality

---

### 2. Gradient Checkpointing

Controls memory usage during backpropagation:

```bash
# Unsloth mode (default): 30% less VRAM
uv run model-garden train --use-gradient-checkpointing unsloth ...

# Standard mode: Better quality
uv run model-garden train --use-gradient-checkpointing true ...

# Disabled: Best quality, most memory
uv run model-garden train --use-gradient-checkpointing false ...
```

**Impact**:
- `unsloth`: 1x memory, minor quality loss
- `true`: 1.3x memory, better quality
- `false`: 1.8x memory, best quality

---

### 3. Optimizer

Controls parameter update algorithm:

```bash
# Default: 8-bit Adam (lowest memory)
uv run model-garden train --optim adamw_8bit ...

# Standard Adam (better quality)
uv run model-garden train --optim adamw_torch ...

# Fused Adam (best quality, fastest)
uv run model-garden train --optim adamw_torch_fused ...
```

**Impact**:
- `adamw_8bit`: 1x memory, minor quality loss
- `adamw_torch`: 2x optimizer memory, better convergence
- `adamw_torch_fused`: 2x optimizer memory, best speed/quality

---

### 4. LoRA Rank

Controls adapter capacity:

```bash
# Low rank (default, fast)
uv run model-garden train --lora-r 16 --lora-alpha 16 ...

# Medium rank (balanced)
uv run model-garden train --lora-r 32 --lora-alpha 32 --use-rslora ...

# High rank (best quality)
uv run model-garden train --lora-r 64 --lora-alpha 64 --use-rslora ...
```

**Impact**:
- `r=16`: Baseline memory/quality
- `r=32`: 2x adapter memory, better capacity
- `r=64`: 4x adapter memory, highest capacity

**Tip**: Always use `--use-rslora` when `--lora-r >= 32`

---

### 5. Learning Rate Scheduler

Controls how learning rate changes over training:

```bash
# Linear (default)
uv run model-garden train --lr-scheduler-type linear ...

# Cosine (better convergence)
uv run model-garden train --lr-scheduler-type cosine ...
```

---

## Recommended Configurations

### Budget GPU (8-12GB VRAM)

```bash
uv run model-garden train \
  --base-model unsloth/llama-3.1-8b-bnb-4bit \
  --dataset ./data/train.jsonl \
  --output-dir ./models/my-model \
  --lora-r 16 \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --epochs 3
```

---

### Mid-Range GPU (16-24GB VRAM)

```bash
uv run model-garden train \
  --base-model unsloth/llama-3.1-8b \
  --dataset ./data/train.jsonl \
  --output-dir ./models/my-model \
  --load-in-8bit \
  --lora-r 32 \
  --use-rslora \
  --optim adamw_torch \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --epochs 3
```

---

### High-End GPU (24GB+ VRAM)

```bash
uv run model-garden train \
  --base-model unsloth/llama-3.1-8b \
  --dataset ./data/train.jsonl \
  --output-dir ./models/my-model \
  --quality-mode \
  --lora-r 64 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --lr-scheduler-type cosine \
  --epochs 3
```

---

## Memory vs Quality Tradeoff Table

| Configuration | VRAM (8B model) | Training Time | Quality | Use Case |
|---------------|-----------------|---------------|---------|----------|
| Default | 6-8GB | Baseline | Good | Quick experiments |
| 8-bit + r32 | 12-16GB | 1.2x | Very Good | Production |
| Quality Mode | 24-32GB | 1.5x | Excellent | Research/Critical |

---

## Tips

1. **Start with defaults** - Test your workflow before optimizing for quality
2. **Monitor VRAM** - Use `nvidia-smi` to check memory usage
3. **Adjust batch size** - If OOM errors, reduce `--batch-size` or increase `--gradient-accumulation-steps`
4. **Quality mode for production** - Use `--quality-mode` for final model training
5. **Higher ranks for complex tasks** - Use `--lora-r 64` for code generation, reasoning, etc.

---

## Common Issues

### Out of Memory (OOM)

Try in order:
1. Reduce `--batch-size` to 1
2. Increase `--gradient-accumulation-steps` to 8
3. Use `--use-gradient-checkpointing unsloth` (default)
4. Reduce `--lora-r` to 8 or 16
5. Use default 4-bit quantization

### Slow Training

1. Increase `--batch-size` if memory allows
2. Use `--optim adamw_torch_fused`
3. Reduce `--gradient-accumulation-steps`
4. Use fewer epochs with higher learning rate

### Poor Model Quality

1. Enable `--quality-mode`
2. Increase `--lora-r` to 32-64
3. Use `--lr-scheduler-type cosine`
4. Train for more epochs
5. Improve dataset quality

---

## See Also

- [UNSLOTH_QUALITY_SETTINGS.md](../UNSLOTH_QUALITY_SETTINGS.md) - Detailed technical explanation
- [05-development-workflow.md](./05-development-workflow.md) - Development guide
- [QUICKSTART.md](../QUICKSTART.md) - Quick start guide
