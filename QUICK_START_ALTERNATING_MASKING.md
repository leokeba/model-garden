# Quick Start: Alternating Masking Strategy

## TL;DR

Train with balanced structure and semantic learning throughout training:

```bash
uv run model-garden train-vision \
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset ./data/vision.jsonl \
  --output-dir ./models/my-model \
  --selective-loss \
  --selective-loss-masking-strategy alternating \
  --selective-loss-mask-every-n-steps 100 \
  --selective-loss-mask-for-n-steps 50 \
  --selective-loss-verbose
```

## What It Does

**Alternating masking** cycles between:
- 🟢 **Masking ON**: Focus on learning semantic content (values, entities, etc.)
- 🔴 **Masking OFF**: Focus on learning JSON structure (braces, formatting, etc.)

This ensures your model learns **both** throughout the entire training process, preventing structure degradation.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--selective-loss-masking-strategy` | `epoch_based` | Choose `alternating` for cycle-based masking |
| `--selective-loss-mask-every-n-steps` | `100` | How often to complete a full cycle |
| `--selective-loss-mask-for-n-steps` | `50` | How many steps per cycle have masking ON |

## Common Patterns

### 50/50 Split (Recommended)
```bash
--selective-loss-mask-every-n-steps 100 \
--selective-loss-mask-for-n-steps 50
```
- Balanced learning
- Good for most use cases

### More Structure Learning
```bash
--selective-loss-mask-every-n-steps 100 \
--selective-loss-mask-for-n-steps 40
```
- 40% semantics, 60% structure
- Use if model struggles with JSON formatting

### More Semantic Learning
```bash
--selective-loss-mask-every-n-steps 100 \
--selective-loss-mask-for-n-steps 70
```
- 70% semantics, 30% structure
- Use if structure is simple but content is complex

### Faster Cycling
```bash
--selective-loss-mask-every-n-steps 50 \
--selective-loss-mask-for-n-steps 25
```
- Switch every 50 steps instead of 100
- More frequent alternation

## Frontend

1. Go to **Training → New Training Job**
2. Enable **Selective Loss Masking**
3. Choose **Alternating (Cycle ON/OFF)**
4. Adjust sliders:
   - **Cycle Length**: Total steps per cycle
   - **Masking ON**: Steps with masking enabled
5. See live preview of pattern

## When to Use

✅ **Use Alternating** when:
- Training for 3+ epochs
- Complex JSON schemas
- Want robust structure throughout training
- Avoiding structure degradation

✅ **Use Epoch-based** when:
- Quick experiments (1-2 epochs)
- Simple, consistent JSON structure
- Understanding masking impact first

## Monitoring

With `--selective-loss-verbose`, you'll see:

```
🔄 Step 0: Masking ON for next 50 steps
🔄 Step 50: Masking OFF for next 50 steps
🔄 Step 100: Masking ON for next 50 steps
```

This confirms the alternating pattern is working correctly.

## Full Example

```bash
# Complete training command with all recommended settings
uv run model-garden train-vision \
  --base-model Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset ./data/my_forms.jsonl \
  --validation-dataset ./data/my_forms_val.jsonl \
  --output-dir ./models/form-extractor \
  --epochs 3 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-5 \
  --selective-loss \
  --selective-loss-level aggressive \
  --selective-loss-masking-strategy alternating \
  --selective-loss-mask-every-n-steps 100 \
  --selective-loss-mask-for-n-steps 50 \
  --selective-loss-verbose \
  --quality-mode
```

That's it! Your model will learn both structure and semantics throughout training. 🎯
