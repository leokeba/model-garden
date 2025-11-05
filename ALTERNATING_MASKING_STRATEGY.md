# Alternating Masking Strategy for Selective Loss

## Overview

The selective loss module now supports two masking strategies:

1. **Epoch-based**: Enable masking after a certain epoch threshold (original behavior)
2. **Alternating**: Cycle between masking ON/OFF throughout training (new feature)

## Why Alternating Masking?

The original epoch-based strategy had a limitation: once masking was enabled, the model would only learn semantic content and stop learning structure. This could lead to:

- Poor JSON formatting in later epochs
- Inability to recover if structure learning was insufficient in early epochs
- Less robust models when dealing with varying output formats

**Alternating masking solves this** by continuously alternating between:
- **Masking ON**: Focus on semantic content (mask structural tokens)
- **Masking OFF**: Learn JSON structure and formatting

This ensures the model learns **both structure and semantics throughout the entire training process**.

## Configuration

### Alternating Strategy

```python
from model_garden.selective_loss import create_selective_loss_collator

collator = create_selective_loss_collator(
    model=model,
    processor=processor,
    mask_level="aggressive",
    masking_strategy="alternating",
    mask_every_n_steps=100,  # Full cycle length
    mask_for_n_steps=50,     # Steps with masking ON
    verbose=True
)
```

**Parameters:**
- `masking_strategy`: Set to `"alternating"`
- `mask_every_n_steps`: Total cycle length in training steps (default: 100)
- `mask_for_n_steps`: Number of steps with masking ON per cycle (default: 50)

**Example Pattern** (with `mask_every_n_steps=100`, `mask_for_n_steps=50`):
- Steps 0-49: Masking **ON** (learn semantics)
- Steps 50-99: Masking **OFF** (learn structure)
- Steps 100-149: Masking **ON** (learn semantics)
- Steps 150-199: Masking **OFF** (learn structure)
- ... and so on

### Epoch-based Strategy (Original)

```python
collator = create_selective_loss_collator(
    model=model,
    processor=processor,
    mask_level="aggressive",
    masking_strategy="epoch_based",
    masking_start_epoch=0.5,  # Start masking halfway through first epoch
    verbose=True
)
```

**Parameters:**
- `masking_strategy`: Set to `"epoch_based"`
- `masking_start_epoch`: Epoch at which to enable masking (default: 0.0)

## Choosing a Strategy

### Use **Alternating** when:
- ✅ You want balanced learning of structure and semantics
- ✅ Training for multiple epochs (3+ epochs)
- ✅ Dataset has varying JSON schemas
- ✅ You need robust structure generation throughout training
- ✅ You want to avoid structure degradation in later epochs

### Use **Epoch-based** when:
- ✅ Doing initial experiments to understand masking impact
- ✅ Training for few epochs (1-2 epochs)
- ✅ Dataset has consistent, simple JSON structure
- ✅ You want to front-load structure learning

## Recommended Settings

### Conservative (recommended starting point)
```python
collator = create_selective_loss_collator(
    mask_level="conservative",      # Only mask {, }, [, ], :, ,, "
    masking_strategy="alternating",
    mask_every_n_steps=100,
    mask_for_n_steps=50,            # 50% masking, 50% structure learning
    verbose=True
)
```

### Aggressive (for complex schemas)
```python
collator = create_selective_loss_collator(
    mask_level="aggressive",        # Mask structure + schema keys
    masking_strategy="alternating",
    mask_every_n_steps=100,
    mask_for_n_steps=60,            # 60% masking, 40% structure learning
    dataset=train_dataset,          # Required for schema key auto-detection
    verbose=True
)
```

### More Structure Learning
If your model struggles with JSON formatting:
```python
collator = create_selective_loss_collator(
    mask_level="conservative",
    masking_strategy="alternating",
    mask_every_n_steps=100,
    mask_for_n_steps=40,            # 40% masking, 60% structure learning
    verbose=True
)
```

### More Semantic Learning
If your model has good structure but poor content:
```python
collator = create_selective_loss_collator(
    mask_level="aggressive",
    masking_strategy="alternating",
    mask_every_n_steps=100,
    mask_for_n_steps=70,            # 70% masking, 30% structure learning
    verbose=True
)
```

## Monitoring

With `verbose=True`, the collator will print:

**Epoch-based:**
```
Epoch 0.50/1.00: Learning structure (masking disabled)
✓ Epoch 1.00: Masking activated! (after 1.0 epochs of structure learning)
```

**Alternating:**
```
🔄 Step 0: Masking ON for next 50 steps
🔄 Step 50: Masking OFF for next 50 steps
🔄 Step 100: Masking ON for next 50 steps
```

## Technical Details

### How It Works

The alternating strategy uses modulo arithmetic to determine masking state:

```python
cycle_position = current_step % mask_every_n_steps
is_masking_on = cycle_position < mask_for_n_steps
```

### Implementation

- Masking state is checked on **every batch** during training
- Step counter increments only during training (not evaluation)
- No trainer state required (unlike epoch-based strategy)
- Zero overhead - just a simple modulo calculation

### Compatibility

- ✅ Works with all mask levels (none, conservative, moderate, aggressive)
- ✅ Compatible with `train_on_responses_only`
- ✅ Supports schema key auto-detection
- ✅ Maintains all Unsloth optimizations

## Migration Guide

If you're currently using epoch-based masking:

**Before:**
```python
collator = create_selective_loss_collator(
    model=model,
    processor=processor,
    mask_level="aggressive",
    masking_start_epoch=0.5,
    verbose=True
)
```

**After (alternating):**
```python
collator = create_selective_loss_collator(
    model=model,
    processor=processor,
    mask_level="aggressive",
    masking_strategy="alternating",  # Add this
    mask_every_n_steps=100,         # Add this
    mask_for_n_steps=50,            # Add this
    # Remove: masking_start_epoch
    verbose=True
)
```

**Note:** The default strategy remains `"epoch_based"` with `masking_start_epoch=0.0` for backward compatibility.

## Testing

Run the test suite to verify the implementation:

```bash
uv run python test_alternating_masking.py
```

This tests:
- ✅ Alternating pattern over 30 steps (3 full cycles)
- ✅ Correct ON/OFF transitions
- ✅ Epoch-based strategy still works

## Examples

### Example 1: Balanced Learning
```python
# 50/50 split between semantics and structure
collator = create_selective_loss_collator(
    mask_level="conservative",
    masking_strategy="alternating",
    mask_every_n_steps=100,
    mask_for_n_steps=50,
    verbose=True
)
```

### Example 2: Short Cycles
```python
# Alternate every 20 steps for more frequent switching
collator = create_selective_loss_collator(
    mask_level="moderate",
    masking_strategy="alternating",
    mask_every_n_steps=20,
    mask_for_n_steps=10,
    verbose=True
)
```

### Example 3: Long Cycles
```python
# Alternate every 200 steps for longer focus periods
collator = create_selective_loss_collator(
    mask_level="aggressive",
    masking_strategy="alternating",
    mask_every_n_steps=200,
    mask_for_n_steps=100,
    dataset=train_dataset,
    verbose=True
)
```

## Summary

- **Alternating masking** ensures continuous learning of both structure and semantics
- **Default recommendation**: `alternating` strategy with `mask_every_n_steps=100`, `mask_for_n_steps=50`
- **Adjust ratio** based on your model's strengths/weaknesses
- **Use verbose mode** to monitor transitions and verify behavior
- **Backward compatible** - existing code continues to work with epoch-based strategy
