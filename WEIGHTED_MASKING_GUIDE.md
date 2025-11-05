# Weighted Masking Strategy Guide

## Overview

Weighted masking is a new **soft masking** approach for selective loss training, complementing the existing binary masking strategies (epoch-based and alternating).

Instead of completely masking structural tokens (setting loss to 0), weighted masking applies **reduced loss weights** to structural tokens, allowing the model to still receive gradient signals from them while prioritizing semantic content.

## Comparison of Strategies

| Strategy | Type | Behavior | Best For |
|----------|------|----------|----------|
| **Epoch-based** | Binary (hard) | Masking OFF → ON after epoch threshold | Clear learning phases, simple baseline |
| **Alternating** | Binary (hard) | Cycles ON/OFF every N steps | Learning both structure & semantics |
| **Weighted** | Continuous (soft) | Always active with reduced weights | Experimental, requires custom trainer |

## How Weighted Masking Works

### Standard Binary Masking (epoch-based/alternating)
```python
# Structural tokens: loss = 0 (completely ignored)
# Semantic tokens: loss = full weight
labels[structural_indices] = -100  # PyTorch ignores these
```

### Weighted Masking
```python
# Structural tokens: loss = reduced weight (e.g., 0.1x)
# Semantic tokens: loss = full weight (1.0x)
weights[structural_indices] = 0.1  # Still contributes to loss
```

## Implementation

### 1. Create Weighted Collator

```python
from model_garden.selective_loss import create_selective_loss_collator

collator = create_selective_loss_collator(
    model=model,
    processor=processor,
    mask_level="aggressive",              # or "conservative", "moderate"
    masking_strategy="weighted",          # KEY: Use weighted strategy
    structural_weight=0.1,                # Structural tokens get 10% weight
    dataset=train_dataset,                # For schema key auto-detection
    verbose=True
)
```

### 2. Implement Custom Trainer

Weighted masking requires a custom `compute_loss` method to handle per-token weights:

```python
from transformers import Trainer
import torch
import torch.nn.functional as F

class WeightedLossTrainer(Trainer):
    """Custom trainer that supports per-token loss weighting."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract labels and weights
        labels = inputs.pop("labels")
        sample_weights = inputs.pop("sample_weights", None)
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute per-token loss (no reduction)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Apply per-token weights if provided
        if sample_weights is not None:
            weights_flat = sample_weights.view(-1)
            loss = loss * weights_flat
            
            # Weighted average: sum(loss * weight) / sum(weight)
            valid_mask = (labels != -100).view(-1)
            loss = loss[valid_mask].sum() / weights_flat[valid_mask].sum()
        else:
            # Standard average over valid tokens
            valid_mask = (labels != -100).view(-1)
            loss = loss[valid_mask].mean()
        
        return (loss, outputs) if return_outputs else loss
```

### 3. Train with Weighted Masking

```python
from model_garden.vision_training import load_model_and_processor, prepare_dataset
from transformers import TrainingArguments

# Load model
model, processor = load_model_and_processor(
    base_model="Qwen/Qwen2.5-VL-3B-Instruct",
    load_in_4bit=True
)

# Prepare dataset
train_dataset = prepare_dataset(
    dataset_path="./data/train.jsonl",
    processor=processor
)

# Create weighted collator
collator = create_selective_loss_collator(
    model=model,
    processor=processor,
    mask_level="aggressive",
    masking_strategy="weighted",
    structural_weight=0.1,
    dataset=train_dataset,
    verbose=True
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
)

# Use custom trainer
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,
)

# Train!
trainer.train()
```

## Parameter Tuning

### `structural_weight` (float, 0.0 - 1.0)

Controls how much structural tokens contribute to the loss:

- **0.0**: Equivalent to binary masking (structural tokens completely ignored)
- **0.05**: Very low weight (5%) - minimal structural signal
- **0.1**: Low weight (10%) - **recommended starting point**
- **0.2**: Moderate weight (20%) - balanced approach
- **0.5**: High weight (50%) - structural tokens matter significantly
- **1.0**: No masking (all tokens weighted equally)

### Recommended Workflow

1. **Start with 0.1** - This gives a small but non-zero signal from structural tokens
2. **Compare to alternating** - Train identical models with both strategies
3. **Experiment** - Try 0.05, 0.2, 0.5 based on results
4. **Monitor loss** - Ensure weighted loss doesn't diverge

## Advantages of Weighted Masking

1. **Soft Constraints**: Model still learns some structural patterns
2. **Continuous Signal**: No abrupt on/off switching
3. **Flexible**: Fine-tune weight to your task
4. **Stable Training**: Gradients flow through all tokens (just weighted)

## Disadvantages

1. **Requires Custom Trainer**: Must override `compute_loss`
2. **More Complex**: Additional hyperparameter to tune
3. **Experimental**: Less battle-tested than binary strategies
4. **May Not Work with All Frameworks**: Custom loss handling needed

## Debugging

### Check if Weights are Being Applied

```python
# In your training script, inspect a batch
batch = next(iter(train_dataloader))
print("Labels shape:", batch["labels"].shape)
print("Has sample_weights:", "sample_weights" in batch)

if "sample_weights" in batch:
    weights = batch["sample_weights"]
    print("Weights shape:", weights.shape)
    print("Unique weights:", torch.unique(weights).tolist())
    print(f"Tokens with 0.1 weight: {(weights == 0.1).sum().item()}")
    print(f"Tokens with 1.0 weight: {(weights == 1.0).sum().item()}")
```

### Verify Custom Loss is Working

```python
# Add debug prints to compute_loss
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    labels = inputs.pop("labels")
    sample_weights = inputs.pop("sample_weights", None)
    
    print(f"Batch has sample_weights: {sample_weights is not None}")
    if sample_weights is not None:
        print(f"Weight range: {sample_weights.min().item():.2f} - {sample_weights.max().item():.2f}")
    
    # ... rest of compute_loss
```

## Example Use Cases

### When to Use Weighted Masking

- **Structured output tasks** where structure matters but should be de-emphasized
- **Form extraction** where JSON brackets are predictable but not zero-information
- **Experimentation** to compare soft vs hard constraints

### When to Use Binary Masking Instead

- **Simple baseline** - Start with epoch-based or alternating
- **Production systems** - More mature, well-tested strategies
- **Clear separation** - When structure is truly trivial (just brackets/commas)

## Testing

Run the test suite to verify weighted masking is working:

```bash
uv run python test_weighted_masking.py
```

This will:
1. Test collator initialization for all three strategies
2. Verify different weight values (0.0, 0.05, 0.1, 0.2, 0.5)
3. Show how to implement a custom trainer
4. Compare all three strategies

## Future Improvements

Potential enhancements for weighted masking:

1. **Dynamic Weights**: Adjust weights during training (e.g., start at 0.5, decay to 0.1)
2. **Token-Specific Weights**: Different weights for different structural token types
3. **Learned Weights**: Let model learn optimal weights via meta-learning
4. **Gradient Scaling**: Scale gradients instead of loss weights

## References

- PyTorch `CrossEntropyLoss` with `reduction='none'`: [Docs](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- Weighted loss in language modeling: [HuggingFace Discussion](https://discuss.huggingface.co/t/custom-loss-weights-for-language-modeling/)
- Soft vs hard constraints in deep learning: Various papers on label smoothing and focal loss

## Support

For questions or issues with weighted masking:
1. Check the test file: `test_weighted_masking.py`
2. Review the implementation: `model_garden/selective_loss.py`
3. Compare with alternating strategy for baseline
