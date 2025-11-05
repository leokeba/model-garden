# Eval Loss Bug Fix - Missing num_items_in_batch in prediction_step

## Problem Summary

Training and validation losses showed a 3-4x discrepancy from the very start of training, with training loss (~0.16→0.08) being significantly lower than validation loss (~0.44→0.35). This was NOT overfitting since the gap existed before any training occurred.

## Root Cause

The bug exists in `transformers/trainer.py` in the difference between `training_step` and `prediction_step`:

### Training Path (CORRECT):
```python
# Line ~4009 in trainer.py
def training_step(...):
    ...
    loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
```

The `num_items_in_batch` parameter is calculated as:
```python
# Line ~5603 in trainer.py
num_items_in_batch = sum([(batch["labels"].ne(-100)).sum()])
```

This counts the number of **non-masked tokens** (i.e., tokens that are NOT -100).

### Evaluation Path (BUGGY):
```python
# Lines ~4881-4882 in trainer.py
def prediction_step(...):
    ...
    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
    loss = loss.detach().mean()  # ❌ num_items_in_batch NOT passed!
```

**The bug:** `num_items_in_batch` is NOT passed to `compute_loss` during evaluation!

## Why This Causes Wrong Loss Values

When using **prompt masking** (train_on_responses_only=True), most tokens in the input are masked with label=-100. For example:

```
Total tokens: 1000
Masked tokens (prompts): 920
Valid tokens (responses): 80
```

### Loss Computation Logic

The loss function uses `num_items_in_batch` to determine reduction strategy:

```python
# transformers/loss/loss_utils.py, line ~30
def fixed_cross_entropy(..., num_items_in_batch=None):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = F.cross_entropy(..., reduction=reduction, ...)
    
    if num_items_in_batch is not None:
        loss = loss / num_items_in_batch  # Normalize by VALID tokens
    
    return loss
```

**Training (with num_items_in_batch):**
- Uses `reduction="sum"` 
- Divides by `num_items_in_batch` (e.g., 80 valid tokens)
- Loss is correctly normalized by **number of valid tokens**

**Evaluation (without num_items_in_batch):**
- Uses `reduction="mean"`
- Averages over ALL tokens including masked ones
- Loss denominator includes masked tokens (e.g., 1000 total tokens)
- Result: Loss is ~12.5x smaller than it should be (80/1000 = 0.08)

Wait, that would make eval loss SMALLER, but we observed it being LARGER. Let me reconsider...

Actually, the model's internal loss computation (before the fixed_cross_entropy) already ignores -100 tokens. The issue is more subtle:

**The real bug:** When `num_items_in_batch` is not passed, the loss uses `reduction="mean"` which averages over the batch dimension WITHOUT accounting for the different number of valid tokens per sample in the batch.

If batch samples have different masking ratios:
- Sample 1: 90% masked (10% valid tokens)
- Sample 2: 95% masked (5% valid tokens)

With `reduction="mean"`, both samples contribute equally to the average, even though sample 2 has fewer valid tokens. This causes inconsistent loss values compared to the training path which properly normalizes by the total valid token count across all samples.

## The Fix

Created `FixedSFTTrainer` that overrides `prediction_step` to pass `num_items_in_batch`:

```python
class FixedSFTTrainer(SFTTrainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Calculate num_items_in_batch from labels
        num_items_in_batch = None
        if "labels" in inputs:
            num_items_in_batch = (inputs["labels"] != -100).sum()
        
        # Pass num_items_in_batch to compute_loss
        if "labels" in inputs:
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(
                    model, inputs, 
                    return_outputs=True, 
                    num_items_in_batch=num_items_in_batch  # ✓ FIX: Now passed!
                )
            loss = loss.detach()  # Don't call .mean() - already normalized
        ...
```

Key changes:
1. Calculate `num_items_in_batch` exactly like training_step does
2. Pass it to `compute_loss` during evaluation
3. Remove the extra `.mean()` call since the loss is already normalized

## Files Modified

- `model_garden/vision_training.py`:
  - Added `FixedSFTTrainer` class (lines 54-120)
  - Changed trainer instantiation to use `FixedSFTTrainer` instead of `SFTTrainer` (line 1111)

## Expected Result

After this fix:
- Training loss and validation loss should use the same normalization
- Loss values should be comparable from the start
- The 3-4x discrepancy should disappear
- Both losses should show similar convergence patterns

## Testing

Run training with evaluation enabled:
```bash
uv run model-garden train-vision \
  --base-model unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit \
  --dataset /root/model-garden/data/vision_test_dataset.jsonl \
  --max-steps 10 \
  --eval-steps 5 \
  --logging-steps 1
```

Check that:
- Training loss ≈ Validation loss (within ~10-20% margin)
- Both losses decrease similarly over time
- No sudden jumps or discrepancies

## Related Files

- `debug_model_accepts_loss.py`: Verified `trainer.model_accepts_loss_kwargs=True` (not the bug)
- `check_trainer_bug.py`: Found the missing `num_items_in_batch` in prediction_step
- `test_trainer_override.py`: Verified FixedSFTTrainer properly overrides prediction_step

## References

- transformers/trainer.py:
  - Line 4009: training_step with num_items_in_batch
  - Line 4882: prediction_step without num_items_in_batch (THE BUG)
  - Line 5603: num_items_in_batch calculation
  - Line 4064: compute_loss method
  
- transformers/loss/loss_utils.py:
  - Line 28-44: fixed_cross_entropy with conditional reduction

## Impact

This bug affects ANY training that uses:
- Prompt masking (train_on_responses_only=True)
- Custom label masking with -100
- Evaluation during training

The bug causes **incorrect loss reporting during evaluation**, which can lead to:
- Misleading evaluation metrics
- Incorrect model selection (wrong eval loss comparison)
- Confusion about model performance
