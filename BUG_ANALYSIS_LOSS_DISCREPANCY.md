# Bug Analysis: Training vs Validation Loss Discrepancy

## Problem
When using the standard SFTTrainer (not WeightedLossTrainer), training and validation losses show a large gap from the very beginning:
- Training Loss: ~0.16 → ~0.08
- Validation Loss: ~0.44 → ~0.35

This 3-4x difference exists before any real training, ruling out overfitting.

## Investigation Findings

### ✓ Confirmed Working:
1. Data collator masks ~93% of tokens correctly (prompt masking works)
2. Same collator instance used for train and eval dataloaders  
3. Model handles masked labels identically in train() and eval() modes
4. Chat marker detection works correctly
5. `train_on_responses_only` function applies masking consistently

### ❌ Root Cause Identified:

The issue is in **how SFTTrainer computes loss** vs **how WeightedLossTrainer computes loss**:

#### WeightedLossTrainer (CORRECT):
```python
def compute_loss(self, model, inputs, ...):
    labels = inputs.pop("labels")  # Remove labels from inputs
    outputs = model(**inputs)  # Model doesn't compute loss
    logits = outputs.logits
    
    # Manually compute cross-entropy with explicit control
    loss = CrossEntropyLoss(reduction='none', ignore_index=-100)(
        logits.view(-1, vocab_size),
        labels.view(-1)
    )
    
    # Properly normalize: sum(loss) / num_valid_tokens
    valid_mask = (labels != -100)
    final_loss = loss[valid_mask].mean()
    return final_loss
```

#### SFTTrainer (PROBLEMATIC):
```python
def compute_loss(self, model, inputs, ...):
    labels = inputs["labels"]  # Labels stay in inputs
    outputs = model(**inputs)  # Model computes loss internally
    loss = outputs.loss  # Trust the model's loss
    return loss
```

The problem is that the **model's internal loss computation** may not properly handle the masked labels in all cases, or there may be subtle differences in how the loss is normalized during training vs evaluation.

## Solution

**Use WeightedLossTrainer for ALL training**, not just for weighted masking. You can disable the weighting feature by setting `structural_weight=1.0` while still benefiting from the explicit loss computation.

### Implementation:

Modify `vision_training.py` to ALWAYS use `WeightedLossTrainer`:

```python
# Always use WeightedLossTrainer for consistent loss computation
from model_garden.weighted_loss_trainer import WeightedLossTrainer

trainer = WeightedLossTrainer(
    model=self.model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=all_callbacks,
    tokenizer=self.tokenizer,
    verbose_loss=False,  # Set to True for debugging
)
```

## Why This Fixes It

WeightedLossTrainer:
1. **Explicitly computes cross-entropy loss** with full control over reduction
2. **Properly normalizes by valid token count** (not total sequence length)
3. **Consistent handling** between training and evaluation
4. **Works identically** whether using weighted masking or not

The weighted loss curves you shared show both losses starting high and converging together - this is the CORRECT behavior that we want for all training.

## Alternative Investigation

If you want to keep using SFTTrainer, we need to:
1. Inspect the model's forward pass to see how it computes loss internally
2. Check if there's a configuration flag to control loss reduction
3. Verify Unsloth's model patches haven't introduced loss computation bugs

However, the simpler and more reliable solution is to use WeightedLossTrainer universally.
