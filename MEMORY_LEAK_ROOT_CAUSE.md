# Memory Leak Root Cause - FOUND!

## Problem
Vision training leaks ~1.2 GB/minute of system RAM, causing OOM in 15-20 minutes.

## Root Cause Identified
The HuggingFace `SFTTrainer` holds references to processed batch data (tensors from the collator) that prevent Python's garbage collector from releasing them.

## Evidence

### Test Results (`test_training_loop_simulation.py`)
```
WITHOUT cleanup: +14816.3 MB for 50 batches (+296.3 MB/batch)
WITH cleanup:    -100.2 MB for 50 batches (-2.0 MB/batch)
```

**Conclusion**: Explicitly deleting batch data and calling `gc.collect()` prevents the leak.

### Why Our MemoryCleanupCallback Doesn't Work
Our callback runs `gc.collect()` after every step, but the leak persists. This means:
1. The Trainer holds strong references to batch data
2. GC can't collect what's still referenced
3. We need to break those references explicitly

## Where Trainer Holds References

Likely culprits in HuggingFace Trainer:
1. **`prediction_step()` return values** - stored in evaluation metrics
2. **`_prepare_inputs()` outputs** - cached for next step
3. **Loss computation intermediates** - kept for logging
4. **Gradient computation** - autograd graph holds references

## Solution

We need a callback that accesses the trainer's internal state and explicitly clears batch-related attributes:

```python
class AggressiveBatchCleanupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Clear any cached inputs/outputs
        model = kwargs.get('model')
        if model is not None:
            # Clear gradients
            model.zero_grad(set_to_none=True)
            
        # Try to access and clear trainer's internal caches
        # (This requires investigating Trainer's private attributes)
        
        # Force GC
        gc.collect()
```

## Next Steps
1. Investigate Trainer's internal attributes to find what's holding references
2. Add explicit cleanup of those attributes in the callback
3. Test if this fixes the leak in actual training

## Alternative Solution
If we can't break the references from within the callback, we could:
1. Modify the collator to return minimal data
2. Use a custom training loop instead of SFTTrainer
3. Periodically restart the training process (workaround, not fix)
